/* ================================================================
 *
 * AtlasWerks Project
 *
 * Copyright (c) Sarang C. Joshi, Bradley C. Davis, J. Samuel Preston,
 * Linh K. Ha. All rights reserved.  See Copyright.txt or for details.
 *
 * This software is distributed WITHOUT ANY WARRANTY; without even the
 * implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 * PURPOSE.  See the above copyright notice for more information.
 *
 * ================================================================ */


#include "LDMMShootingAtlasThreadGPU.h"

#include "cudaSplat.h"

/* ################ LDMMAtlasManagerGPU ################ */

unsigned int 
LDMMShootingAtlasManagerGPU::
CheckNumThreads(unsigned int requestedThreads, unsigned int nImages)
{

  // determine the number of GPUs
  unsigned int systemGPUs = getNumberOfCapableCUDADevices();
  unsigned int nThreads = requestedThreads;
  if(requestedThreads == 0){
    nThreads = systemGPUs;
  }else if(requestedThreads > systemGPUs){
    LOGNODE(logWARNING) << "Cannot use " << requestedThreads << " GPUs, only " << systemGPUs << " available";
    nThreads = systemGPUs;
  }

  if(nThreads > nImages){
    LOGNODE(logWARNING) << "Number of threads (" << nThreads 
			<< ") greater than number of images (" << nImages
			<< "), using " << nImages << " threads" << std::endl;
    nThreads = nImages;
  }

  return nThreads;
}

/* ################ LDMMShootingAtlasDefDataGPU ################ */

LDMMShootingAtlasDefDataGPU::
LDMMShootingAtlasDefDataGPU(const RealImage *I, 
		    const RealImage *IHat,
		    Real weight,
		    const LDMMParam &param) :
  LDMMShootingDefDataGPU(IHat, I, param),
  mJacDet(new RealImage()),
  mWeight(weight)
{
  // IHat is I0, do not automatically rescale this
  this->ScaleI0(false);
}

LDMMShootingAtlasDefDataGPU::
~LDMMShootingAtlasDefDataGPU()
{
  delete mJacDet;
}

RealImage&
LDMMShootingAtlasDefDataGPU::
JacDet(){
  unsigned int nVox = mJacDet->getSize().productOfElements();
  if(mDeviceDataInitialized){
    copyArrayFromDevice<float>(mJacDet->getDataPointer(), mdJacDet, nVox);
  }
  return *mJacDet;
}

void 
LDMMShootingAtlasDefDataGPU::
InitializeWarp()
{
  LDMMShootingDefDataGPU::InitializeWarp();
  unsigned int memSize = sizeof(Real) * mI1Orig->getSize().productOfElements();
  allocateDeviceArray((void**)&mdJacDet, memSize);;
}

void 
LDMMShootingAtlasDefDataGPU::
FinalizeWarp()
{
  LDMMShootingDefDataGPU::FinalizeWarp();
  cudaSafeDelete(mdJacDet);
}

void 
LDMMShootingAtlasDefDataGPU::
SetScaleLevel(const MultiscaleManager &scaleManager)
{
  // call parent function
  LDMMShootingDefDataGPU::SetScaleLevel(scaleManager);

  if(scaleManager.InitialScaleLevel()){
    mJacDet->resize(mCurSize);
    mJacDet->setOrigin(this->I0().getOrigin());
    mJacDet->setSpacing(mCurSpacing);
    mJacDet->fill(1.0);
  }else{
    scaleManager.UpsampleToLevel(*mJacDet, scaleManager.CurScaleLevel());
  }
  copyArrayToDevice(mdJacDet, mJacDet->getDataPointer(), mCurVox);
}

/** ################ LDMMShootingAtlasThreadGPU ################ **/

LDMMShootingAtlasThreadGPU::
LDMMShootingAtlasThreadGPU(std::vector<DeformationDataType*> defData,
			   const ParamType &param,
			   AtlasBuilderInterface &builder,
			   RealImage *globalMean,
			   unsigned int nodeId, unsigned int nNodes,
			   unsigned int threadId, unsigned int nThreads,
			   unsigned int nTotalImages)
  : AtlasThread<LDMMShootingAtlasManagerGPU>(defData, param, builder, globalMean, 
					     nodeId, nNodes, threadId, nThreads, nTotalImages),
    mNTimeSteps(param.NTimeSteps()),
    mThreadMean(NULL),
    mJacDetSum(NULL),
    mThreadEnergy(NULL)
{
  mNVox = mImSize.productOfElements();
  
  mThreadMean = new RealImage();
  mJacDetSum = new RealImage();

  if(mParam.WriteVelocityFields()){
    throw AtlasWerksException(__FILE__,__LINE__,"Writing of velocity fields not supported for shooting optimization");
  }
}

LDMMShootingAtlasThreadGPU::
~LDMMShootingAtlasThreadGPU()
{
  delete mThreadMean;
  delete mJacDetSum;
}

void
LDMMShootingAtlasThreadGPU::
InitDeviceData()
{
  // Set the device contex
  CUDAUtilities::SetCUDADevice(mThreadId);
  // ensure device supports CUDA capability version 1.2
  CUDAUtilities::AssertMinCUDACapabilityVersion(1,2);

  if(mIterator) delete mIterator;
  mIterator = new IteratorType(mImSize, mImOrigin, mImSpacing, mNTimeSteps, true);

  unsigned int memSize = mNVox*sizeof(Real);
  allocateDeviceArray((void**)&mdThreadMean, memSize);
  allocateDeviceArray((void**)&mdJacDetSum, memSize);
  allocateDeviceArray((void**)&mdScratchI, memSize);
  for (unsigned int imIdx=0; imIdx < mNImages; ++imIdx){
    mDeformationData[imIdx]->InitializeWarp();
  }
}

void 
LDMMShootingAtlasThreadGPU::
FreeDeviceData()
{
  cudaSafeDelete(mdThreadMean);
  cudaSafeDelete(mdJacDetSum);
  cudaSafeDelete(mdScratchI);
  for (unsigned int imIdx=0; imIdx < mNImages; ++imIdx){
    mDeformationData[imIdx]->FinalizeWarp();
  }
}

void
LDMMShootingAtlasThreadGPU::
InitThread()
{
  this->InitDeviceData();

  mInitialScaleLevel = mParam.StartScaleLevel();
  mInitialIter = mParam.StartIter();
}

void 
LDMMShootingAtlasThreadGPU::
LoadInitData()
{
  if(mParam.InputVFieldFormat().size() > 0){
    throw AtlasWerksException(__FILE__,__LINE__,"Error, cannot initialize shooting warp with VFields");
  }
  
  if(mParam.InputAlpha0Format().size() > 0){
    if(mParam.InputMeanImage().size() == 0){
      throw AtlasWerksException(__FILE__,__LINE__,"Error, cannot initialize shooting warp without mean image");
    }
    
    // load the mean
    LOGNODETHREAD(logDEBUG2) << "Loading mean image " << mParam.InputMeanImage().c_str();
    ApplicationUtils::LoadImageITK(mParam.InputMeanImage().c_str(), *mGlobalMean);
    LOGNODETHREAD(logDEBUG2) << "Updating global mean";
    this->GlobalMeanUpdated();
    
    // load the alphas
    for(unsigned int imIdx=0; imIdx < mNImages; ++imIdx){
      DeformationDataType &curDefData = *mDeformationData[imIdx];
      std::string alpha0Name = 
	StringUtils::strPrintf(mParam.InputAlpha0Format().c_str(), 
			       curDefData.GetName().c_str());
      LOGNODETHREAD(logDEBUG2) << "Loading alpha0 " << alpha0Name;
      RealImage alpha0;
      ApplicationUtils::LoadImageITK(alpha0Name.c_str(), alpha0);
      
      LOGNODETHREAD(logDEBUG2) << "Copying alpha0 to deformation data " << alpha0Name;
      curDefData.Alpha0() = alpha0;
      copyArrayToDevice(curDefData.dAlpha0(), alpha0.getDataPointer(), mCurVox);
    }

  }
  
}

void
LDMMShootingAtlasThreadGPU::
FinishThread()
{
  this->FreeDeviceData();
}

void
LDMMShootingAtlasThreadGPU::
SetScaleLevel(const MultiscaleManagerType &scaleManager)
{
  AtlasThread<LDMMShootingAtlasManagerGPU>::SetScaleLevel(scaleManager);

  mCurVox = mCurSize.productOfElements();

  for(unsigned int i=0;i<mNImages;i++){
    mDeformationData[i]->StepSize(mCurScaleParam->StepSize());
  }

  mThreadMean->resize(mCurSize);
  mThreadMean->setOrigin(mImOrigin);
  mThreadMean->setSpacing(mCurSpacing);
  mJacDetSum->resize(mCurSize);
  mJacDetSum->setOrigin(mImOrigin);
  mJacDetSum->setSpacing(mCurSpacing);

  if(mThreadEnergy) delete mThreadEnergy;
  mThreadEnergy = new LDMMEnergy(mNTimeSteps, mCurScaleParam->Sigma());

  // still need to generate the mean at this scale level...
}

void
LDMMShootingAtlasThreadGPU::
ComputeThreadMean()
{
  CUDAUtilities::CheckCUDAError(__FILE__,__LINE__);

  // create the weighted average for all images on this node
  cplVectorOpers::SetMem(mdThreadMean, 0.0f, mCurVox);
  cplVectorOpers::SetMem(mdJacDetSum, 0.0f, mCurVox);
  mThreadEnergy->Clear();
  for (unsigned int imIdx=0; imIdx < mNImages; ++imIdx){

    LDMMShootingAtlasDefDataGPU &curData = *mDeformationData[imIdx];

    this->ComputeJacDet(curData);

    // TEST
    float imSum = CUDAUtilities::DeviceImageSum(curData.dI1(), mCurVox);
    LOGNODETHREAD(logDEBUG2) << "image " << imIdx << " sum is " << imSum;
    // END TEST

    cudaHField3DUtils::apply(mdScratchI, curData.dI1(), curData.dPhi0T(), mCurSize);

    // TEST
    imSum = CUDAUtilities::DeviceImageSum(mdScratchI, mCurVox);
    LOGNODETHREAD(logDEBUG2) << "deformed image " << imIdx << " sum is " << imSum;
    imSum = CUDAUtilities::DeviceImageSum(curData.dJacDet(), mCurVox);
    LOGNODETHREAD(logDEBUG2) << "jac. det. " << imIdx << " sum is " << imSum;
    LOGNODETHREAD(logDEBUG2) << "weight is " << curData.Weight();
    // END TEST

    if(mParam.JacobianScale()){
      cplVectorOpers::Add_MulMulC_I(mdThreadMean, mdScratchI, curData.dJacDet(), curData.Weight(), mCurVox);
      
      cplVectorOpers::Add_MulC_I(mdJacDetSum, curData.dJacDet(), curData.Weight(), mCurVox);
    }else{
      cplVectorOpers::Add_MulC_I(mdThreadMean, mdScratchI, curData.Weight(), mCurVox);
    }
    //cplVectorOpers::AddScaledArray(mdJacDetSum, curData.Weight(), curData.dJacDet(), mCurVox);

    if(mDeformationData[imIdx]->HasEnergy()){
      const LDMMEnergy &curEnergy = 
	dynamic_cast<const LDMMEnergy&>(mDeformationData[imIdx]->LastEnergy());
      // sum energies
      (*mThreadEnergy) += curEnergy * curData.Weight();
      // test for NaN
      if(curEnergy.GetEnergy() != curEnergy.GetEnergy()){
	LOGNODETHREAD(logERROR) << "Error, NaN encountered in image " << imIdx << " energy.";
      }
    }

  }
  
  // TEST
  float meanSum = CUDAUtilities::DeviceImageSum(mdThreadMean, mCurVox);
  LOGNODETHREAD(logDEBUG2) << "mean sum (on device) is " << meanSum;
  meanSum = CUDAUtilities::DeviceImageSum(mdJacDetSum, mCurVox);
  LOGNODETHREAD(logDEBUG2) << "mean sum (on device) is " << meanSum;
  // END TEST

  // copy to host
  copyArrayFromDevice(mJacDetSum->getDataPointer(), mdJacDetSum, mCurVox);
  copyArrayFromDevice(mThreadMean->getDataPointer(), mdThreadMean, mCurVox);
  CUDAUtilities::CheckCUDAError(__FILE__,__LINE__);

}

void 
LDMMShootingAtlasThreadGPU::
GlobalMeanUpdated()
{
  for (unsigned int imIdx=0; imIdx < mNImages; ++imIdx){
    // copy the results back to the GPU
    copyArrayToDevice(mDeformationData[imIdx]->dI0(), mGlobalMean->getDataPointer(), mCurVox);
  }
}

void 
LDMMShootingAtlasThreadGPU::
ComputeJacDet(LDMMShootingAtlasDefDataGPU &data)
{
  if(data.I0().getSize() != mCurSize){
    throw AtlasWerksException(__FILE__,__LINE__,"Error, jacobian determinant "
			   "calculation must be performed on images of the "
			   "same size as the iterator");
  }
  
  cplVectorOpers::SetMem(mdScratchI, 1.0f, mCurVox);
  cplSplat3DH(data.dJacDet(), mdScratchI, data.dPhiT0(), mCurSize);
  // cplSplatingHFieldAtomicSigned(data.dJacDet(), mdScratchI, 
  //                               data.dPhiT0().x, data.dPhiT0().y, data.dPhiT0().z,
  //                               mCurSize.x, mCurSize.y, mCurSize.z);
}

