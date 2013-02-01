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

#include "LDMMAtlasThreadGPU.h"
#include "cudaSplat.h"
#include <cudaDownSample.h>
#include <cutil_inline.h>


/* ################ LDMMAtlasManagerGPU ################ */

unsigned int 
LDMMAtlasManagerGPU::
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


/* ################ LDMMAtlasDefDataGPU ################ */

LDMMAtlasDefDataGPU::
LDMMAtlasDefDataGPU(const RealImage *I, 
		    const RealImage *IHat,
		    Real weight,
		    const LDMMParam &param) :
  LDMMDeformationDataGPU(IHat, I, param),
  mJacDet(new RealImage()),
  mWeight(weight)
{
  // IHat is I0, do not automatically rescale this
  this->ScaleI0(false);
}

LDMMAtlasDefDataGPU::
~LDMMAtlasDefDataGPU()
{
  delete mJacDet;
}

RealImage&
LDMMAtlasDefDataGPU::
JacDet(){
  unsigned int nVox = mJacDet->getSize().productOfElements();
  if(mDeviceDataInitialized){
    copyArrayFromDevice<float>(mJacDet->getDataPointer(), mdJacDet, nVox);
  }
  return *mJacDet;
}

void 
LDMMAtlasDefDataGPU::
InitializeWarp()
{
  LDMMDeformationDataGPU::InitializeWarp();
  unsigned int memSize = sizeof(Real) * mI1Orig->getSize().productOfElements();
  allocateDeviceArray((void**)&mdJacDet, memSize);;
}

void 
LDMMAtlasDefDataGPU::
FinalizeWarp()
{
  LDMMDeformationDataGPU::FinalizeWarp();
  cudaSafeDelete(mdJacDet);
}

void 
LDMMAtlasDefDataGPU::
SetScaleLevel(const MultiscaleManager &scaleManager)
{
  // call parent function
  LDMMDeformationDataGPU::SetScaleLevel(scaleManager);

  // scale jac. det.
  if(scaleManager.InitialScaleLevel()){
    mJacDet->resize(mCurSize);
    mJacDet->setOrigin(mImOrigin);
    mJacDet->setSpacing(mCurSpacing);
    mJacDet->fill(1.0);
  }else{
    scaleManager.UpsampleToLevel(*mJacDet, scaleManager.CurScaleLevel());
  }
}

/* ################ LDMMAtlasThreadGPU ################ */

LDMMAtlasThreadGPU::
LDMMAtlasThreadGPU(std::vector<DeformationDataType*> defData,
		   const ParamType &param,
		   AtlasBuilderInterface &builder,
		   RealImage *globalMean,
		   unsigned int nodeId, unsigned int nNodes,
		   unsigned int threadId, unsigned int nThreads,
		   unsigned int nTotalImages)
  : AtlasThread<LDMMAtlasManagerGPU>(defData, param, builder, globalMean, 
				     nodeId, nNodes, threadId, nThreads, nTotalImages),
    mNTimeSteps(param.NTimeSteps()),
    mThreadMean(NULL),
    mJacDetSum(NULL),
    mThreadEnergy(NULL),
    mdThreadMean(NULL),
    mdJacDetSum(NULL),
    mdScratchI(NULL),
    mDist(NULL),
    mTMWeights(NULL)
{
  mNVox = mImSize.productOfElements();
  
  mThreadMean = new RealImage();
  mJacDetSum = new RealImage();
  mDist = new Real[mNImages];
  mTMWeights = new Real[mNImages];
  memset(mDist, 0, mNImages*sizeof(Real));
  memset(mTMWeights, 0, mNImages*sizeof(Real));
}

LDMMAtlasThreadGPU::
~LDMMAtlasThreadGPU()
{
  delete mThreadMean;
  delete mJacDetSum;
  delete [] mDist;
  delete [] mTMWeights;
}

void
LDMMAtlasThreadGPU::
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
LDMMAtlasThreadGPU::
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
LDMMAtlasThreadGPU::
InitThread()
{
  this->InitDeviceData();
}

void 
LDMMAtlasThreadGPU::
LoadInitData()
{
  // TODO: load VFields
  if(mParam.InputVFieldFormat().size() > 0){
    for(unsigned int imIdx=0; imIdx < mNImages; ++imIdx){
      DeformationDataType &curDefData = *mDeformationData[imIdx];
      std::string velPattern = 
	StringUtils::strPrintf(mParam.InputVFieldFormat().c_str(), 
			       curDefData.GetName().c_str());
      for(unsigned int t=0; t < mNTimeSteps; ++t){
	
	std::string velName = 
	  StringUtils::strPrintf(velPattern.c_str(), t);
	
	LOGNODETHREAD(logDEBUG2) << "Loading vector field " << velName;
	VectorField v;
	ApplicationUtils::LoadHFieldITK(velName.c_str(), v);
	
	LOGNODETHREAD(logDEBUG2) << "Copying v to deformation data";
	curDefData.v(t) = v;
	CUDAUtilities::CopyVectorFieldToDevice(v, curDefData.dV(t), true);
      }
    }
   
    // now we need to compute the mean from these velocity fields
    
    LOGNODETHREAD(logDEBUG2) << "Computing the mean from newly loaded vfields";

    this->ComputeThreadMean();
    mBuilder.ComputeMean();
    this->GlobalMeanUpdated();
    
  }
  
  if(mParam.InputMeanImage().size() > 0){
    throw AtlasWerksException(__FILE__,__LINE__,"Error, cannot initialize LDMM relaxation from alpha0s");
  }
  if(mParam.InputAlpha0Format().size() > 0){
    throw AtlasWerksException(__FILE__,__LINE__,"Error, cannot initialize LDMM relaxation from alpha0s");
  }
}


void
LDMMAtlasThreadGPU::
FinishThread()
{
  if(mParam.WriteAlphas() || mParam.WriteAlpha0s()){
    this->ComputeAtlasAlphas();
  }
  this->FreeDeviceData();
}

void
LDMMAtlasThreadGPU::
BeginImageIteration(int iteration, int imageIndex)
{
  if(iteration == 0 && mCurScaleParam->UseAdaptiveStepSize()){
    LOGNODETHREAD(logINFO) << "Calculating step size";
    mIterator->UpdateStepSizeNextIteration();
    mIterator->Iterate(*mDeformationData[imageIndex]);
    LOGNODETHREAD(logINFO) << "Done calculating step size";
  }
}

void
LDMMAtlasThreadGPU::
FinishImageIteration(int iteration, int imageIndex)
{
  // test for increasing energy
  if(iteration > 0){
    
    EnergyHistory &hist = mDeformationData[imageIndex]->GetEnergyHistory();
    Real energyDiff = hist.LastEnergyChange();
    if(energyDiff > 0.f){
      LOGNODETHREAD(logWARNING) << "Increasing energy detected: image " << imageIndex 
				<< " (" << mDeformationData[imageIndex]->GetName()
				<< "), increase = " << energyDiff;
    }
  }

  // calculate distance
  if(mDeformationData[imageIndex]->HasEnergy()){
    mDist[imageIndex] = sqrt(mDeformationData[imageIndex]->LastEnergy().GetEnergy());
  }else{
    mDist[imageIndex] = 0.f;
  }
}

void
LDMMAtlasThreadGPU::
SetScaleLevel(const MultiscaleManagerType &scaleManager)
{
  AtlasThread<LDMMAtlasManagerGPU>::SetScaleLevel(scaleManager);

  mCurVox = mCurSize.productOfElements();
  
  for(unsigned int i=0;i<mNImages;i++){
    mDeformationData[i]->StepSize(mCurScaleParam->StepSize());
  }
  
  mThreadMean->resize(mCurSize);
  mThreadMean->setSpacing(mCurSpacing);
  mThreadMean->setOrigin(mImOrigin);
  mJacDetSum->resize(mCurSize);
  mJacDetSum->setSpacing(mCurSpacing);
  mJacDetSum->setOrigin(mImOrigin);
  
  if(mThreadEnergy) delete mThreadEnergy;
  mThreadEnergy = new LDMMEnergy(mNTimeSteps, mCurScaleParam->Sigma());

  // still need to generate the mean at this scale level...
}

void
LDMMAtlasThreadGPU::
ComputeThreadMean()
{
  if(mParam.TrimmedMeanSize() > 0){
    // sychronizes threads
    mBuilder.ComputeWeights();
  }
  
  // create the weighted average for all images on this node
  cplVectorOpers::SetMem(mdThreadMean, 0.f, mCurVox);
  cplVectorOpers::SetMem(mdJacDetSum, 0.f, mCurVox);
  mThreadEnergy->Clear();
  //TEST
  LOGNODETHREAD(logDEBUG2) << "Starting to iterate over thread images";
  //END TEST
  for (unsigned int imIdx=0; imIdx < mNImages; ++imIdx){

    // getn the data
    LDMMAtlasDefDataGPU &curData = *mDeformationData[imIdx];
    Real imWeight = mDeformationData[imIdx]->Weight();
    if(mParam.ComputeMedian()){
      Real d = mDist[imIdx];
      // on first mean calculation energy is zero, don't reweight (just compute mean)
      if(d == 0) d = 1.f;
      d = 1.0/d;
      imWeight *= d;
      LOGNODETHREAD(logDEBUG2) << "individual InvDistSum: " << d;
    }

    if(mParam.TrimmedMeanSize()){
      imWeight = mTMWeights[imIdx];
    }

    // deform I1 to mean image space
    curData.GetDefTTo1(mIterator->dH(),mIterator->dScratchV(),0);
    cudaHField3DUtils::apply(mdScratchI, curData.dI1(), mIterator->dH(), mCurSize);

    // splat I1 back to mean image space
    //curData.GetDefTTo0(mIterator->dH(),mIterator->dScratchV(),mNTimeSteps);
    //cplvSplatingHFieldAtomicSigned(mdScratchI, curData.dI1(), 
    //  mIterator->dH().x, mIterator->dH().y, mIterator->dH().z,
    //  mCurSize.x, mCurSize.y, mCurSize.z);

    // this is out of place (and overwrites mdScratchI)
    this->ComputeJacDet(*mDeformationData[imIdx]);

    if(mParam.JacobianScale()){
      cplVectorOpers::Add_MulMulC_I(mdThreadMean, mdScratchI, curData.dJacDet(), imWeight, mCurVox);
      cplVectorOpers::Add_MulC_I(mdJacDetSum, curData.dJacDet(), imWeight, mCurVox);
    }else{
      cplVectorOpers::Add_MulC_I(mdThreadMean, mdScratchI, imWeight, mCurVox);
    }

    // sum energies
    if(mDeformationData[imIdx]->HasEnergy()){
      // get the energy
      const LDMMEnergy &curEnergy = 
	dynamic_cast<const LDMMEnergy&>(mDeformationData[imIdx]->LastEnergy());
      
      //(*mThreadEnergy) += curEnergy * curData.Weight();
      (*mThreadEnergy) += curEnergy * imWeight;
    }
    if(mDeformationData[imIdx]->SaveDefToMean()){
      RealImage *defToMean = mDeformationData[imIdx]->GetDefToMean();
      copyArrayFromDevice(defToMean->getDataPointer(), mdScratchI, mCurVox);
    }
    
  } // end loop over images

  if(mThreadEnergy->GetEnergy() != mThreadEnergy->GetEnergy()){
    LOGNODETHREAD(logERROR) << "Energy has NaN: " << *mThreadEnergy;
  }

  // copy to host
  copyArrayFromDevice(mJacDetSum->getDataPointer(), mdJacDetSum, mCurVox);
  copyArrayFromDevice(mThreadMean->getDataPointer(), mdThreadMean, mCurVox);

  LOGNODETHREAD(logDEBUG2) << "Done computing thread mean";
}

void 
LDMMAtlasThreadGPU::
GlobalMeanUpdated()
{
  for (unsigned int imIdx=0; imIdx < mNImages; ++imIdx){
    // copy the results back to the GPU
    copyArrayToDevice(mDeformationData[imIdx]->dI0(), mGlobalMean->getDataPointer(), mCurVox);
  }
}

void 
LDMMAtlasThreadGPU::
ComputeJacDet(LDMMAtlasDefDataGPU &data)
{
  if(data.I0().getSize() != mCurSize){
    throw AtlasWerksException(__FILE__,__LINE__,"Error, jacobian determinant "
			   "calculation must be performed on images of the "
			   "same size as the iterator");
  }
  LOGNODETHREAD(logDEBUG2) << "LDMMAtlasThread Computing jac. det";
  mIterator->ComputeJacDet(data.dV(), data.dJacDet());
  LOGNODETHREAD(logDEBUG2) << "LDMMAtlasThread Done computing jac. det";
}

void
LDMMAtlasThreadGPU::
ComputeAtlasAlphas()
{
  LOGNODETHREAD(logDEBUG) << "Computing alphas";
  for(unsigned int imIdx=0; imIdx < mNImages; ++imIdx){
    LOGNODETHREAD(logDEBUG) << "Computing alphas for image " << imIdx << "...";
    // compute the alphas
    mDeformationData[imIdx]->ComputeAlphas(true);
    mIterator->Iterate(*mDeformationData[imIdx], true);
  }
}

