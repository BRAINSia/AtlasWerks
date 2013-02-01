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


#include "LDMMShootingAtlasThreadCPU.h"

/* ################ LDMMShootingAtlasManagerCPU ################ */

unsigned int 
LDMMShootingAtlasManagerCPU::
CheckNumThreads(unsigned int requestedThreads, unsigned int nImages)
{
  unsigned int nThreads = requestedThreads;
  if(requestedThreads == 0){
    nThreads = nImages;
    LOGNODE(logINFO) << "Using " << nThreads << " threads.";
  }

  if(nThreads > nImages){
    LOGNODE(logWARNING) << "Number of threads (" << nThreads 
			<< ") greater than number of images (" << nImages
			<< "), using " << nImages << " threads" << std::endl;
    nThreads = nImages;
  }

  return nThreads;
}

/* ################ LDMMShootingAtlasDefDataCPU ################ */

LDMMShootingAtlasDefDataCPU::
LDMMShootingAtlasDefDataCPU(const RealImage *I, 
		    const RealImage *IHat,
		    Real weight,
		    const LDMMParam &param) :
  LDMMShootingDefData(IHat, I, param),
  mJacDet(new RealImage()),
  mWeight(weight)
{
  // IHat is I0, do not automatically rescale this
  this->ScaleI0(false);
}

LDMMShootingAtlasDefDataCPU::
~LDMMShootingAtlasDefDataCPU()
{
  delete mJacDet;
}

// static
unsigned int 
LDMMShootingAtlasThreadCPU::
CheckNumThreads(unsigned int requestedThreads, unsigned int nImages)
{
  unsigned int nThreads = requestedThreads;
  if(requestedThreads == 0){
    nThreads = nImages;
    LOGNODE(logINFO) << "Using " << nThreads << " threads.";
  }

  if(nThreads > nImages){
    LOGNODE(logWARNING) << "Number of threads (" << nThreads 
			<< ") greater than number of images (" << nImages
			<< "), using " << nImages << " threads" << std::endl;
    nThreads = nImages;
  }

  return nThreads;
}

void 
LDMMShootingAtlasDefDataCPU::
SetScaleLevel(const MultiscaleManager &scaleManager)
{
  // call parent function
  LDMMShootingDefData::SetScaleLevel(scaleManager);

  // scale jac. det.
  SizeType curSize = scaleManager.CurScaleSize();
  SizeType curSpacing = scaleManager.CurScaleSpacing();
  if(scaleManager.InitialScaleLevel()){
    mJacDet->resize(curSize);
    mJacDet->setOrigin(this->I0().getOrigin());
    mJacDet->setSpacing(curSpacing);
    mJacDet->fill(1.0);
  }else{
    scaleManager.UpsampleToLevel(*mJacDet, scaleManager.CurScaleLevel());
  }

}

/** ################ LDMMShootingAtlasThreadCPU ################ **/

LDMMShootingAtlasThreadCPU::
LDMMShootingAtlasThreadCPU(std::vector<DeformationDataType*> defData,
			   const ParamType &param,
			   AtlasBuilderInterface &builder,
			   RealImage *globalMean,
			   unsigned int nodeId, unsigned int nNodes,
			   unsigned int threadId, unsigned int nThreads,
			   unsigned int nTotalImages)
  : AtlasThread<LDMMShootingAtlasManagerCPU>(defData, param, builder, globalMean,
					     nodeId, nNodes, threadId, nThreads, nTotalImages),
    mNTimeSteps(param.NTimeSteps()),
    mThreadMean(NULL),
    mJacDetSum(NULL),
    mThreadEnergy(NULL)
{
  mThreadMean = new RealImage();
  mJacDetSum = new RealImage();

  if(mParam.WriteVelocityFields()){
    throw AtlasWerksException(__FILE__,__LINE__,"Writing of velocity fields not supported for shooting optimization");
  }
  
}

LDMMShootingAtlasThreadCPU::
~LDMMShootingAtlasThreadCPU()
{
  delete mThreadMean;
  delete mJacDetSum;
}

void 
LDMMShootingAtlasThreadCPU::
InitThread()
{
  if(mIterator) delete mIterator;
  mIterator = new IteratorType(mImSize, mImOrigin, mImSpacing, mNTimeSteps, true);
}

void 
LDMMShootingAtlasThreadCPU::
LoadInitData()
{
  if(mParam.InputVFieldFormat().size() > 0){
    throw AtlasWerksException(__FILE__,__LINE__,"Error, cannot initialize LDMM shooting from VFields");
  }
  
  // TODO: load VFields
  if(mParam.InputMeanImage().size() > 0){
    throw AtlasWerksException(__FILE__,__LINE__,"Error, LDMM Shooting Alpha initialization not implemented yet");
  }
  if(mParam.InputAlpha0Format().size() > 0){
    throw AtlasWerksException(__FILE__,__LINE__,"Error, LDMM Shooting Alpha initialization not implemented yet");
  }
}

void
LDMMShootingAtlasThreadCPU::
SetScaleLevel(const MultiscaleManagerType &scaleManager)
{

  AtlasThread<LDMMShootingAtlasManagerCPU>::SetScaleLevel(scaleManager);

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
LDMMShootingAtlasThreadCPU::
ComputeThreadMean()
{
  // create the weighted average for all images on this node
  mThreadMean->fill(0.0);
  mJacDetSum->fill(0.0);
  mThreadEnergy->Clear();
  Real *averageData = mThreadMean->getDataPointer();
  Real *jacSumData = mJacDetSum->getDataPointer();
  unsigned int size = mThreadMean->getNumElements();
  RealImage defIm(mImSize, mImOrigin, mImSpacing);
  for(unsigned int imIdx=0; imIdx<mNImages; imIdx++){
    // get the weight
    Real curWeight = mDeformationData[imIdx]->Weight();
    // generate the deformed image
    mDeformationData[imIdx]->GetI1At0(defIm);
    const Real *imageData = defIm.getDataPointer();
    mIterator->ComputeJacDet(*mDeformationData[imIdx], mDeformationData[imIdx]->JacDet());
    // get the jac det for this deformation
    const Real *jacData = mDeformationData[imIdx]->JacDet().getDataPointer();
    // update weighted image and jac det sums
    if(mParam.JacobianScale()){
      for(unsigned int j = 0; j < size; j++){
	averageData[j] += imageData[j] * jacData[j] * curWeight;
	jacSumData[j] += jacData[j] * curWeight;
      }
    }else{
      for(unsigned int j = 0; j < size; j++){
	averageData[j] += imageData[j] * curWeight;
      }
    }
    // sum energies
    if(mDeformationData[imIdx]->HasEnergy()){
      const LDMMEnergy &lastEnergy = 
	dynamic_cast<const LDMMEnergy&>(mDeformationData[imIdx]->LastEnergy());
      (*mThreadEnergy) += lastEnergy*curWeight;
    }
       
    
  }
}

