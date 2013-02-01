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


#include "GreedyAtlasThreadGPU.h"

/* ################ GreedyAtlasManagerGPU ################ */

unsigned int 
GreedyAtlasManagerGPU::
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

/* ################ GreedyAtlasDefDataGPU ################ */

GreedyAtlasDefDataGPU::
GreedyAtlasDefDataGPU(const RealImage *I, 
		      const RealImage *IHat,
		      Real weight,
		      const GreedyParam &param) :
  GreedyDeformationDataGPU(I, IHat, param),
  mWeight(weight)
{
  // IHat is I0, do not automatically rescale this
  this->ScaleI1(false);
}

/* ################ GreedyAtlasThreadGPU ################ */

GreedyAtlasThreadGPU::
GreedyAtlasThreadGPU(std::vector<DeformationDataType*> defData,
		   const ParamType &param,
		   AtlasBuilderInterface &builder,
		   RealImage *globalMean,
		   unsigned int nodeId, unsigned int nNodes,
		   unsigned int threadId, unsigned int nThreads,
		   unsigned int nTotalImages)
  : AtlasThread<GreedyAtlasManagerGPU>(defData, param, builder, globalMean, 
				     nodeId, nNodes, threadId, nThreads, nTotalImages),
    mThreadMean(NULL),
    mdThreadMean(NULL),
    mdScratchI(NULL)
{
  mNVox = mImSize.productOfElements();

  mThreadMean = new RealImage();
}

GreedyAtlasThreadGPU::
~GreedyAtlasThreadGPU()
{
  delete mThreadMean;
}

void
GreedyAtlasThreadGPU::
InitDeviceData()
{
  // Set the device contex
  CUDAUtilities::SetCUDADevice(mThreadId);

  if(mIterator) delete mIterator;
  mIterator = new IteratorType(mImSize, mImOrigin, mImSpacing, true);

  unsigned int memSize = mNVox*sizeof(Real);
  allocateDeviceArray((void**)&mdThreadMean, memSize);
  allocateDeviceArray((void**)&mdScratchI, memSize);
  for (unsigned int imIdx=0; imIdx < mNImages; ++imIdx){
    mDeformationData[imIdx]->InitializeWarp();
  }
}

void 
GreedyAtlasThreadGPU::
FreeDeviceData()
{
  cudaSafeDelete(mdThreadMean);
  cudaSafeDelete(mdScratchI);
  for (unsigned int imIdx=0; imIdx < mNImages; ++imIdx){
    mDeformationData[imIdx]->FinalizeWarp();
  }
}

void 
GreedyAtlasThreadGPU::
InitThread()
{
  this->InitDeviceData();
}

void 
GreedyAtlasThreadGPU::
FinishThread()
{
  this->FreeDeviceData();
}

void 
GreedyAtlasThreadGPU::
BeginImageIteration(int iteration, int imageIndex){
  if(iteration == 0){
    // have iterator update 
    mIterator->UpdateStepSizeNextIteration();
  }
}

void 
GreedyAtlasThreadGPU::
FinishImageIteration(int iteration, int imageIndex){
  if(iteration == 0){
    LOGNODE(logINFO) << "Step Size is " << mDeformationData[imageIndex]->StepSize();
  }
}

	
void
GreedyAtlasThreadGPU::
SetScaleLevel(const MultiscaleManagerType &scaleManager)
{
  AtlasThread<GreedyAtlasManagerGPU>::SetScaleLevel(scaleManager);

  mCurVox = mCurSize.productOfElements();

  mThreadMean->resize(mCurSize);
  mThreadMean->setSpacing(mCurSpacing);
  mThreadMean->setOrigin(mImOrigin);
  
  // still need to generate the mean at this scale level...
}

void
GreedyAtlasThreadGPU::
ComputeThreadMean()
{
  // create the weighted average for all images on this node
  cplVectorOpers::SetMem(mdThreadMean, 0.f, mCurVox);
  mThreadEnergy.Clear();
  for (unsigned int imIdx=0; imIdx < mNImages; ++imIdx){
    GreedyAtlasDefDataGPU &curData = *mDeformationData[imIdx];
    // deform I1 to mean image space
    
    cudaHField3DUtils::apply(mdScratchI, curData.dI0(), curData.dDef1To0(), mCurSize);

    // splat I1 back to mean image space
    //curData.ComputeDefTTo0(mIterator->dH(),mIterator->dScratchV(),mNTimeSteps);
    //cplvSplatingHFieldAtomicSigned(mdScratchI, curData.dI1(), 
    //  mIterator->dH().x, mIterator->dH().y, mIterator->dH().z,
    //  mCurSize.x, mCurSize.y, mCurSize.z);

    cplVectorOpers::Add_MulC_I(mdThreadMean, mdScratchI, curData.Weight(), mCurVox);

    // sum energies
    if(mDeformationData[imIdx]->HasEnergy()){
      mThreadEnergy += mDeformationData[imIdx]->LastEnergy() * curData.Weight();
    }
    if(mDeformationData[imIdx]->SaveDefToMean()){
      RealImage *defToMean = mDeformationData[imIdx]->GetDefToMean();
      copyArrayFromDevice(defToMean->getDataPointer(), mdScratchI, mCurVox);
    }
  }

  // copy to host
  copyArrayFromDevice(mThreadMean->getDataPointer(), mdThreadMean, mCurVox);
}

void 
GreedyAtlasThreadGPU::
GlobalMeanUpdated()
{
  for (unsigned int imIdx=0; imIdx < mNImages; ++imIdx){
    // copy the results back to the GPU
    copyArrayToDevice(mDeformationData[imIdx]->dI1(), 
		      mDeformationData[imIdx]->I1().getDataPointer(), 
		      mCurVox);
  }
}

