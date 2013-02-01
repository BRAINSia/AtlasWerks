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


#include "GreedyAtlasThreadCPU.h"

/* ################ GreedyAtlasManagerCPU ################ */

unsigned int 
GreedyAtlasManagerCPU::
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

/* ################ GreedyAtlasDefDataCPU ################ */

GreedyAtlasDefDataCPU::
GreedyAtlasDefDataCPU(const RealImage *I, 
		      const RealImage *IHat,
		      Real weight,
		      const GreedyParam &param) :
  GreedyDeformationData(I, IHat, param),
  mWeight(weight)
{
  // IHat is I0, do not automatically rescale this
  this->ScaleI1(false);
}

/* ################ GreedyAtlasThreadCPU ################ */

GreedyAtlasThreadCPU::
GreedyAtlasThreadCPU(std::vector<DeformationDataType*> defData,
		   const ParamType &param,
		   AtlasBuilderInterface &builder,
		   RealImage *globalMean,
		   unsigned int nodeId, unsigned int nNodes,
		   unsigned int threadId, unsigned int nThreads,
		   unsigned int nTotalImages)
  : AtlasThread<GreedyAtlasManagerCPU>(defData, param, builder, globalMean, 
				     nodeId, nNodes, threadId, nThreads, nTotalImages),
    mThreadMean(NULL)
{
  mThreadMean = new RealImage();
}

GreedyAtlasThreadCPU::
~GreedyAtlasThreadCPU()
{
  delete mThreadMean;
}

void 
GreedyAtlasThreadCPU::
InitThread()
{
  if(mIterator) delete mIterator;
  mIterator = new IteratorType(mImSize, mImOrigin, mImSpacing, true);
}

void 
GreedyAtlasThreadCPU::
BeginImageIteration(int iteration, int imageIndex){
  if(iteration == 0){
    // have iterator update 
    mIterator->UpdateStepSizeNextIteration();
  }
}

void 
GreedyAtlasThreadCPU::
FinishImageIteration(int iteration, int imageIndex){
  if(iteration == 0){
    LOGNODE(logINFO) << "Step Size is " << mDeformationData[imageIndex]->StepSize();
  }
}

	
void
GreedyAtlasThreadCPU::
SetScaleLevel(const MultiscaleManagerType &scaleManager)
{
  AtlasThread<GreedyAtlasManagerCPU>::SetScaleLevel(scaleManager);

  mThreadMean->resize(mCurSize);
  mThreadMean->setSpacing(mCurSpacing);
  mThreadMean->setOrigin(mImOrigin);
  
  // still need to generate the mean at this scale level...
}

void
GreedyAtlasThreadCPU::
ComputeThreadMean()
{
  // create the weighted average for all images on this node
  mThreadMean->fill(0.0);
  mThreadEnergy.Clear();
  Real *averageData = mThreadMean->getDataPointer();
  unsigned int size = mThreadMean->getNumElements();
  RealImage defIm(mImSize, mImOrigin, mImSpacing);
  for(unsigned int imIdx=0; imIdx<mNImages; imIdx++){
    // get the weight
    Real curWeight = mDeformationData[imIdx]->Weight();
    // generate the deformed image
    mDeformationData[imIdx]->GetI0At1(defIm);
    const Real *imageData = defIm.getDataPointer();
    // update weighted image and jac det sums
    for(unsigned int j = 0; j < size; j++){
      averageData[j] += imageData[j] * curWeight;
    }
    // sum energies
    if(mDeformationData[imIdx]->HasEnergy()){
      mThreadEnergy += mDeformationData[imIdx]->LastEnergy() * curWeight;
    }
    if(mDeformationData[imIdx]->SaveDefToMean()){
      *mDeformationData[imIdx]->GetDefToMean() = defIm;
    }
  }
}

