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

// included in AtlasThread.h

/* ################ AtlasThread ################ */

template<class AtlasManagerType>
void*
AtlasThread<AtlasManagerType>::
StartThread(AtlasThreadType* atlasThread)
{
  // set the thread name for logging
  std::string threadname = StringUtils::strPrintf("GPURegThread%d",atlasThread->mThreadId);
  ErrLog::SetThreadName(threadname);
  
  atlasThread->AtlasThreadMain();

  return NULL;
}

template<class AtlasManagerType>
AtlasThread<AtlasManagerType>::
AtlasThread(std::vector<DeformationDataType*> defData,
	    const ParamType &param,
	    AtlasBuilderInterface &builder,
	    RealImage *globalMean,
	    unsigned int nodeId, unsigned int nNodes,
	    unsigned int threadId, unsigned int nThreads,
	    unsigned int nTotalImages)
  : mNNodes(nNodes),
    mNodeId(nodeId),
    mNThreads(nThreads),
    mThreadId(threadId),
    mNImages(defData.size()),
    mNTotalImages(nTotalImages),
    mNScaleLevels(param.GetNumberOfScaleLevels()),
    mInitialScaleLevel(0),
    mInitialIter(0),
    mBuilder(builder),
    mParam(param),
    mIterator(NULL),
    mCurScaleParam(NULL),
    mDeformationData(defData),
    mGlobalMean(globalMean)
{
  if(mNImages < 1){
    std::string errMsg = StringUtils::strPrintf("Error, node %d thread %d has %d images",mNodeId,mThreadId,mNImages);
    throw AtlasWerksException(__FILE__,__LINE__,errMsg);
  }
  mImSize = defData[0]->I1Orig().getSize();
  mImOrigin = defData[0]->I1Orig().getOrigin();
  mImSpacing = defData[0]->I1Orig().getSpacing();
  
}

template<class AtlasManagerType>
AtlasThread<AtlasManagerType>::
~AtlasThread()
{
  delete mIterator;
}

template<class AtlasManagerType>
void
AtlasThread<AtlasManagerType>::
SetScaleLevel(const MultiscaleManagerType &scaleManager)
{
  // get scale level information
  mCurSize = scaleManager.CurScaleSize();
  mCurSpacing = scaleManager.CurScaleSpacing();
  
  // scale individual image deformaiton data
  for(unsigned int i=0;i<mNImages;i++){
    mDeformationData[i]->SetScaleLevel(scaleManager);
  }
  
  int scale = scaleManager.CurScaleLevel();
  mCurScaleParam = &mParam.GetScaleLevel(scale).Iterator();
  mIterator->SetScaleLevel(scaleManager, *mCurScaleParam);

  // still need to generate the mean at this scale level...
}

template<class AtlasManagerType>
void
AtlasThread<AtlasManagerType>::
AtlasThreadMain()
{

  this->InitThread();
  
  for(unsigned int scaleLevel = mInitialScaleLevel; scaleLevel < mNScaleLevels; scaleLevel++)
    {
      // have one thread set the builder scale level (synchronized
      // within the builder function)
      mBuilder.SetScaleLevel(scaleLevel);
      // set the scale level for this thread
      const MultiscaleManager &m = mBuilder.GetScaleManager();
      const MultiscaleManagerType &manager = dynamic_cast<const MultiscaleManagerType&>(m);
      this->SetScaleLevel(manager);
      
      // Compute the thread mean
      this->ComputeThreadMean();

      // have one thread compute the global mean
      mBuilder.ComputeMean();

      this->GlobalMeanUpdated();

      // load in data from checkpoint if needed
      if(scaleLevel == mInitialScaleLevel){
	this->LoadInitData();
      }

      if(mThreadId == 0){
	mBuilder.BeginScaleLevel(scaleLevel);
      }
      
      unsigned int nIters = mParam.GetScaleLevel(scaleLevel).NIterations();

      unsigned int iter=0;
      if(scaleLevel == mInitialScaleLevel){
	iter = mInitialIter;
      }

      // iterate
      for(;iter<nIters;iter++){
	
	if(mThreadId == 0){
	  mBuilder.BeginIteration(iter);
	}
	
	// Update deformation for all local images
	for(unsigned int imIdx=0; imIdx < mNImages; ++imIdx){

	  mDeformationData[imIdx]->SetCurIter(iter);

	  this->BeginImageIteration(iter, imIdx);

	  //
	  // Perform Iteration
	  //
	  mIterator->Iterate(*mDeformationData[imIdx]);

	  this->FinishImageIteration(iter, imIdx);
	  
	}

	// Compute the thread mean
	this->ComputeThreadMean();

	// have one thread compute the global mean
	mBuilder.ComputeMean();

	this->GlobalMeanUpdated();

	if(mThreadId == 0){
	  mBuilder.FinishIteration(iter);
	}
	  
      }// end iteration loop
	
      if(mThreadId == 0){
	mBuilder.FinishScaleLevel(scaleLevel);
      }
      
    } // end iterate over scale levels
  
  this->FinishThread();
}

// template<class AtlasManagerType>
// int
// AtlasThread<AtlasManagerType>::
// LocalToGlobalMapping(int localIdx)
// {
//   // compute global index

//   // calc distribution to this node
//   int nodeBId,nNodeImages;
//   ApplicationUtils::Distribute(mNTotalImages, mNNodes, mNodeId, nodeBId, nNodeImages);

//   // calc distribution to this thread
//   int threadBId,nThreadImages;
//   ApplicationUtils::Distribute(nNodeImages, mNGPUs, mGPUId, threadBId, nThreadImages);

//   assert(nThreadImages == (int)mNImages);

//   return nodeBId + threadBId + localIdx;
// }  

// template<class AtlasManagerType>
// void
// AtlasThread<AtlasManagerType>::
// ComputeAtlasAlphas()
// {
//   LOGNODETHREAD(logDEBUG) << "Computing alphas";
//   for(unsigned int imIdx=0; imIdx < mNImages; ++imIdx){
//     LOGNODETHREAD(logDEBUG) << "Computing alphas for image " << imIdx << "...";
//     // compute the alphas
//     Real imEnergy, vecEnergy;
//     mIterator->Iterate(mdMean, mdI[imIdx], mdV[imIdx], mdH, imEnergy, vecEnergy, true);
//     copyArrayFromDevice(mAlpha0[imIdx]->getDataPointer(), mIterator->GetAlpha(0), mCurVox);
//   }
// }

