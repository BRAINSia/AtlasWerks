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

#ifndef __ATLAS_THREAD_H__
#define __ATLAS_THREAD_H__

#include "AtlasWerksTypes.h"
#include "AtlasBuilderInterface.h"

/* ################ AtlasThread ################ */

template<class AtlasManagerType>
class AtlasThread {

public:

  typedef typename AtlasManagerType::MultiscaleManagerType MultiscaleManagerType;
  typedef typename AtlasManagerType::ParamType ParamType;
  typedef typename AtlasManagerType::IteratorType IteratorType;
  typedef typename IteratorType::ParamType IteratorParamType;
  typedef typename AtlasManagerType::DeformationDataType DeformationDataType;
  typedef typename AtlasManagerType::AtlasThreadType AtlasThreadType;
  
  AtlasThread(std::vector<DeformationDataType*> defData,
	      const ParamType &param,
	      AtlasBuilderInterface &builder,
	      RealImage *globalMean,
	      unsigned int nodeId, unsigned int nNodes,
	      unsigned int threadId, unsigned int nThreads,
	      unsigned int nTotalImages);
  virtual ~AtlasThread();
  
  unsigned int GetNImages(){ return mNImages; }

  DeformationDataType* GetDeformationData(int i){
    return mDeformationData[i];
  }

  static void *StartThread(AtlasThreadType* threadPointer);
  
protected:

  /** Used as main routine for thread */
  void AtlasThreadMain();

  /** Set parameters for the given scale level */
  virtual void SetScaleLevel(const MultiscaleManagerType &scaleManager);

  /** Initialization called from within thread */
  virtual void InitThread(){};

  /** Finalization called from within thread */
  virtual void FinishThread(){};

  /** Called at the beginning of a single image iteration */
  virtual void BeginImageIteration(int iteration, int imageIndex){}

  /** Called at the end of a single image iteration */
  virtual void FinishImageIteration(int iteration, int imageIndex){}
  
  /** Compute the average image */
  virtual void ComputeThreadMean(){};
  
  /** Called after the global mean has been computed */
  virtual void GlobalMeanUpdated(){};
  
  /** Load data for restart from checkpoint */
  virtual void LoadInitData(){};

  // Atlas building info
  unsigned int mNNodes;
  unsigned int mNodeId;
  unsigned int mNThreads;
  unsigned int mThreadId;
  unsigned int mNImages;
  unsigned int mNTotalImages;
  unsigned int mNScaleLevels;

  // for restarting from checkpoint
  unsigned int mInitialScaleLevel;
  unsigned int mInitialIter;

  // Reference to parent builder
  AtlasBuilderInterface &mBuilder;

  const ParamType &mParam;
  IteratorType *mIterator;
  const IteratorParamType *mCurScaleParam;

  std::vector<DeformationDataType*> mDeformationData;

  RealImage *mGlobalMean;
  
  // Image Info
  SizeType mImSize;
  OriginType mImOrigin;
  SpacingType mImSpacing;
  SizeType mCurSize;
  SpacingType mCurSpacing;

};

#include "AtlasThread.txx"

#endif // __ATLAS_THREAD_H__
