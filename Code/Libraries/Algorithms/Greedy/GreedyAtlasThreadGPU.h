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

#ifndef __GREEDY_ATLAS_THREAD_GPU__
#define __GREEDY_ATLAS_THREAD_GPU__

#include "AtlasThread.h"
#include "GreedyIteratorGPU.h"
#include "GreedyDeformationData.h"
#include "GreedyAtlasParam.h"
#include "MultiscaleManager.h"

class GreedyAtlasDefDataGPU;
class GreedyAtlasThreadGPU;

/* ################ GreedyAtlasManagerGPU ################ */

class GreedyAtlasManagerGPU
{
public:
  typedef MultiscaleManager MultiscaleManagerType;
  typedef GreedyAtlasParam ParamType;
  typedef GreedyIteratorGPU IteratorType;
  typedef GreedyAtlasDefDataGPU DeformationDataType;
  typedef GreedyAtlasThreadGPU AtlasThreadType;

  static unsigned int CheckNumThreads(unsigned int requestedThreads, unsigned int nImages);

};

/* ################ GreedyAtlasDefDataGPU ################ */
 
class GreedyAtlasDefDataGPU : public GreedyDeformationDataGPU
{
public:
  GreedyAtlasDefDataGPU(const RealImage *I, 
			const RealImage *IHat,
			Real weight,
			const GreedyParam &param);
  virtual ~GreedyAtlasDefDataGPU(){}
  Real Weight(){ return mWeight; }
protected:
  Real mWeight;
};

/* ################ GreedyAtlasThreadGPU ################ */

/**
 * Class managing one GPU thread during atlas building.  May contain
 * multiple warps, which are updated sequentially.
 */
class GreedyAtlasThreadGPU :
  public AtlasThread<GreedyAtlasManagerGPU>
{
public:
  /**
   * Constructor
   *
   * \param defData The deformation data for the deformations this thread is responsible for
   *
   * \param paramThe parameter settings controlling the deformaions
   *
   * \param builder The AtlasBuilder responsible for managing this thread
   *
   * \param nodeId the node number this thread is running on.  0 for
   * single-node builds.
   *
   * \param nNodes The total number of nodes for this build, 1 for
   * single-node builds
   *
   * \param threadId The thread ID of this thread
   *
   * \param nThreads the number of threads running in this AtlasBuilder instance
   *
   * \param nTotalImages the total number of input images across all nodes
   */
  GreedyAtlasThreadGPU(std::vector<DeformationDataType*> defData,
		     const ParamType &param,
		     AtlasBuilderInterface &builder,
		     RealImage *globalMean,
		     unsigned int nodeId, unsigned int nNodes,
		     unsigned int threadId, unsigned int nThreads,
		     unsigned int nTotalImages);
  
  virtual ~GreedyAtlasThreadGPU();
  const RealImage *GetMean(){ return mThreadMean; }
  const Energy &GetEnergy(){ return mThreadEnergy; }
  
protected:

  /** Initialization called from within thread */
  virtual void InitThread();

  /** Finalization called from within thread */
  virtual void FinishThread();

  /** Called at the beginning of a single image iteration */
  virtual void BeginImageIteration(int iteration, int imageIndex);

  /** Called at the end of a single image iteration */
  virtual void FinishImageIteration(int iteration, int imageIndex);

  /** Set parameters for the given scale level */
  virtual void SetScaleLevel(const MultiscaleManagerType &scaleManager);

  /** Compute the average image */
  virtual void ComputeThreadMean();
  
  /** Compute the average image */
  virtual void GlobalMeanUpdated();
  
  /** deal with device data */
  void InitDeviceData();
  void FreeDeviceData();

  // Thread Data
  RealImage *mThreadMean;
  Energy mThreadEnergy;

  // Device Data
  float *mdThreadMean;
  float *mdScratchI;

  unsigned int mNVox;
  unsigned int mCurVox;
  
};

#endif // __GREEDY_ATLAS_THREAD_GPU__
