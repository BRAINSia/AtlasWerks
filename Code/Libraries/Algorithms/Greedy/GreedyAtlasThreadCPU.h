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

#ifndef __GREEDY_ATLAS_THREAD_CPU__
#define __GREEDY_ATLAS_THREAD_CPU__

#include "AtlasThread.h"
#include "GreedyIteratorCPU.h"
#include "GreedyDeformationData.h"
#include "GreedyAtlasParam.h"
#include "MultiscaleManager.h"

class GreedyAtlasDefDataCPU;
class GreedyAtlasThreadCPU;

/* ################ GreedyAtlasManagerCPU ################ */

class GreedyAtlasManagerCPU
{
public:
  typedef MultiscaleManager MultiscaleManagerType;
  typedef GreedyAtlasParam ParamType;
  typedef GreedyIteratorCPU IteratorType;
  typedef GreedyAtlasDefDataCPU DeformationDataType;
  typedef GreedyAtlasThreadCPU AtlasThreadType;

  static unsigned int CheckNumThreads(unsigned int requestedThreads, unsigned int nImages);

};

/* ################ GreedyAtlasDefDataCPU ################ */
 
class GreedyAtlasDefDataCPU : public GreedyDeformationData
{
public:
  GreedyAtlasDefDataCPU(const RealImage *I, 
			const RealImage *IHat,
			Real weight,
			const GreedyParam &param);
  virtual ~GreedyAtlasDefDataCPU(){}
  Real Weight(){ return mWeight; }
protected:
  Real mWeight;
};

/* ################ GreedyAtlasThreadCPU ################ */

/**
 * Class managing one CPU thread during atlas building.  May contain
 * multiple warps, which are updated sequentially.
 */
class GreedyAtlasThreadCPU :
  public AtlasThread<GreedyAtlasManagerCPU>
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
  GreedyAtlasThreadCPU(std::vector<DeformationDataType*> defData,
		     const ParamType &param,
		     AtlasBuilderInterface &builder,
		     RealImage *globalMean,
		     unsigned int nodeId, unsigned int nNodes,
		     unsigned int threadId, unsigned int nThreads,
		     unsigned int nTotalImages);
  
  virtual ~GreedyAtlasThreadCPU();
  const RealImage *GetMean(){ return mThreadMean; }
  const Energy &GetEnergy(){ return mThreadEnergy; }
  
protected:

  /** Initialization called from within thread */
  virtual void InitThread();

  /** Finalization called from within thread */
  virtual void FinishThread(){};

  /** Called at the beginning of a single image iteration */
  virtual void BeginImageIteration(int iteration, int imageIndex);

  /** Called at the end of a single image iteration */
  virtual void FinishImageIteration(int iteration, int imageIndex);

  /** Set parameters for the given scale level */
  virtual void SetScaleLevel(const MultiscaleManagerType &scaleManager);

  /** Compute the average image */
  virtual void ComputeThreadMean();
  
  /** Compute the average image */
  virtual void GlobalMeanUpdated(){}
  
  // Thread Data
  RealImage *mThreadMean;
  Energy mThreadEnergy;
  
};

#endif // __GREEDY_ATLAS_THREAD_CPU__
