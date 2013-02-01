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

#ifndef __LDMM_ATLAS_THREAD_CPU__
#define __LDMM_ATLAS_THREAD_CPU__

#include "AtlasThread.h"
#include "LDMMIteratorCPU.h"
#include "LDMMDeformationData.h"
#include "LDMMAtlasParam.h"
#include "MultiscaleManager.h"

class LDMMAtlasDefDataCPU;
class LDMMAtlasThreadCPU;

/* ################ LDMMAtlasManagerCPU ################ */

class LDMMAtlasManagerCPU
{
public:
  typedef MultiscaleManager MultiscaleManagerType;
  typedef LDMMAtlasParam ParamType;
  typedef LDMMIteratorCPU IteratorType;
  typedef LDMMAtlasDefDataCPU DeformationDataType;
  typedef LDMMAtlasThreadCPU AtlasThreadType;

  static unsigned int CheckNumThreads(unsigned int requestedThreads, unsigned int nImages);

};

/* ################ LDMMAtlasDefDataCPU ################ */
 
class LDMMAtlasDefDataCPU : public LDMMDeformationData
{
public:
  LDMMAtlasDefDataCPU(const RealImage *I, 
		      const RealImage *IHat,
		      Real weight,
		      const LDMMParam &param);
  virtual ~LDMMAtlasDefDataCPU();
  virtual void SetScaleLevel(const MultiscaleManager &scaleManager);
  Real Weight(){ return mWeight; }
  RealImage &JacDet(){ return *mJacDet; }
protected:
  RealImage *mJacDet;
  Real mWeight;
};

/* ################ LDMMAtlasThreadCPU ################ */

/**
 * Class managing one CPU thread during atlas building.  May contain
 * multiple warps, which are updated sequentially.
 */
class LDMMAtlasThreadCPU :
  public AtlasThread<LDMMAtlasManagerCPU>
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
  LDMMAtlasThreadCPU(std::vector<DeformationDataType*> defData,
		     const ParamType &param,
		     AtlasBuilderInterface &builder,
		     RealImage *globalMean,
		     unsigned int nodeId, unsigned int nNodes,
		     unsigned int threadId, unsigned int nThreads,
		     unsigned int nTotalImages);
  
  virtual ~LDMMAtlasThreadCPU();
  const RealImage *GetMean(){ return mThreadMean; }
  const RealImage *GetJacDetSum(){ return mJacDetSum; }
  const LDMMEnergy &GetEnergy(){ return *mThreadEnergy; }

  const Real* GetDistances(){ return mDist; }
  Real* GetTMWeights(){ return mTMWeights; }
  
protected:

  /** Initialization called from within thread */
  virtual void InitThread();

  /** Finalization called from within thread */
  virtual void FinishThread();

  /** Set parameters for the given scale level */
  virtual void SetScaleLevel(const MultiscaleManagerType &scaleManager);

  /** Compute the average image */
  virtual void ComputeThreadMean();
  
  /** Compute the average image */
  virtual void GlobalMeanUpdated(){}
  
  /** Load data for restart from checkpoint */
  virtual void LoadInitData();

  /** Called at the beginning of a single image iteration */
  virtual void BeginImageIteration(int iteration, int imageIndex);

  /** Called at the end of a single image iteration */
  virtual void FinishImageIteration(int iteration, int imageIndex);

  /** Compute alphas for all deformations */
  void ComputeAtlasAlphas();

  unsigned int mNTimeSteps;

  // Thread Data
  RealImage *mThreadMean;
  RealImage *mJacDetSum;
  LDMMEnergy *mThreadEnergy;

  Real *mDist;
  Real *mTMWeights;
  
};

#endif // __LDMM_ATLAS_THREAD_CPU__
