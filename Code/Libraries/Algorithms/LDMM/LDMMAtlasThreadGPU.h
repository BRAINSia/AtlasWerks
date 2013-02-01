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

#ifndef __LDMM_ATLAS_THREAD_GPU_H__
#define __LDMM_ATLAS_THREAD_GPU_H__

#include "AtlasThread.h"
#include "LDMMIteratorGPU.h"
#include "LDMMDeformationDataGPU.h"
#include "LDMMAtlasParam.h"
#include "MultiscaleManagerGPU.h"

class LDMMAtlasDefDataGPU;
class LDMMAtlasThreadGPU;

/* ################ LDMMAtlasManagerGPU ################ */

class LDMMAtlasManagerGPU
{
public:
  typedef MultiscaleManagerGPU MultiscaleManagerType;
  typedef LDMMAtlasParam ParamType;
  typedef LDMMIteratorGPU IteratorType;
  typedef LDMMAtlasDefDataGPU DeformationDataType;
  typedef LDMMAtlasThreadGPU AtlasThreadType;

  static unsigned int CheckNumThreads(unsigned int requestedThreads, unsigned int nImages);

};

/* ################ LDMMAtlasDefDataGPU ################ */

class LDMMAtlasDefDataGPU : public LDMMDeformationDataGPU
{
public:
  LDMMAtlasDefDataGPU(const RealImage *I, 
			      const RealImage *IHat,
			      Real weight,
			      const LDMMParam &param);
  virtual ~LDMMAtlasDefDataGPU();
  virtual void SetScaleLevel(const MultiscaleManager &scaleManager);
  virtual void InitializeWarp();
  virtual void FinalizeWarp();
  Real Weight(){ return mWeight; }
  RealImage &JacDet();
  float *dJacDet(){ return mdJacDet; }
  //void ComputeJacDet();
protected:
  RealImage *mJacDet;
  Real mWeight;
  // device data
  float* mdJacDet;
};

/* ################ LDMMAtlasThreadGPU ################ */

class LDMMAtlasThreadGPU :
  public AtlasThread<LDMMAtlasManagerGPU>
{
  
public:
  
  LDMMAtlasThreadGPU(std::vector<DeformationDataType*> defData,
		     const ParamType &param,
		     AtlasBuilderInterface &builder,
		     RealImage *globalMean,
		     unsigned int nodeId, unsigned int nNodes,
		     unsigned int threadId, unsigned int nThreads,
		     unsigned int nTotalImages);
  
  virtual ~LDMMAtlasThreadGPU();
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

  /** Compute jacobian determinant of the given data deformation */
  void ComputeJacDet(LDMMAtlasDefDataGPU &data);

  /** Set parameters for the given scale level */
  virtual void SetScaleLevel(const MultiscaleManagerType &scaleManager);

  /** Compute the average image */
  virtual void ComputeThreadMean();
  
  /** Called after the global mean has been computed */
  virtual void GlobalMeanUpdated();
 
  /** deal with device data */
  virtual void InitDeviceData();
  virtual void FreeDeviceData();

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

  // Device Data
  float *mdThreadMean;
  float *mdJacDetSum;
  float *mdScratchI;

  unsigned int mNVox;
  unsigned int mCurVox;

  Real *mDist;
  Real *mTMWeights;
  
};

#endif // __LDMM_ATLAS_THREAD_GPU_H__
