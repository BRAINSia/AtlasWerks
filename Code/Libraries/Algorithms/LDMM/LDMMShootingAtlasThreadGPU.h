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

#ifndef __LDMM_SHOOTING_ATLAS_THREAD_GPU__
#define __LDMM_SHOOTING_ATLAS_THREAD_GPU__

#include "AtlasThread.h"
#include "LDMMShootingIteratorGPU.h"
#include "LDMMShootingDefData.h"
#include "LDMMAtlasParam.h"
#include "MultiscaleManagerGPU.h"

class LDMMShootingAtlasDefDataGPU;
class LDMMShootingAtlasThreadGPU;

/* ################ LDMMAtlasManagerGPU ################ */

class LDMMShootingAtlasManagerGPU
{
public:
  typedef MultiscaleManagerGPU MultiscaleManagerType;
  typedef LDMMAtlasParam ParamType;
  typedef LDMMShootingIteratorGPU IteratorType;
  typedef LDMMShootingAtlasDefDataGPU DeformationDataType;
  typedef LDMMShootingAtlasThreadGPU AtlasThreadType;

  static unsigned int CheckNumThreads(unsigned int requestedThreads, unsigned int nImages);

};

/** ################ LDMMShootingAtlasDefDataGPU ################ **/

class LDMMShootingAtlasDefDataGPU : public LDMMShootingDefDataGPU
{
public:
  LDMMShootingAtlasDefDataGPU(const RealImage *I, 
			      const RealImage *IHat,
			      Real weight,
			      const LDMMParam &param);
  virtual ~LDMMShootingAtlasDefDataGPU();
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

/** ################ LDMMShootingAtlasThreadGPU ################ **/

class LDMMShootingAtlasThreadGPU :
  public AtlasThread<LDMMShootingAtlasManagerGPU>
{

public:

  LDMMShootingAtlasThreadGPU(std::vector<DeformationDataType*> defData,
			     const ParamType &param,
			     AtlasBuilderInterface &builder,
			     RealImage *globalMean,
			     unsigned int nodeId, unsigned int nNodes,
			     unsigned int threadId, unsigned int nThreads,
			     unsigned int nTotalImages);
  
  virtual ~LDMMShootingAtlasThreadGPU();
  const RealImage *GetMean(){ return mThreadMean; }
  const RealImage *GetJacDetSum(){ return mJacDetSum; }
  const LDMMEnergy &GetEnergy(){ return *mThreadEnergy; }

  const Real* GetDistances(){
    throw AtlasWerksException(__FILE__,__LINE__,"Error, GetDistances unimplemented in shooting optimization");
    return NULL;
  }
  Real* GetTMWeights(){ 
    throw AtlasWerksException(__FILE__,__LINE__,"Error, GetTMWeights unimplemented in shooting optimization");
    return NULL;
 }

protected:

  /** Initialization called from within thread */
  virtual void InitThread();

  /** Finalization called from within thread */
  virtual void FinishThread();

  /** Compute jacobian determinant of the given data deformation */
  void ComputeJacDet(LDMMShootingAtlasDefDataGPU &data);

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

};

#endif // __LDMM_SHOOTING_ATLAS_THREAD_GPU__
