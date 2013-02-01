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


#ifndef __LDMM_VEL_SHOOTING_DEF_DATA_H__
#define __LDMM_VEL_SHOOTING_DEF_DATA_H__

#include "AtlasWerksTypes.h"
#include "DeformationIteratorInterface.h"
#include "MultiscaleManager.h"
#include "LDMMEnergy.h"
#include "LDMMParam.h"

/**
 * Per-image deformation data
 */
class LDMMVelShootingDefData 
  : public DeformationDataInterface
{
  
public:
  
  LDMMVelShootingDefData(const RealImage *I0, 
			 const RealImage *I1,
			 const LDMMParam &param);
  ~LDMMVelShootingDefData();
  
  void SetScaleLevel(MultiscaleManager &scaleManager);

  const RealImage& I0(){ return *mI0Ptr; }
  const RealImage& I1(){ return *mI1Ptr; }
  const RealImage& I0Orig(){ return *mI0Orig; }
  const RealImage& I1Orig(){ return *mI1Orig; }

  virtual VectorField &PhiT0(){ return *mPhiT0; }
  virtual VectorField &Phi0T(){ return *mPhi0T; }
  virtual VectorField &V0(){ return *mV0; }

  /** not implemented, throws exception if called */
  virtual RealImage& Alpha(unsigned int t);

  virtual void GetVField(VectorField &v, unsigned int t);

  virtual void GetI0At1(RealImage& iDef);
  virtual void GetI1At0(RealImage& iDef);

  virtual void GetI0AtT(RealImage& iDef, unsigned int tIdx);
  virtual void GetI1AtT(RealImage& iDef, unsigned int tIdx);

  virtual RealImage* GetDefToMean(){
    throw AtlasWerksException(__FILE__,__LINE__,"Error, GetDefToMean not implemented");
  }

  void SaveDefToMean(bool save, RealImage *defToMeanIm=NULL){
    throw AtlasWerksException(__FILE__,__LINE__,"Error, SaveDefToMean not implemented");
  }

  /** 
   * These are run from the warp thread, primarily a place for device
   * data to be initialized in the GPU version
   */
  virtual void InitializeWarp(){}
  virtual void FinalizeWarp(){}

  virtual void GetDef0To1(VectorField &hField);
  virtual void GetDef1To0(VectorField &hField);

  void ScaleI0(bool scale) { mScaleI0 = scale; }
  bool ScaleI0() { return mScaleI0; }
  void ScaleI1(bool scale) { mScaleI1 = scale; }
  bool ScaleI1() { return mScaleI1; }

  void SetScaleLevel(const MultiscaleManager &scaleManager);

  // Used to set/get flag for whether this deformation is used in
  // trimmed mean calculation
  void SetUsedInMeanCalc(bool used){ mUsedInMeanCalc = used; }
  bool GetUsedInMeanCalc(){return mUsedInMeanCalc; }
  
protected:

  const LDMMParam &mParam;
  // # of timesteps
  unsigned int mNTimeSteps;
  
  // original images
  const RealImage *mI0Orig;
  const RealImage *mI1Orig;
  // scaled images
  RealImage *mI0Scaled;
  RealImage *mI1Scaled;
  // pointers to Orig or Scaled images, depending on scale level
  const RealImage *mI0Ptr;
  const RealImage *mI1Ptr;
 
  // should we scale the initial/final images in SetScaleLevel?
  bool mScaleI0;
  bool mScaleI1;

  // v0
  VectorField *mV0;
  // deformation pulling I0 to I1
  VectorField *mPhiT0;
  // deformation pulling I1 to I0
  VectorField *mPhi0T;
  
  // this is used to flag deformations which aren't used in the
  // trimmed mean calculation
  bool mUsedInMeanCalc;

};

#endif // __LDMM_VEL_SHOOTING_DEF_DATA_H__
