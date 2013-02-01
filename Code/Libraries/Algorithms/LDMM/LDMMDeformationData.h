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


#ifndef __LDMM_DEFORMATION_DATA_H__
#define __LDMM_DEFORMATION_DATA_H__

#include <vector>

#include "AtlasWerksTypes.h"
#include "DeformationIteratorInterface.h"
#include "MultiscaleManager.h"
#include "LDMMEnergy.h"
#include "LDMMParam.h"

/*
 * Per-image deformation data
 */
#define USE_LV 0

class LDMMDeformationData 
  : public DeformationDataInterface
{
  
public:
  
  LDMMDeformationData(const RealImage *I0, 
		      const RealImage *I1,
		      const LDMMParam &param);
  
  virtual ~LDMMDeformationData();

  virtual void SetScaleLevel(const MultiscaleManager &scaleManager);
  
  virtual void InitializeWarp(){};
  virtual void FinalizeWarp(){};

  virtual const RealImage& I0(){ return *mI0Ptr; }
  virtual const RealImage& I1(){ return *mI1Ptr; }
  virtual const RealImage& I0Orig(){ return *mI0Orig; }
  virtual const RealImage& I1Orig(){ return *mI1Orig; }
  // TEST
  virtual void GetI0(RealImage &im){ im = *mI0Ptr; }
  virtual void GetI1(RealImage &im){ im = *mI1Ptr; }
  // END TEST
  virtual RealImage& Alpha(unsigned int t);
  virtual RealImage& Alpha0();
  virtual std::vector<VectorField*> &v(){ return mV; }
  virtual VectorField &v(int i){ return *mV[i]; }

  virtual void GetI0At1(RealImage& iDef);
  virtual void GetI1At0(RealImage& iDef);
  virtual void GetI0AtT(RealImage& iDef, unsigned int tIdx);
  virtual void GetI1AtT(RealImage& iDef, unsigned int tIdx);

  virtual void GetDef0To1(VectorField &hField);
  virtual void GetDef1To0(VectorField &hField);
  
  virtual void GetDef0ToT(VectorField &hField, unsigned int tIdx);
  virtual void GetDefTTo0(VectorField &hField, unsigned int tIdx);
  virtual void GetDef1ToT(VectorField &hField, unsigned int tIdx);
  virtual void GetDefTTo1(VectorField &hField, unsigned int tIdx);

  virtual void GetVField(VectorField &v, unsigned int t){ v = this->v(t); }

  virtual RealImage* GetDefToMean();
  
  void ScaleI0(bool scale) { mScaleI0 = scale; }
  bool ScaleI0() { return mScaleI0; }
  void ScaleI1(bool scale) { mScaleI1 = scale; }
  bool ScaleI1() { return mScaleI1; }
  void ComputeAlphas(bool compute){ 
    mComputeAlphas = compute; 
    this->InitAlphas();
  }
  bool ComputeAlphas(){ return mComputeAlphas; }
  void ComputeAlpha0(bool compute){ 
    mComputeAlpha0 = compute; 
    this->InitAlphas();
  }
  bool ComputeAlpha0(){ return mComputeAlpha0; }
  void SaveDefToMean(bool save, RealImage *defToMeanIm=NULL);
  bool SaveDefToMean(){return mSaveDefToMean; }

  virtual void AddEnergy(const Energy &e);

  // Used to set/get flag for whether this deformation is used in
  // trimmed mean calculation
  void SetUsedInMeanCalc(bool used){ mUsedInMeanCalc = used; }
  bool GetUsedInMeanCalc(){return mUsedInMeanCalc; }

  // Interp velocity fields at the given timepoint.  Currently just
  // linear interpolation.
  void InterpV(VectorField &v, Real tIdx);

protected:
  
  virtual void InitAlphas();
  
  const LDMMParam &mParam;

  // # of timesteps
  unsigned int mNTimeSteps;

  const RealImage *mI0Orig;
  const RealImage *mI1Orig;
  RealImage *mI0Scaled;
  RealImage *mI1Scaled;
  const RealImage *mI0Ptr;
  const RealImage *mI1Ptr;
  // vector field
    std::vector<VectorField*> mV;
#if USE_LV
    std::vector<VectorField*> mLV;
#endif
    
  std::vector<RealImage*> mAlpha;

  RealImage *mDefToMean;

  bool mComputeAlphas;
  bool mComputeAlpha0;
  bool mScaleI0;
  bool mScaleI1;
  bool mSaveDefToMean;
  // this is used to flag deformations which aren't used in the
  // trimmed mean calculation
  bool mUsedInMeanCalc;
  
};

#endif // __LDMM_DEFORMATION_DATA_H__
