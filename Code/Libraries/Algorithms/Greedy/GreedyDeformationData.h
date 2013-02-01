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


#ifndef __GREEDY_DEFORMATION_DATA_H__
#define __GREEDY_DEFORMATION_DATA_H__

#include "AffineTransform3D.h"
#include "AtlasWerksTypes.h"
#include "DeformationIteratorInterface.h"
#include "Energy.h"
#include "EnergyHistory.h"
#include "GreedyParam.h"
#include "MultiscaleManager.h"

typedef AffineTransform3D<Real> Affine3D;

/*
 * Per-image deformation data
 */
class GreedyDeformationData 
  : public DeformationDataInterface
{
  
public:
  
  GreedyDeformationData(const RealImage *I0, 
			const RealImage *I1,
			const GreedyParam &param);
  
  ~GreedyDeformationData();
  
  virtual void SetScaleLevel(const MultiscaleManager &scaleManager);
  
  const RealImage& I0(){ return *mI0Ptr; }
  const RealImage& I1(){ return *mI1Ptr; }
  const RealImage& I0Orig(){ return *mI0Orig; }
  const RealImage& I1Orig(){ return *mI1Orig; }

  virtual VectorField &Def0To1(){ return *mDef0To1; }
  virtual VectorField &Def1To0(){ return *mDef1To0; }
  virtual void GetI0At1(RealImage& iDef);
  virtual void GetI1At0(RealImage& iDef);

  virtual RealImage* GetDefToMean();

  /** 
   * These are run from the warp thread, primarily a place for device
   * data to be initialized in the GPU version
   */
  virtual void InitializeWarp() {}
  virtual void FinalizeWarp() {}

  virtual void GetDef1To0(VectorField &h);
  virtual void GetDef0To1(VectorField &h);

  void ScaleI0(bool scale) { mScaleI0 = scale; }
  bool ScaleI0() { return mScaleI0; }
  void ScaleI1(bool scale) { mScaleI1 = scale; }
  bool ScaleI1() { return mScaleI1; }

  bool ComputeInverseHField();
  void ComputeInverseHField(bool computeInv);

  void SaveDefToMean(bool save, RealImage *defToMeanIm=NULL);
  bool SaveDefToMean(){return mSaveDefToMean; }

  void SetInitialAffine(const Affine3D &aff);
  const Affine3D &GetInitialAffine();

  virtual void AddEnergy(const Energy &e);
  
protected:
  
  const GreedyParam &mParam;
  
  // image data
  const RealImage *mI0Orig;
  const RealImage *mI1Orig;
  RealImage *mI0Scaled;
  RealImage *mI1Scaled;

  // these either point to the 'Scaled' or 'Orig' data, depending on
  // scale level
  const RealImage *mI0Ptr;
  const RealImage *mI1Ptr;

  // deformation field
  VectorField* mDef1To0;
  // inverse deformation field
  VectorField* mDef0To1;
  // compute the inverse deformation field?
  bool mComputeInverseHField;

  bool mSaveDefToMean;
  RealImage *mDefToMean;
  
  bool mScaleI0;
  bool mScaleI1;

  Affine3D *mInitialAffine;
  
};

#endif // __GREEDY_DEFORMATION_DATA_H__
