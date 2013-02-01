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

#ifndef __LDMM_GEODESIC_SHOOTING_H__
#define __LDMM_GEODESIC_SHOOTING_H__

#include "AtlasWerksTypes.h"
#include "DiffOper.h"
#include "HField3DUtils.h"
#include "LDMMEnergy.h"
#include "log.h"

class LDMMGeodesicShooting {

public:
  LDMMGeodesicShooting(SizeType imSize, 
		       SpacingType imSpacing);
  virtual ~LDMMGeodesicShooting();

  virtual void ShootImage(const RealImage &I0,
			  const RealImage &alpha0,
			  unsigned int nTimeSteps) = 0;

  virtual void ShootImage(const RealImage &I0,
 			  const VectorField &v0,
 			  unsigned int nTimeSteps) = 0;
  
  bool SaveAlphas(){ return mSaveAlphas; }
  void SaveAlphas(bool save){ mSaveAlphas = save; }
  bool SaveImages(){ return mSaveImages; }
  void SaveImages(bool save){ mSaveImages = save; }
  bool SaveVecs(){ return mSaveVecs; }
  void SaveVecs(bool save){ mSaveVecs = save; }
  
  virtual const RealImage &GetFinalImage() = 0;
  virtual const VectorField &GetPhi0To1() = 0;
  virtual const VectorField &GetPhi1To0() = 0;
  
  virtual const RealImage &GetAlphaT(unsigned int t);
  virtual const RealImage &GetIT(unsigned int t);
  virtual const VectorField &GetVT(unsigned int t);

  void GetJacDetPhi0To1(RealImage &jacDet);
  void GetJacDetPhi1To0(RealImage &jacDet);

protected:
  
  SizeType mImSize;
  SpacingType mImSpacing;
  
  int mNTimeSteps;

  bool mSaveAlphas;
  bool mSaveImages;
  bool mSaveVecs;

  std::vector<RealImage*> mAlphaVec;
  std::vector<RealImage*> mImVec;
  std::vector<VectorField*> mVVec;
  
};

#endif  

