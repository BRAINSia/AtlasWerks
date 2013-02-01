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

#ifndef __LDMM_GEODESIC_SHOOTING_CPU_H__
#define __LDMM_GEODESIC_SHOOTING_CPU_H__

#include "LDMMGeodesicShooting.h"

class LDMMGeodesicShootingCPU : 
  public LDMMGeodesicShooting 
{

public:
  LDMMGeodesicShootingCPU(SizeType imSize, 
			  SpacingType imSpacing, 
			  DiffOperParam &diffOpParam);
  virtual ~LDMMGeodesicShootingCPU();

  virtual void ShootImage(const RealImage &I0,
			  const RealImage &alpha0,
			  unsigned int nTimeSteps);
  
  virtual void ShootImage(const RealImage &I0,
			  const VectorField &v0,
			  unsigned int nTimeSteps);
  
  virtual const RealImage &GetFinalImage(){ return mIT; }
  virtual const VectorField &GetPhi0To1(){ return mPhi0T; }
  virtual const VectorField &GetPhi1To0(){ return mPhiT0; }
  
protected:

  unsigned int mNVox;
  
  DiffOper mDiffOp;

  VectorField mPhi0T;
  VectorField mPhiT0;
  VectorField mVT;
  VectorField mScratchV;
  RealImage mAlphaT;
  RealImage mIT;

};

#endif  

