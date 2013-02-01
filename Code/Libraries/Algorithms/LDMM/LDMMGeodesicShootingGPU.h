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

#ifndef __LDMM_GEODESIC_SHOOTING_GPU_H__
#define __LDMM_GEODESIC_SHOOTING_GPU_H__

#include "LDMMGeodesicShooting.h"
#include "AtlasWerksTypes.h"
#include "DiffOperGPU.h"
#include "HField3DUtils.h"
#include "LDMMEnergy.h"
#include "cudaInterface.h"
#include "cudaReduce.h"
#include "log.h"

class LDMMGeodesicShootingGPU : 
  public LDMMGeodesicShooting 
{

public:
  LDMMGeodesicShootingGPU(SizeType imSize, 
			  SpacingType imSpacing, 
			  DiffOperParam &diffOpParam);
  virtual ~LDMMGeodesicShootingGPU();

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
  
  DiffOperGPU mdDiffOp;
  cplReduce mdRd; 

  RealImage mIT;
  VectorField mPhi0T;
  VectorField mPhiT0;

  cplVector3DArray mdPhi0T;
  cplVector3DArray mdPhiT0;
  cplVector3DArray mdVT;
  cplVector3DArray mdScratchV;
  float *mdAlphaT;
  float *mdIT;
  float *mdAlpha0;
  float *mdI0;
  float *mdScratchI;
};

#endif  

