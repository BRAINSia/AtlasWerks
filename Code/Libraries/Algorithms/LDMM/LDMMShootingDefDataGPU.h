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

#ifndef __LDMM_SHOOTING_DEF_DATA_GPU_H__
#define __LDMM_SHOOTING_DEF_DATA_GPU_H__

#include "LDMMShootingDefData.h"
#include "cudaInterface.h"
#include "cutil_comfunc.h"
#include "cudaHField3DUtils.h"
#include "CUDAUtilities.h"
#include "VectorMath.h"

class LDMMShootingDefDataGPU
  : public LDMMShootingDefData
{
public:
  
  LDMMShootingDefDataGPU(const RealImage *I0, 
			 const RealImage *I1,
			 const LDMMParam &param);
  
  ~LDMMShootingDefDataGPU();
  
  virtual void SetScaleLevel(const MultiscaleManager &scaleManager);
  
  virtual VectorField &PhiT0();
  virtual VectorField &Phi0T();
  virtual RealImage& Alpha0();

  virtual void GetI0At1(RealImage& iDef);
  virtual void GetI1At0(RealImage& iDef);

  virtual void InitializeWarp();
  virtual void FinalizeWarp();
  
  float* dI0(){ return mdI0Ptr; }
  float* dI1(){ return mdI1Ptr; }
  cplVector3DArray& dPhiT0(){ return mdPhiT0; }
  cplVector3DArray& dPhi0T(){ return mdPhi0T; }
  float* dAlpha0(){ return mdAlpha0; }

  void InitializeDeviceData();
  void FreeDeviceData();
  
protected:

  unsigned int mNVox;
  unsigned int mCurVox;

  bool mDeviceDataInitialized;

  // image data
  float *mdI0Orig;
  float *mdI1Orig;
  float *mdI0Scaled;
  float *mdI1Scaled;
  
  // these either point to the 'Scaled' or 'Orig' data, depending on
  // scale level
  float *mdI0Ptr;
  float *mdI1Ptr;

  // Forward and reverse deformation fields
  cplVector3DArray mdPhiT0;
  cplVector3DArray mdPhi0T;
  float *mdAlpha0;

};

#endif // __LDMM_SHOOTING_DEF_DATA_GPU_H__
