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

#ifndef __LDMM_DEFORMATION_DATA_GPU_H__
#define __LDMM_DEFORMATION_DATA_GPU_H__

#include "LDMMDeformationData.h"
#include "cudaInterface.h"
#include "cutil_comfunc.h"
#include "cudaHField3DUtils.h"
#include "CUDAUtilities.h"
#include "VectorMath.h"


class LDMMDeformationDataGPU
  : public LDMMDeformationData
{
public:
  
  LDMMDeformationDataGPU(const RealImage *I0, 
			 const RealImage *I1,
			 const LDMMParam &param);
  
  ~LDMMDeformationDataGPU();
  
  virtual void SetScaleLevel(const MultiscaleManager &scaleManager);
  
  // TEST
  virtual void GetI0(RealImage &im);
  virtual void GetI1(RealImage &im);
  // END TEST

  float* dI0(){ return mdI0Ptr; }
  float* dI1(){ return mdI1Ptr; }
  std::vector<cplVector3DArray>& dV(){ return mdV; }

    cplVector3DArray& dV(int i){ return mdV[i]; }
#if USE_LV
    cplVector3DArray& dLV(int i){ return mdLV[i]; }
#endif
  virtual void GetI0AtT(RealImage& iDef, unsigned int tIdx);
  virtual void GetI1AtT(RealImage& iDef, unsigned int tIdx);
  
  virtual void GetDef0ToT(VectorField &hField, unsigned int tIdx);
  virtual void GetDefTTo0(VectorField &hField, unsigned int tIdx);
  virtual void GetDef1ToT(VectorField &hField, unsigned int tIdx);
  virtual void GetDefTTo1(VectorField &hField, unsigned int tIdx);

  void GetDef0ToT(cplVector3DArray &dH, cplVector3DArray &dScratchV, unsigned int tIdx);
  void GetDefTTo0(cplVector3DArray &dH, cplVector3DArray &dScratchV, unsigned int tIdx);
  void GetDef1ToT(cplVector3DArray &dH, cplVector3DArray &dScratchV, unsigned int tIdx);
  void GetDefTTo1(cplVector3DArray &dH, cplVector3DArray &dScratchV, unsigned int tIdx);

  virtual void InitializeWarp();
  virtual void FinalizeWarp();
  
  void InitializeDeviceData();
  void FreeDeviceData();

  void InterpV(cplVector3DArray &dV, Real tIdx);
  
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

    // mNTimeSteps Velocity fields
    std::vector<cplVector3DArray> mdV;
#if USE_LV
    std::vector<cplVector3DArray> mdLV;
#endif
};


#endif // __LDMM_DEFORMATION_DATA_GPU_H__
