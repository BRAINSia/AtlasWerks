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

#ifndef __GREEDY_DEFORMATION_DATA_GPU_H__
#define __GREEDY_DEFORMATION_DATA_GPU_H__

#include "GreedyDeformationData.h"
#include "cudaInterface.h"
#include "cutil_comfunc.h"
#include "cudaHField3DUtils.h"
#include "CUDAUtilities.h"
#include "VectorMath.h"

class GreedyDeformationDataGPU
  : public GreedyDeformationData
{
public:
  
  GreedyDeformationDataGPU(const RealImage *I0, 
			   const RealImage *I1,
			   const GreedyParam &param);
  
  ~GreedyDeformationDataGPU();
  
  virtual void SetScaleLevel(const MultiscaleManager &scaleManager);
  
  virtual VectorField &Def1To0();
  virtual VectorField &Def0To1();
  virtual void GetI0At1(RealImage& iDef);
  virtual void GetI1At0(RealImage& iDef);

  virtual void InitializeWarp();
  virtual void FinalizeWarp();
  
  float* dI0(){ return mdI0Ptr; }
  float* dI1(){ return mdI1Ptr; }
  cplVector3DArray& dDef1To0(){ return mdDef1To0; }
  cplVector3DArray& dDef0To1(){ return mdDef0To1; }

  void InitializeDeviceData();
  void FreeDeviceData();
  
protected:

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

  // deformation field
  cplVector3DArray mdDef1To0;
  // inverse deformation field
  cplVector3DArray mdDef0To1;

};

#endif // __GREEDY_DEFORMATION_DATA_GPU_H__
