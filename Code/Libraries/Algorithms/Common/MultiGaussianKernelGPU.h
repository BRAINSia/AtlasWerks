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

#ifndef __MULTI_GAUSSIAN_KERNEL_GPU_H__
#define __MULTI_GAUSSIAN_KERNEL_GPU_H__

#include "AtlasWerksTypes.h"
#include "KernelInterfaceGPU.h"
#include "KernelParam.h"
#include "MultiGaussianKernelParam.h"
#include "GaussianFilter3D.h"

#include <cudaVector3DArray.h>
#include <VectorMath.h>
#include <cudaInterface.h>
#include <cudaFFTWrapper.h>
#include <cutil_comfunc.h>

class MultiGaussianKernelGPU : 
  public KernelInterfaceGPU 
{

public:

  MultiGaussianKernelGPU();
  
  ~MultiGaussianKernelGPU();
  
  void SetSize(const SizeType &size, 
	       const SpacingType &spacing,
	       const KernelParam &params);
  
  /**
   * f = Lv
   * 
   * v field is overwritten in this operation (holds f).
   */
  void ApplyOperator(cplVector3DArray& dV){
    throw AtlasWerksException(__FILE__, __LINE__, 
			      "Error, ApplyOperator not implemented in "
			      "MultiGaussianKernel");
  }
  
  /**
   * v = Kf
   * 
   * f field is overwritten in this operation (holds v).
   */
  void ApplyInverseOperator(cplVector3DArray& dF);

  void FreeDeviceData();
  
protected:

  void InitDeviceData();

  void GenerateGaussianSum();

  SizeType mSize;
  unsigned int mNVox;
  SpacingType mSpacing;
  unsigned int mNGaussians;
  
  std::vector<Vector3D<float> > mSigma;
  std::vector<float> mWeight;

  RealImage mGaussian;

  cplComplex* mdKernelC;
  float *mdScratch;
  cplFFT3DConvolutionWrapper mdFFTFilter;
  
  // TEST
  int mCallCount;
  int mItersBetweenUpdate;
  std::vector<float> mWeightChange;
  // END TEST
};

#endif // __MULTI_GAUSSIAN_KERNEL_GPU_H__
