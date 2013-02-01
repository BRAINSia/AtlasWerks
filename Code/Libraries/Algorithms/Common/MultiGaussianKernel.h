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

#ifndef __MULTI_GAUSSIAN_KERNEL_H__
#define __MULTI_GAUSSIAN_KERNEL_H__

#include "AtlasWerksTypes.h"
#include "KernelInterface.h"
#include "MultiGaussianKernelParam.h"
#include "GaussianFilter3D.h"

template<class T>
class MultiGaussianKernelT : 
  public KernelInterfaceT<T> 
{

public:

  typedef typename KernelInterfaceT<T>::KernelInternalVF KernelInternalVF;

  MultiGaussianKernelT(const SizeType &size,
		       const SpacingType &spacing,
		       const MultiGaussianKernelParam &param);
  
  ~MultiGaussianKernelT();
  
  /**
   * Copy vf into the internal processing buffer.
   */
  void CopyIn(const VectorField &vf);
  
  /**
   * Copy the internal result to vf
   */
  void CopyOut(VectorField &vf);
  
  /**
   * This should be called before any data copied into kernel.
   */
  void Initialize(){};

  /**
   * Apply L operator
   */
  void ApplyOperator(){
    throw AtlasWerksException(__FILE__, __LINE__, 
			      "Error, ApplyOperator not implemented in MultiGaussianKernel");
  };

  /**
   * Apply inverse L operator
   */
  void ApplyInverseOperator();

  KernelInternalVF *GetInternalFFTWVectorField()
  {
    return this->mVFBuffer;
  }

  void pointwiseMultiplyBy_FFTW_Safe(const Array3D<Real> &rhs){
    this->mVFBuffer->pointwiseMultiplyBy(rhs);
  }
  
protected:

  SizeType mSize;
  SpacingType mSpacing;
  unsigned int mNGaussians;
  
  KernelInternalVF *mVFBuffer;
  KernelInternalVF *mSumBuffer;

  std::vector<Vector3D<float> > mSigma;
  std::vector<float> mWeight;

  RealImage *mXComponentImage;
  RealImage *mYComponentImage;
  RealImage *mZComponentImage;

  GaussianFilter3D mXGaussFilter;
  GaussianFilter3D mYGaussFilter;
  GaussianFilter3D mZGaussFilter;
};

typedef MultiGaussianKernelT<Real> MultiGaussianKernel;

#endif // __MULTI_GAUSSIAN_KERNEL_H__
