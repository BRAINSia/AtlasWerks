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

#ifndef __KERNEL_INTERFACE_H__
#define __KERNEL_INTERFACE_H__

#include "AtlasWerksTypes.h"

template<class T>
class KernelInterfaceT {

public:
  
  typedef Array3D<Vector3D<T> > KernelInternalVF;

  virtual ~KernelInterfaceT(){}

  /**
   * Copy vf into the internal processing buffer.
   */
  virtual void CopyIn(const VectorField &vf) = 0;
  
  /**
   * Copy the internal result to vf
   */
  virtual void CopyOut(VectorField &vf) = 0;
  
  /**
   * This should be called before any data copied into kernel.
   */
  virtual void Initialize() = 0;

  /**
   * Apply L operator
   */
  virtual void ApplyOperator() = 0;

  /**
   * Apply inverse L operator
   */
  virtual void ApplyInverseOperator() = 0;

  /**
   * Pointwise multiply the internal vector field by rhs
   */
  virtual void pointwiseMultiplyBy_FFTW_Safe(const Array3D<Real> &rhs) = 0;

  /**
   * Get pointer to internal vector field.
   */
  virtual KernelInternalVF *GetInternalFFTWVectorField() = 0;
  
};

typedef KernelInterfaceT<Real> KernelInterface;

#endif // __KERNEL_INTERFACE_H__
