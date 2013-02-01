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

#ifndef __KERNEL_INTERFACE_GPU_H__
#define __KERNEL_INTERFACE_GPU_H__

#include <cudaVector3DArray.h>
#include "KernelParam.h"

class KernelInterfaceGPU {
  
public:

  virtual ~KernelInterfaceGPU(){}
  
  virtual void SetSize(const SizeType &logicalSize, 
		       const SpacingType &spacing,
		       const KernelParam &params) = 0;
  /**
   * f = Lv
   * 
   * v field is overwritten in this operation (holds f).
   */
  virtual void ApplyOperator(cplVector3DArray& dV) = 0;
  
  /**
   * v = Kf
   * 
   * f field is overwritten in this operation (holds v).
   */
  virtual void ApplyInverseOperator(cplVector3DArray& dF) = 0;
  
};

#endif // __KERNEL_INTERFACE_GPU_H__
