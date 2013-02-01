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

#ifndef __KERNEL_FACTORY_H__
#define __KERNEL_FACTORY_H__

#include "AtlasWerksTypes.h"
#include "KernelParam.h"
#include "KernelInterface.h"

#ifdef CUDA_ENABLED
#include "KernelInterfaceGPU.h"
#endif // CUDA_ENABLED

/**
 * Simple class for generating the correct type of kernel based on a
 * KernelParam
 */
class KernelFactory {
public:
  static KernelInterface *NewKernel(const KernelParam &param, 
				    const SizeType &size,
				    const SpacingType &spacing);
#ifdef CUDA_ENABLED  
  static KernelInterfaceGPU *NewGPUKernel(const KernelParam &param);
#endif // CUDA_ENABLED
};

#endif // __KERNEL_FACTORY_H__
