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

#include "KernelFactory.h"

#include "DiffOper.h"
#include "MultiGaussianKernel.h"

#ifdef CUDA_ENABLED
#include "DiffOperGPU.h"
#include "MultiGaussianKernelGPU.h"
#endif // CUDA_ENABLED

KernelInterface*
KernelFactory::
NewKernel(const KernelParam &param, 
	  const SizeType &size,
	  const SpacingType &spacing)
{
  KernelInterface *kernel;
  
  if(param.IsDiffOperParam()){
    kernel = new DiffOper(size, spacing, *param.AsDiffOperParam());
  }else if(param.IsMultiGaussianKernelParam()){
    kernel = new MultiGaussianKernel(size, spacing, 
				     *param.AsMultiGaussianKernelParam());
  }else{
    throw AtlasWerksException(__FILE__, __LINE__, "Error, Unknown kernel type");
  }
  return kernel;
}


#ifdef CUDA_ENABLED
KernelInterfaceGPU*
KernelFactory::
NewGPUKernel(const KernelParam &param)
{
  KernelInterfaceGPU *kernel;
  
  if(param.IsDiffOperParam()){
    kernel = new DiffOperGPU();
  }else if(param.IsMultiGaussianKernelParam()){
    kernel = new MultiGaussianKernelGPU();
  }else{
    throw AtlasWerksException(__FILE__, __LINE__, "Error, Unknown kernel type");
  }
  return kernel;
}
#endif // CUDA_ENABLED
