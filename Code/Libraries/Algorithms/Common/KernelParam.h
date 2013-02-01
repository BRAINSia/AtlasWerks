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


#ifndef __KERNEL_PARAM_H__
#define __KERNEL_PARAM_H__

#include "DiffOperParam.h"
#include "MultiGaussianKernelParam.h"
#include "XORParam.h"

class KernelParam : 
  public XORParam 
{
public:
  KernelParam(const std::string& name = "Kernel", 
	      const std::string& desc = "kernel parameters", 
	      ParamLevel level = PARAM_COMMON) :
    XORParam(name, desc, level)
  {
    // add options for kernel
    this->AddChild(DiffOperParam()); // DiffOper
    this->AddChild(MultiGaussianKernelParam()); // MultiGaussian

  }

  CopyFunctionMacro(KernelParam)

  bool IsDiffOperParam() const {
    if(!mSetParam){
      throw AtlasWerksException(__FILE__, __LINE__, 
				"Error, no param set for KernelParam XORParam");
    }
    return this->GetSetParam()->MatchesName("DiffOper");
  }

  bool IsMultiGaussianKernelParam() const {
    if(!mSetParam){
      throw AtlasWerksException(__FILE__, __LINE__, 
				"Error, no param set for KernelParam XORParam");
    }
    return this->GetSetParam()->MatchesName("MultiGaussian");
  }
  
  DiffOperParam *AsDiffOperParam()
  {
    if(!mSetParam){
      throw AtlasWerksException(__FILE__, __LINE__, 
				"Error, no param set for KernelParam XORParam");
    }
    return dynamic_cast<DiffOperParam*>(this->GetSetParam());
  }

  const DiffOperParam *AsDiffOperParam() const
  {
    if(!mSetParam){
      throw AtlasWerksException(__FILE__, __LINE__, 
				"Error, no param set for KernelParam XORParam");
    }
    return dynamic_cast<const DiffOperParam*>(this->GetSetParam());
  }

  MultiGaussianKernelParam *AsMultiGaussianKernelParam()
  {
    if(!mSetParam){
      throw AtlasWerksException(__FILE__, __LINE__, 
				"Error, no param set for KernelParam XORParam");
    }
    return dynamic_cast<MultiGaussianKernelParam*>(this->GetSetParam());
  }

  const MultiGaussianKernelParam *AsMultiGaussianKernelParam() const
  {
    if(!mSetParam){
      throw AtlasWerksException(__FILE__, __LINE__, 
				"Error, no param set for KernelParam XORParam");
    }
    return dynamic_cast<const MultiGaussianKernelParam*>(this->GetSetParam());
  }

};

#endif // __KERNEL_PARAM_H__

