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


#ifndef __MULTI_GAUSSIAN_KERNEL_PARAM_H__
#define __MULTI_GAUSSIAN_KERNEL_PARAM_H__

#include "CompoundParam.h"
#include "ValueParam.h"
#include "MultiParam.h"

class GaussianParam :
  public CompoundParam
{
public:
  GaussianParam(const std::string& name = "Gaussian", 
		const std::string& desc = "single gaussian parameters", 
		ParamLevel level = PARAM_COMMON)
    : CompoundParam(name, desc, level)
  {
    this->AddChild(ValueParam<float>("Sigma",
				     "sigma for isotropic gaussian",
				     PARAM_COMMON,
				     1.0));
    this->AddChild(ValueParam<float>("Weight",
				     "Weight of this gaussian",
				     PARAM_COMMON,
				     1.0));

    // TEST
    this->AddChild(ValueParam<float>("WeightChange",
				     "How much do we change the weight of this gaussian every update?",
				     PARAM_DEBUG,
				     0.0));
    // END TEST
  }

  ValueParamAccessorMacro(float, Sigma)
  ValueParamAccessorMacro(float, Weight)
  // TEST
  ValueParamAccessorMacro(float, WeightChange)
  // END TEST

  CopyFunctionMacro(GaussianParam)

};

class MultiGaussianKernelParam : 
  public CompoundParam 
{
public:
  
  MultiGaussianKernelParam(const std::string& name = "MultiGaussian", 
			   const std::string& desc = "kernel parameters", 
			   ParamLevel level = PARAM_COMMON)
    : CompoundParam(name, desc, level)
  {
    this->AddChild
      (MultiParam<GaussianParam>
       (GaussianParam("Gaussian")));
    // TEST
    this->AddChild(ValueParam<int>("ItersBetweenUpdate",
				   "How much do we change the weight of this gaussian every update?",
				   PARAM_DEBUG,
				   0));
    // END TEST
  }
  
  ParamAccessorMacro(MultiParam<GaussianParam >, Gaussian)
  // TEST
  ValueParamAccessorMacro(int, ItersBetweenUpdate)
  // END TEST
  
  CopyFunctionMacro(MultiGaussianKernelParam)

};

#endif // __MULTI_GAUSSIAN_KERNEL_PARAM_H__
