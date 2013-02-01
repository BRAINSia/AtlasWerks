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

#ifndef __LDMM_ITERATOR_PARAM_H__
#define __LDMM_ITERATOR_PARAM_H__

#include "AtlasWerksTypes.h"
#include "CompoundParam.h"
#include "KernelParam.h"

class LDMMIteratorParam : public CompoundParam {
public:
  LDMMIteratorParam(const std::string& name = "LDMMIterator", 
		    const std::string& desc = "Settings for LDMM iteration", 
		    ParamLevel level = PARAM_COMMON)
    : CompoundParam(name, desc, level)
  {
    this->AddChild(KernelParam("Kernel"));
    
    this->
      AddChild(ValueParam<Real>("Sigma",
				"Controls tradeoff between image matching and velocity field smoothness",
				PARAM_COMMON,
				5.0));
    this->
      AddChild(ValueParam<Real>("StepSize",
				"Gradient descent step size, or maximum step size when using adaptive step size",
				PARAM_RARE,
				0.005));
    this->
      AddChild(ValueParam<bool>("UseAdaptiveStepSize",
				"Use an adaptive step size, where step size is scaled to be maxPert*minSpacing of the first iteration",
				PARAM_RARE,
				true));
    this->
      AddChild(ValueParam<Real>("MaxPert",
				"when using adaptive step size, step will be scaled to maxPert*minSpacing",
				PARAM_COMMON,
				0.1));
    this->
      AddChild(ValueParam<bool>("VerboseEnergy",
				"Print individual velocity step energies as well as total deformation energy",
				PARAM_RARE,
				false));
  }
  
  ParamAccessorMacro(KernelParam, Kernel)
  ValueParamAccessorMacro(Real, Sigma)
  ValueParamAccessorMacro(Real, StepSize)
  ValueParamAccessorMacro(bool, UseAdaptiveStepSize)
  ValueParamAccessorMacro(Real, MaxPert)
  ValueParamAccessorMacro(bool, VerboseEnergy)

  CopyFunctionMacro(LDMMIteratorParam)

};

#endif // __LDMM_ITERATOR_PARAM_H__

