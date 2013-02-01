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

#ifndef __GREEDY_ITERATOR_PARAM_H__
#define __GREEDY_ITERATOR_PARAM_H__

#include "CompoundParam.h"
#include "KernelParam.h"

class GreedyIteratorParam : public CompoundParam {
public:
  GreedyIteratorParam(const std::string& name = "GreedyIterator", 
		      const std::string& desc = "Settings for greedy iterator", 
		      ParamLevel level = PARAM_COMMON)
    : CompoundParam(name, desc, level)
  {
    this->AddChild(KernelParam("Kernel"));

    this->
      AddChild(ValueParam<Real>("MaxPert",
				"Scale factor on the maximum velocity in a given "
				"deformation for computing delta",
				PARAM_COMMON,
				0.1));
  }
  ParamAccessorMacro(KernelParam, Kernel)
  ValueParamAccessorMacro(Real, MaxPert)
  
  CopyFunctionMacro(GreedyIteratorParam)
};

#endif
