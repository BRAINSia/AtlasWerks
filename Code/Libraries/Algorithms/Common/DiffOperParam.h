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


#ifndef __DIFF_OPER_PARAM_H__
#define __DIFF_OPER_PARAM_H__

#ifndef SWIG

#include "AtlasWerksTypes.h"
#include "ValueParam.h"
#include "CompoundParam.h"

class DiffOperParam : public CompoundParam {
public:
  
  DiffOperParam(const std::string& name = "DiffOper", 
		const std::string& desc = "Differential operator parameters", 
		ParamLevel level = PARAM_COMMON)
    : CompoundParam(name, desc, level)
  {
    this->
      AddChild(ValueParam<Real>("Alpha",
				"Controls the 'viscosity' of the fluid",
				PARAM_COMMON,
				0.01));
    this->
      AddChild(ValueParam<Real>("Beta",
				"Controls the 'viscosity' of the fluid (penalizes expansion/compression)",
				PARAM_COMMON,
				0.01));
    this->
      AddChild(ValueParam<Real>("Gamma",
				"ensures inverability of the operator",
				PARAM_COMMON,
				0.001));
    this->
      AddChild(ValueParam<Real>("LPow",
				"'power' of L to use",
				PARAM_RARE,
				1.0));
    this->
      AddChild(ValueParam<bool>("UseEigenLUT",
				"Use lookup table of eigenvalues, trades memory for speed",
				PARAM_RARE,
				false));
    this->
      AddChild(ValueParam<bool>("DivergenceFree",
				"Compute a divergence-free deformation",
				PARAM_RARE,
				false));
    this->
      AddChild(ValueParam<unsigned int>("FFTWNumberOfThreads",
					"Number of threads for FFTW library to use",
					PARAM_RARE,
					1));
    this->
      AddChild(ValueParam<bool>("FFTWMeasure",
				"Do performance measuring during plan construction",
				PARAM_RARE,
				true));

  }
  
  ValueParamAccessorMacro(Real, Alpha)
  ValueParamAccessorMacro(Real, Beta)
  ValueParamAccessorMacro(Real, Gamma)
  ValueParamAccessorMacro(Real, LPow)
  ValueParamAccessorMacro(bool, UseEigenLUT)
  ValueParamAccessorMacro(bool, DivergenceFree)
  ValueParamAccessorMacro(unsigned int, FFTWNumberOfThreads)
  ValueParamAccessorMacro(bool, FFTWMeasure)

  CopyFunctionMacro(DiffOperParam)

};

#endif // SWIG

#endif // __DIFF_OPER_PARAM_H__
