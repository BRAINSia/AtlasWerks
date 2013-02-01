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

#ifndef __LDMM_PARAM__
#define __LDMM_PARAM__

#include "MultiscaleManager.h"
#include "LDMMIteratorParam.h"

class LDMMScaleLevelParam : public ScaleLevelSettingsParam 
{
public:
  LDMMScaleLevelParam(const std::string& name = "LDMMScaleLevel", 
		      const std::string& desc = "Settings for single-scale LDMM registration", 
		      ParamLevel level = PARAM_COMMON)
    : ScaleLevelSettingsParam(name, desc, level)
  {
    LDMMIteratorParam iterParam("Iterator");
    // for backwards compatability
    iterParam.AddAlias("LDMMIterator");
    this->AddChild(iterParam);
    this->
      AddChild(ValueParam<unsigned int>("NIterations",
					"Number of iterations",
					PARAM_COMMON,
					50));
    this->AddChild(ValueParam<unsigned int>("OutputEveryNIterations", "If nonzero, write output every N iterations", PARAM_DEBUG, 0));
    this->AddChild(ValueParam<unsigned int>("ReparameterizeEveryNIterations", "If nonzero, reparameterize for constant speed every N iterations", PARAM_COMMON, 0));
  }
  
  ParamAccessorMacro(LDMMIteratorParam, Iterator)
  ValueParamAccessorMacro(unsigned int, NIterations)
  ValueParamAccessorMacro(unsigned int, ReparameterizeEveryNIterations)
  ValueParamAccessorMacro(unsigned int, OutputEveryNIterations)
  
  CopyFunctionMacro(LDMMScaleLevelParam)
  
};

class LDMMParam : public MultiscaleParamBase<LDMMScaleLevelParam> {

public:

  LDMMParam(const std::string& name = "LDMM", 
	    const std::string& desc = "Settings for LDMM registration/atlas building", 
	    ParamLevel level = PARAM_COMMON)
    : MultiscaleParamBase<LDMMScaleLevelParam>(LDMMScaleLevelParam("LDMMScaleLevel"), name, desc, level)
  {
    this->
      AddChild(ValueParam<unsigned int>("NTimeSteps",
					"Number of timesteps (and therefore intermediate vector fields) to use",
					PARAM_COMMON,
					5));

    this->
      AddChild(ValueParam<bool>("AutoStepReduce",
				"Automatically reduce the step size if energy increases",
				PARAM_COMMON,
				false));

    
    this->AddChild(ValueParam<std::string>("OutputPrefix", "filename prefix to use", PARAM_COMMON, ""));  
    this->AddChild(ValueParam<std::string>("OutputSuffix", "filename extension to use (determines format)", PARAM_COMMON, "mha"));
  }

  ValueParamAccessorMacro(unsigned int, NTimeSteps)
  ValueParamAccessorMacro(bool, AutoStepReduce)
  ValueParamAccessorMacro(std::string, OutputPrefix)
  ValueParamAccessorMacro(std::string, OutputSuffix)
  
  CopyFunctionMacro(LDMMParam)

};

#endif // __LDMM_PARAM__
