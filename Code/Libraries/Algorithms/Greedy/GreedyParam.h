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

#ifndef __GREEDY_PARAM_H__
#define __GREEDY_PARAM_H__

#include "MultiscaleManager.h"
#include "GreedyIteratorParam.h"

class GreedyScaleLevelParam : public ScaleLevelSettingsParam 
{
public:
  GreedyScaleLevelParam(const std::string& name = "GreedyScaleLevel", 
			const std::string& desc = "Settings for single-scale Greedy registration", 
			 ParamLevel level = PARAM_COMMON)
    : ScaleLevelSettingsParam(name, desc, level)
  {
    GreedyIteratorParam iterParam("Iterator");
    // for backwards compatability
    iterParam.AddAlias("GreedyIterator");
    this->AddChild(iterParam);
    this->
      AddChild(ValueParam<unsigned int>("NIterations",
					"Number of iterations",
					PARAM_COMMON,
					50));
  }
  
  ParamAccessorMacro(GreedyIteratorParam, Iterator)
  ValueParamAccessorMacro(unsigned int, NIterations)
  
  CopyFunctionMacro(GreedyScaleLevelParam)
  
};

#ifdef SWIG
%template(GreedyScaleLevel_MultiscaleParam) MultiscaleParamBase<GreedyScaleLevelParam>;
%template(GreedyScaleLevel_MultiParam) MultiParam<GreedyScaleLevelParam>;
#endif // SWIG

class GreedyParam : public MultiscaleParamBase<GreedyScaleLevelParam> {

public:
  GreedyParam(const std::string& name = "Greedy", 
	    const std::string& desc = "Settings for Greedy registration/atlas building", 
	    ParamLevel level = PARAM_REQUIRED)
    : MultiscaleParamBase<GreedyScaleLevelParam>(GreedyScaleLevelParam("GreedyScaleLevel"), name, desc, level)
  {
    this->AddChild(ValueParam<std::string>("OutputPrefix", "filename prefix to use", PARAM_COMMON, "Greedy"));  
    this->AddChild(ValueParam<std::string>("OutputSuffix", "filename extension to use (determines format)", PARAM_COMMON, "mha"));
    this->
      AddChild(ValueParam<bool>("AutoStepReduce",
				"Automatically reduce the step size if energy increases",
				PARAM_COMMON,
				false));
  }

  ValueParamAccessorMacro(bool, AutoStepReduce)
  ValueParamAccessorMacro(std::string, OutputPrefix)
  ValueParamAccessorMacro(std::string, OutputSuffix)
  
  CopyFunctionMacro(GreedyParam)
  
};

#endif // __GREEDY_PARAM_H__
