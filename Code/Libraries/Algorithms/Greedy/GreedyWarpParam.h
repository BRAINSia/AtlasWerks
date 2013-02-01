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

#ifndef __GREEDY_WARP_PARAM_H__
#define __GREEDY_WARP_PARAM_H__

#include "GreedyParam.h"

/** ################ GreedyWarpParam ################ **/

class GreedyWarpParam : public GreedyParam {
public:
  GreedyWarpParam(const std::string& name = "GreedyWarp", 
		  const std::string& desc = "Settings for Greedy warp", 
		  ParamLevel level = PARAM_REQUIRED)
    : GreedyParam(name, desc, level)
  {
    this->AddChild(ValueParam<bool>("WriteDefImage", "Write out deformed (I0 at 1) image?", PARAM_COMMON, true));
    this->AddChild(ValueParam<bool>("WriteInvDefImage", "Write out inverse (I1 at 0) deformation field?", PARAM_COMMON, true));
    this->AddChild(ValueParam<bool>("WriteDefField", "Write out deformation (1To0) field?", PARAM_COMMON, true));
    this->AddChild(ValueParam<bool>("WriteInvDefField", "Write out inverse (0To1) deformation field?", PARAM_COMMON, true));
    this->AddChild(ValueParam<bool>("WriteEnergy", "Write out energy history for this warp?", PARAM_COMMON, true));
    // debug
    this->AddChild(ValueParam<bool>("WriteScaleLevelImages", "Write out I0 and IT at each scale level?", PARAM_DEBUG, false));
    this->AddChild(ValueParam<bool>("WriteInitialScaleLevelDefImage", "Write out initial scale level deformed image?", PARAM_DEBUG, false));
    this->AddChild(ValueParam<bool>("WriteInitialScaleLevelInvDefImage", "Write out initial scale level deformed image?", PARAM_DEBUG, false));
    this->AddChild(ValueParam<bool>("WriteInitialScaleLevelDefField", "Write out initial scale level deformed image?", PARAM_DEBUG, false));
    this->AddChild(ValueParam<bool>("WriteInitialScaleLevelInvDefField", "Write out initial scale level deformed image?", PARAM_DEBUG, false));
    this->AddChild(ValueParam<bool>("WriteFinalScaleLevelDefImage", "Write out initial scale level deformed image?", PARAM_DEBUG, false));
    this->AddChild(ValueParam<bool>("WriteFinalScaleLevelInvDefImage", "Write out initial scale level deformed image?", PARAM_DEBUG, false));
    this->AddChild(ValueParam<bool>("WriteFinalScaleLevelDefField", "Write out initial scale level deformed image?", PARAM_DEBUG, false));
    this->AddChild(ValueParam<bool>("WriteFinalScaleLevelInvDefField", "Write out initial scale level deformed image?", PARAM_DEBUG, false));
  }
  
  ValueParamAccessorMacro(bool, WriteDefImage)
  ValueParamAccessorMacro(bool, WriteInvDefImage)
  ValueParamAccessorMacro(bool, WriteDefField)
  ValueParamAccessorMacro(bool, WriteInvDefField)
  ValueParamAccessorMacro(bool, WriteEnergy)
  // debug
  ValueParamAccessorMacro(bool, WriteScaleLevelImages);
  ValueParamAccessorMacro(bool, WriteInitialScaleLevelDefImage);
  ValueParamAccessorMacro(bool, WriteInitialScaleLevelInvDefImage);
  ValueParamAccessorMacro(bool, WriteInitialScaleLevelDefField);
  ValueParamAccessorMacro(bool, WriteInitialScaleLevelInvDefField);
  ValueParamAccessorMacro(bool, WriteFinalScaleLevelDefImage);
  ValueParamAccessorMacro(bool, WriteFinalScaleLevelInvDefImage);
  ValueParamAccessorMacro(bool, WriteFinalScaleLevelDefField);
  ValueParamAccessorMacro(bool, WriteFinalScaleLevelInvDefField);
};

#endif // __GREEDY_WARP_PARAM_H__
