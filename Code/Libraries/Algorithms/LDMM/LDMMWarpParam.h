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

#ifndef __LDMM_WARP_PARAM_H__
#define __LDMM_WARP_PARAM_H__

#include "LDMMParam.h"

/** ################ LDMMWarpParam ################ **/

class LDMMWarpParam : public LDMMParam {
  // Debugging params...
public:
  LDMMWarpParam(const std::string& name = "LDMMWarp", 
		const std::string& desc = "Settings for LDMM warp", 
		ParamLevel level = PARAM_COMMON)
    : LDMMParam(name, desc, level)
  {
    this->AddChild(ValueParam<bool>("WriteDefImage", "Write out deformed (I0 at 1) image?", PARAM_COMMON, true));
    this->AddChild(ValueParam<bool>("WriteInvDefImage", "Write out inverse deformed (I1 at 0) image?", PARAM_COMMON, true));
    this->AddChild(ValueParam<bool>("WriteDefField", "Write out deformation (1 to 0) field?", PARAM_COMMON, true));
    this->AddChild(ValueParam<bool>("WriteInvDefField", "Write out inverse deformation (0 to 1) field?", PARAM_COMMON, true));
    this->AddChild(ValueParam<bool>("WriteVelocityFields", "Write out all (nTimesteps) velocity fields?", PARAM_COMMON, false));
    this->AddChild(ValueParam<bool>("WriteIntermediateImages", "Write out all (nTimesteps) intermediate deformed images?", PARAM_COMMON, false));
    this->AddChild(ValueParam<bool>("WriteAlphas", "Write out all (nTimesteps) Alpha images?", PARAM_COMMON, false));
    this->AddChild(ValueParam<bool>("WriteAlpha0", "Write out Alpha0 image?", PARAM_COMMON, false));
    this->AddChild(ValueParam<bool>("WriteEnergy", "Write out energy history for this warp?", PARAM_COMMON, true));

    // debug
    this->AddChild(ValueParam<bool>("WriteInitialScaleLevelDefImage", "Write out initial deformed image at each scale level?", PARAM_DEBUG, false));
    this->AddChild(ValueParam<bool>("WriteFinalScaleLevelDefImage", "Write out final deformed image at each scale level?", PARAM_DEBUG, false));
  }

  ValueParamAccessorMacro(bool, WriteDefImage)
  ValueParamAccessorMacro(bool, WriteInvDefImage)
  ValueParamAccessorMacro(bool, WriteDefField)
  ValueParamAccessorMacro(bool, WriteInvDefField)
  ValueParamAccessorMacro(bool, WriteVelocityFields)
  ValueParamAccessorMacro(bool, WriteIntermediateImages)
  ValueParamAccessorMacro(bool, WriteAlphas)
  ValueParamAccessorMacro(bool, WriteAlpha0)
  ValueParamAccessorMacro(bool, WriteEnergy)
  ValueParamAccessorMacro(bool, WriteInitialScaleLevelDefImage)
  ValueParamAccessorMacro(bool, WriteFinalScaleLevelDefImage)
};

#endif // __LDMM_WARP_PARAM_H__
