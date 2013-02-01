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

#ifndef __GREEDY_ATLAS_PARAM_H__
#define __GREEDY_ATLAS_PARAM_H__

#include "GreedyParam.h"

/** ################ GreedyAtlasParam ################ **/

class GreedyAtlasParam : public GreedyParam {
public:
  GreedyAtlasParam(const std::string& name = "GreedyAtlas", 
		  const std::string& desc = "Settings for Greedy warp", 
		  ParamLevel level = PARAM_REQUIRED)
    : GreedyParam(name, desc, level)
  {

    this->AddChild(ValueParam<bool>("DoPCAStep", "Run PCA on deformed images, projecting onto first component for deformation", PARAM_RARE, false));
    this->AddChild(ValueParam<bool>("DoMeanSubtraction", "Subtract mean before doing PCA?", PARAM_RARE, false));
    this->AddChild(ValueParam<unsigned int>("NPCAComponents", "If DoPCAStep is true, the number of PCA components to retain", PARAM_RARE, 1));
    this->AddChild(ValueParam<unsigned int>("NPowerPCAIters", "number of power method iterations per PCA component", PARAM_RARE, 20));

    this->AddChild(ValueParam<bool>("WriteMeanImage", "Write out mean image?", PARAM_COMMON, true));
    this->AddChild(ValueParam<bool>("WriteDefImages", "Write out all (nImages) images deformed to the mean?", PARAM_COMMON, true));

    // This param used to be WriteInvDefImages, but was a confusing
    // name.  Keep the old name as an alias.
    ValueParam<bool> WriteDefMeanParam("WriteDefMean", "Write out mean deformed to all (nImages) images?  Note inverse deformation used to calculate these images is only approximate.", PARAM_COMMON, true);
    WriteDefMeanParam.AddAlias("WriteInvDefImages");
    this->AddChild(WriteDefMeanParam);

    // Def / InvDef is confusing, make direction explicit, but keep
    // old name as alias
    ValueParam<bool> WriteDefFields0To1Param("WriteDefFieldsImToMean", "Write out all (nImages) deformation fields, 0 to 1 direction?  Note that this is only a rough approximation of the inverse of 1 to 0 deformation.", PARAM_COMMON, true);
    WriteDefFields0To1Param.AddAlias("WriteInvDefFields");
    this->AddChild(WriteDefFields0To1Param);
    ValueParam<bool> WriteDefFields1To0Param("WriteDefFieldsMeanToIm", "Write out all (nImages) deformation fields, 1 to 0 direction?", PARAM_COMMON, true);
    WriteDefFields1To0Param.AddAlias("WriteDefFields");
    this->AddChild(WriteDefFields1To0Param);

    this->AddChild(ValueParam<bool>("WriteEnergy", "Write total energy history for atlas as an XML file?", PARAM_COMMON, true));
    this->AddChild(ValueParam<bool>("WriteIndividualEnergies", "Write separate XML file of energy history for each image deformation?", PARAM_COMMON, false));

    // debug
    this->AddChild(ValueParam<bool>("WriteScaleLevelImages", "Write out I0 and IT at each scale level?", PARAM_DEBUG, false));
    this->AddChild(ValueParam<bool>("WriteInitialScaleLevelMean", "Write out initial scale level mean image?", PARAM_DEBUG, false));
    this->AddChild(ValueParam<bool>("WriteFinalScaleLevelMean", "Write out final scale level mean image?", PARAM_DEBUG, false));
  }
  
  ValueParamAccessorMacro(bool, DoPCAStep)
  ValueParamAccessorMacro(bool, DoMeanSubtraction)
  ValueParamAccessorMacro(unsigned int, NPCAComponents)
  ValueParamAccessorMacro(unsigned int, NPowerPCAIters)

  ValueParamAccessorMacro(bool, WriteMeanImage)
  ValueParamAccessorMacro(bool, WriteDefImages)
  ValueParamAccessorMacro(bool, WriteDefMean)
  ValueParamAccessorMacro(bool, WriteDefFieldsImToMean)
  ValueParamAccessorMacro(bool, WriteDefFieldsMeanToIm)
  ValueParamAccessorMacro(bool, WriteEnergy)
  ValueParamAccessorMacro(bool, WriteIndividualEnergies)

  // debug
  ValueParamAccessorMacro(bool, WriteScaleLevelImages);
  ValueParamAccessorMacro(bool, WriteInitialScaleLevelMean);
  ValueParamAccessorMacro(bool, WriteFinalScaleLevelMean);
};

#endif // __GREEDY_WARP_PARAM_H__
