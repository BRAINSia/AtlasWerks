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

#ifndef __LDMM_ATLAS_PARAM_H__
#define __LDMM_ATLAS_PARAM_H__

#include "LDMMParam.h"

/* ################ LDMMAtlasParam ################ */

class LDMMAtlasParam : public LDMMParam {
  // Debugging params...
public:

  LDMMAtlasParam(const std::string& name = "LDMMAtlas", 
		 const std::string& desc = "Settings for LDMM atlas building", 
		 ParamLevel level = PARAM_COMMON)
    : LDMMParam(name, desc, level)
  {
    //
    // Atlas Generation
    // 
    this->AddChild(ValueParam<bool>("ComputeMedian", 
				    "Compute the geometric median image instead of the mean?",
				    PARAM_COMMON, false));
    this->AddChild(ValueParam<unsigned int>("TrimmedMeanSize", 
					    "If nonzero, compute trimmed mean using only this many images",
					    PARAM_COMMON, 0));
    this->AddChild(ValueParam<bool>("UsePerVoxelMedian", 
				    "Compute the voxel-wise median image instead of the mean?",
				    PARAM_COMMON, false));
    this->
      AddChild(ValueParam<bool>("JacobianScale",
				"Scale each image by jac. det. during atlas estimation",
				PARAM_COMMON,
				true));
    
    //
    // Restart Parameters
    //
    // for restarting from velocities
    this->AddChild(ValueParam<std::string>("InputVFieldFormat", 
					   "Format for the initial velocity fields, for example "
					   "VelFieldFor%sTime%%02d.mha",
					   PARAM_COMMON, ""));
    // for restarting from alpha0s and mean image
    this->AddChild(ValueParam<std::string>("InputAlpha0Format", 
					   "Format for the initial momentum, for example "
					   "LDMMAtlas%sAlpha0.mha",
					   PARAM_COMMON, ""));
    this->AddChild(ValueParam<std::string>("InputMeanImage", 
					   "Name of the mean image used for restarting from Alpha0s",
					   PARAM_COMMON, ""));
    // for restarting from either
    this->AddChild(ValueParam<unsigned int>("StartScaleLevel", 
					    "Initial scale level to start at (ie '1' would start at "
					    "second-from-lowest-resolution level, '0' means run as normal",
					    PARAM_RARE, 0));
    this->AddChild(ValueParam<unsigned int>("StartIter", 
					    "Initial iteration number to start on, used for restarting",
					    PARAM_RARE, 0));
    //
    // PCA
    //
    this->AddChild(ValueParam<bool>("DoPCAStep", "Run PCA on deformed images, projecting onto first component for deformation", PARAM_RARE, false));
    this->AddChild(ValueParam<unsigned int>("NPCAComponents", "If DoPCAStep is true, the number of PCA components to retain", PARAM_RARE, 1));
    this->AddChild(ValueParam<unsigned int>("NPowerPCAIters", "number of power method iterations per PCA component", PARAM_RARE, 20));
    this->AddChild(ValueParam<bool>("DoMeanSubtraction", "Subtract mean before doing PCA?", PARAM_RARE, true));

    //
    // output params
    //
    this->AddChild(ValueParam<bool>("WriteMeanImage", "Write out the mean image?", PARAM_COMMON, true));
    this->AddChild(ValueParam<bool>("WriteDefImages", "Write out all (nImages) images deformed to mean?", PARAM_COMMON, true));

    // This param used to be WriteInvDefImages, but was a confusing
    // name.  Keep the old name as an alias.
    ValueParam<bool> WriteDefMeanParam("WriteDefMean", "Write out mean deformed to all (nImages) images?", PARAM_COMMON, true);
    WriteDefMeanParam.AddAlias("WriteInvDefImages");
    this->AddChild(WriteDefMeanParam);

    // Def / InvDef is confusing, make direction explicit, but keep
    // old name as alias
    ValueParam<bool> WriteDefFields0To1Param("WriteDefFieldsMeanToIm", "Write out all (nImages) deformation fields, 0 to 1 direction?", PARAM_COMMON, true);
    WriteDefFields0To1Param.AddAlias("WriteInvDefFields");
    this->AddChild(WriteDefFields0To1Param);
    ValueParam<bool> WriteDefFields1To0Param("WriteDefFieldsImToMean", "Write out all (nImages) deformation fields, 1 to 0 direction?", PARAM_COMMON, true);
    WriteDefFields1To0Param.AddAlias("WriteDefFields");
    this->AddChild(WriteDefFields1To0Param);

    this->AddChild(ValueParam<bool>("WriteEnergy", "Write total energy history for atlas as an XML file?", PARAM_COMMON, true));
    this->AddChild(ValueParam<bool>("WriteIndividualEnergies", "Write separate XML file of energy history for each image deformation?", PARAM_COMMON, false));

    this->AddChild(ValueParam<bool>("WriteVelocityFields", "Write out all (nImages*nTimesteps) velocity fields?", PARAM_COMMON, false));
    this->AddChild(ValueParam<bool>("WriteIntermediateImages", "Write out all (nImages*nTimesteps) intermediate deformed images?", PARAM_COMMON, false));
    this->AddChild(ValueParam<bool>("WriteAlphas", "Write out all (nImages*nTimesteps) Alpha images?", PARAM_COMMON, false));
    this->AddChild(ValueParam<bool>("WriteAlpha0s", "Write out all (nImages) Alpha0 images?", PARAM_COMMON, false));

    // debug
    this->AddChild(ValueParam<bool>("WriteInputImages", "Write out preprocessed input images?", PARAM_DEBUG, false));
    this->AddChild(ValueParam<bool>("WriteInitialScaleLevelMean", "Write out initial mean image at each scale level?", PARAM_DEBUG, false));
    this->AddChild(ValueParam<bool>("WriteFinalScaleLevelMean", "Write out final mean image at each scale level?", PARAM_DEBUG, false));

  }

  ValueParamAccessorMacro(bool, ComputeMedian)
  ValueParamAccessorMacro(unsigned int, TrimmedMeanSize)
  ValueParamAccessorMacro(bool, UsePerVoxelMedian)
  ValueParamAccessorMacro(bool, JacobianScale)
  ValueParamAccessorMacro(std::string, InputVFieldFormat)
  ValueParamAccessorMacro(std::string, InputAlpha0Format)
  ValueParamAccessorMacro(std::string, InputMeanImage)
  ValueParamAccessorMacro(unsigned int, StartScaleLevel)
  ValueParamAccessorMacro(unsigned int, StartIter)

  ValueParamAccessorMacro(bool, DoPCAStep)
  ValueParamAccessorMacro(bool, DoMeanSubtraction)
  ValueParamAccessorMacro(unsigned int, NPCAComponents)
  ValueParamAccessorMacro(unsigned int, NPowerPCAIters)

  ValueParamAccessorMacro(bool, WriteMeanImage)
  ValueParamAccessorMacro(bool, WriteDefImages)
  ValueParamAccessorMacro(bool, WriteDefMean)
  ValueParamAccessorMacro(bool, WriteDefFieldsMeanToIm)
  ValueParamAccessorMacro(bool, WriteDefFieldsImToMean)
  ValueParamAccessorMacro(bool, WriteEnergy)
  ValueParamAccessorMacro(bool, WriteIndividualEnergies)
  ValueParamAccessorMacro(bool, WriteVelocityFields)
  ValueParamAccessorMacro(bool, WriteIntermediateImages)
  ValueParamAccessorMacro(bool, WriteAlphas)
  ValueParamAccessorMacro(bool, WriteAlpha0s)
  ValueParamAccessorMacro(bool, WriteInputImages)
  ValueParamAccessorMacro(bool, WriteInitialScaleLevelMean)
  ValueParamAccessorMacro(bool, WriteFinalScaleLevelMean)
};

#endif // __LDMM_ATLAS_PARAM_H__
