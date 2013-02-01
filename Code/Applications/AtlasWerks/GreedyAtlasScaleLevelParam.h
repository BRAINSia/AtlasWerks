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


#ifndef __GREEDY_ATLAS_SCALE_LEVEL_PARAM_H__
#define __GREEDY_ATLAS_SCALE_LEVEL_PARAM_H__

#include "AtlasBuilder.h"

class GreedyAtlasScaleLevelParam : public AtlasBuilderParam {
public:
  GreedyAtlasScaleLevelParam() 
    : AtlasBuilderParam("GreedyAtlasScaleLevel")
  {
    this->AddChild(ScaleLevelParam("ScaleLevel"));
  }
  ParamAccessorMacro(ScaleLevelParam, ScaleLevel)
};

#endif // __GREEDY_ATLAS_SCALE_LEVEL_PARAM_H__
