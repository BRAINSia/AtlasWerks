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

#ifndef __GREEDY_ATLAS_BUILDER_INTERFACE_H__
#define __GREEDY_ATLAS_BUILDER_INTERFACE_H__

#include "AtlasWerksTypes.h"

class GreedyAtlasBuilderInterface {
public:
  
  virtual ~GreedyAtlasBuilderInterface(){}
  
  virtual void BuildAtlas() = 0;

  virtual void SetComputeInverseHFields(bool computeInverseHFields) = 0;
  virtual bool GetComputeInverseHFields() = 0;
  
  virtual void GetDeformedImage(int imIdx, RealImage &im)=0;
  virtual void GetHField(int imIdx, VectorField &vf)=0;
  virtual void GetInvHField(int imIdx, VectorField &vf)=0;
  virtual void GetMeanImage(RealImage &mean)=0;

};

#endif 
