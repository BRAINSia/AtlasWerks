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


#ifndef __ATLAS_BUILDER_INTERFACE_H__
#define __ATLAS_BUILDER_INTERFACE_H__

#include "AtlasWerksTypes.h"
#include "WeightedImageSet.h"
#include "MultiscaleManager.h"

class AtlasBuilderInterface
{
  
public:
  
  AtlasBuilderInterface(const WeightedImageSet &imageSet,
		  unsigned int nThreads,
		  unsigned int nodeId = 0,
		  unsigned int nNodes = 1,
		  unsigned int nTotalImages = 0);

  virtual ~AtlasBuilderInterface(){}
  
  virtual void BuildAtlas() = 0;

  virtual void GenerateOutput() = 0;
  
  virtual const MultiscaleManager &GetScaleManager() const = 0;
  
  // called from individual threads, performs thread synchronization
  virtual void ComputeMean() = 0;
  virtual void SetScaleLevel(int scaleLevel) = 0;
  virtual void ComputeWeights(){
    throw 
      AtlasWerksException(__FILE__,__LINE__,
			  "Error, ComputeWeights unimplemented!");
  }
  
  /** Called at the beginning of a scale level */
  virtual void BeginScaleLevel(int scaleLevel){};
  
  /** Called at the beginning of an iteration */
  virtual void BeginIteration(int iteration){};
  
  /** Called at the end of an iteration (after new mean image is generated) */
  virtual void FinishIteration(int iteration){};
  
  /** Called at the end of a scale level */
  virtual void FinishScaleLevel(int scaleLevel){};

protected:

  //
  // Data
  //
  
  const WeightedImageSet &mImageSet;
  
  unsigned int mNThreads;
  unsigned int mNImages;
  unsigned int mNodeId;
  unsigned int mNNodes;
  unsigned int mNTotalImages;

  SizeType mImSize;
  OriginType mImOrigin;
  SpacingType mImSpacing;
  
};

#endif // __ATLAS_BUILDER_INTERFACE_H__
