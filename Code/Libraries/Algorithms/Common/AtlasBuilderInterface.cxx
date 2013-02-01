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


#include "AtlasBuilderInterface.h"

#include "log.h"

AtlasBuilderInterface::
AtlasBuilderInterface(const WeightedImageSet &imageSet,
		unsigned int nThreads,
		unsigned int nodeId,
		unsigned int nNodes,
		unsigned int nTotalImages)
  : mImageSet(imageSet),
    mNThreads(nThreads),
    mNImages(0),
    mNodeId(nodeId),
    mNNodes(nNodes),
    mNTotalImages(nTotalImages),
    mImSize(0,0,0),
    mImOrigin(0,0,0),
    mImSpacing(1,1,1)
{
  mNImages = mImageSet.NumImages();
  mImSize = mImageSet.GetImageSize();
  mImOrigin = mImageSet.GetImageOrigin();
  mImSpacing = mImageSet.GetImageSpacing();
  if(nTotalImages == 0){
    if(nNodes == 1){
      nTotalImages = mImageSet.NumImages();
    }else{
      throw AtlasWerksException(__FILE__, __LINE__, "Error, nTotalImages = 0 but nNodes > 1");
    }
  }
}
  


