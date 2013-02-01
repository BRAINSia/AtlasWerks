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

#ifndef __IMAGE_PCA_H__
#define __IMAGE_PCA_H__

#include <iostream>
#include <vector>
#include <cstdlib>

#include "AtlasWerksTypes.h"
#include "Matrix.h"

class ImagePCA {

public:
  ImagePCA(unsigned int nImages, 
	   SizeType imSize = SizeType(1,1,1),
	   unsigned int nComponents = 1);

  void SetImageSize(SizeType imSize);

  void SetNComponents(unsigned int nComponents);

  void ComputePCA(std::vector<RealImage*> &images,
		  bool computeProjections = true);
  
  void ComputePCAPower(std::vector<RealImage*> &images);

  RealImage& GetComponent(unsigned int i);

  void RandomizeComponents();

  void SetNPowerIters(unsigned int iters){ mNPowerIters = iters; }
  unsigned int GetNPowerIters(){ return mNPowerIters;}

protected:

  unsigned int mNImages;
  SizeType mImSize;
  unsigned int mNVox;
  unsigned int mNComponents;
  unsigned int mNThreads;
  unsigned int mNPowerIters;
  
  Matrix mXXt;
  Matrix mU;
  Vector mS;
  std::vector<RealImage*> mComponent;
};

#endif // __IMAGE_PCA_H__
