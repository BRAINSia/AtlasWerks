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

#include <cmath>

#include "AtlasWerksTypes.h"
#include "Array3D.h"
#include "Array3DUtils.h"
#include "ImageUtils.h"
#include "ApplicationUtils.h"
#include "HField3DUtils.h"
#include "ImageUtils.h"

#include <tclap/CmdLine.h>


int main(int argc, char *argv[]){

  const char *in_filename = "myfile.mha";
  const char *out_filename = "myoutfile.mha";
      
  RealImage inputImage;
  // Load the input image
  ApplicationUtils::LoadImageITK(in_filename,
				 inputImage);
      
  SizeType size = inputImage.getSize();
  OriginType origin = inputImage.getOrigin();
  SpacingType spacing = inputImage.getSpacing();

  std::cout << "Input image size: " << size << std::endl;
  std::cout << "Input image origin: " << origin << std::endl;
  std::cout << "Input image spacing: " << spacing << std::endl;

  SizeType newSize = size/2;
  SpacingType newSpacing = spacing*2;
  OriginType newOrigin = origin + OriginType(0.25, 0.5, 0.75);
  RealImage outImage(newSize, newOrigin, newSpacing);

  ImageUtils::resample(inputImage, outImage);

  ApplicationUtils::SaveImageITK(out_filename, outImage);

}

