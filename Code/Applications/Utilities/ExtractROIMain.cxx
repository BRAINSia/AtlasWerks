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

  try
    {
      TCLAP::CmdLine cmd("ExtractROI");
      
      TCLAP::ValueArg<std::string>
	originArg("o","origin",
		  "origin (in world coordinates) of the start of the ROI, specify as -o ox,oy,oz",
		  true,"","float x 3", cmd);

      TCLAP::ValueArg<std::string>
	sizeArg("s","size",
		  "size, in pixels, of the ROI, specify as -s sx,sy,sz",
		  true,"","int x 3", cmd);
      
      TCLAP::ValueArg<std::string>
	spacingArg("p","spacing",
		  "spacing of output image, defaults to spacing of input.  specify as -p sx,sy,sz",
		  false,"","float x 3", cmd);
      
      TCLAP::ValueArg<float>
	backgroundArg("b","background",
		      "background value to use, zero by default",
		      false,0,"float", cmd);
      
      TCLAP::UnlabeledValueArg<std::string>
	inputImageArg("inputImage",
		      "image to crop",
		      true,"","file",cmd);
      
      TCLAP::UnlabeledValueArg<std::string>
	outputImageArg("outputImage",
		       "cropped image",
		       true,"","file",cmd);
      
      cmd.parse(argc, argv);

      SizeType size;
      OriginType origin;
      SpacingType spacing;

      RealImage inputImage;
      // Load the input image
      ApplicationUtils::LoadImageITK(inputImageArg.getValue().c_str(),
				     inputImage);
      
      // spacing defaults to input spacing
      spacing = inputImage.getSpacing();
      
      
      // parse vectors

      int parsed;

      parsed = sscanf(sizeArg.getValue().c_str(), "%u,%u,%u", &(size.x), &(size.y), &(size.z));
      if(parsed != 3){
	throw AtlasWerksException(__FILE__,__LINE__,"Error, problem reading size vector");
      }
      std::cout << "Size is " << size << std::endl;

      parsed = sscanf(originArg.getValue().c_str(), "%lf,%lf,%lf", &(origin.x), &(origin.y), &(origin.z));
      if(parsed != 3){
	throw AtlasWerksException(__FILE__,__LINE__,"Error, problem reading origin vector");
      }
      std::cout << "Origin is " << origin << std::endl;
      
      std::cout << "Spacing arg is " << spacingArg.getValue() << std::endl;
      if(spacingArg.isSet()){
	parsed = sscanf(spacingArg.getValue().c_str(), "%lf,%lf,%lf", &(spacing.x), &(spacing.y), &(spacing.z));
      }
      std::cout << "Spacing is " << spacing << std::endl;
      
      RealImage outputImage(size, origin, spacing);
      ImageUtils::resampleNew<Real, Array3DUtils::BACKGROUND_STRATEGY_VAL, DEFAULT_SCALAR_INTERP>
	(inputImage, outputImage, backgroundArg.getValue());
      
      ApplicationUtils::SaveImageITK(outputImageArg.getValue().c_str(), outputImage);
    }
  catch (TCLAP::ArgException &e)
    {
      std::cerr << "error: " << e.error() << " for arg " << e.argId()
		<< std::endl;
      exit(1);
    }
}

