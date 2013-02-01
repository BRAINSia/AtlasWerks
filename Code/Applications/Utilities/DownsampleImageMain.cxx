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
      TCLAP::CmdLine cmd("DownsampleImage");
      
      TCLAP::SwitchArg
	integerDSArg("i","int",
		     "Use integer downsampling (gaussian downsampling used by default)",
		     cmd);
      TCLAP::ValueArg<std::string>
	factorArg("f","factor",
		  "downsampel factor in each direction.  specify as -f fx,fy,fz",
		  true,"","int x 3", cmd);
      
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
      
      Vector3D<unsigned int> factor;
      
      RealImage inputImage;
      // Load the input image
      ApplicationUtils::LoadImageITK(inputImageArg.getValue().c_str(),
				     inputImage);
      
      size = inputImage.getSize();
      origin = inputImage.getOrigin();
      spacing = inputImage.getSpacing();
      std::cout << "Input image size: " << size << std::endl;
      std::cout << "Input image origin: " << origin << std::endl;
      std::cout << "Input image spacing: " << spacing << std::endl;

      // parse factor
      int parsed;
      parsed = sscanf(factorArg.getValue().c_str(), "%u,%u,%u", &(factor.x), &(factor.y), &(factor.z));
      if(parsed != 3){
	throw AtlasWerksException(__FILE__,__LINE__,"Error, problem reading factor vector");
      }
      std::cout << "Downsampling by " << factor << std::endl;
      
      SpacingType newSpacing = spacing*factor;

      RealImage outputImage;
      if(integerDSArg.getValue()){
	Array3DUtils::downsampleByInts(inputImage, outputImage, factor);
      }else{
	SizeType newSize(static_cast<unsigned int>(size.x/factor.x),
			 static_cast<unsigned int>(size.y/factor.y),
			 static_cast<unsigned int>(size.z/factor.z));
	Vector3D<double> sigma = factor;
	sigma.x = sqrt(factor.x/2);
	sigma.y = sqrt(factor.y/2);
	sigma.z = sqrt(factor.z/2);
	Vector3D<int> kernel;
	kernel.x = (int)ceil(2.0*factor.x);
	kernel.y = (int)ceil(2.0*factor.y);
	kernel.z = (int)ceil(2.0*factor.z);
	ImageUtils::gaussianDownsample(inputImage, outputImage, factor, sigma, kernel);
      }
      outputImage.setOrigin(origin);
      outputImage.setSpacing(newSpacing);

      std::cout << "Output image size: " << outputImage.getSize() << std::endl;
      std::cout << "Output image origin: " << outputImage.getOrigin() << std::endl;
      std::cout << "Output image spacing: " << outputImage.getSpacing() << std::endl;

      ApplicationUtils::SaveImageITK(outputImageArg.getValue().c_str(), outputImage);

    }
  catch (TCLAP::ArgException &e)
    {
      std::cerr << "error: " << e.error() << " for arg " << e.argId()
		<< std::endl;
      exit(1);
    }
}

