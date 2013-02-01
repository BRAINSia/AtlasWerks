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


#include "AtlasWerksTypes.h"
#include "CmdLineParser.h"
#include "ApplicationUtils.h"

/**
 * \page IntensityAverage
 * Generate simple intensity average of a series of images.  
 */
int main(int argc, char ** argv)
{

  ValueParam<std::string> *imageFormat = new ValueParam<std::string>("ImageFormatString", "printf-style format string for input image filenames", PARAM_REQUIRED, "LDMMWarpImage%02d.mha");
  ValueParam<unsigned int> *nImages = 
    new ValueParam<unsigned int>("nImages", "Number of images to be averaged", PARAM_REQUIRED, 0); 
  ValueParam<unsigned int> *baseImageNum = 
    new ValueParam<unsigned int>("baseImageNum", "base image number (used in generating input filenames)", PARAM_COMMON, 0); 
  ValueParam<std::string> *outputFile = new ValueParam<std::string>("OutputFileName", "output filename for the mean image", PARAM_COMMON, "MeanImage.mha");

  CompoundParam pf("ParameterFile", "top-level node", PARAM_REQUIRED);
  pf.AddChild(imageFormat);
  pf.AddChild(nImages);
  pf.AddChild(baseImageNum);
  pf.AddChild(outputFile);

  CmdLineParser parser(pf);
  try{
    parser.Parse(argc,argv);
  }catch(ParamException &e){
    std::cerr << "Error parsing arguments:" << std::endl;
    std::cerr << "   " << e.what() << std::endl;
    std::exit(-1);
  }

  RealImage *averageIm = NULL;
  Vector3D<unsigned int> imSize;
  Vector3D<double> imSpacing;
  Vector3D<double> imOrigin;

  Real scaleFactor = 1.0/nImages->Value();
  char buff[1024];
  RealImage *curIm = new RealImage();
  for(unsigned int imIdx=baseImageNum->Value();imIdx<nImages->Value();imIdx++){
    sprintf(buff, imageFormat->Value().c_str(), imIdx);
    ApplicationUtils::LoadImageITK(buff,*curIm);
    // if this is the first image..
    if(imIdx == baseImageNum->Value()){
      imSize = curIm->getSize();
      imSpacing = curIm->getSpacing();
      imOrigin = curIm->getOrigin();
      averageIm = new RealImage(imSize, imOrigin, imSpacing);
      averageIm->fill(0.0);
    }else{ // otherwise test...
      if(imSize != curIm->getSize()){
	std::cerr << "Error: image " << imIdx << ": image sizes do not match" << std::endl;
	std::exit(-1);
      }
      if(imSpacing != curIm->getSpacing()){
	std::cerr << "Error: image " << imIdx << ": image spacings do not match" << std::endl;
	std::exit(-1);
      }
      if(imOrigin != curIm->getOrigin()){
	std::cerr << "Error: image " << imIdx << ": image origins do not match" << std::endl;
	std::exit(-1);
      }
    }
    
    curIm->scale(scaleFactor);
    averageIm->pointwiseAdd(*curIm);
  }
  
  ApplicationUtils::SaveImageITK(outputFile->Value().c_str(), *averageIm);

}

