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


#include "ApplicationUtils.h"
#include "CmdLineParser.h"
#include "AtlasWerksTypes.h"
#include "ImagePreprocessor.h"

/** Simple parameter class to hold parameters for preprocessing */
class ImagePreprocessingParamFile : public CompoundParam {
public:
  ImagePreprocessingParamFile()
    : CompoundParam("AlignCentroidsParameterFile", "top-level node", PARAM_REQUIRED)
  {
    this->AddChild(ValueParam<std::string>("InputImageName", "input image filename", PARAM_REQUIRED, ""));
    this->AddChild(ValueParam<std::string>("OutputImageName", "output image filename", PARAM_REQUIRED, ""));
    this->AddChild(ImagePreprocessorParam("ImagePreprocessor"));
  }

  ValueParamAccessorMacro(std::string, InputImageName)
  ValueParamAccessorMacro(std::string, OutputImageName)
  ParamAccessorMacro(ImagePreprocessorParam, ImagePreprocessor)

  CopyFunctionMacro(ImagePreprocessingParamFile)
};

/**
 * \page ImagePreprocessing
 * Run image preprocessing
 */
int main(int argc, char ** argv)
{
  
  ImagePreprocessingParamFile pf;
  
  CmdLineParser parser(pf);
  try{
    parser.Parse(argc,argv);
  }catch(ParamException &e){
    std::cerr << "Error parsing arguments:" << std::endl;
    std::cerr << "   " << e.what() << std::endl;
    std::exit(-1);
  }
  
  //
  // Load images
  //
  RealImage *inputImage = new RealImage(); 
  ApplicationUtils::LoadImageITK(pf.InputImageName().c_str(),*inputImage);
  
  Vector3D<unsigned int> size = inputImage->getSize();
  Vector3D<double> spacing = inputImage->getSpacing();
  Vector3D<double> origin = inputImage->getOrigin();
  
  // preprocess
  ImagePreprocessor preprocessor(pf.ImagePreprocessor());

  preprocessor.Process(*inputImage, pf.InputImageName());

  ApplicationUtils::SaveImageITK(pf.OutputImageName().c_str(), *inputImage);
}

