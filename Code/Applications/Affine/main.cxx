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

/*
  Inorder to get Affine transform just include Affine.h in your program and create an object for that class
  you can call the default constructor in two ways
  1) image1(filename),image2(filename),type of transformation
  2) image1 (Imagepointer), image2(image pointer), type of transform
  Here the type of transform can be
  Affine
  Translation
  Rigid

  Create the object accordingly to get the desired result.
*/

#include<stdio.h>
#include<Affine.h>
#include "CmdLineParser.h"
#include<string.h>
#include "ImagePreprocessor.h"
#include "ApplicationUtils.h"
#include "WeightedImageSet.h"


class AffineParamFile : public CompoundParam {
public:
  AffineParamFile()
    : CompoundParam("ParameterFile", "top-level node", PARAM_REQUIRED)
  {
    
    this->AddChild(WeightedImageSetParam("WeightedImageSet"));
    this->AddChild(ImagePreprocessorParam("ImagePreprocessor"));
    this->AddChild(ValueParam<std::string>("RegistrationType", "Resistration Type: Affine, Translation, Rigid",PARAM_REQUIRED, ""));
    this->AddChild(ValueParam<bool>("WriteDefImages", "Write out transformed image(s)?", PARAM_COMMON, true));
    this->AddChild(ValueParam<bool>("WriteTransforms", "Write out transformation matrix?", PARAM_COMMON, true));
    this->AddChild(ValueParam<std::string>("OutputPrefix", "filename prefix to use", PARAM_COMMON, "Affine"));  
    this->AddChild(ValueParam<std::string>("OutputSuffix", "image filename extension to use (determines format)", PARAM_COMMON, "mha"));
  }
  
  ParamAccessorMacro(WeightedImageSetParam, WeightedImageSet)
  ParamAccessorMacro(ImagePreprocessorParam, ImagePreprocessor)
  ValueParamAccessorMacro(std::string, RegistrationType)
  ValueParamAccessorMacro(bool, WriteDefImages)
  ValueParamAccessorMacro(bool, WriteTransforms)
  ValueParamAccessorMacro(std::string, OutputPrefix)
  ValueParamAccessorMacro(std::string, OutputSuffix)
  CopyFunctionMacro(AffineParamFile)

};

int main(int argc, char ** argv)
{
  AffineParamFile pf;

  CmdLineParser parser(pf);

  try{
    parser.Parse(argc,argv);
  }catch(ParamException &e){
    std::cerr << "Error parsing arguments:" << std::endl;
    std::cerr << "   " << e.what() << std::endl;
    return EXIT_FAILURE;
  }


	
  //
  // load/preprocess images
  //
  WeightedImageSet imageSet(pf.WeightedImageSet());
  // verbose load
  imageSet.Load(true);
  // preprocess
  ImagePreprocessor preprocessor(pf.ImagePreprocessor());
  {
    std::vector<RealImage*> imVec = imageSet.GetImageVec();
    std::vector<std::string> imNames = imageSet.GetImageNameVec();
    preprocessor.Process(imVec, imNames);
  }
	
  // copy out loaded, preprocessed images
  unsigned int numImages = imageSet.NumImages();
  Affine::ImageType** images = new Affine::ImageType*[numImages];
  //std::string paths[numImages];
  //double *imageWeights = new double[numImages];
  for(unsigned int i=0;i<numImages;i++){
    //paths[i] = preprocessor.
    images[i] = imageSet.GetImage(i);
    //imageWeights[i] = imageSet.GetWeight(i);
  }

  std::string regtype = pf.RegistrationType();
	
  parser.GenerateFile("AffineParsedOutput.xml");
	
  std::cout<<"Writing Original Base Image "<<" ...."<<std::endl;
  ApplicationUtils::SaveImageITK((char*)"base_1.mhd", *images[0]);
  std::cout<<"Writing done"<<std::endl;
	
  for(unsigned int i=1;i<numImages;i++){
    std::cout<<"Computing Affine Registration between base image and image "<<i<<std::endl;
    Affine abc(images[0], images[i], regtype);
    std::stringstream ss;
	  
    //writing transformations
    std::cout<<"Writing transform "<<i<<" ...."<<std::endl;
    ss <<pf.OutputPrefix()<< "Transform" <<i<< ".plunc";
    try
      {
	abc.registrationTransform.writePLUNCStyle(ss.str().c_str());
      }
    catch (...)
      {
	std::cout<<"Failed to save matrix"<<std::endl;
	return 0;
      }
	  
    ss.str("");
    ss<<pf.OutputPrefix()<<"TransformedImage"<<i<<"."<<pf.OutputSuffix();
	  
    std::cout<<"Writing Affined Image "<<i<<" ...."<<std::endl;
    ApplicationUtils::SaveImageITK(ss.str().c_str(), *abc.finalImage);
    std::cout<<"Writing done"<<std::endl;
  }
	
  return 1;
}
