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

#include "GreedyWarp.h"
#include "ImagePreprocessor.h"
#include "CmdLineParser.h"
#include "GreedyIteratorCPU.h"

#ifdef CUDA_ENABLED
#include "GreedyIteratorGPU.h"
#endif

/**
 * \page GreedyWarp GreedyWarp
 * Simple frontend program for computing Greedy image registraton
 *
 * If tests have been run (`make test'), an example page has been generated
 * \htmlonly
 * <a href="../../Testing/Applications/Greedy/GreedyWarp/index.html"><b>here.</b></a>
 * \endhtmlonly
 */

/** Simple parameter class to hold parameters for GreedyWarp program */
class GreedyWarpParamFile : public GreedyWarpParam {
public:
  GreedyWarpParamFile()
    : GreedyWarpParam("GreedyWarpParameterFile", "top-level node", PARAM_REQUIRED)
  {

    ValueParam<std::string> MovingImageParam("MovingImage", "moving image filename", PARAM_REQUIRED, "");
    MovingImageParam.AddAlias("InitialImage");
    this->AddChild(MovingImageParam);

    ValueParam<std::string> StaticImageParam("StaticImage", "static (template) image filename", PARAM_REQUIRED, "");
    StaticImageParam.AddAlias("FinalImage");
    this->AddChild(StaticImageParam);

    this->AddChild(ImagePreprocessorParam("ImagePreprocessor"));

    ValueParam<bool> PreprocessMovingImageParam("PreprocessMovingImage", "Run preprocessing on initial image", PARAM_COMMON, true);
    PreprocessMovingImageParam.AddAlias("PreprocessInitialImage");
    this->AddChild(PreprocessMovingImageParam);

    ValueParam<bool> PreprocessStaticImageParam("PreprocessStaticImage", "Run preprocessing on initial image", PARAM_COMMON, true);
    PreprocessStaticImageParam.AddAlias("PreprocessFinalImage");
    this->AddChild(PreprocessStaticImageParam);

    this->AddChild(ValueParam<std::string>("AffineTransform", "filename of initial affine transform", PARAM_COMMON, ""));
    this->AddChild(ValueParam<bool>("ItkTransform", "is this an ITK-style transform file vs. an AffineTransform3D-style file?", PARAM_COMMON, false));
    this->AddChild(ValueParam<bool>("UseGPU", "Compute warp on the GPU.  Only a subset of normal settings are applicable", PARAM_COMMON, false));
    this->AddChild(ValueParam<unsigned int>("GPUId", "If UseGPU is true, this determines which device to use", PARAM_COMMON, 0));
  }

  ValueParamAccessorMacro(std::string, MovingImage)
  ValueParamAccessorMacro(std::string, StaticImage)
  ParamAccessorMacro(ImagePreprocessorParam, ImagePreprocessor)
  ValueParamAccessorMacro(bool, PreprocessMovingImage)
  ValueParamAccessorMacro(bool, PreprocessStaticImage)
  ValueParamAccessorMacro(std::string, AffineTransform)
  ValueParamAccessorMacro(bool, ItkTransform)
  ValueParamAccessorMacro(bool, UseGPU)
  ValueParamAccessorMacro(unsigned int, GPUId)

  CopyFunctionMacro(GreedyWarpParamFile)
};

int main(int argc, char ** argv)
{

  GreedyWarpParamFile pf;

  CmdLineParser parser(pf);

  try{
    parser.Parse(argc,argv);
  }catch(ParamException &e){
    std::cerr << "Error parsing arguments:" << std::endl;
    std::cerr << "   " << e.what() << std::endl;
    std::exit(-1);
  }

  Timer totalTimer;
  totalTimer.start();

  RealImage *images[2]; 
  images[0] = new RealImage();
  images[1] = new RealImage();
  
  ApplicationUtils::LoadImageITK(pf.MovingImage().c_str(),*images[0]);
  ApplicationUtils::LoadImageITK(pf.StaticImage().c_str(),*images[1]);
  
  ImagePreprocessor preprocessor(pf.ImagePreprocessor());
  
  if(pf.PreprocessMovingImage()){
    preprocessor.Process(*images[0]);
  }
  if(pf.PreprocessStaticImage()){
    preprocessor.Process(*images[1]);
  }

  // get the base filename of moving image
  std::string path, movingNameBase, nameExt;
  ApplicationUtils::SplitName(pf.MovingImage().c_str(), path, movingNameBase, nameExt);
  std::string staticNameBase;
  ApplicationUtils::SplitName(pf.StaticImage().c_str(), path, staticNameBase, nameExt);
  std::string nameBase = movingNameBase + "_to_" + staticNameBase;

  Affine3D *aff = NULL;
  // see if we have an affine transformation
  if(pf.AffineTransform().length() > 0){
    aff = new Affine3D();
    if(pf.ItkTransform()){
      aff->readITKStyle(pf.AffineTransform());
    }else{
      aff->readPLUNCStyle(pf.AffineTransform());
    }
  }

  // Run Warp
  WarpInterface *warper = NULL;
  if(pf.UseGPU()){
#ifdef CUDA_ENABLED    
    // Set the GPU to use
    CUDAUtilities::SetCUDADevice(pf.GPUId());
    
    warper = new GreedyWarp<GreedyIteratorGPU>(images[0], images[1], pf, aff, nameBase);
#else
    throw AtlasWerksException(__FILE__, __LINE__, "Error, GPU code not built.  Select USE_CUDA in CMake settings.");
#endif
  }else{
    warper = new GreedyWarp<GreedyIteratorCPU>(images[0], images[1], pf, aff, nameBase);
  }

  std::cerr << "Begin Warp Time: " << totalTimer.getTime() << std::endl;

  warper->RunWarp();

  std::cerr << "End Warp Time: " << totalTimer.getTime() << std::endl;
  
  // save images/deformations
  std::cout << "Writing out results..." << std::endl;
  warper->GenerateOutput();

  std::cerr << "Total Time: " << totalTimer.getTime() << std::endl;
  
  std::cout << "GreedyWarpNew exiting successfully" << std::endl;

}

