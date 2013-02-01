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


#include "LDMMWarpCPU.h"
#include "LDMMWarpAd.h"
#include "LDMMVelShootingIteratorCPU.h"
#include "LDMMAdShootingIteratorCPU.h"
#include "LDMMShootingIteratorCPU.h"
#include "ImagePreprocessor.h"
#include "CmdLineParser.h"

#ifdef CUDA_ENABLED
#include "LDMMAdShootingIteratorGPU.h"
#include "LDMMWarpGPU.h"
#include "LDMMShootingIteratorGPU.h"
#endif

/**
 * \page LDMMWarp LDMMWarp
 * Simple frontend program for computing LDMM image registraton
 *
 * If tests have been run (`make test'), an example page has been generated
 * \htmlonly
 * <a href="../../Testing/Applications/LDMM/LDMMWarp/index.html"><b>here.</b></a>
 * \endhtmlonly
 */

/** Simple parameter class to hold parameters for LDMMWarp program */
class LDMMWarpParamFile : public LDMMWarpParam {
public:
  LDMMWarpParamFile()
    : LDMMWarpParam("LDMMWarpParameterFile", "top-level node", PARAM_REQUIRED)
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

    this->AddChild(ValueParam<bool>("ShootingOptimization", "Use shooting optimization instead of relaxation?", PARAM_COMMON, false));
    this->AddChild(ValueParam<bool>("UseVelocityShootingOptimization", "If ShootingOptimization is true, this parameter chooses between Alpha0 shooting and V0 shooting", PARAM_RARE, false));
    this->AddChild(ValueParam<bool>("UseAdjointShootingOptimization", "If ShootingOptimization is true, this parameter chooses between Alpha0 shooting (old style) and Alpha0 shooting (using Adjoint equations as in Francios Xavier 2011 et al)", PARAM_COMMON, false));

    this->AddChild(ValueParam<bool>("UseGPU", "Compute warp on the GPU.  Only a subset of normal settings are applicable", PARAM_COMMON, false));
    this->AddChild(ValueParam<unsigned int>("GPUId", "If UseGPU is true, this determines which device to use", PARAM_COMMON, 0));
  }

  ValueParamAccessorMacro(std::string, MovingImage)
  ValueParamAccessorMacro(std::string, StaticImage)
  ParamAccessorMacro(ImagePreprocessorParam, ImagePreprocessor)
  ValueParamAccessorMacro(bool, PreprocessMovingImage)
  ValueParamAccessorMacro(bool, PreprocessStaticImage)
  ValueParamAccessorMacro(bool, ShootingOptimization)
  ValueParamAccessorMacro(bool, UseVelocityShootingOptimization)
  ValueParamAccessorMacro(bool, UseAdjointShootingOptimization)
  ValueParamAccessorMacro(bool, UseGPU)
  ValueParamAccessorMacro(unsigned int, GPUId)

  CopyFunctionMacro(LDMMWarpParamFile)
};

int main(int argc, char ** argv)
{

  LDMMWarpParamFile pf;
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

  std::cerr << "Start the warper **********" << std::endl;

  // get the base filename of moving image
  std::string path, movingNameBase, nameExt;
  ApplicationUtils::SplitName(pf.MovingImage().c_str(), path, movingNameBase, nameExt);
  std::string staticNameBase;
  ApplicationUtils::SplitName(pf.StaticImage().c_str(), path, staticNameBase, nameExt);
  std::string nameBase = movingNameBase + "_to_" + staticNameBase;

  // Run Warp
  WarpInterface *warper = NULL;
  if(pf.UseGPU()){
#ifdef CUDA_ENABLED
    // Set the GPU to use
    CUDAUtilities::SetCUDADevice(pf.GPUId());
    // ensure device supports CUDA capability version 1.2
    CUDAUtilities::AssertMinCUDACapabilityVersion(1,2);

    if(pf.ShootingOptimization()){
      if(pf.UseVelocityShootingOptimization()){
	throw AtlasWerksException(__FILE__,__LINE__,"Error, velocity shooting optimization not supported on the GPU yet");
      }
      else if(pf.UseAdjointShootingOptimization()){
	warper = new LDMMWarpAd<LDMMAdShootingIteratorGPU>(images[0], images[1], pf, nameBase);
      }else{
	warper = new LDMMWarp<LDMMShootingIteratorGPU>(images[0], images[1], pf, nameBase);
      }
    }else{
      warper = new LDMMWarpGPU(images[0], images[1], pf, nameBase);
    }
#else
    throw AtlasWerksException(__FILE__, __LINE__, "Error, GPU code not built.  Select USE_CUDA in CMake settings.");
#endif
  }else{
    if(pf.ShootingOptimization()){
      if(pf.UseVelocityShootingOptimization()){
	warper = new LDMMWarp<LDMMVelShootingIteratorCPU>(images[0], images[1], pf, nameBase);
      }
      else if(pf.UseAdjointShootingOptimization()){
	warper = new LDMMWarpAd<LDMMAdShootingIteratorCPU>(images[0], images[1], pf, nameBase);
      }
      else{
	warper = new LDMMWarp<LDMMShootingIteratorCPU>(images[0], images[1], pf, nameBase);
      }
    }else{
      warper = new LDMMWarpCPU(images[0], images[1], pf, nameBase);
    }
  }

  std::cerr << "Begin Warp Time: " << totalTimer.getTime() << std::endl;

  warper->RunWarp();

  std::cerr << "End Warp Time: " << totalTimer.getTime() << std::endl;
  
  // save images/deformations
  std::cout << "Writing out results..." << std::endl;
  warper->GenerateOutput();

  std::cerr << "Total Time: " << totalTimer.getTime() << std::endl;
  
  std::cout << "LDMMWarpNew exiting successfully" << std::endl;

}

