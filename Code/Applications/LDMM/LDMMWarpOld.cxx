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


#include "LDMM.h"
#include "ImagePreprocessor.h"
#include "CmdLineParser.h"

// void buildGPU(const RealImage *I0, const RealImage *IT, const LDMMOldParam *ldmmParam, 
// 	      std::vector<RealImage*> &defImage, 
// 	      std::vector<VectorField*> &defField,
// 	      std::vector<RealImage*> &alpha)
// {
// #ifdef CUDA_ENABLED
//   LDMMWarpGPU warper(I0, IT, *ldmmParam);
//   warper.RunWarp();
//   std::cout << "Retrieving data from warper..." << std::endl;
//   for(unsigned int t=0;t<ldmmParam->NTimeSteps();t++){
//     defImage.push_back(new RealImage(*warper.GetDefImage(t)));
//     defField.push_back(new VectorField(*warper.GetVField(t)));
//     if(ldmmParam->WriteAlphas()){
//       alpha.push_back(new RealImage(*warper.GetAlpha(t)));
//     }
//   }
//   defImage.push_back(new RealImage(*warper.GetDefImage(ldmmParam->NTimeSteps())));
//   std::cout << "done." << std::endl;
// #else
//   throw AtlasWerksException(__FILE__, __LINE__, "Cannot run warp on GPU, CUDA code not compiled");
// #endif
// }

/**
 * \page LDMMWarp LDMMWarp
 * Simple frontend program for computing LDMM image registraton
 *
 * If tests have been run (`make test'), an example page has been generated
 * \htmlonly
 * <a href="../../Testing/Applications/LDMM/LDMMWarp/index.html"><b>here.</b></a>
 * \endhtmlonly
 */

int main(int argc, char ** argv)
{

  ValueParam<std::string> *initialImageParam = new ValueParam<std::string>("InitialImage", "initial image filename", PARAM_REQUIRED, "");
  ValueParam<std::string> *finalImageParam = new ValueParam<std::string>("FinalImage", "final image filename", PARAM_REQUIRED, "");
  ImagePreprocessorParam *preprocessorParam = new ImagePreprocessorParam("ImagePreprocessor", "Preprocessing to perform on ImageToDeform");
  ValueParam<bool> *preprocessInitialImage = new ValueParam<bool>("PreprocessInitialImage", "Run preprocessing on initial image", PARAM_COMMON, true);
  ValueParam<bool> *preprocessFinalImage = new ValueParam<bool>("PreprocessFinalImage", "Run preprocessing on final image", PARAM_COMMON, true);

  LDMMOldParam *ldmmParam = new LDMMOldParam();

  ValueParam<std::string> *outPrefix  = new ValueParam<std::string>("outputPrefix", "filename prefix to use", PARAM_COMMON, "LDMMWarp");  
  ValueParam<std::string> *outSuffix = new ValueParam<std::string>("outputSuffix", "filename extension to use (determines format)", PARAM_COMMON, "mha");
  ValueParam<bool> *writeVelFields = new ValueParam<bool>("WriteVelocityFields", "Write out all (nImages*nTimesteps) velocity fields?", PARAM_COMMON, false);
  ValueParam<bool> *writeIntermediateImages = new ValueParam<bool>("WriteIntermediateImages", "Write out all (nImages*nTimesteps) intermediate deformed images?", PARAM_COMMON, false);
  ValueParam<bool> *useGPU = new ValueParam<bool>("UseGPU", "compute warp on the GPU", PARAM_COMMON, false);

  //ValueParam<unsigned int> *testParam = new ValueParam<unsigned int>("TestParam", "unused test param", false, 999);

  CompoundParam pf("ParameterFile", "top-level node", PARAM_REQUIRED);
  pf.AddChild(initialImageParam);
  pf.AddChild(finalImageParam);
  pf.AddChild(preprocessorParam);
  pf.AddChild(preprocessInitialImage);  
  pf.AddChild(preprocessFinalImage);
  pf.AddChild(outPrefix);
  pf.AddChild(outSuffix);
  pf.AddChild(ldmmParam);
  pf.AddChild(writeVelFields);
  pf.AddChild(writeIntermediateImages);
  pf.AddChild(useGPU);

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
  
  ApplicationUtils::LoadImageITK(initialImageParam->Value().c_str(),*images[0]);
  ApplicationUtils::LoadImageITK(finalImageParam->Value().c_str(),*images[1]);
  
  ImagePreprocessor preprocessor(*preprocessorParam);
  
  if(preprocessInitialImage->Value()){
    preprocessor.Process(*images[0]);
  }
  if(preprocessFinalImage->Value()){
    preprocessor.Process(*images[1]);
  }

//   Vector3D<unsigned int> imSz = preprocessor.GetImageSize();
//   Vector3D<Real> origin = preprocessor.GetImageOrigin();
//   Vector3D<Real> spacing = preprocessor.GetImageSpacing();
  
  // run LDMM
  std::vector<RealImage*> defImage;
  std::vector<VectorField*> defField;
  std::vector<RealImage*> alpha;

  if(useGPU->Value()){
    throw AtlasWerksException(__FILE__, __LINE__, "GPU not supported, please use CPU");
  }else{
    if(ldmmParam->WriteAlphas()){
      throw AtlasWerksException(__FILE__, __LINE__, "WriteAlphas parameter not implemented in this version");
    }
    //LDMM::LDMMShootingRegistration2(images[0], images[1], ldmmParam->NTimeSteps(), ldmmParam->GetScaleLevel(0), defImage, defField);
    //LDMM::LDMMShootingRegistration(images[0], images[1], ldmmParam->NTimeSteps(), ldmmParam->GetScaleLevel(0), defImage, defField);
    LDMM::LDMMMultiscaleRegistration(images[0], images[1], *ldmmParam, defImage, defField);
  }

  // save images/deformations
  std::cout << "Writing out results..." << std::endl;

  char fname[256];
  if(writeVelFields->Value()){
    for(unsigned int i=0;i<defField.size();i++){
      sprintf(fname, "%sVelField%02d.%s", outPrefix->Value().c_str(), i, outSuffix->Value().c_str());
      ApplicationUtils::SaveHFieldITK(fname, *defField[i]);
    }
  }

  if(writeIntermediateImages->Value()){
    std::cout << "Writing " << defImage.size() << " deformed images" << std::endl;
    for(unsigned int i=0;i<defImage.size();i++){
      sprintf(fname, "%sImage%02d.%s", outPrefix->Value().c_str(), i, outSuffix->Value().c_str());
      ApplicationUtils::SaveImageITK(fname, *defImage[i]);
    }
  }

  if(ldmmParam->WriteAlphas()){
    for(unsigned int i=0;i<alpha.size();i++){
      sprintf(fname, "%sAlpha%02d.%s", outPrefix->Value().c_str(), i, outSuffix->Value().c_str());
      ApplicationUtils::SaveImageITK(fname, *alpha[i]);
    }
  }
  
  std::cerr << "Total Time: " << totalTimer.getTime() << std::endl;

  std::cout << "LDMMWarp exiting successfully" << std::endl;

}

