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
#include "CmdLineParser.h"

/**
 * \page LDMMWarp LDMMWarp
 * Simple frontend program for computing LDMM image registraton
 */

int main(int argc, char ** argv)
{

  ValueParam<std::string> *initialImageParam = new ValueParam<std::string>("InitialImage", "initial image filename", PARAM_REQUIRED, "");
  ValueParam<std::string> *finalImageParam = new ValueParam<std::string>("FinalImage", "final image filename", PARAM_REQUIRED, "");

  ValueParam<unsigned int> *nTimeSteps = new ValueParam<unsigned int>("nTimeSteps", "number of timesteps to use", PARAM_REQUIRED, 10);
  LDMMScaleLevelParam *scaleLevel = new LDMMScaleLevelParam();

  ValueParam<std::string> *outPrefix  = new ValueParam<std::string>("outputPrefix", "filename prefix to use", PARAM_COMMON, "LDMMWarp");  
  ValueParam<std::string> *outSuffix = new ValueParam<std::string>("outputSuffix", "filename extension to use (determines format)", PARAM_COMMON, "mha");

  ValueParam<bool> *readInitialVelocities = new ValueParam<bool>("ReadInInitialVelocities", "Read in velocity fields to use for initialization?", PARAM_COMMON, false);
  ValueParam<std::string> *inputVelocityPattern  = new ValueParam<std::string>("inputVelocityFilenamePattern", "printf-style pattern to use for input velocity filename generation", PARAM_COMMON, "LDMMWarpDefField%02d.mha");  

  //ValueParam<unsigned int> *testParam = new ValueParam<unsigned int>("TestParam", "unused test param", false, 999);

  CompoundParam pf("ParameterFile", "top-level node", PARAM_REQUIRED);
  pf.AddChild(initialImageParam);
  pf.AddChild(finalImageParam);
  pf.AddChild(outPrefix);
  pf.AddChild(outSuffix);
  pf.AddChild(nTimeSteps);
  pf.AddChild(scaleLevel);
  pf.AddChild(readInitialVelocities);
  pf.AddChild(inputVelocityPattern);

  CmdLineParser parser(pf);

  try{
    parser.Parse(argc,argv);
  }catch(ParamException e){
    std::cerr << "Error parsing arguments:" << std::endl;
    std::cerr << "   " << e.what() << std::endl;
    std::exit(-1);
  }

  RealImage *images[2]; 
  images[0] = new RealImage();
  images[1] = new RealImage();

  ApplicationUtils::LoadImageITK(initialImageParam->Value().c_str(),*images[0]);
  ApplicationUtils::LoadImageITK(finalImageParam->Value().c_str(),*images[1]);
  
  // scale images, test that they're the same size
  Real min, max;
  Vector3D<unsigned int> imSz = images[0]->getSize();
  Vector3D<Real> origin = images[0]->getOrigin();
  for(unsigned int i=0;i<2;i++){
    Vector3D<unsigned int> newSize = images[i]->getSize();
    Vector3D<Real> newOrigin = images[i]->getOrigin();
    if(newSize != imSz ||  newOrigin != origin){
      std::cerr << "Error, image dimensions/origin do not match!! Aborting." << std::endl;
      for(i = 0; i < 2; i++){
	delete images[i];
	images[i] = NULL;
      }
      exit(-1);
    }
    min = max = 0; // make compiler happy
    Array3DUtils::getMinMax(*images[i],min,max);
    std::cout << "Image " << i << ": size=" << newSize 
	      << ", origin=" << newOrigin
	      << ", min/max=" << min << "/" << max
	      << std::endl;
    // scale to maximum value of 1
    // TODO: do we need to change this to scale to [0,1]?
    // images[i]->scale(1.0/max);
  }

  char fname[256];

  // run LDMM
  std::vector<RealImage*> defImage;
  std::vector<VectorField*> defField;

  VectorField *vel;
  if(readInitialVelocities->Value()){
    for(unsigned int i=0;i<nTimeSteps->Value();i++){
      vel = new VectorField(imSz);
      sprintf(fname, inputVelocityPattern->Value().c_str(), i);
      ApplicationUtils::LoadHFieldITK(fname, *vel);
      defField.push_back(vel);
    }
  }

  LDMM::LDMMShootingRegistration(images[0],
				 images[1],
				 nTimeSteps->Value(),
				 *scaleLevel,
				 defImage,
				 defField);

  // save images/deformations
  for(unsigned int i=0;i<defField.size();i++){
    sprintf(fname, "%sImage%02d.%s", outPrefix->Value().c_str(), i, outSuffix->Value().c_str());
    ApplicationUtils::SaveImageITK(fname, *defImage[i]);
    sprintf(fname, "%sDefField%02d.%s", outPrefix->Value().c_str(), i, outSuffix->Value().c_str());
    ApplicationUtils::SaveHFieldITK(fname, *defField[i]);
  }
  sprintf(fname, "%sImage%02d.%s", outPrefix->Value().c_str(), (int)defField.size(), outSuffix->Value().c_str());
  ApplicationUtils::SaveImageITK(fname, *defImage[defField.size()]);
  
}

