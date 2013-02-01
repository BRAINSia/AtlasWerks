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
#include "ImagePreprocessor.h"

#define NO_VAL -1

/**
 * \page AlphaFromVelocities
 *
 * Computes \f$\alpha\f$ fields ('momentum') for a series of
 * velocities.  \f$\alpha\f$ is defined via the equation: 
 *
 *\f[
 * \mathbf{L}v_t = a_t = \alpha_t\nabla J^0_t 
 * \f]
 *
 * where \f$J^0_t is I_0(\phi^v_{t,0})\f$
 *
 * the initial momentum, \f$\alpha_0\f$, is used as the initial
 * condition in geodesic shooting.
 */
int main(int argc, char ** argv)
{

  ValueParam<std::string> *initialImageParam = new ValueParam<std::string>("InitialImage", "initial image filename", PARAM_REQUIRED, "");
  ValueParam<std::string> *finalImageParam = new ValueParam<std::string>("FinalImage", "initial image filename", PARAM_REQUIRED, "");
  ImagePreprocessorParam *preprocessorParam = new ImagePreprocessorParam("ImagePreprocessor", "Preprocessing to perform on the initial and final images", PARAM_COMMON);
  ValueParam<bool> *preprocessInitialImage  = new ValueParam<bool>("PreprocessInitialImage", "run preprocessing on initial image?", PARAM_COMMON, true);  
  ValueParam<bool> *preprocessFinalImage  = new ValueParam<bool>("PreprocessFinalImage", "run preprocessing on final image?", PARAM_COMMON, true);  

  ValueParam<std::string> *velocityFormat = new ValueParam<std::string>("VelocityFormatString", "printf-style format string for velocity field filenames", PARAM_COMMON, "LDMMWarpDefField%02d.mha");
  ValueParam<unsigned int> *nTimeSteps = 
    new ValueParam<unsigned int>("nTimeSteps", "Number of timesteps used (ie number of velocity images)", PARAM_COMMON, 5); 
  ValueParam<Real> *sigma = new ValueParam<Real>("sigma", "sigma value used in LDMM computation of velocity fields", PARAM_COMMON, 5.0);
  ValueParam<bool> *inverse  = new ValueParam<bool>("Inverse", "Compute the inverse alphas, from final image to initial image?", PARAM_COMMON, false);  
  ValueParam<std::string> *alphaFormat  = new ValueParam<std::string>("AlphaFormatString", "printf-style format strin for outputting alpha field files", PARAM_COMMON, "Alpha%02d.mha");  
  ValueParam<bool> *outputInitialAlphaOnly = new ValueParam<bool>("outputInitialAlphaOnly", "Only write out the initial alpha field?", PARAM_COMMON, false);

  CompoundParam pf("ParameterFile", "top-level node", PARAM_REQUIRED);
  pf.AddChild(initialImageParam);
  pf.AddChild(finalImageParam);
  pf.AddChild(preprocessorParam);
  pf.AddChild(preprocessInitialImage);
  pf.AddChild(preprocessFinalImage);
  pf.AddChild(velocityFormat);
  pf.AddChild(nTimeSteps);
  pf.AddChild(sigma);
  pf.AddChild(inverse);
  pf.AddChild(alphaFormat);
  pf.AddChild(outputInitialAlphaOnly);

  CmdLineParser parser(pf);

  try{
    parser.Parse(argc,argv);
  }catch(ParamException &e){
    std::cerr << "Error parsing arguments:" << std::endl;
    std::cerr << "   " << e.what() << std::endl;
    std::exit(-1);
  }

  parser.GenerateFile("AlphaFromVelocitiesParsedParams.xml");

  char buff[1024];

  // read in the images
  RealImage *I0, *IT;
  I0 = new RealImage();  
  IT = new RealImage();
  
  ApplicationUtils::LoadImageITK(initialImageParam->Value().c_str(),*I0);
  ApplicationUtils::LoadImageITK(finalImageParam->Value().c_str(),*IT);

  ImagePreprocessor preprocessor(*preprocessorParam);
  if(preprocessInitialImage->Value()) preprocessor.Process(*I0);
  if(preprocessFinalImage->Value()) preprocessor.Process(*IT);
  
  Vector3D<unsigned int> size = I0->getSize();

  // read in the velocity fields
  std::vector<VectorField *> v;
  VectorField *curV;

  // TEST
  VectorField hField(size);
  VectorField scratchV(size);
  RealImage scratchI(size, I0->getOrigin(), I0->getSpacing());
  HField3DUtils::setToIdentity(hField);
  // END TEST
  for(unsigned int t=0;t<nTimeSteps->Value();t++){
    sprintf(buff, velocityFormat->Value().c_str(), t);
    std::cout << "reading vector field " << buff << std::endl;
    curV = new VectorField(size);
    ApplicationUtils::LoadHFieldITK(buff, *curV);
    v.push_back(curV);
    
    // TEST
    HField3DUtils::composeHVInv(hField, *curV, scratchV, I0->getSpacing(),
				HField3DUtils::BACKGROUND_STRATEGY_PARTIAL_ID);
    hField = scratchV;
    // create the deformed image from source to this timepoint
    HField3DUtils::apply(*I0, hField, scratchI);
    sprintf(buff, "TestForDef%02d.mha", t+1);
    ApplicationUtils::SaveImageITK(buff, scratchI);
    // END TEST
  }

  // create the inverse field
  if(inverse->Value()){
    std::cout << "Swapping images and inverting and reversing velocity fields" << std::endl;
    RealImage *tmp = I0;
    I0 = IT;
    IT = tmp;

    std::vector<VectorField *> vInv;
    for(int i=v.size()-1;i>=0;i--){
      v[i]->scale(-1.0);
      vInv.push_back(v[i]);
      v = vInv;
    }
  }

  // compute the alpha fields
  std::vector<RealImage*> alphaVec;
  LDMM::computeAlphaFields(v,I0,IT,sigma->Value(),alphaVec);

  // output the alpha field(s)
  unsigned int outputLimit = nTimeSteps->Value();
  if(outputInitialAlphaOnly->Value()){
    outputLimit = 1;
  }
  for(size_t i=0;i<outputLimit;i++){
    sprintf(buff, alphaFormat->Value().c_str(), (int)i);
    ApplicationUtils::SaveImageITK(buff, *alphaVec[i]);
  }
  
}

