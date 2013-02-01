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
#include "ForwardSolve.h"
#include "StringUtils.h"
#include "HField3DUtils.h"

/**
 * \page ForwardSolve
 * An iterative solution to pushing an image forward by a velocity field
 */
int main(int argc, char ** argv)
{

  ValueParam<std::string> *InitialImageParam = new ValueParam<std::string>("InitialImage", "input image filename", PARAM_REQUIRED, "");
  ValueParam<std::string> *VelocityFieldParam = new ValueParam<std::string>("VelocityField", "velocity field", PARAM_REQUIRED, "");
  ValueParam<std::string> *OutputFileParam = new ValueParam<std::string>("OutputFileName", "output filename for deformed image", PARAM_COMMON, "ForwardDeformedImage.mha");
  ValueParam<unsigned int> *nIters = new ValueParam<unsigned int>("nIters", "number of iterations to run", PARAM_COMMON, 5);
  ValueParam<Real> *StepSize = new ValueParam<Real>("StepSize", "Iteration step size", PARAM_COMMON, 1.0);
  ValueParam<bool> *WriteDebugImages = new ValueParam<bool>("WriteDebugImages", "Should intermediate images be written each iteration?", PARAM_COMMON, false);
  ValueParam<std::string> *DebugImageFormat = new ValueParam<std::string>("DebugImageFormat", "printf-style format for debug images", PARAM_COMMON, "ForwardSolveIter%02d.nhdr");
  
  CompoundParam pf("ParameterFile", "top-level node", PARAM_REQUIRED);
  pf.AddChild(InitialImageParam);
  pf.AddChild(VelocityFieldParam);
  pf.AddChild(OutputFileParam);
  pf.AddChild(nIters);
  pf.AddChild(StepSize);
  pf.AddChild(WriteDebugImages);
  pf.AddChild(DebugImageFormat);

  CmdLineParser parser(pf);
  try{
    parser.Parse(argc,argv);
  }catch(ParamException &e){
    std::cerr << "Error parsing arguments:" << std::endl;
    std::cerr << "   " << e.what() << std::endl;
    std::exit(-1);
  }

  RealImage inputImage;
  VectorField vField;

  ApplicationUtils::LoadImageITK(InitialImageParam->Value().c_str(), inputImage);
  ApplicationUtils::LoadHFieldITK(VelocityFieldParam->Value().c_str(), vField);

  Vector3D<unsigned int> imSize = inputImage.getSize();
  Vector3D<double> imOrigin = inputImage.getOrigin();
  Vector3D<double> imSpacing = inputImage.getSpacing();

  // initial estimage of final image
  RealImage finalImage(imSize, imOrigin, imSpacing);
  // test inverse apply
  VectorField vInv = vField;
  vInv.scale(-1.0);
  HField3DUtils::applyU(inputImage,vInv,finalImage, imSpacing);
  RealImage tmp(imSize, imOrigin, imSpacing);
  HField3DUtils::applyU(finalImage,vField,tmp, imSpacing);
  ApplicationUtils::SaveImageITK("ImagePulledBackByNegField.mha", tmp);

  ForwardSolve solver(imSize, imOrigin, imSpacing);
  if(WriteDebugImages->Value()){
    solver.SetIterationFileFormat(DebugImageFormat->Value().c_str());
  }
  solver.SetPrintError(true);
  solver.SetStepSize(StepSize->Value());
  solver.Solve(inputImage, vField, finalImage, nIters->Value());
  ApplicationUtils::SaveImageITK(OutputFileParam->Value().c_str(), finalImage);

  HField3DUtils::applyU(finalImage,vField,tmp, imSpacing);
  ApplicationUtils::SaveImageITK("FinalImageDeformedBack.mha", tmp);
  
}

