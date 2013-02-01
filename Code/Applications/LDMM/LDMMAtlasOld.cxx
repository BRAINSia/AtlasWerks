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

#include <stdlib.h>

#include "LDMM.h"
#include "WeightedImageSet.h"
#include "ImagePreprocessor.h"
#include "CmdLineParser.h"

#include "log.h"

#ifdef MPI_ENABLED
#include <mpi.h>
#endif // MPI_ENABLED

/**
 * \page LDMMAtlas LDMMAtlas
 * Simple frontend program for computing an atlas via LDMM
 *
 * If tests have been run (`make test'), an example page has been generated
 * \htmlonly
 * <a href="../../Testing/Applications/LDMM/LDMMAtlas/index.html"><b>here.</b></a>
 * \endhtmlonly
 */

void setupMPI(int argc, char** argv, int& nNodes, int& nodeID){
#ifdef MPI_ENABLED
  MPI_Init( &argc, &argv );                //Init MPI
  MPI_Comm_size( MPI_COMM_WORLD, &nNodes); //Get number of node 
  MPI_Comm_rank( MPI_COMM_WORLD, &nodeID); //Get node id
#else
  nNodes = 1;
  nodeID = 0;
#endif // MPI_ENABLED
}

void cleanupMPI(){
#ifdef MPI_ENABLED
  MPI_Finalize();
#endif // MPI_ENABLED
}

/** Simple parameter class to hold parameters for LDMMAtlas program */
class LDMMAtlasParamFile : public CompoundParam {
public:
  LDMMAtlasParamFile()
    : CompoundParam("ParameterFile", "top-level node", PARAM_REQUIRED)
  {
    this->AddChild(WeightedImageSetParam("WeightedImageSet"));
    this->AddChild(ImagePreprocessorParam("ImagePreprocessor"));
    this->AddChild(LDMMOldParam()); // name: "LDMM"
    this->AddChild(ValueParam<bool>("UseGPU", "Compute atlas on the GPU.  Only a subset of normal settings are applicable", PARAM_COMMON, false));
    this->AddChild(ValueParam<unsigned int>("nGPUs", "If UseGPU is true, use this many GPUs (0 lets the system self-select)", PARAM_COMMON, 0));
    this->AddChild(ValueParam<unsigned int>("nThreads", "number of threads to use, 0=one per image", PARAM_COMMON, 0));
    this->AddChild(ValueParam<bool>("WriteMeanImage", "Write out the mean image?", PARAM_COMMON, true));
    this->AddChild(ValueParam<bool>("WriteDefImages", "Write out all (nImages) deformed images?", PARAM_COMMON, true));
    this->AddChild(ValueParam<bool>("WriteDefFields", "Write out all (nImages) deformation fields?", PARAM_COMMON, true));
    this->AddChild(ValueParam<bool>("WriteInvDefFields", "Write out all (nImages) deformation fields?", PARAM_COMMON, true));
    this->AddChild(ValueParam<bool>("WriteVelocityFields", "Write out all (nImages*nTimesteps) velocity fields?", PARAM_COMMON, false));
    this->AddChild(ValueParam<bool>("WriteIntermediateImages", "Write out all (nImages*nTimesteps) intermediate deformed images?", PARAM_COMMON, false));
    this->AddChild(ValueParam<std::string>("OutputPrefix", "filename prefix to use", PARAM_COMMON, "LDMMAtlas"));  
    this->AddChild(ValueParam<std::string>("OutputSuffix", "filename extension to use (determines format)", PARAM_COMMON, "mha"));
  }
  
  ParamAccessorMacro(WeightedImageSetParam, WeightedImageSet)
  ParamAccessorMacro(ImagePreprocessorParam, ImagePreprocessor)
  ParamAccessorMacro(LDMMOldParam, LDMM)
  ValueParamAccessorMacro(bool, UseGPU)
  ValueParamAccessorMacro(unsigned int, nGPUs)
  ValueParamAccessorMacro(unsigned int, nThreads)
  ValueParamAccessorMacro(bool, WriteMeanImage)
  ValueParamAccessorMacro(bool, WriteDefImages)
  ValueParamAccessorMacro(bool, WriteDefFields)
  ValueParamAccessorMacro(bool, WriteInvDefFields)
  ValueParamAccessorMacro(bool, WriteVelocityFields)
  ValueParamAccessorMacro(bool, WriteIntermediateImages)
  ValueParamAccessorMacro(std::string, OutputPrefix)
  ValueParamAccessorMacro(std::string, OutputSuffix)

  CopyFunctionMacro(LDMMAtlasParamFile)
};

// void BuildWithGPU(const WeightedImageSet &imageSet, 
// 		  const LDMMAtlasParamFile &param, 
// 		  unsigned int nTotalImages,
// 		  unsigned int nodeId,
// 		  unsigned int nNodes,
// 		  RealImage *MeanImage)
// {
// #ifndef CUDA_ENABLED
//   throw AtlasWerksException(__FILE__, __LINE__, "Error, GPU code not built.  Please use CPU version.");
// #else

//   LDMMAtlasBuilderGPU builder(imageSet, param.LDMM(), nodeId, nNodes, nTotalImages, param.nGPUs());
//   //cudaLDMM3DMultiGPUs builder(imageSet, param, nTotalImages, nodeId, nNodes);
//   builder.BuildAtlas();
//   if(MeanImage){
//     builder.GetMeanImage(*MeanImage);
//   }else{
//     throw AtlasWerksException(__FILE__, __LINE__, "Mean image not initialized -- you don't want any data?");
//   }
//   if(param.WriteVelocityFields()){
//     builder.WriteVFields("");
//   }
//   if(param.WriteDefFields()){
//     builder.WriteDefFields("");
//   }
//   if(param.WriteInvDefFields()){
//     builder.WriteInvDefFields("");
//   }

// #endif
// }

#define HOSTNAME_SIZE 256

int main(int argc, char ** argv)
{

  int nNodes, nodeID; // Number of node and node ID
  setupMPI(argc, argv, nNodes, nodeID);  //MPI Setup
  ErrLog::SetNodeID(nodeID);

  char hostname[HOSTNAME_SIZE];
  gethostname(hostname, HOSTNAME_SIZE);
  LOGNODE(logINFO) << "Host " << hostname << " Running as Node " << nodeID;

  Timer totalTimer;
  if(nodeID == 0)
    totalTimer.start();

  LDMMAtlasParamFile pf;

  CmdLineParser parser(pf);

  try{
    parser.Parse(argc,argv);
  }catch(ParamException &e){
    LOGNODE(logERROR) << "Error parsing arguments:" << std::endl
		      << "   " << e.what() << std::endl;
    std::exit(-1);
  }

  // a little extra parameter handling
  if(pf.LDMM().OutputPrefix().size() > 0){
    pf.OutputPrefix() = pf.LDMM().OutputPrefix();
  }else{
    pf.LDMM().OutputPrefix() = pf.OutputPrefix();
  }

  if(pf.LDMM().OutputSuffix().size() > 0){
    pf.OutputSuffix() = pf.LDMM().OutputSuffix();
  }else{
    pf.LDMM().OutputSuffix() = pf.OutputSuffix();
  }

  //
  // load/preprocess images
  //
  WeightedImageSet imageSet(pf.WeightedImageSet());
  // number of images across all nodes
  unsigned int nTotalImages = imageSet.NumImages();
  // number of images on this node
  int nImages = 0;
  // beginning image index for this node
  int bid = 0;

  // distribute among multiple nodes
  ApplicationUtils::Distribute(nTotalImages, nNodes, nodeID, bid, nImages);
  imageSet.Load(true, bid, nImages);
  LOGNODE(logDEBUG) << "Node ID " << nodeID << " Number of nodes: " << nNodes << nImages;
  LOGNODE(logDEBUG) << "Node ID " << nodeID << " Total " << nTotalImages << " NInputs " << nImages;

  // preprocess
  ImagePreprocessor preprocessor(pf.ImagePreprocessor());
  {
    std::vector<RealImage*> imVec = imageSet.GetImageVec();
    std::vector<std::string> imNames = imageSet.GetImageNameVec();
    preprocessor.Process(imVec, imNames);
  }
  

  std::vector<RealImage*> images = imageSet.GetImageVec();
  std::vector<Real> weights = imageSet.GetWeightVec();
  if(imageSet.HasTransforms()){
    LOGNODE(logERROR) << "Error, LDMMAtlas does not accept transforms with initial images";
    std::exit(-1);
  }

  // run LDMM
  std::vector<const RealImage*> constImages;
  constImages.assign(images.begin(),images.end());
  RealImage *meanImage = NULL;
  if(pf.WriteMeanImage()){
    LOGNODE(logDEBUG) << "Will write mean image";
    meanImage = new RealImage();
  }
  std::vector<RealImage*> *defImage = NULL;
  if(pf.WriteDefImages()){
    LOGNODE(logDEBUG) << "Will write deformed images";
    defImage = new std::vector<RealImage*>();
  }
  std::vector<VectorField*> *defField = NULL;
  if(pf.WriteDefFields()){
    LOGNODE(logDEBUG) << "Will write deformation fields";
    defField = new std::vector<VectorField*>();
  }
  std::vector<std::vector<VectorField*> > *velField = NULL;
  if(pf.WriteVelocityFields() || pf.WriteIntermediateImages()){
    LOGNODE(logDEBUG) << "Will write velocity fields";
    velField = new std::vector<std::vector<VectorField*> >();
  }

  if(pf.UseGPU()){
    throw AtlasWerksException(__FILE__, __LINE__, "Error, GPU not supported.  Please use CPU version.");
  }else{
    LDMM::LDMMMultiscaleMultithreadedAtlas(constImages, weights, pf.LDMM(), pf.nThreads(), meanImage, defImage, defField, velField);
  }

  // save images/deformations
  char fname[1024];
  
  if(pf.WriteMeanImage()){
    LOGNODE(logDEBUG) << "Writing mean image.";
    sprintf(fname, "%sMeanImage.%s", pf.OutputPrefix().c_str(), pf.OutputSuffix().c_str());
    ApplicationUtils::SaveImageITK(fname, *meanImage);
  }

  if(pf.WriteDefImages()){
    LOGNODE(logDEBUG) << "Writing " << defImage->size() << " deformed images:";
    for(unsigned int i=0;i<defImage->size();i++){
      sprintf(fname, "%sImage%02d.%s", pf.OutputPrefix().c_str(), i, pf.OutputSuffix().c_str());
      ApplicationUtils::SaveImageITK(fname, *(*defImage)[i]);
    }
  }

  if(pf.WriteDefFields()){
    LOGNODE(logDEBUG) << "Writing " << defField->size() << " deformation fields:";
    for(unsigned int i=0;i<defField->size();i++){
      sprintf(fname, "%sDefField%02d.%s", pf.OutputPrefix().c_str(), i, pf.OutputSuffix().c_str());
      ApplicationUtils::SaveHFieldITK(fname, *(*defField)[i], preprocessor.GetImageSpacing());
    }
  }

  if(pf.WriteVelocityFields() || pf.WriteIntermediateImages()){
    // temp vars for computing deformed images
    VectorField hField(preprocessor.GetImageSize());
    VectorField scratchV(preprocessor.GetImageSize());
    RealImage scratchI(preprocessor.GetImageSize(), preprocessor.GetImageOrigin(), preprocessor.GetImageSpacing());
    LOGNODE(logDEBUG) << "Writing " << velField->size() << " velocity field series...";
    for(unsigned int image=0;image<velField->size();image++){
      LOGNODE(logDEBUG) << "Writing a series of velocity fields for " << (*velField)[image].size() << " timesteps:";
      HField3DUtils::setToIdentity(hField);
      int nTimeSteps = (*velField)[image].size();
      for(int time=nTimeSteps-1;time>=0;time--){
	
	VectorField &curVelField = *(*velField)[image][time];
	if(pf.WriteVelocityFields()){
	  sprintf(fname, "%sImage%02dTime%02dVelField.%s", pf.OutputPrefix().c_str(), image, time, pf.OutputSuffix().c_str());
	  ApplicationUtils::SaveHFieldITK(fname, curVelField, preprocessor.GetImageSpacing());
	}
	
	if(pf.WriteIntermediateImages()){
	  HField3DUtils::composeHV(hField, curVelField, scratchV, preprocessor.GetImageSpacing(),
				   HField3DUtils::BACKGROUND_STRATEGY_PARTIAL_ID);
	  hField = scratchV;
	  HField3DUtils::apply(*constImages[image], hField, scratchI);
	  sprintf(fname, "%sImage%02dTime%02d.%s", pf.OutputPrefix().c_str(), image, time, pf.OutputSuffix().c_str());
	  ApplicationUtils::SaveImageITK(fname, scratchI);
	}
      }
    }
  }

  if(nodeID == 0){
    LOGNODE(logINFO) << "Total Time: " << totalTimer.getTime();
  }

  cleanupMPI();

  return 0;
}

