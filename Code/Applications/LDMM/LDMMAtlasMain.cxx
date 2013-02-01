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

#include "LDMMAtlasBuilder.h"
#include "LDMMShootingAtlasThreadCPU.h"
#include "LDMMAtlasThreadCPU.h"
#include "WeightedImageSet.h"
#include "ImagePreprocessor.h"
#include "CmdLineParser.h"

#ifdef CUDA_ENABLED
#include "LDMMAtlasThreadGPU.h"
#include "LDMMShootingAtlasThreadGPU.h"
#endif

#ifdef MPI_ENABLED
#include <mpi.h>
#endif // MPI_ENABLED

#include "log.h"

/**
 * \page LDMMAtlasNew LDMMAtlasNew
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
class LDMMAtlasParamFile : public LDMMAtlasParam {
public:
  LDMMAtlasParamFile()
    : LDMMAtlasParam("ParameterFile", "top-level node", PARAM_REQUIRED)
  {
    this->AddChild(WeightedImageSetParam("WeightedImageSet"));
    this->AddChild(ImagePreprocessorParam("ImagePreprocessor"));
    this->AddChild(ValueParam<bool>("ShootingOptimization", "Use shooting optimization instead of relaxation?", PARAM_COMMON, false));
    this->AddChild(ValueParam<bool>("UseGPU", "Compute atlas on the GPU.  Only a subset of normal settings are applicable", PARAM_COMMON, false));
    this->AddChild(ValueParam<unsigned int>("nThreads", "number of threads to use, 0=one per image", PARAM_COMMON, 0));
  }
  
  ParamAccessorMacro(WeightedImageSetParam, WeightedImageSet)
  ParamAccessorMacro(ImagePreprocessorParam, ImagePreprocessor)
  ValueParamAccessorMacro(bool, UseGPU)
  ValueParamAccessorMacro(bool, ShootingOptimization)
  ValueParamAccessorMacro(unsigned int, nThreads)

  CopyFunctionMacro(LDMMAtlasParamFile)
};

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
    std::cerr << "Error parsing arguments:" << std::endl;
    std::cerr << "   " << e.what() << std::endl;
    std::exit(-1);
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
  LOGNODE(logINFO) << "Node ID " << nodeID << " Number of nodes: " << nNodes << " Number of images: "<< nImages;
  LOGNODE(logINFO) << "Node ID " << nodeID << " Total " << nTotalImages << " NInputs " << nImages;

  // preprocess
  ImagePreprocessor preprocessor(pf.ImagePreprocessor());
  {
    std::vector<RealImage*> imVec = imageSet.GetImageVec();
    std::vector<std::string> imNames = imageSet.GetImageNameVec();
    preprocessor.Process(imVec, imNames);
  }
  

  if(imageSet.HasTransforms()){
    LOGNODE(logERROR) << "Error, LDMMAtlas does not accept transforms with initial images";
    std::exit(-1);
  }

  AtlasBuilderInterface *builder = NULL;
  if(pf.UseGPU()){
#ifdef CUDA_ENABLED
    if(pf.ShootingOptimization()){
      builder = 
 	new LDMMAtlasBuilder<LDMMShootingAtlasManagerGPU>
 	(imageSet, pf, pf.nThreads(), nodeID, nNodes, nTotalImages);
    }else{
      builder = 
 	new LDMMAtlasBuilder<LDMMAtlasManagerGPU>
 	(imageSet, pf, pf.nThreads(), nodeID, nNodes, nTotalImages);
    }
#else
    throw AtlasWerksException(__FILE__, __LINE__, "Error, GPU code not built.  Select USE_CUDA in CMake settings.");
#endif
  }else{
    if(pf.ShootingOptimization()){
      builder = new LDMMAtlasBuilder<LDMMShootingAtlasManagerCPU>
	(imageSet, pf, pf.nThreads(), nodeID, nNodes, nTotalImages);
    }else{
      builder = new LDMMAtlasBuilder<LDMMAtlasManagerCPU>
 	(imageSet, pf, pf.nThreads(), nodeID, nNodes, nTotalImages);
    }
  }

  //builder->Init();
  builder->BuildAtlas();

  builder->GenerateOutput();
  
  if(nodeID == 0)
    std::cerr << "Total Time: " << totalTimer.getTime() << std::endl;
  
  cleanupMPI();
  
  return 0;
}


