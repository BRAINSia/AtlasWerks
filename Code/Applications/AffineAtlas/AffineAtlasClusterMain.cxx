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


#include <ctime>
#include <string>
#include <sstream>
#include <vector>
#include <iostream>
#include <iomanip>
#include <exception>

#include "AtlasWerksTypes.h"
#include "ApplicationUtils.h"
#include "Image.h"
#include "Vector3D.h"
#include "Array3D.h"
#include "Array3DUtils.h"
#include "ImageUtils.h"
#include "MultiscaleManager.h"
#include "ScaleLevelParamOrderingConstraint.h"
#include "CmdLineParser.h"
#include "WeightedImageSet.h"
#include "ImagePreprocessor.h"
#include "Timer.h"

#include "AffineAtlasBuilderCPU.h"

#include <itkMultiThreader.h>
#include <itkTransformFileReader.h>
#include <itkTransformBase.h>
#include <itkAffineTransform.h>

#ifdef MPI_ENABLED
#include <mpi.h>
#endif // MPI_ENABLED

#ifdef CUDA_ENABLED
#include <cudaInterface.h>
#endif // CUDA_ENABLED

/**
 * \page AtlasWerks
 * 
 * Create an atlas image using greedy fluid deformation method.  Run
 * AtlasWerks with no parameters for a list of options.  Supports
 * multiple scale levels.  Multithreaded.
 *
 * If tests have been run (`make test'), an example page has been generated
 * \htmlonly
 * <a href="../../Testing/Applications/Greedy/AtlasWerks/index.html"><b>here.</b></a>
 * \endhtmlonly
 */

#ifndef appout
#define appout std::cerr
#endif

void setupCUDA(int argc, char **argv){
#ifdef CUDA_ENABLED
  cudaInit(argc, argv);
#endif// CUDA_ENABLED
}

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

/** Simple parameter class to hold parameters for greedy atlas building */
class AffineAtlasClusterParamFile : public CompoundParam {
public:
        AffineAtlasClusterParamFile()
        : CompoundParam("ParameterFile", "top-level node", PARAM_REQUIRED)
        {
                this->AddChild(WeightedImageSetParam("WeightedImageSet"));
                this->AddChild(ImagePreprocessorParam("ImagePreprocessor"));
                this->AddChild(ValueParam<std::string>("RegistrationType", "Resistration Type: Affine, Translation, Rigid",PARAM_REQUIRED, ""));
                this->AddChild(ValueParam<unsigned int>("nIterations", "number of Iterations", PARAM_REQUIRED, 50));

                this->AddChild(ValueParam<bool>("UseGPU", "Compute atlas on the GPU.  Only a subset of normal settings are applicable", PARAM_COMMON, false));
                this->AddChild(ValueParam<unsigned int>("nGPUs", "If UseGPU is true, use this many GPUs (0 lets the system self-select)", PARAM_COMMON, 0));
                this->AddChild(ValueParam<unsigned int>("nThreads", "number of threads to use, 0=one per processor (only forCPU computation)", PARAM_COMMON, 0));
                this->AddChild(ValueParam<std::string>("OutputImageNamePrefix", "prefix for the mean image", PARAM_COMMON, "TransformedImage"));
		this->AddChild(ValueParam<bool>("WriteTransformedImages", "If the value is true the final transformed images are written", PARAM_COMMON, false));

        }

        ParamAccessorMacro(WeightedImageSetParam, WeightedImageSet)
        ParamAccessorMacro(ImagePreprocessorParam, ImagePreprocessor)
        ValueParamAccessorMacro(std::string, RegistrationType)
        ValueParamAccessorMacro(unsigned int, nIterations)

        ValueParamAccessorMacro(bool, UseGPU)
        ValueParamAccessorMacro(unsigned int, nGPUs)
        ValueParamAccessorMacro(unsigned int, nThreads)
        ValueParamAccessorMacro(bool, ScaleImageWeights)
        ValueParamAccessorMacro(std::string, OutputImageNamePrefix)
	ValueParamAccessorMacro(bool, WriteTransformedImages)

        CopyFunctionMacro(AffineAtlasClusterParamFile)

};

int main(int argc, char **argv)
{

  int nNodes, nodeID; // Number of node and node ID
  //std::cout<<"argc: "<<argc<<"   argv: "<<*argv[1]<<std::endl;
  setupMPI(argc, argv, nNodes, nodeID);  //MPI Setup
  //std::cout<<"nNodes: "<<nNodes<<"   nodeID: "<<nodeID;
  Timer totalTimer;
  if(nodeID == 0)
    totalTimer.start();

  AffineAtlasClusterParamFile pf;

  CmdLineParser parser(pf);

  try{
    parser.Parse(argc,argv);
  }catch(ParamException &e){
    std::cerr << "Error parsing arguments:" << std::endl;
    std::cerr << "   " << e.what() << std::endl;
    return EXIT_FAILURE;
  }


  int  numberOfThreads = pf.nThreads();
  if(numberOfThreads == 0){
    numberOfThreads = itk::MultiThreader::GetGlobalDefaultNumberOfThreads();
  }

  //
  // load images
  //
  WeightedImageSet imageSet(pf.WeightedImageSet());
  // number of images across all nodes
  unsigned int nTotalImages = imageSet.NumImages();
  // number of images on this node
  int nImages = 0;
  // beginning image index for this node
  int bid = 0;

  std::cerr << "Node ID " << nodeID << " Number of nodes: " << nNodes << ", Total number of images: " << nTotalImages << std::endl;
  ApplicationUtils::Distribute(nTotalImages, nNodes, nodeID, bid, nImages);
  std::cerr << "Node ID " << nodeID << ", offset " << bid << ", nInputs " << nImages << std::endl;
  imageSet.Load(true, bid, nImages);
  ImagePreprocessor preprocessor(pf.ImagePreprocessor());
  {
    std::vector<RealImage*> imVec = imageSet.GetImageVec();
    std::vector<std::string> imNames = imageSet.GetImageNameVec();
    preprocessor.Process(imVec, imNames);
  }

  SpacingType spacing = imageSet.GetImageSpacing();

  std::string regtype = pf.RegistrationType();
  int  nIterations = pf.nIterations();


  AffineAtlasBuilderInterface *builder = NULL;

  if(pf.UseGPU()){

	//setupCUDA(argc, argv);
	//The Program with cuda implementation should be called here
	// As of now its not coded.

  }else{
    builder = new AffineAtlasBuilderCPU(nodeID, nNodes, nTotalImages,
					imageSet, numberOfThreads,regtype,nIterations,pf.WriteTransformedImages());
  }
    
  builder->BuildAtlas();
  if (nodeID == 0)  
	appout << "DONE Computing Atlas." << std::endl;

  // Write out mean image if this is node 0
  if(nodeID == 0){
    //
    // write atlas image
    //
	RealImage mean;
	builder->GetMeanImage(mean);
	appout << "DONE" << std::endl;
  } // end if node 0
  
  // delete images
  imageSet.Clear();

  if(nodeID == 0)
  {
   	 appout << "Total Time: " << totalTimer.getTime() << std::endl;
  
	appout << "Deleting atlas builder..." << std::endl;
	// delete builder
	delete builder;

	appout << "Done deleting atlas builder." << std::endl;
  }
  
  cleanupMPI();
  
  return 0;
}


