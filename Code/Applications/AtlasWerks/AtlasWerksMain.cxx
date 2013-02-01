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
#include "HField3DUtils.h"
#include "HField3DIO.h"
#include "MultiscaleManager.h"
#include "ScaleLevelParamOrderingConstraint.h"
#include "GreedyAtlasScaleLevelParam.h"
#include "CmdLineParser.h"
#include "WeightedImageSet.h"
#include "ImagePreprocessor.h"
#include "Timer.h"

#include "GreedyAtlasBuilderCPU.h"
#include "GreedyAtlasBuilderGPU.h"

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
class GreedyAtlasParamFile : public CompoundParam {

public:
  GreedyAtlasParamFile()
    : CompoundParam("ParameterFile", "top-level node", PARAM_REQUIRED)
  {
    this->AddChild(WeightedImageSetParam("WeightedImageSet"));
    this->AddChild(ImagePreprocessorParam("ImagePreprocessor"));
    this->AddChild(MultiParam<GreedyAtlasScaleLevelParam>(GreedyAtlasScaleLevelParam())); // name: "GreedyAtlasScaleLevel"
    this->GreedyAtlasScaleLevel().SetConstraint(new ScaleLevelParamOrderingConstraint<GreedyAtlasScaleLevelParam>());
    this->AddChild(ValueParam<bool>("UseGPU", "Compute atlas on the GPU.  Only a subset of normal settings are applicable", PARAM_COMMON, false));
    this->AddChild(ValueParam<unsigned int>("nGPUs", "If UseGPU is true, use this many GPUs (0 lets the system self-select)", PARAM_COMMON, 0));
    this->AddChild(ValueParam<unsigned int>("nThreads", "number of threads to use, 0=one per processor (only for CPU computation)", PARAM_COMMON, 0));
    this->AddChild(ValueParam<bool>("ScaleImageWeights", "If true, scale the image weights to 1.0", PARAM_RARE, true));
    this->AddChild(ValueParam<std::string>("OutputImageNamePrefix", "prefix for the mean image", PARAM_COMMON, "GreedyMeanImage"));  
    this->AddChild(ValueParam<std::string>("OutputDeformedImageNamePrefix", "prefix for each of the deformed images", PARAM_COMMON, ""));  
    this->AddChild(ValueParam<std::string>("OutputHFieldImageNamePrefix", "prefix for each of the deformation fields", PARAM_COMMON, ""));  
    this->AddChild(ValueParam<std::string>("OutputInvHFieldImageNamePrefix", "prefix for each of the inverse deformation fields", PARAM_COMMON, ""));
    this->AddChild(ValueParam<std::string>("OutputFileType", "filename extension to use (determines format)", PARAM_COMMON, "mha"));
  }
  
  ParamAccessorMacro(WeightedImageSetParam, WeightedImageSet)
  ParamAccessorMacro(ImagePreprocessorParam, ImagePreprocessor)
  //ParamAccessorMacro(MultiParam<WeightedInputImageParam>, WeightedInputImage)
  ParamAccessorMacro(MultiParam<GreedyAtlasScaleLevelParam>, GreedyAtlasScaleLevel)
  //ParamAccessorMacro(IntensityWindowParam, IntensityWindow)
  ValueParamAccessorMacro(bool, UseGPU)
  ValueParamAccessorMacro(unsigned int, nGPUs)
  ValueParamAccessorMacro(unsigned int, nThreads)
  ValueParamAccessorMacro(bool, ScaleImageWeights)
  ValueParamAccessorMacro(std::string, OutputImageNamePrefix)
  ValueParamAccessorMacro(std::string, OutputDeformedImageNamePrefix)
  ValueParamAccessorMacro(std::string, OutputHFieldImageNamePrefix)
  ValueParamAccessorMacro(std::string, OutputInvHFieldImageNamePrefix)
  ValueParamAccessorMacro(std::string, OutputFileType)

  CopyFunctionMacro(GreedyAtlasParamFile)
};


int main(int argc, char **argv)
{

  int nNodes, nodeID; // Number of node and node ID
  setupMPI(argc, argv, nNodes, nodeID);  //MPI Setup

  Timer totalTimer;
  if(nodeID == 0)
    totalTimer.start();

  GreedyAtlasParamFile pf;

  CmdLineParser parser(pf);

  try{
    parser.Parse(argc,argv);
  }catch(ParamException &e){
    std::cerr << "Error parsing arguments:" << std::endl;
    std::cerr << "   " << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  bool computeInverseHFields = false;

  std::string outputImageFilenamePrefix = pf.OutputImageNamePrefix();
  std::string outputDeformedImageFilenamePrefix = pf.OutputDeformedImageNamePrefix();
  std::string outputHFieldFilenamePrefix = pf.OutputHFieldImageNamePrefix();
  std::string outputHInvFieldFilenamePrefix = pf.OutputInvHFieldImageNamePrefix();
  std::string outputFileType = pf.OutputFileType();
  if(outputHInvFieldFilenamePrefix.length() > 0){
    computeInverseHFields = true;
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

  OriginType origin = imageSet.GetImageOrigin();
  SpacingType spacing = imageSet.GetImageSpacing();

  std::cout << "ImageSet Origin is " << origin << std::endl;
  std::cout << "ImageSet Spacing is " << spacing << std::endl;

  GreedyAtlasBuilderInterface *builder = NULL;

  if(pf.UseGPU()){

    setupCUDA(argc, argv);

    builder = new GreedyAtlasBuilderGPU(nodeID, nNodes, nTotalImages,
					imageSet,
					pf.GreedyAtlasScaleLevel(),
					pf.nGPUs());
  }else{
    builder = new GreedyAtlasBuilderCPU(nodeID, nNodes, nTotalImages,
					imageSet, pf.GreedyAtlasScaleLevel(), numberOfThreads);
  }
    
  builder->SetComputeInverseHFields(computeInverseHFields);

  builder->BuildAtlas();
    
  appout << "DONE Computing Atlas." << std::endl;

  // Write out mean image if this is node 0
  if(nodeID == 0){
    //
    // write atlas image
    //
    if (outputImageFilenamePrefix != "")
      {
	RealImage mean;
	builder->GetMeanImage(mean);
	std::ostringstream oss;
	oss << outputImageFilenamePrefix << "." << outputFileType;
	appout << "Writing Average Image...";
	ApplicationUtils::SaveImageITK(oss.str().c_str(), mean);
	appout << "DONE" << std::endl;
      }
  } // end if node 0
  
  //
  // write deformed images
  //
  if (outputDeformedImageFilenamePrefix != "")
    {
      appout << "Node " << nodeID << " Writing " << nImages << " deformed images" << std::endl;
      RealImage defImg;
      for (unsigned int imageIndex = 0; imageIndex < (uint)nImages; ++imageIndex)
	{
	  builder->GetDeformedImage(imageIndex, defImg);
	  std::ostringstream oss;
	  oss << outputDeformedImageFilenamePrefix
	      << std::setw(4) << std::setfill('0') << bid+imageIndex << "." << outputFileType;;
	  appout << "Writing Deformed Image " << outputDeformedImageFilenamePrefix << "...";
	  ApplicationUtils::SaveImageITK(oss.str().c_str(), defImg);
	  appout << "DONE" << std::endl;      
	}
    }
  else
    {
      appout << "Node " << nodeID << " Not writing deformed images" <<  std::endl;
    }
  
  //
  // write h fields at last scale level
  //
  if (outputHFieldFilenamePrefix != "") 
    {
      appout << "Node " << nodeID << " Writing " << nImages << " deformation fields" << std::endl;
      VectorField hField;
      for (int i = 0; i < (int) nImages; ++i) 
	{
	  builder->GetHField(i, hField);
	  std::stringstream ss;
	  ss << outputHFieldFilenamePrefix << std::setw(4) << std::setfill('0') << bid+i << "." << outputFileType;
	  appout << "Writing H Field " << outputHFieldFilenamePrefix << "...";
	  ApplicationUtils::SaveHFieldITK(ss.str().c_str(), hField, origin, spacing);
	  appout << "DONE" << std::endl;      
	}
    }
  else
    {
      appout << "Node " << nodeID << " Not writing deformation fields" <<  std::endl;
    }
 
  if (computeInverseHFields && outputHInvFieldFilenamePrefix != "") 
    {
      appout << "Node " << nodeID << " Writing " << nImages << " inverse deformation fields" << std::endl;
      VectorField invHField;
      for (int i = 0; i < (int) nImages; ++i) 
	{
	  builder->GetInvHField(i, invHField);
	  std::stringstream ss;
	  ss << outputHInvFieldFilenamePrefix << std::setw(4) << std::setfill('0') << bid+i << "." << outputFileType;
	  appout << "Writing inverse H Field " << outputHInvFieldFilenamePrefix << "...";
	  ApplicationUtils::SaveHFieldITK(ss.str().c_str(), invHField, origin, spacing);
	  appout << "DONE" << std::endl;      
	}
    }
  else
    {
      appout << "Node " << nodeID << " Not writing inverse deformation fields" <<  std::endl;
    }
  
  
  if(nodeID == 0)
    appout << "Total Time: " << totalTimer.getTime() << std::endl;
  
  // delete images
  imageSet.Clear();
  
  appout << "Deleting atlas builder..." << std::endl;
  // delete builder
  delete builder;

  appout << "Done deleting atlas builder." << std::endl;
  
  cleanupMPI();
  
  return 0;
}


