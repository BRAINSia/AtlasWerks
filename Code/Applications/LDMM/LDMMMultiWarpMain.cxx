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


#include "ImagePreprocessor.h"
#include "CmdLineParser.h"
#include <multithreading.h>
#include "LDMMWarpAd.h"

#include <stdlib.h>
#include <sstream>

#include <log.h>

#ifdef CUDA_ENABLED
#include "LDMMWarpGPU.h"
#include <cudaInterface.h>
#include "LDMMAdShootingIteratorGPU.h"
#endif

#ifdef MPI_ENABLED
#include <mpi.h>
#endif // MPI_ENABLED

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

class WarpInputParam : public CompoundParam {
public:
  WarpInputParam(std::string name = "WarpInput", 
		 std::string desc = "Input images for single warp", 
		 ParamLevel level = PARAM_COMMON)
    : CompoundParam(name, desc, level)
  {
    ValueParam<std::string> MovingImageParam("MovingImage", "moving image filename", PARAM_REQUIRED, "");
    MovingImageParam.AddAlias("InitialImage");
    this->AddChild(MovingImageParam);

    ValueParam<std::string> StaticImageParam("StaticImage", "static (template) image filename", PARAM_REQUIRED, "");
    StaticImageParam.AddAlias("FinalImage");
    this->AddChild(StaticImageParam);
  }
  ValueParamAccessorMacro(std::string, MovingImage)
  ValueParamAccessorMacro(std::string, StaticImage)
  
  CopyFunctionMacro(WarpInputParam)
};

class LDMMMultiWarpParamFile : public LDMMWarpParam {
public:
  LDMMMultiWarpParamFile()
    : LDMMWarpParam()
  {
    this->AddChild(MultiParam<WarpInputParam>(WarpInputParam("WarpInput")));
    this->AddChild(ImagePreprocessorParam("ImagePreprocessor"));
    ValueParam<bool> PreprocessMovingImageParam("PreprocessMovingImage", "Run preprocessing on moving image", PARAM_COMMON, true);

    PreprocessMovingImageParam.AddAlias("PreprocessInitialImage");
    this->AddChild(PreprocessMovingImageParam);

    ValueParam<bool> PreprocessStaticImageParam("PreprocessStaticImage", "Run preprocessing on static image", PARAM_COMMON, true);

    PreprocessStaticImageParam.AddAlias("PreprocessFinalImage");
    this->AddChild(PreprocessStaticImageParam);

    this->AddChild(ValueParam<bool>("ShootingOptimization", "Use shooting optimization instead of relaxation?", PARAM_COMMON, false));
    this->AddChild(ValueParam<bool>("UseVelocityShootingOptimization", "If ShootingOptimization is true, this parameter chooses between Alpha0 shooting and V0 shooting", PARAM_RARE, false));
    this->AddChild(ValueParam<bool>("UseAdjointShootingOptimization", "If ShootingOptimization is true, this parameter chooses between Alpha0 shooting (old style) and Alpha0 shooting (using Adjoint equations as in Francios Xavier 2011 et al)", PARAM_COMMON, false));

    this->AddChild(ValueParam<bool>("UseGPU", "Compute atlas on the GPU.  Only a subset of normal settings are applicable", PARAM_COMMON, true));
  }
  
  ParamAccessorMacro(MultiParam<WarpInputParam>, WarpInput)
  ParamAccessorMacro(ImagePreprocessorParam, ImagePreprocessor)
  ValueParamAccessorMacro(bool, PreprocessMovingImage)
  ValueParamAccessorMacro(bool, PreprocessStaticImage)
  ValueParamAccessorMacro(bool, ShootingOptimization)
  ValueParamAccessorMacro(bool, UseVelocityShootingOptimization)
  ValueParamAccessorMacro(bool, UseAdjointShootingOptimization)
  ValueParamAccessorMacro(bool, UseGPU)

  CopyFunctionMacro(LDMMMultiWarpParamFile)
};


class LDMMWarpData {
public:
  unsigned int nodeId;
  unsigned int imIdx;
  unsigned int threadIdx;
  bool useAdjointShootingOpt;
  bool writeVelocityFields;
  bool writeDefImage;
  bool writeDefField;
  bool writeInvDefField;
  bool writeIntermediateImages;
  std::string outPrefix;
  std::string outSuffix;
  std::string movingImageName;
  std::string staticImageName;
  RealImage *I0;
  RealImage *IT;
  const LDMMWarpParam *param;
};

/**
 * Main routine for running GPU warp
 */
void buildGPU(LDMMWarpData *data)
{

#ifdef CUDA_ENABLED
  std::string threadname = StringUtils::strPrintf("GPUThread%d",data->threadIdx);
  ErrLog::SetThreadName(threadname);

  LOGNODETHREAD(logINFO) << "Starting warp " << data->imIdx << ", warping image " 
			 << data->movingImageName << " to "
			 << data->staticImageName;

  // set the GPU to use
  CUDAUtilities::SetCUDADevice(data->threadIdx);
  // ensure device supports CUDA capability version 1.2
  CUDAUtilities::AssertMinCUDACapabilityVersion(1,2);

  std::string path, movingNameBase, nameExt;
  ApplicationUtils::SplitName(data->movingImageName.c_str(), path, movingNameBase, nameExt);
  std::string staticNameBase;
  ApplicationUtils::SplitName(data->staticImageName.c_str(), path, staticNameBase, nameExt);
  std::string nameBase = movingNameBase + "_to_" + staticNameBase;

  if(data->useAdjointShootingOpt){
    LDMMWarpAd<LDMMAdShootingIteratorGPU> 
      warper(data->I0, data->IT, *data->param, nameBase);
    warper.RunWarp();
    warper.GenerateOutput();
  }else{
    LDMMWarpGPU warper(data->I0, data->IT, *data->param, nameBase);
    warper.RunWarp();
    warper.GenerateOutput();
  }
  
  LOGNODETHREAD(logDEBUG) << "LDMMWarp exiting successfully";

#else
  throw AtlasWerksException(__FILE__, __LINE__, "Cannot run warp on GPU, CUDA code not compiled");
#endif
}

/**
 * \page LDMMMultiWarp LDMMMultiWarp 
 *
 * Simple frontend program for computing many LDMM image registrations
 * distributed on the cluster
 *
 */

#define HOSTNAME_SIZE 256

int main(int argc, char ** argv)
{
  
  int nNodes, nodeID; // Number of node and node ID
  setupMPI(argc, argv, nNodes, nodeID);  //MPI Setup
  ErrLog::SetNodeID(nodeID);

  char hostname[HOSTNAME_SIZE];
  gethostname(hostname, HOSTNAME_SIZE);
  LOGNODE(logINFO) << "Host " << hostname << " Running as Node " << nodeID;

  LDMMMultiWarpParamFile pf;

  CmdLineParser parser(pf);

  try{
    parser.Parse(argc,argv);
  }catch(ParamException &e){
    LOGNODE(logERROR) << "Error parsing arguments:" << std::endl
		      << "   " << e.what();
    std::exit(-1);
  }

  if(!pf.UseGPU()){
    throw AtlasWerksException(__FILE__, __LINE__, 
			   "Error, LDMMMultiWarp currently only supports GPU warping.");
  }

  bool useShooting = false;

  // only adjoing shooting optimization supported
  if(pf.ShootingOptimization()){
    if(pf.UseVelocityShootingOptimization()){
      throw AtlasWerksException(__FILE__, __LINE__, 
				"Error, LDMMMultiWarp does not currently "
				"support velocity shooting optimization");
    }
    if(pf.UseAdjointShootingOptimization()){
      useShooting = true;
    }else{
      throw AtlasWerksException(__FILE__, __LINE__, 
				"Error, LDMMMultiWarp only supports "
				"adjoint shooting optimization");
    }
  }




  // debugging output
  if(nodeID == 0){
    for(unsigned int i=0;i<pf.WarpInput().size();i++){
      LOGNODE(logINFO) << "Warp " << i << ": " 
		       << pf.WarpInput()[i].MovingImage() << " --> "
		       << pf.WarpInput()[i].StaticImage();
    }
  }

  // total number of warps
  unsigned int nTotalWarps = pf.WarpInput().size();
  // number of warps on this node
  int nWarps = 0;
  // beginning warp index for this node
  int bid = 0;

  // distribute among multiple nodes
  ApplicationUtils::Distribute(nTotalWarps, nNodes, nodeID, bid, nWarps);
  LOGNODE(logINFO) << "Number of nodes: " << nNodes;
  LOGNODE(logINFO) << "Total # of warps:" << nTotalWarps << " number of local warps: " << nWarps;

  // determine the number of GPUs
  unsigned int nSystemGPUs = getNumberOfCapableCUDADevices();
  LOGNODE(logINFO) << "Number of GPUs: " << nSystemGPUs;

  if(nSystemGPUs < 1){
    std::stringstream ss;
    ss << "Error, no GPUs found on host " << hostname << "!";
    throw AtlasWerksException(__FILE__, __LINE__, ss.str().c_str());
  }

  //
  // loop over warps for this node, running them serially
  //
  unsigned int warpIdx = bid;
  while((int)warpIdx < bid+nWarps){

    unsigned int nThreads = nSystemGPUs;
    if((int)(warpIdx+nSystemGPUs) > bid+nWarps) nThreads = bid+nWarps - warpIdx;
    
    LOGNODE(logINFO) << "Starting " << nThreads << " warps, base global ID is " << warpIdx;
    
    CUTThread*  threadID = new CUTThread[nThreads];
    LDMMWarpData* threadData = new LDMMWarpData[nThreads];

    for(unsigned int threadId = 0; threadId < nThreads; threadId++){
      
      RealImage *images[2]; 
      images[0] = new RealImage();
      images[1] = new RealImage();
      
      std::string movingImageName = pf.WarpInput()[warpIdx+threadId].MovingImage();
      std::string staticImageName = pf.WarpInput()[warpIdx+threadId].StaticImage();

      ApplicationUtils::LoadImageITK(movingImageName.c_str(),*images[0]);
      ApplicationUtils::LoadImageITK(staticImageName.c_str(),*images[1]);
      
      ImagePreprocessor preprocessor(pf.ImagePreprocessor());
      if(pf.PreprocessMovingImage()){
	preprocessor.Process(*images[0]);
      }
      if(pf.PreprocessStaticImage()){
	preprocessor.Process(*images[1]);
      }

      threadData[threadId].nodeId = nodeID;
      threadData[threadId].imIdx = warpIdx+threadId;
      threadData[threadId].threadIdx = threadId;
      threadData[threadId].writeVelocityFields = pf.WriteVelocityFields();
      threadData[threadId].writeDefImage = pf.WriteDefImage();
      threadData[threadId].writeDefField = pf.WriteDefField();
      threadData[threadId].writeInvDefField = pf.WriteInvDefField();
      threadData[threadId].writeIntermediateImages = pf.WriteIntermediateImages();
      threadData[threadId].outPrefix = pf.OutputPrefix();
      threadData[threadId].outSuffix = pf.OutputSuffix();
      threadData[threadId].movingImageName = movingImageName;
      threadData[threadId].staticImageName = staticImageName;
      threadData[threadId].I0 = images[0];
      threadData[threadId].IT = images[1];
      threadData[threadId].param = &pf;
      threadData[threadId].useAdjointShootingOpt = useShooting;

      LOGNODE(logINFO) << "Running local warp #" << warpIdx-bid+threadId+1 << " of " << nWarps << ", global ID " << warpIdx+threadId;

      threadID[threadId] = cutStartThread((CUT_THREADROUTINE)buildGPU, (void *)&threadData[threadId]);
    }

    LOGNODE(logINFO) << "Waiting for warps to finish.";
    cutWaitForThreads(threadID, nThreads);
    LOGNODE(logINFO) << "Warps have finished.";

    warpIdx+= nThreads;

    delete [] threadID;
    delete [] threadData;

  }

  LOGNODE(logINFO) << "Finishing.";
  
  cleanupMPI();

}

