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

#include "GreedyAtlasBuilderSingleGPUThread.h"
/*
 * This is just a null implementation for CPU-only builds which just
 * throws and exception if the GPU code is called.
 */


#include "AtlasWerksException.h"
#include <vector>

GreedyAtlasBuilderSingleGPUThread::
GreedyAtlasBuilderSingleGPUThread(int node_id, int thread_id,
                                  std::vector<const RealImage*> images,
                                  const MultiParam<GreedyAtlasScaleLevelParam> &param,
                                  RealImage* h_avgL, RealImage* h_avgG, Real* h_sqrErr, int nTotalImgs)
  :  mParams(param)
{
  throw AtlasWerksException(__FILE__, __LINE__, 
			 "Error: GreedyAtlasBuilderSingleGPUThread: CUDA implementation not built, "
			 "please use CPU version!");
}

GreedyAtlasBuilderSingleGPUThread::
~GreedyAtlasBuilderSingleGPUThread()
{
  throw AtlasWerksException(__FILE__, __LINE__, 
			 "Error: GreedyAtlasBuilderSingleGPUThread: CUDA implementation not built, "
			 "please use CPU version!");
}

void 
GreedyAtlasBuilderSingleGPUThread::
InitDeviceData()
{
  throw AtlasWerksException(__FILE__, __LINE__, 
			 "Error: GreedyAtlasBuilderSingleGPUThread: CUDA implementation not built, "
			 "please use CPU version!");
}

void 
GreedyAtlasBuilderSingleGPUThread::
FreeDeviceData()
{
  throw AtlasWerksException(__FILE__, __LINE__, 
			 "Error: GreedyAtlasBuilderSingleGPUThread: CUDA implementation not built, "
			 "please use CPU version!");
}

void 
GreedyAtlasBuilderSingleGPUThread::
BuildAtlas()
{
  throw AtlasWerksException(__FILE__, __LINE__, 
			 "Error: GreedyAtlasBuilderSingleGPUThread: CUDA implementation not built, "
			 "please use CPU version!");
}



