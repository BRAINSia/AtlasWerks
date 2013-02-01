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

#include "GreedyAtlasBuilderGPU.h"
#include "AtlasWerksException.h"

GreedyAtlasBuilderGPU::
GreedyAtlasBuilderGPU(
    unsigned int nodeId,
    unsigned int nNodes,
    unsigned int nTotalImgs,
    WeightedImageSet &imageSet,
    const MultiParam<GreedyAtlasScaleLevelParam> &param,
    unsigned int nGPUs)
    :m_nodeId(nodeId),
     m_nNodes(nNodes),
     m_nTotalImgs(nTotalImgs),
     m_imageSet(imageSet),
     mParams(param),
     m_nGPUs(nGPUs),
     mUseGlobalDelta(false),
     m_deltaG_i(FLT_MAX),
     m_deltaG_o(FLT_MAX)
{
  init();
}

GreedyAtlasBuilderGPU::
GreedyAtlasBuilderGPU(
    WeightedImageSet &imageSet,
    const MultiParam<GreedyAtlasScaleLevelParam> &param,
    unsigned int nGPUs)
    :m_nodeId(0),
     m_nNodes(1),
     m_imageSet(imageSet),
     mParams(param),
     m_nGPUs(nGPUs),
     mUseGlobalDelta(false),
     m_deltaG_i(FLT_MAX),
     m_deltaG_o(FLT_MAX)
{
  init();
}

GreedyAtlasBuilderGPU::
~GreedyAtlasBuilderGPU()
{
}

void
GreedyAtlasBuilderGPU::
checkParams()
{
}

void 
GreedyAtlasBuilderGPU::
init()
{
  throw AtlasWerksException(__FILE__, __LINE__, 
			 "Error: CUDA implementation not built, please use CPU version!");
}

void 
GreedyAtlasBuilderGPU::
initHostData()
{
}

void 
GreedyAtlasBuilderGPU::
FreeHostData()
{
}

void 
GreedyAtlasBuilderGPU::
SumAcrossNodes(int nP)
{
}

void 
GreedyAtlasBuilderGPU::
UpdateGlobalDelta()
{
}

// Compute the avergage image using the reduction 
// result return in the h_avgL[0] array 
void 
GreedyAtlasBuilderGPU::
computeAverageThread()
{
}

//
// Compute the sum on a this node, divided by the total number of
// images across *all* nodes.  The result is saved in h_avgL[0].
//
void 
GreedyAtlasBuilderGPU::
computeAverage(int nVox)
{
}

void 
GreedyAtlasBuilderGPU::
BuildAtlas()
{
}


void 
GreedyAtlasBuilderGPU::
GetMeanImage(RealImage &mean)
{
}

void 
GreedyAtlasBuilderGPU::
GetDeformedImage(int imIdx, RealImage &im)
{
}

void 
GreedyAtlasBuilderGPU::
GetImageLocation(int imIdx, int &gpuIdx, int &localIdx)
{
}
void 
GreedyAtlasBuilderGPU::
GetHField(int imIdx, VectorField &vf)
{
}

void 
GreedyAtlasBuilderGPU::
GetInvHField(int imIdx, VectorField &vf)
{
}

void 
GreedyAtlasBuilderGPU::
SetComputeInverseHFields(bool computeInverseHFields)
{
}

bool 
GreedyAtlasBuilderGPU::
GetComputeInverseHFields()
{
  return false;
}


