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

#ifndef __GREEDY_ATLAS_BUILDER_CLUSTER_H__
#define __GREEDY_ATLAS_BUILDER_CLUSTER_H__

#include "GreedyAtlasBuilderInterface.h"
#include "GreedyAtlasBuilderSingleGPUThread.h"
#include "WeightedImageSet.h"

class GreedyAtlasBuilderGPU 
  : public GreedyAtlasBuilderInterface 
{
public:
  
  /**
   * Initialization for MPI version (multiple nodes)
   */
  GreedyAtlasBuilderGPU(unsigned int nodeId, unsigned int nNodes, 
			    unsigned int nTotalImgs,
			    WeightedImageSet &imageSet,
			    const MultiParam<GreedyAtlasScaleLevelParam> &param,
			    unsigned int nGPUs=0);

  /**
   * Single-node version
   */
  GreedyAtlasBuilderGPU(WeightedImageSet &imageSet,
			    const MultiParam<GreedyAtlasScaleLevelParam> &param,
			    unsigned int nGPUs=0);

  ~GreedyAtlasBuilderGPU();

  /**
   * Run the atlas building
   */
  void BuildAtlas();

  /**
   * Main routine for thread that computes average across threads,
   * should not be called by user
   */
  void computeAverageThread();

  void SetComputeInverseHFields(bool computeInverseHFields);
  bool GetComputeInverseHFields();

  void GetImageLocation(int imIdx, int &gpuIdx, int &localIdx);
  void GetDeformedImage(int imIdx, RealImage &im);
  void GetHField(int imIdx, VectorField &vf);
  void GetInvHField(int imIdx, VectorField &vf);
  void GetMeanImage(RealImage &mean);
  
private:
  
  int m_nodeId;
  unsigned int m_nNodes;         // Number of nodes
  unsigned int m_nTotalImgs;         // Number of images

  void checkParams();
  void init();
    
  void initHostData();
  void FreeHostData();

  void computeAverage(int size);
  void SumAcrossNodes(int nP);
  void UpdateGlobalDelta();
    
  // image information
  WeightedImageSet &m_imageSet;
  std::vector<const RealImage*> h_I0;
  std::vector<float> m_weights;
  MultiParam<GreedyAtlasScaleLevelParam> mParams;
    
  unsigned int m_nVox;            // Number of voxels in images
  unsigned int m_nImages;         // Number of images
  unsigned int m_nGPUs;           // Number of GPU device
  std::vector<int> m_bid;         // Index of first image for each GPU

  Vector3Di m_vSize;
  Vector3Df m_vOrg;
  Vector3Df m_vSp;
  
  // this will point to h_avg:
  RealImage* h_avgNode_o;
  // buffer for receiving sum from MPI
  RealImage* h_avgNode_i;
  // Used to hold the squared error from each thread
  Real *m_sqrErr;
    
  std::vector<RealImage*> h_avgL;

  GreedyAtlasBuilderSingleGPUThread** m_builders;
    
  // parameters
  bool mUseGlobalDelta;
  float m_deltaG_i;                 // Global delta value input from MPI
  float m_deltaG_o;                 // Global delta value output to MPI

};

#endif 
