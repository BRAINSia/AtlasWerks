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

#ifndef __GREEDY_ATLAS_BUILDER_CPU_H__
#define __GREEDY_ATLAS_BUILDER_CPU_H__

#include "GreedyAtlasBuilderInterface.h"

#include "AtlasWerksTypes.h"
#include "Array3DUtils.h"
#include "ImageUtils.h"
#include "HField3DUtils.h"
#include "MultiscaleManager.h"
#include "GreedyAtlasScaleLevelParam.h"
#include "WeightedImageSet.h"
#include "AtlasBuilder.h"

class GreedyAtlasBuilderCPU 
  : public GreedyAtlasBuilderInterface 
{
public:
  
  /**
   * Initialization for MPI version (multiple nodes)
   */
  GreedyAtlasBuilderCPU(unsigned int nodeId, unsigned int nNodes, 
			unsigned int nTotalImgs,
			WeightedImageSet &imageSet,
			const MultiParam<GreedyAtlasScaleLevelParam> &param,
			unsigned int nThreads);
  
  ~GreedyAtlasBuilderCPU();

  /**
   * Run atlas building
   */
  void BuildAtlas();

  void SetComputeInverseHFields(bool computeInverseHFields){ m_computeInverseHFields = computeInverseHFields; }
  bool GetComputeInverseHFields(){ return m_computeInverseHFields; }

  void GetDeformedImage(int imIdx, RealImage &im);
  void GetHField(int imIdx, VectorField &vf);
  void GetInvHField(int imIdx, VectorField &vf);
  void GetMeanImage(RealImage &mean);
  
private:
  
  static void invert(RealAffineTransform *invtransform);
  
  /**
   * initialize h and hinv to a vector field representation of affine
   * and inv(affine), respectively
   */
  static void initializeHField(VectorField* h,
			       VectorField* hinv,
			       RealAffineTransform *affine,
			       Vector3D<float> spacing,
			       Vector3D<float> origin);

  int m_nodeId;                  // Node ID, if using multiple nodes (MPI)
  unsigned int m_nNodes;         // Number of nodes
  unsigned int m_nTotalImgs;     // Number of images
  unsigned int m_nImages;        // Number of images
  unsigned int m_nThreads;
  
  Vector3D<unsigned int> m_imSize;
  Vector3D<double> m_imOrigin;
  Vector3D<double> m_imSpacing;

  // image information
  WeightedImageSet &m_imageSet;
  const MultiParam<GreedyAtlasScaleLevelParam> &m_scaleLevelParams;
  
  RealImage *m_iHat;
  RealImage** m_scaledImages;
  VectorField** m_h;
  VectorField** m_hinv;

  bool m_computeInverseHFields;
  
};

#endif 
