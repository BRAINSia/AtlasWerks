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

#ifndef __AFFINE_ATLAS_BUILDER_CPU_H__
#define __AFFINE_ATLAS_BUILDER_CPU_H__

#include "AffineAtlasBuilderInterface.h"

#include "AtlasWerksTypes.h"
#include "Array3DUtils.h"
#include "ImageUtils.h"
#include<stdio.h>
#include<AffineAtlas.h>
#include "CmdLineParser.h"
#include "WeightedImageSet.h"

#ifdef MPI_ENABLED
#include <mpi.h>
#endif // MPI_ENABLED


class AffineAtlasBuilderCPU
  : public AffineAtlasBuilderInterface
{
public:

  /**
   * Initialization for MPI version (multiple nodes)
   */
  AffineAtlasBuilderCPU(unsigned int nodeId, unsigned int nNodes,
                      unsigned int nTotalImgs,
                      WeightedImageSet &imageSet,
                      unsigned int nThreads,
                      std::string registrationtype,
                      unsigned int iterations, bool WriteFinalImages);
  ~AffineAtlasBuilderCPU();

  void BuildAtlas();
  void SumAcrossNodes();
  void SumAcrossNodesWithJacobian(double);
  void GetMeanImage(RealImage &mean); 
private:
  
  int m_nodeId;                  // Node ID, if using multiple nodes (MPI)
  unsigned int m_nNodes;         // Number of nodes
  unsigned int m_nTotalImgs;     // Number of images
  unsigned int m_nImages;        // Number of images
  unsigned int m_nThreads;
  bool WriteTransformedImages;

  WeightedImageSet &m_imageSet;
  std::string regtype;
  unsigned int nIterations;
  
  Vector3D<unsigned int> m_imSize;
  Vector3D<double> m_imOrigin;
  Vector3D<double> m_imSpacing;


  //RealImage *m_iavg;
  //RealImage *mMPIMeanImage;
  //RealImage** m_Images; 
  AffineAtlas::ImageType** m_Images;
  AffineAtlas::ImageType *m_iavg;
  AffineAtlas::ImageType *mPIMeanImage;
};

#endif 
