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

#ifndef MultiscaleFluid_h
#define MultiscaleFluid_h

#include "DataTypes/Image.h"
#include "Array3D.h"
#include "FluidWarpParameters.h"
#include "Vector3D.h"
#include "ROI.h"

class MultiscaleFluid
{

public:
  typedef float                                VoxelType;
  typedef Image<VoxelType>                     ImageType;
  typedef float                                DisplacementType;
  typedef Array3D<Vector3D<DisplacementType> > HFieldType;
  typedef ROI<int, unsigned int>               ROIType;

  enum AlgorithmType { DeflateReverseFluid, DeflateForwardFluid };

  static
  void
  registerImages(HFieldType& h,
                 const ImageType& fixed,
                 const ImageType& moving,
                 const ROIType& roi,
                 const VoxelType& fixedIWMin,
                 const VoxelType& fixedIWMax,
                 const VoxelType& movingIWMin,
                 const VoxelType& movingIWMax,
                 unsigned int numScaleLevels,
                 const FluidWarpParameters fluidParameters[],
                 bool resliceToIsotropicVoxels);
  
  static
  void
  deflateImage(HFieldType& s,
               const ImageType& image,
               const ROIType& roi,
               const VoxelType& iwMin,
               const VoxelType& iwMax,
               unsigned int numScaleLevels,
               const FluidWarpParameters fluidParameters[],
               unsigned int numDilations,
               unsigned int numErosions,
               bool resliceToIsotropicVoxels,
               bool shrinkLightRegions,
               MultiscaleFluid::AlgorithmType deflationAlgorithm);

  static
  void
  deflateImageTwoWay(HFieldType& s,
                     HFieldType& sinv,
                     const ImageType& image,
                     const ROIType& roi,
                     const VoxelType& iwMin,
                     const VoxelType& iwMax,
                     unsigned int numScaleLevels,
                     const FluidWarpParameters fluidParameters[],
                     unsigned int numDilations,
                     unsigned int numErosions,
                     bool resliceToIsotropicVoxels,
                     bool shrinkLightRegions);

//   static
//   void
//   deflateImageForward(HFieldType& s,
//                       const ImageType& image,
//                       const ROIType& roi,
//                       const VoxelType& iwMin,
//                       const VoxelType& iwMax,
//                       unsigned int numScaleLevels,
//                       const FluidWarpParameters fluidParameters[],
//                       bool resliceToIsotropicVoxels,
//                       bool shrinkLightRegions);

//   static
//   void
//   generateMeanImage(unsigned int numImages,
//                     const ImageType** images,
//                     const MultiscaleFluidWarpParameters& params,
//                     ImageType& iHat,
//                     VectorField** h,
//                     VectorField** hinv);

  static
  void
  writeDebugImages(bool val);

  static
  void
  outputFluidVerbose(bool val);

  static 
  void
  doFFTWMeasure(bool shouldMeasure);

private:
  static bool _writeDebugImages;
  static bool _outputFluidVerbose;
  static bool _doFFTWMeasure;
};

#endif
