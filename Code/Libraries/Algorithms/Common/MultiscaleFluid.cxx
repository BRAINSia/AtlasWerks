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


#ifdef WIN32
#pragma warning (disable: 4786) // truncated debug info
#endif

#include "MultiscaleFluid.h"
#include "ImageUtils.h"
#include "HField3DUtils.h"
#include "Array3DUtils.h"
#include "Vector3D.h"
#include "FluidWarp.h"
#include <sstream>

bool MultiscaleFluid::_writeDebugImages = false;
bool MultiscaleFluid::_outputFluidVerbose = false;
bool MultiscaleFluid::_doFFTWMeasure = false;

void
MultiscaleFluid
::writeDebugImages(bool val)
{
  _writeDebugImages = val;
}

void
MultiscaleFluid
::outputFluidVerbose(bool val)
{
  _outputFluidVerbose = val;
}

void
MultiscaleFluid
::doFFTWMeasure(bool shouldMeasure)
{
  _doFFTWMeasure = shouldMeasure;
}


// void
// generateMeanImage(unsigned int numImages,
//                   const ImageType** images,
//                   const MultiscaleFluidWarpParameters& params,
//                   ImageType& iHat,
//                   VectorField** h,
//                   VectorField** hinv)
// {

// }

void
MultiscaleFluid
::registerImages(HFieldType& h,
                 const ImageType& fixed,
                 const ImageType& moving,
                 const ROIType& roi,
                 const VoxelType& fixedIWMin,
                 const VoxelType& fixedIWMax,
                 const VoxelType& movingIWMin,
                 const VoxelType& movingIWMax,
                 unsigned int numScaleLevels,
                 const FluidWarpParameters fluidParameters[],
                 bool resliceToIsotropicVoxels)
{
  //
  // resample moving image into space of fixed image
  //
  ImageType resampledMoving(fixed.getSize(), 
                            fixed.getOrigin(), 
                            fixed.getSpacing());
  ImageUtils::resample(moving, resampledMoving);

  //
  // extract ROIs
  //
  ImageType fixedROI(roi.getSize());
  ImageUtils::extractROIVoxelCoordinates(fixed, fixedROI, roi);
  ImageType movingROI(roi.getSize());
  ImageUtils::extractROIVoxelCoordinates(resampledMoving, movingROI, roi);

  // clear memory for image no longer used
  resampledMoving.resize(0,0,0);

  if (_writeDebugImages)
    { 
      ImageUtils::writeMETA(fixedROI, "FixedROI");
      ImageUtils::writeMETA(movingROI, "MovingROI");
    }
    
  //
  // reslice for isotropic voxels if desired
  //
  Vector3D<unsigned int> roiSizeOriginal(fixedROI.getSize());
  if (resliceToIsotropicVoxels)
    {
      ImageUtils::resliceZMakeIsotropic(fixedROI);
      ImageUtils::resliceZMakeIsotropic(movingROI);
    }

  if (_writeDebugImages)
    {
      ImageUtils::writeMETA(fixedROI, "FixedReslicedROI");
      ImageUtils::writeMETA(movingROI, "MovingReslicedROI");
    }

  //
  // apply intensity windowing to ROIs
  //
  Array3DUtils::rescaleElements(fixedROI, fixedIWMin, fixedIWMax, 
                                (VoxelType) 0.0, (VoxelType) 1.0);
  Array3DUtils::rescaleElements(movingROI, movingIWMin, movingIWMax, 
                                (VoxelType) 0.0, (VoxelType) 1.0);

  if (_writeDebugImages)
    {
      ImageUtils::writeMETA(fixedROI, "FixedRescaledROI");
      ImageUtils::writeMETA(movingROI, "MovingRescaledROI");
    }

  //
  // create downsampled images
  //
  ImageType** downsampledFixedImages = new ImageType*[numScaleLevels];
  ImageType** downsampledMovingImages = new ImageType*[numScaleLevels];
  downsampledFixedImages[0] = &fixedROI;
  downsampledMovingImages[0] = &movingROI;
  for (unsigned int level = 1; level < numScaleLevels; ++level)
    {
      int s = 2;
      downsampledFixedImages[level] = new ImageType;
      downsampledFixedImages[level]->setOrigin(fixedROI.getOrigin());
      ImageUtils::gaussianDownsample(*downsampledFixedImages[level - 1],
                                     *downsampledFixedImages[level],
                                     Vector3D<int>(s, s, s),
                                     Vector3D<double>(s, s, s),
                                     Vector3D<int>(2*s, 2*s, 2*s));
      downsampledMovingImages[level] = new ImageType;
      downsampledMovingImages[level]->setOrigin(movingROI.getOrigin());
      ImageUtils::gaussianDownsample(*downsampledMovingImages[level - 1],
                                     *downsampledMovingImages[level],
                                     Vector3D<double>(s, s, s),
                                     Vector3D<double>(s, s, s),
                                     Vector3D<double>(2*s, 2*s, 2*s));
    }

  //
  // initialize h field
  //
  h.resize(downsampledFixedImages[numScaleLevels-1]->getSize());
  HField3DUtils::setToIdentity(h);

  //
  // fluid registration from coarse to fine
  //
  for (int level = (int) numScaleLevels - 1; level >= 0; --level)
    {
      //
      // upsample h field to this scale level
      //
      if (level != (int) numScaleLevels - 1)
        {
          HFieldType tmpH(h);
          HField3DUtils::resample(tmpH, h,
                                  downsampledFixedImages[level]->getSize());
        }

      if (_writeDebugImages)
        {
          std::ostringstream oss;
          oss << "FixedROILevel" << level;
          ImageUtils::writeMETA(*downsampledFixedImages[level], 
                                oss.str().c_str());
          oss.str("");
          oss << "MovingROILevel" << level;
          ImageUtils::writeMETA(*downsampledMovingImages[level], 
                                oss.str().c_str());          
        }

      //
      // run fluid registration at this scale level
      //
      FluidWarp fluidWarper;
      if (_outputFluidVerbose)
        {
          fluidWarper.setOutputMode(FluidWarp::FW_OUTPUT_MODE_VERBOSE);
        }
      if (_doFFTWMeasure)
        {
          fluidWarper.setFFTWMeasure(true);
        }
      fluidWarper.computeHFieldAsymmetric(*downsampledFixedImages[level],
                                          *downsampledMovingImages[level],
                                          fluidParameters[level],
                                          h);
    }

  //
  // resample h to original resolution
  //
  if (resliceToIsotropicVoxels)
    {
      HFieldType tmpH(h);
      HField3DUtils::resample(tmpH, h, roiSizeOriginal);        
    }

  //
  // delete image arrays
  //
  for (unsigned int i = 1; i < numScaleLevels; ++i)
    {
      delete downsampledFixedImages[i];
      delete downsampledMovingImages[i];
    }
  delete [] downsampledFixedImages;
  delete [] downsampledMovingImages;
}        

//
// after run, s will be def field in roi
//
void
MultiscaleFluid
::deflateImage(HFieldType& s,
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
               MultiscaleFluid::AlgorithmType deflationAlgorithm)
{
  if (deflationAlgorithm != MultiscaleFluid::DeflateReverseFluid &&
      deflationAlgorithm != MultiscaleFluid::DeflateForwardFluid)
    {
      throw std::runtime_error("invalid deflation type");
    }

  //
  // extract ROI
  //
  ImageType imageROI(roi.getSize());
  ImageUtils::extractROIVoxelCoordinates(image, imageROI, roi);

  if (_writeDebugImages)
    {
      ImageUtils::writeMETA(imageROI, "DeflateROI");
    }

  //
  // reslice for isotropic voxels if desired
  //
  Vector3D<unsigned int> roiSizeOriginal(imageROI.getSize());
  if (resliceToIsotropicVoxels)
    {
      ImageUtils::resliceZMakeIsotropic(imageROI);
    }

  if (_writeDebugImages)
    {
      ImageUtils::writeMETA(imageROI, "DeflateReslicedROI");
    }

  //
  // apply intensity windowing to ROI
  //
  Array3DUtils::rescaleElements(imageROI, iwMin, iwMax, 
                                (VoxelType) 0.0, (VoxelType) 1.0);

  if (_writeDebugImages)
    {
      ImageUtils::writeMETA(imageROI, "DeflateRescaledROI");
    }

  //
  // use morphological operators to clean images
  //
  for (unsigned int i = 0; i < numDilations; ++i)
    {
      Array3DUtils::maxFilter3D(imageROI);
    }
  for (unsigned int i = 0; i < numErosions; ++i)
    {
      Array3DUtils::minFilter3D(imageROI);
    }

  if (_writeDebugImages)
    {
      ImageUtils::writeMETA(imageROI, "DeflateClosedROI");
    }

  //
  // create downsampled images
  //
  ImageType** downsampledImages = new ImageType*[numScaleLevels];
  downsampledImages[0] = &imageROI;
  for (unsigned int level = 1; level < numScaleLevels; ++level)
    {
      double s = 2.0;
      downsampledImages[level] = new ImageType;
      downsampledImages[level]->setOrigin(imageROI.getOrigin());
      ImageUtils::gaussianDownsample(*downsampledImages[level - 1],
                                     *downsampledImages[level],
                                     Vector3D<double>(s, s, s),
                                     Vector3D<double>(s, s, s),
                                     Vector3D<double>(2*s, 2*s, 2*s));
    }

  //
  // initialize h field to identity at coarsest scale level
  //
  s.resize(downsampledImages[numScaleLevels-1]->getSize());
  HField3DUtils::setToIdentity(s);

  //
  // fluid deflation from coarse to fine
  //
  for (int level = (int) numScaleLevels - 1; level >= 0; --level)
    {
      if (_writeDebugImages)
        {
          std::ostringstream oss;
          oss << "DeflateROILevel" << level;
          ImageUtils::writeMETA(*downsampledImages[level], 
                                oss.str().c_str());
        }

      //
      // upsample h field to this scale level
      //
      if (level != (int) numScaleLevels - 1)
        {
          HFieldType tmpH(s);
          HField3DUtils::resample(tmpH, s, 
                                  downsampledImages[level]->getSize());
        }

      //
      // run fluid registration at this scale level
      //
      FluidWarp fluidWarper;
      fluidWarper.setFilePrefix("~/play/forward_test/");
      fluidWarper.setWriteZSlices(true);
      fluidWarper.setWritePerIter(100);
      fluidWarper.setZSlice(9);
      fluidWarper.setWriteDeformedImageFiles(true);

      if (_outputFluidVerbose)
        {
          fluidWarper.setOutputMode(FluidWarp::FW_OUTPUT_MODE_VERBOSE);
        }
      if (_doFFTWMeasure)
        {
          fluidWarper.setFFTWMeasure(true);
        }
      switch (deflationAlgorithm)
        {
        case (MultiscaleFluid::DeflateReverseFluid):
          fluidWarper.shrinkRegion(*downsampledImages[level],
                                   fluidParameters[level],
                                   s, shrinkLightRegions);
          break;
        case (MultiscaleFluid::DeflateForwardFluid):
          fluidWarper.shrinkRegionForward(*downsampledImages[level],
                                          fluidParameters[level],
                                          s, shrinkLightRegions);          
          break;
        default:
          throw std::runtime_error("invalid deflation type");
        }
    }

  //
  // resample s to original roi resolution
  //
  if (resliceToIsotropicVoxels)
    {
      HFieldType tmpH(s);
      HField3DUtils::resample(tmpH, s, roiSizeOriginal);        
    }

  //
  // delete image arrays
  //
  for (unsigned int i = 1; i < numScaleLevels; ++i)
    {
      delete downsampledImages[i];
    }
  delete [] downsampledImages;
}        

void
MultiscaleFluid
::deflateImageTwoWay(HFieldType& h,
                     HFieldType& hinv,
                     const ImageType& image,
                     const ROIType& roi,
                     const VoxelType& iwMin,
                     const VoxelType& iwMax,
                     unsigned int numScaleLevels,
                     const FluidWarpParameters fluidParameters[],
                     unsigned int numDilations,
                     unsigned int numErosions,
                     bool resliceToIsotropicVoxels,
                     bool shrinkLightRegions)
{

  //
  // extract ROI
  //
  ImageType imageROI(roi.getSize());
  ImageUtils::extractROIVoxelCoordinates(image, imageROI, roi);

  if (_writeDebugImages)
  {
    ImageUtils::writeMETA(imageROI, "DeflateROI");
  }

  //
  // reslice for isotropic voxels if desired
  //
  Vector3D<unsigned int> roiSizeOriginal(imageROI.getSize());
  if (resliceToIsotropicVoxels)
  {
    ImageUtils::resliceZMakeIsotropic(imageROI);
  }

  if (_writeDebugImages)
  {
    ImageUtils::writeMETA(imageROI, "DeflateReslicedROI");
  }

  //
  // apply intensity windowing to ROI
  //
  Array3DUtils::rescaleElements(imageROI, iwMin, iwMax, 
                                (VoxelType) 0.0, (VoxelType) 1.0);

  if (_writeDebugImages)
  {
    ImageUtils::writeMETA(imageROI, "DeflateRescaledROI");
  }

  //
  // use morphological operators to clean images
  //
  for (unsigned int i = 0; i < numDilations; ++i)
  {
    Array3DUtils::maxFilter3D(imageROI);
  }
  for (unsigned int i = 0; i < numErosions; ++i)
  {
    Array3DUtils::minFilter3D(imageROI);
  }

  if (_writeDebugImages)
  {
    ImageUtils::writeMETA(imageROI, "DeflateClosedROI");
  }

  //
  // create downsampled images
  //
  ImageType** downsampledImages = new ImageType*[numScaleLevels];
  downsampledImages[0] = &imageROI;
  for (unsigned int level = 1; level < numScaleLevels; ++level)
  {
    double s = 2.0;
    downsampledImages[level] = new ImageType;
    downsampledImages[level]->setOrigin(imageROI.getOrigin());
    ImageUtils::gaussianDownsample(*downsampledImages[level - 1],
                                   *downsampledImages[level],
                                   Vector3D<double>(s, s, s),
                                   Vector3D<double>(s, s, s),
                                   Vector3D<double>(2*s, 2*s, 2*s));
  }

  //
  // initialize h and hinv to identity at coarsest scale level
  //
  h.resize(downsampledImages[numScaleLevels-1]->getSize());
  HField3DUtils::setToIdentity(h);
  hinv.resize(downsampledImages[numScaleLevels-1]->getSize());
  HField3DUtils::setToIdentity(hinv);

  //
  // fluid deflation from coarse to fine
  //
  for (int level = (int) numScaleLevels - 1; level >= 0; --level)
  {
    if (_writeDebugImages)
    {
      std::ostringstream oss;
      oss << "DeflateROILevel" << level;
      ImageUtils::writeMETA(*downsampledImages[level], 
                            oss.str().c_str());
    }

    //
    // upsample h field to this scale level
    //
    if (level != (int) numScaleLevels - 1)
    {
      HFieldType tmpH(h);
      HField3DUtils::resample(tmpH, h, 
                              downsampledImages[level]->getSize());
      HFieldType tmpHinv(hinv);
      HField3DUtils::resample(tmpHinv, hinv, 
                              downsampledImages[level]->getSize());
    }

    //
    // run fluid registration at this scale level
    //
    FluidWarp fluidWarper;

    if (_outputFluidVerbose)
    {
      fluidWarper.setOutputMode(FluidWarp::FW_OUTPUT_MODE_VERBOSE);
    }
    if (_doFFTWMeasure)
    {
      fluidWarper.setFFTWMeasure(true);
    }

    fluidWarper.shrinkRegion(*downsampledImages[level],
                             fluidParameters[level],
                             h, hinv, shrinkLightRegions);
  }

  //
  // resample h and hinv to original roi resolution
  //
  if (resliceToIsotropicVoxels)
  {
    HFieldType tmpH(h);
    HField3DUtils::resample(tmpH, h, roiSizeOriginal);        
    HFieldType tmpHinv(hinv);
    HField3DUtils::resample(tmpHinv, hinv, roiSizeOriginal);        
  }

  //
  // delete image arrays
  //
  for (unsigned int i = 1; i < numScaleLevels; ++i)
  {
    delete downsampledImages[i];
  }
  delete [] downsampledImages;
}        

