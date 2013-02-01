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

#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <algorithm>
#include <sstream>

#include "FluidWarpParameters.h"
#include "FluidWarp.h"

#include "Timer.h"
#include "Array3DUtils.h"
#include "HField3DUtils.h"
#include "DiffOper.h"
//#include "PlatformCompatibility.h"
#include <time.h>

#include <cfloat> // old g++ compiler cant find limits

#include "Array3DIO.h"
#include "HField3DIO.h"
#include <sstream>

#include "ApplicationUtils.h"
#include "TestUtils.h"

FluidWarp
::FluidWarp()
{
  this->_updateAverageAfterEverySubIteration = true;

  _outputMode = FW_OUTPUT_MODE_NORMAL;
  _filePrefix = "";
  _writeVolumes = false;
  _writeXSlices = false;
  _writeYSlices = false;
  _writeZSlices = false;
  _writeErrorSpreadsheet = false;
  _writePerIter = false;
  _writePerRMSDecline = false;
  _writePerIterSize = 0;
  _writePerRMSDeclineSize = 0;
  _writeOneAtlasPerIter = true;
  _xSliceNumber = 0;
  _ySliceNumber = 0;
  _zSliceNumber = 0;
  _writeDeformedImageFiles = false;
  _writeAtlasFiles = false;
  _writeJacobianFiles = false;
  _writeDivergenceFiles = false;
  _writeCurrentHFieldFiles = false;
  _writeCurrentHInvFieldFiles = false;
  _writeLogFile = false;
  _lastWrittenIter = 0;
  _lastWrittenRMSError = 0;

  _imageOrigin.set(0.0,0.0,0.0);
  _imageSpacing.set(1.0,1.0,1.0);

  _FFTWDoMeasure       = false;
  _FFTWNumberOfThreads = 1;

}

FluidWarp::
~FluidWarp()
{
}

//
// methods to control algorithm output
//
void
FluidWarp
::setOutputMode(OutputMode mode)
{
  _outputMode = mode;
}

void
FluidWarp
::setFilePrefix(const char* prefix)
{
  _filePrefix = std::string(prefix);
}

void
FluidWarp
::setWriteVolumes(bool shouldWrite)
{
  _writeVolumes = shouldWrite;
}

void
FluidWarp
::setWriteXSlices(bool shouldWrite)
{
  _writeXSlices = shouldWrite;
}

void
FluidWarp
::setWriteYSlices(bool shouldWrite)
{
  _writeYSlices = shouldWrite;
}

void
FluidWarp
::setWriteZSlices(bool shouldWrite)
{
  _writeZSlices = shouldWrite;
}

void
FluidWarp
::setWriteErrorSpreadsheet(bool shouldWrite)
{
  _writeErrorSpreadsheet = shouldWrite;
}

void
FluidWarp
::setWriteLogFile(bool shouldWrite)
{
  _writeLogFile = shouldWrite;
}

void 
FluidWarp
::setWritePerIter(unsigned int numIters)
{
  _writePerIter = true;
  _writePerRMSDecline = false;
  _writePerIterSize = numIters;
}

void 
FluidWarp
::setWritePerRMSDecline(double decline)
{
  _writePerIter = false;
  _writePerRMSDecline = true;
  _writePerRMSDeclineSize = decline;
}

void 
FluidWarp
::setXSlice(unsigned int sliceNumber)
{
  _xSliceNumber = sliceNumber;
}

void 
FluidWarp
::setYSlice(unsigned int sliceNumber)
{
  _ySliceNumber = sliceNumber;
}

void 
FluidWarp
::setZSlice(unsigned int sliceNumber)
{
  _zSliceNumber = sliceNumber;
}

void
FluidWarp
::setWriteDeformedImageFiles(bool shouldWrite)
{
  _writeDeformedImageFiles = shouldWrite;
}

void
FluidWarp
::setWriteAtlasFiles(bool shouldWrite)
{
  _writeAtlasFiles = shouldWrite;
}

void
FluidWarp
::setWriteJacobianFiles(bool shouldWrite)
{
  _writeJacobianFiles = shouldWrite;
}

void
FluidWarp
::setWriteDivergenceFiles(bool shouldWrite)
{
  _writeDivergenceFiles = shouldWrite;
}

void
FluidWarp
::setWriteCurrentHFieldFiles(bool shouldWrite)
{
  _writeCurrentHFieldFiles = shouldWrite;
}

void
FluidWarp
::setWriteCurrentHInvFieldFiles(bool shouldWrite)
{
  _writeCurrentHInvFieldFiles = shouldWrite;
}


void
FluidWarp
::setWriteOneAtlasPerIter(bool shouldWriteOnlyOne)
{
  _writeOneAtlasPerIter = shouldWriteOnlyOne;
}

void
FluidWarp
::setFFTWMeasure(bool shouldMeasure)
{
  _FFTWDoMeasure = shouldMeasure;
}

void
FluidWarp
::setFFTWNumberOfThreads(int numberOfThreads)
{
  this->_FFTWNumberOfThreads = numberOfThreads;
}

void
FluidWarp
::setUpdateAverageAfterEverySubIteration(bool shouldUpdate)
{
  _updateAverageAfterEverySubIteration = shouldUpdate;
}

void
FluidWarp
::setImageOrigin(const Vector3D<double>& origin) {
  _imageOrigin = origin;
}

void
FluidWarp
::setImageSpacing(const Vector3D<double>& spacing) {
  _imageSpacing = spacing;
}

//
// shrink region
//
// - compute h field that shrinks the light regions of image
// - h and hinv fields must be the same size as the image
// - h and hinv, when passed in, should either both be the identity 
//   or be specialized initial h and hinv fields
// - at the end of the routine, h and hinv will hold the final h and hinv
//   fields; the initial h field will be overwritten
//

void 
FluidWarp
::shrinkRegion(const ImageType& image,
	       const Parameters& parameters,
	       VectorField& h,
	       bool shrinkLightRegions)
{
  _shrinkRegion(image, parameters, &h, 0, shrinkLightRegions);
}

void 
FluidWarp
::shrinkRegion(const ImageType& image,
	       const Parameters& parameters,
	       VectorField& h,
	       VectorField& hinv,
	       bool shrinkLightRegions)
{
  _shrinkRegion(image, parameters, &h, &hinv, shrinkLightRegions);
}

void 
FluidWarp
::_shrinkRegion(const ImageType& moving,
		const Parameters& parameters,
		VectorField* h,
		VectorField* hinv,
		bool shrinkLightRegions)
{
  Timer totalTimer;
  totalTimer.start();

  //
  // make sure that h and hinv are the same size as the image
  //
  Vector3D<unsigned int> imageSize = moving.getSize();
  // we don't actually check spacing, but we probably should..
  Vector3D<CoordinateType> imageSpacing(1,1,1);
  if (imageSize != h->getSize())
    {
      _report("Incompatible h and image array sizes.\n", FW_OUTPUT_TYPE_ERROR);
      throw std::runtime_error("incompatible h and image array sizes");
    }
  if (hinv && imageSize != hinv->getSize())
    {
      _report("Incompatible hinv and array sizes.\n", FW_OUTPUT_TYPE_ERROR);
      throw std::runtime_error("incompatible hinv and image array sizes");
    }

  //
  // allocate memory for velocity field and deformed image
  //
  _time("allocating memory...");

  DiffOper op(imageSize, imageSpacing);
  op.SetAlpha(parameters.alpha);
  op.SetBeta(parameters.beta);
  op.SetGamma(parameters.gamma);
  op.SetUseEigenLUT(true);
  op.SetFFTWNumberOfThreads(this->_FFTWNumberOfThreads);
  op.SetFFTWMeasure(this->_FFTWDoMeasure);
  op.Initialize();

  // this multipurpose array is used to hold
  // gradient, body force, and velocity arrays (in that order)
  // this saves LOTS of memory
  //
  // VERY IMPORTANT: however, because it is used to hold the result of an fft
  // it needs to be padded just a little bit, see www.fftw.org
  // this wont affect its use as gradient and body force arrays
  // as long as logical size (image size) is used for this array
  // and access into this array is done **ONLY** via the (x, y, z) operator
  // and not via incremented pointers.  I REPEAT, don't access the vector
  // field via incremented pointers unless you *know* what you are doing.
  VectorField *vf = op.GetInternalFFTWVectorField();

  // allocate deformed image (I(hk(x)))
  ImageType def(imageSize);
  //VectorField hhinv(imageSize);
  _stopTime();

  float delta = 0;
  bool haveDelta = false;
  _openSpreadsheetOfstream();
  ImageType dummyArray;
  bool breakNextRound = false;
  for (unsigned int iter = 0; iter < parameters.numIterations; ++iter)
    {
      //
      // deform the moving image
      //
      _time("deforming image...");
      HField3DUtils::apply(moving, *h, def);
      _stopTime();

      // don't start if nothing to do
      if (iter == 0)
        {
          float minEle, maxEle;
	  minEle = maxEle = 0; // make compiler happy
          Array3DUtils::getMinMax(def, minEle, maxEle);
          std::cerr << "Min: " << minEle << ", Max: " << maxEle 
                    << std::endl;
          if (minEle > (maxEle / 2.0)) break;
        }

      //
      // remove all shades of gray--keep this binary
      //
      for (unsigned int i = 0; i < def.getNumElements(); ++i)
        {
          if (def(i) > 0) def(i) = 1;
        }

      // break after next round if only tiny gas to shrink
      if (iter % 5 == 0)
        {
          ImageType tmp(def);
          Array3DUtils::maxFilter3D(tmp);
          float minEle, maxEle;
	  minEle = maxEle = 0; // make compiler happy
          Array3DUtils::getMinMax(tmp, minEle, maxEle);
          std::cerr << "Min: " << minEle << ", Max: " << maxEle 
                    << std::endl;

          if (breakNextRound)
            {
              std::cerr << "Stopping Condition Reached: no gas to shrink" 
                        << std::endl;
              break;
            }
          if (minEle > (maxEle / 2.0)) 
            {
              breakNextRound = true;
            }
        }

      //
      // testing: write out deformed image
      //
      // std::ostringstream tmpname;
      // tmpname << "defImageReverse" << iter;
      // Array3DIO::writeMETAVolume(def, tmpname.str().c_str());

      //
      // compute the gradient of the deformed image
      //
      _time("computing gradient...");
      Array3DUtils::computeGradient(def, *vf);
      _stopTime();

      //
      // use gradient as body force
      //
      if (shrinkLightRegions)
	{
	  // flip the gradient to shrink light regions
	  vf->scale(-1);
	}

      //
      // testing: write out gradient
      //
      // tmpname.str("");
      // tmpname << "gradFlippedReverseX" << iter;
      // HField3DIO::writeMETAXComponent(vf, tmpname.str().c_str());
      // tmpname.str("");
      // tmpname << "gradFlippedReverseY" << iter;
      // HField3DIO::writeMETAYComponent(vf, tmpname.str().c_str());
      // tmpname.str("");
      // tmpname << "gradFlippedReverseZ" << iter;
      // HField3DIO::writeMETAZComponent(vf, tmpname.str().c_str());

      //
      // compute the velocity field
      //
      _time("computing velocity field...");
      op.ApplyInverseOperator();
      _stopTime();

      //
      // testing: write out velocity
      //
      // tmpname.str("");
      // tmpname << "velReverseX" << iter;
      // HField3DIO::writeMETAXComponent(vf, tmpname.str().c_str());
      // tmpname.str("");
      // tmpname << "velReverseY" << iter;
      // HField3DIO::writeMETAYComponent(vf, tmpname.str().c_str());
      // tmpname.str("");
      // tmpname << "velReverseZ" << iter;
      // HField3DIO::writeMETAZComponent(vf, tmpname.str().c_str());

      //
      // compute delta (currently only once at the beginning)
      //
      if (!haveDelta)
	{
	  _time("finding delta...");
	  delta = _computeDelta(*vf, imageSize, parameters.maxPerturbation);
	  _stopTime();

	  std::ostringstream deltaString;
	  deltaString << "Computed delta: " << delta << std::endl;
	  _report(deltaString, FW_OUTPUT_TYPE_VERBOSE);
	  haveDelta = true;

          if (delta > 100)
            {
              std::cerr << "No gas to deflate." << std::endl;
              break;
            }
	}

      //
      // update current h and hinv fields
      //
      _time("updating h fields...");
      if (hinv)
	{
	  _updateHField(*h, *hinv, *vf, delta);
	}
      else
	{
	  _updateHField(*h, *vf, delta);
	}
      _stopTime();

      //
      // output iter results
      //
      _reportIterResults(iter, 0, 1, totalTimer.getSeconds(), 
			 delta, 0, 0, 0);

      //
      // optionally write out volumes and/or slices for movies, etc.
      //
      _time("Writing per iteration data (if requested)...");
      _writePerIterationData(iter, 0, 1, 0, def, dummyArray, *h, *hinv);
      _stopTime();
    }

  _closeSpreadsheetOfstream();

  //
  // report the total time
  //
  totalTimer.stop();
  std::ostringstream totalTimeString;
  totalTimeString << "Total Time: " << totalTimer.getSeconds() 
		  << " (sec)" << std::endl;
  _report(totalTimeString, FW_OUTPUT_TYPE_STANDARD);
}

void 
FluidWarp
::shrinkRegionForward(const ImageType& image,
                      const Parameters& parameters,
                      VectorField& h,
                      bool shrinkLightRegions)
{
  _shrinkRegionForward(image, parameters, &h, 
                       shrinkLightRegions);
}

void 
FluidWarp
::_shrinkRegionForward(const ImageType& moving,
                       const Parameters& parameters,
                       VectorField* h,
                       bool shrinkLightRegions)
{
  Timer totalTimer;
  totalTimer.start();

  //
  // make sure that h is the same size as the image
  //
  Vector3D<unsigned int> imageSize = moving.getSize();
  // we don't actually check spacing, but we probably should..
  Vector3D<CoordinateType> imageSpacing(1,1,1);
  if (imageSize != h->getSize())
    {
      _report("Incompatible h and image array sizes.\n", FW_OUTPUT_TYPE_ERROR);
      throw std::runtime_error("incompatible h and image array sizes");
    }

  //
  // allocate memory for velocity field and deformed image
  //
  _time("allocating memory...");

  DiffOper op(imageSize, imageSpacing);
  op.SetAlpha(parameters.alpha);
  op.SetBeta(parameters.beta);
  op.SetGamma(parameters.gamma);
  op.SetUseEigenLUT(true);
  op.SetFFTWNumberOfThreads(this->_FFTWNumberOfThreads);
  op.SetFFTWMeasure(this->_FFTWDoMeasure);
  op.Initialize();

  // this multipurpose array is used to hold
  // gradient, body force, and velocity arrays (in that order)
  // this saves LOTS of memory
  //
  // VERY IMPORTANT: however, because it is used to hold the result of an fft
  // it needs to be padded just a little bit, see www.fftw.org
  // this wont affect its use as gradient and body force arrays
  // as long as logical size (image size) is used for this array
  // and access into this array is done **ONLY** via the (x, y, z) operator
  // and not via incremented pointers.  I REPEAT, don't access the vector
  // field via incremented pointers unless you *know* what you are doing.
  VectorField *vf = op.GetInternalFFTWVectorField();

  ImageType def(moving);
  _stopTime();

  float delta = 0;
  bool haveDelta = false;
  _openSpreadsheetOfstream();
  ImageType dummyArray;

  for (unsigned int iter = 0; iter < parameters.numIterations; ++iter)
    {
      //
      // deform the moving image
      //
//       _time("deforming image...");
//       HField3DUtils::forwardApply(moving, *h, def, 0, 0, 0);
//       float imin, imax;
//       Array3DUtils::getMinMax(moving, imin, imax);      
//       std::cerr << "moving min: " << imin << ", max: " << imax << std::endl;
//       Array3DUtils::getMinMax(def, imin, imax);
//       std::cerr << "df     min: " << imin << ", max: " << imax << std::endl;
//       _stopTime();

      //
      // remove all shades of gray---keep this binary
      //
      for (unsigned int i = 0; i < def.getNumElements(); ++i)
        {
          //if (def(i) > 0) def(i) = 1;
          if (def(i) < 1) def(i) = 0;
        }

      //
      // testing: write out deformed image
      //
      // std::ostringstream tmpname;
      // tmpname << "defImageForward" << iter;
      // Array3DIO::writeMETAVolume(def, tmpname.str().c_str());

      //
      // compute the gradient of the deformed image
      //
      _time("computing gradient...");
      Array3DUtils::computeGradient(def, *vf);
      _stopTime();

      //
      // use gradient as body force
      //
      if (!shrinkLightRegions)
	{
          // flip the gradient to shrink dark regions
	  vf->scale(-1);
	}

      //
      // testing: write body force for display
      //
      // Array3D<CoordinateType> bx(imageSize);
      // for (unsigned int z = 0; z < imageSize.z; ++z) {
      //   for (unsigned int y = 0; y < imageSize.y; ++y) {
      //     for (unsigned int x = 0; x < imageSize.x; ++x) {
      //       bx(x,y,z) = vf(x,y,z).x;
      //     }
      //   }
      // }
      // Array3DUtils::rescaleElements(bx, 0.0F, 255.0F);
      // std::ostringstream tmpnamebx;
      // tmpnamebx << "bodyForceX" << iter;
      // Array3DIO::writeMETAVolume(bx, tmpnamebx.str().c_str());      

      //
      // compute the velocity field
      //
      _time("computing velocity field...");
      op.ApplyInverseOperator();
      _stopTime();

      //
      // testing: write velocity field for display
      //
      // Array3D<CoordinateType> vx(imageSize);
      // for (unsigned int z = 0; z < imageSize.z; ++z) {
      //   for (unsigned int y = 0; y < imageSize.y; ++y) {
      //     for (unsigned int x = 0; x < imageSize.x; ++x) {
      //       vx(x,y,z) = vf(x,y,z).x;
      //     }
      //   }
      // }
      //Array3DUtils::rescaleElements(vx, 0.0F, 255.0F);
      // std::ostringstream tmpnamevx;
      // tmpnamevx << "velocityFieldX" << iter;
      // Array3DIO::writeMETAVolume(vx, tmpnamevx.str().c_str());      

      //
      // compute delta (currently only once at the beginning)
      //
      if (!haveDelta)
	{
	  _time("finding delta...");
	  delta = _computeDelta(*vf, imageSize, parameters.maxPerturbation);
	  _stopTime();

	  std::ostringstream deltaString;
	  deltaString << "Computed delta: " << delta << std::endl;
	  _report(deltaString, FW_OUTPUT_TYPE_VERBOSE);
	  haveDelta = true;
	}

      //
      // compute incremental h field
      //
      _time("computing incremental h field...");
      HField3DUtils::velocityToH(*vf, delta);
      double minL2, maxL2;
      HField3DUtils::minMaxDeformationL2Norm(*vf, minL2, maxL2);
      std::cerr << "Min/Max change L2 norm: " << minL2 << "/" << maxL2
                << std::endl;
      _stopTime();      

      //
      // incrementally deform image
      //
      ImageType oldDef(def);
      
      HField3DUtils::forwardApply(oldDef, *vf, def, 0, 0, 0);      

      //
      // update current h field
      //
      _time("updating h field...");
      //_updateHField(*h, vf, delta);
      // h = vf o h
      HField3DUtils::compose(*vf, *h, *h, HField3DUtils::BACKGROUND_STRATEGY_PARTIAL_ID);
      _stopTime();

      VectorField ident(vf->getSize());
      HField3DUtils::setToIdentity(ident);
      VectorField tmpVF(*vf); 
      tmpVF.pointwiseSubtract(ident);
      //
      // output iter results
      //
      _reportIterResults(iter, 0, 1, totalTimer.getSeconds(), 
			 delta, 0, 0, 0);

      //
      // optionally write out volumes and/or slices for movies, etc.
      //
      _time("Writing per iteration data (if requested)...");
      _writePerIterationData(iter, 0, 1, 0, def, dummyArray, *h, *h);
      _stopTime();
    }

  _closeSpreadsheetOfstream();

  //
  // report the total time
  //
  totalTimer.stop();
  std::ostringstream totalTimeString;
  totalTimeString << "Total Time: " << totalTimer.getSeconds() 
		  << " (sec)" << std::endl;
  _report(totalTimeString, FW_OUTPUT_TYPE_STANDARD);
}

//
// elastic shrink region with mask
//


void 
FluidWarp
::elasticShrinkRegionWithMask(const ImageType& image,
	       const Parameters& parameters,
         MaskType& mask,
	       VectorField& h,
	       bool shrinkLightRegions)
{
  _elasticShrinkRegionWithMask(image,  parameters,mask, &h);
}

void 
FluidWarp
::_elasticShrinkRegionWithMask(const ImageType& moving,
		const Parameters& parameters,
    MaskType& mask,
		VectorField* h,
		bool shrinkLightRegions)
{
  Timer totalTimer;
  totalTimer.start();

  //
  // make sure that h and hinv are the same size as the image
  //
  Vector3D<unsigned int> imageSize = moving.getSize();
  // we don't actually check spacing, but we probably should..
  Vector3D<CoordinateType> imageSpacing(1,1,1);
  if (imageSize != h->getSize())
    {
      _report("Incompatible h and image array sizes.\n", FW_OUTPUT_TYPE_ERROR);
      throw std::runtime_error("incompatible h and image array sizes");
    }
  //
  // allocate memory for velocity field and deformed image
  //
  _time("allocating memory...");


  // allocate deformed image (def Image - fixed Image)
  VectorField tmp(imageSize);
  // allocate deformed image (I(dk(x)))
  ImageType def(imageSize);
  // allocate laplacian of the displacement  laplacian(dk(x))
  VectorField laplacianD(imageSize);
  _stopTime();



  float delta = 0;
  bool haveDelta = false;
  _openSpreadsheetOfstream();
  ImageType dummyArray;
  for (unsigned int iter = 0; iter < parameters.numIterations; ++iter)
    {
      //
      // deform the moving image
      //
      _time("deforming image...");
      HField3DUtils::apply(moving, *h, def);
      Array3DUtils::computeLaplacian(*h,laplacianD);
      _stopTime();

      //
      // compute the gradient of the deformed image
      //
      _time("computing gradient...");
      Array3DUtils::computeGradient(def, tmp);
      _stopTime();

      //
      // use gradient as body force
      //
      if (!shrinkLightRegions)
	{
	  // flip the gradient to shrink dark regions
	  tmp.scale(-1);
	}


      //
      // compute delta (currently only once at the beginning)
      //
      if (!haveDelta)
	{
	  _time("finding delta...");
	  delta = _computeDelta(tmp, imageSize, parameters.maxPerturbation);
	  _stopTime();

	  std::ostringstream deltaString;
	  deltaString << "Computed delta: " << delta << std::endl;
	  _report(deltaString, FW_OUTPUT_TYPE_VERBOSE);
	  haveDelta = true;
	}

      //
      // update current h and hinv fields
      //
      _time("updating h fields...");

	  _updateHFieldElasticWithMask(*h,tmp, mask, laplacianD,parameters.alpha, delta);

      _stopTime();

      //
      // output iter results
      //
      _reportIterResults(iter, 0, 1, totalTimer.getSeconds(), 
			 delta, 0, 0, 0);

      //
      // optionally write out volumes and/or slices for movies, etc.
      //
      _time("Writing per iteration data (if requested)...");
      _writePerIterationData(iter, 0, 1, 0, def, dummyArray, *h, *h);
      _stopTime();
    }

  _closeSpreadsheetOfstream();



  //
  // report the total time
  //
  totalTimer.stop();
  std::ostringstream totalTimeString;
  totalTimeString << "Total Time: " << totalTimer.getSeconds() 
		  << " (sec)" << std::endl;
  _report(totalTimeString, FW_OUTPUT_TYPE_STANDARD);
}

//
// asymmetric version
//
// - compute h field such that fixed(x) corresponds with moving(h(x)) given
//   initial h and hinv fields
// - fixed and moving images must be the same size
// - h and hinv fields must be the same size as the images
// - h and hinv, when passed in, should either both be the identity 
//   or be specialized initial h and hinv fields
// - at the end of the routine, h and hinv will hold the final h and hinv
//   fields; the initial h field will be overwritten
//

void 
FluidWarp
::computeHFieldAsymmetric(const ImageType& fixed,
			  const ImageType& moving,
			  const Parameters& parameters,
			  VectorField& h)
{
  _computeHFieldAsymmetric(fixed, moving, parameters, &h, 0);
}

void 
FluidWarp
::computeHFieldAsymmetric(const ImageType& fixed,
			  const ImageType& moving,
			  const Parameters& parameters,
			  VectorField& h,
			  VectorField& hinv)
{
  _computeHFieldAsymmetric(fixed, moving, parameters, &h, &hinv);
}


void 
FluidWarp
::_computeHFieldAsymmetric(const ImageType& fixed,
			   const ImageType& moving,
			   const Parameters& parameters,
			   VectorField* h,
			   VectorField* hinv)
{
  if (parameters.numIterations == 0)
    {
      return;
    }

  Timer totalTimer;
  totalTimer.start();

  //
  // make sure images are the same size, and that h and hinv are the
  // same size as the images
  //
  Vector3D<unsigned int> imageSize = fixed.getSize();
  // we don't actually check spacing, but we probably should..
  Vector3D<CoordinateType> imageSpacing(1,1,1);
  if (imageSize != moving.getSize())
    {
      _report("Incompatible array sizes.\n", FW_OUTPUT_TYPE_ERROR);
      throw std::runtime_error("incompatible image array sizes");
    }
  if (imageSize != h->getSize())
    {
      _report("Incompatible h and image array sizes.\n", FW_OUTPUT_TYPE_ERROR);
      throw std::runtime_error("incompatible h and image array sizes");
    }
  if (hinv && imageSize != hinv->getSize())
    {
      _report("Incompatible hinv and array sizes.\n", FW_OUTPUT_TYPE_ERROR);
      throw std::runtime_error("incompatible hinv and image array sizes");
    }

  //
  // allocate memory for velocity field and deformed image
  //
  _time("allocating memory...");

  DiffOper op(imageSize, imageSpacing);
  op.SetAlpha(parameters.alpha);
  op.SetBeta(parameters.beta);
  op.SetGamma(parameters.gamma);
  op.SetUseEigenLUT(true);
  op.SetFFTWNumberOfThreads(this->_FFTWNumberOfThreads);
  op.SetFFTWMeasure(this->_FFTWDoMeasure);
  op.Initialize();

  // this multipurpose array is used to hold
  // gradient, body force, and velocity arrays (in that order)
  // this saves LOTS of memory
  //
  // VERY IMPORTANT: however, because it is used to hold the result of an fft
  // it needs to be padded just a little bit, see www.fftw.org
  // this wont affect its use as gradient and body force arrays
  // as long as logical size (image size) is used for this array
  // and access into this array is done **ONLY** via the (x, y, z) operator
  // and not via incremented pointers.  I REPEAT, don't access the vector
  // field via incremented pointers unless you *know* what you are doing.
  VectorField *vf = op.GetInternalFFTWVectorField();

  // allocate deformed image (I(hk(x)))
  ImageType def(imageSize);
  //VectorField hhinv(imageSize);
  _stopTime();

  std::stringstream ss;

  float delta = 0;
  bool haveDelta = false;
  double lastSquaredError = FLT_MAX;
  _openSpreadsheetOfstream();
  for (unsigned int iter = 0; iter < parameters.numIterations; ++iter)
    {
      //
      // deform the moving image
      //
      _time("deforming image...");
      HField3DUtils::apply(moving, *h, def, (float)0.0);
      _stopTime();

      //
      // debug write images
      //
//       std::ostringstream oss;
//       oss << "Def" << iter;
//       Array3DIO::writeMETAVolume(def, oss.str().c_str());
      //

      if( parameters.jacobianScale ) 
        {
          std::cerr << "Scaling by Jac." << std::endl;
          ImageType jacobian;
          HField3DUtils::jacobian(*h, jacobian);
          float min, max;
	  min = max = 0; // make compiler happy
          Array3DUtils::getMinMax( jacobian, min, max );
          std::cout << "min, max = " << min << ", " << max << std::endl;

          ImageType::SizeType size = def.getSize();
          for(unsigned int k = 0; k < size[2]; ++k) {
            for(unsigned int j = 0; j < size[1]; ++j) {
              for(unsigned int i = 0; i < size[0]; ++i) {
                def(i, j, k) *= jacobian(i, j, k);
              }
            }
          }
        }

      double squaredError;
      if( parameters.jacobianScale ) 
        {
          _time("generating body force...");
          // squared error gets set here
          _generateBodyForceJacobianScale(fixed, def, *vf, squaredError);
          _stopTime();
	} 
      else 
        {
          //
          // compute the gradient of the deformed image
          //
          _time("computing gradient...");
          Array3DUtils::computeGradient(def, *vf);
          _stopTime();

	  std::cout << "Sum of squares of def: " << Array3DUtils::sumOfSquaredElements(*vf) << std::endl;

          //
          // debug: write gradient for display
          //
//           Array3D<CoordinateType> gy(imageSize);
//           for (unsigned int z = 0; z < imageSize.z; ++z) {
//             for (unsigned int y = 0; y < imageSize.y; ++y) {
//               for (unsigned int x = 0; x < imageSize.x; ++x) {
//                 gy(x,y,z) = vf(x,y,z).normL2();
//               }
//             }
//           }
//           //Array3DUtils::rescaleElements(gy, 0.0F, 1.0F);
//           std::ostringstream tmpnamegy;
//           tmpnamegy << "GradY" << iter;
//           Array3DIO::writeMETAVolume(gy, tmpnamegy.str().c_str());      
          //

          //
          // generate the body force
          //
          _time("generating body force...");
          // squared error gets set here
          _generateBodyForce(fixed, def, *vf, squaredError);
          _stopTime(); 

          //
          // debug: write body force for display
          //
//            Array3D<CoordinateType> by(imageSize);
//            for (unsigned int z = 0; z < imageSize.z; ++z) {
//              for (unsigned int y = 0; y < imageSize.y; ++y) {
//                for (unsigned int x = 0; x < imageSize.x; ++x) {
//                  by(x,y,z) = vf(x,y,z).normL2();
//                }
//              }
//            }
//            //Array3DUtils::rescaleElements(by, 0.0F, 1.0F);
//            std::ostringstream tmpnameby;
//            tmpnameby << "BodyForceY" << iter;
//            Array3DIO::writeMETAVolume(by, tmpnameby.str().c_str());      
          //
	}

      //
      // compute the velocity field
      //
      _time("computing velocity field using DiffOper...");
      op.ApplyInverseOperator();
      _stopTime();

      //
      // debug: write velocity for display
      //
//       Array3D<CoordinateType> vy(imageSize);
//       for (unsigned int z = 0; z < imageSize.z; ++z) {
//         for (unsigned int y = 0; y < imageSize.y; ++y) {
//           for (unsigned int x = 0; x < imageSize.x; ++x) {
//             vy(x,y,z) = vf(x,y,z).normL2();
//           }
//         }
//       }
//       //Array3DUtils::rescaleElements(vy, 0.0F, 1.0F);
//       std::ostringstream tmpnamevy;
//       tmpnamevy << "VelocityY" << iter;
//       Array3DIO::writeMETAVolume(vy, tmpnamevy.str().c_str());      
      //

      //
      // compute delta (currently only once at the beginning)
      //
      if (!haveDelta)
	{
	  _time("finding delta...");
	  delta = _computeDelta(*vf, imageSize, parameters.maxPerturbation);
	  _stopTime();

	  std::ostringstream deltaString;
	  deltaString << "Computed delta: " << delta << std::endl;
	  _report(deltaString, FW_OUTPUT_TYPE_VERBOSE);
	  haveDelta = true;

          if (delta > 100)
            {
              std::cerr << "Images are too similar to process." << std::endl;
              break;
            }
	}

      //
      // update current h and hinv fields
      //
      _time("updating h fields...");
      if (hinv)
	{
	  _updateHField(*h, *hinv, *vf, delta);
	}
      else
	{
	  _updateHField(*h, *vf, delta);
	}
      _stopTime();

      //
      // output iter results
      //
      float rmsError = sqrt(squaredError / 
			    (imageSize.x * imageSize.y * imageSize.z));
      _reportIterResults(iter, 0, 1, totalTimer.getSeconds(), 
			 delta, squaredError, lastSquaredError, rmsError);
      lastSquaredError = squaredError;

      //
      // optionally write out volumes and/or slices for movies, etc.
      //
      _time("Writing per iteration data (if requested)...");
      _writePerIterationData(iter, 0, 1, rmsError, def, fixed, *h, *hinv);
      _stopTime();
    }

  _closeSpreadsheetOfstream();

  //
  // report the total time
  //
  totalTimer.stop();
  std::ostringstream totalTimeString;
  totalTimeString << "Total Time: " << totalTimer.getSeconds() 
		  << " (sec)" << std::endl;
  _report(totalTimeString, FW_OUTPUT_TYPE_STANDARD);
}

//
// elastic version
//

void 
FluidWarp
::computeHFieldElastic(const ImageType& fixed,
			  const ImageType& moving,
			  const Parameters& parameters,
			  VectorField& h)
{
  _computeHFieldElastic(fixed, moving, parameters, &h);
}

void 
FluidWarp
::_computeHFieldElastic(const ImageType& fixed,
                        const ImageType& moving,
                        const Parameters& parameters,
                        VectorField* h)
{
  Timer totalTimer;
  totalTimer.start();

  //
  // make sure images are the same size, and that h and hinv are the
  // same size as the images
  //
  Vector3D<unsigned int> imageSize = fixed.getSize();
  if (imageSize != moving.getSize())
  {
    _report("Incompatible array sizes.\n", FW_OUTPUT_TYPE_ERROR);
    throw std::runtime_error("incompatible image array sizes");
  }
  if (imageSize != h->getSize())
  {
    _report("Incompatible h and image array sizes.\n", FW_OUTPUT_TYPE_ERROR);
    throw std::runtime_error("incompatible h and image array sizes");
  }


  _time("allocating memory...");

  // allocate deformed image (def Image - fixed Image)
  VectorField tmp(imageSize);
  // allocate deformed image (I(dk(x)))
  ImageType def(imageSize);
  // allocate laplacian of the displacement  laplacian(dk(x))
  VectorField laplacianD(imageSize);



  _stopTime();



  float delta = 0;
  bool haveDelta = false;
  double lastSquaredError = FLT_MAX;
  _openSpreadsheetOfstream();
  for (unsigned int iter = 0; iter < parameters.numIterations; ++iter)
  {
    //
    // deform the moving image
    //
    _time("deforming image...");
    HField3DUtils::apply(moving, *h, def);
    Array3DUtils::computeLaplacian(*h,laplacianD);

    double squaredError;

    _stopTime();


    //
    // compute the gradient of the deformed image
    //
    _time("computing gradient...");
    Array3DUtils::computeGradient(def, tmp);
    _stopTime();

    //
    // generate the body force
    //
    _time("generating body force...");
    // squared error gets set here
    _generateBodyForce(fixed, def, tmp, squaredError);
    _stopTime();

    //
    // compute delta (currently only once at the beginning)
    //
    if (!haveDelta)
    {
      _time("finding delta...");
      delta = _computeDelta(tmp, imageSize, parameters.maxPerturbation);
      _stopTime();

      std::ostringstream deltaString;
      deltaString << "Computed delta: " << delta << std::endl;
      _report(deltaString, FW_OUTPUT_TYPE_VERBOSE);
      haveDelta = true;
    }

    //
    // update current h and hinv fields
    //
    _time("updating h fields...");

      _updateHFieldElastic(*h, tmp, laplacianD,parameters.alpha, delta);

    _stopTime();

    //
    // output iter results
    //
    float rmsError = sqrt(squaredError / 
      (imageSize.x * imageSize.y * imageSize.z));
    _reportIterResults(iter, 0, 1, totalTimer.getSeconds(), 
      delta, squaredError, lastSquaredError, rmsError);
    lastSquaredError = squaredError;

    //
    // optionally write out volumes and/or slices for movies, etc.
    //
    _time("Writing per iteration data (if requested)...");
    _writePerIterationData(iter, 0, 1, rmsError, def, fixed, *h, *h);
    _stopTime();
  }

  _closeSpreadsheetOfstream();

  //
  // report the total time
  //
  totalTimer.stop();
  std::ostringstream totalTimeString;
  totalTimeString << "Total Time: " << totalTimer.getSeconds() 
		  << " (sec)" << std::endl;
  _report(totalTimeString, FW_OUTPUT_TYPE_STANDARD);
}

//
// elastic version with mask
//

void 
FluidWarp
::computeHFieldElasticWithMask(const ImageType& fixed,
			  const ImageType& moving,
        MaskType& mask,
			  const Parameters& parameters,
			  VectorField& h)
{
  _computeHFieldElasticWithMask(fixed, moving, mask, parameters, &h);
}

void 
FluidWarp
::_computeHFieldElasticWithMask(const ImageType& fixed,
                        const ImageType& moving,
                        MaskType& mask,
                        const Parameters& parameters,
                        VectorField* h)
{
  Timer totalTimer;
   totalTimer.start();

  //
  // make sure images are the same size, and that h and hinv are the
  // same size as the images
  //
  Vector3D<unsigned int> imageSize = fixed.getSize();
  if (imageSize != moving.getSize())
  {
    _report("Incompatible array sizes.\n", FW_OUTPUT_TYPE_ERROR);
    throw std::runtime_error("incompatible image array sizes");
  }
  if (imageSize != h->getSize())
  {
    _report("Incompatible h and image array sizes.\n", FW_OUTPUT_TYPE_ERROR);
    throw std::runtime_error("incompatible h and image array sizes");
  }


  _time("allocating memory...");

  // allocate deformed image (def Image - fixed Image)
  VectorField tmp(imageSize);
  // allocate deformed image (I(dk(x)))
  ImageType def(imageSize);
  // allocate laplacian of the displacement  laplacian(dk(x))
  VectorField laplacianD(imageSize);



  _stopTime();



  float delta = 0;
  bool haveDelta = false;
  double lastSquaredError = FLT_MAX;
  _openSpreadsheetOfstream();
  for (unsigned int iter = 0; iter < parameters.numIterations; ++iter)
  {
    //
    // deform the moving image
    //
    _time("deforming image...");
    HField3DUtils::apply(moving, *h, def);
    Array3DUtils::computeLaplacian(*h,laplacianD);

    double squaredError;

    _stopTime();


    //
    // compute the gradient of the deformed image
    //
    _time("computing gradient...");
    Array3DUtils::computeGradient(def, tmp);
    _stopTime();

    //
    // generate the body force
    //
    _time("generating body force...");
    // squared error gets set here
    _generateBodyForce(fixed, def, tmp, squaredError);
    _stopTime();

    //
    // compute delta (currently only once at the beginning)
    //
    if (!haveDelta)
    {
      _time("finding delta...");
      delta = _computeDelta(tmp, imageSize, parameters.maxPerturbation);
      _stopTime();

      std::ostringstream deltaString;
      deltaString << "Computed delta: " << delta << std::endl;
      _report(deltaString, FW_OUTPUT_TYPE_VERBOSE);
      haveDelta = true;
    }

    //
    // update current h and hinv fields
    //
    _time("updating h fields...");

      _updateHFieldElasticWithMask(*h, tmp, mask, laplacianD,parameters.alpha, delta);

    _stopTime();

    //
    // output iter results
    //
    float rmsError = sqrt(squaredError / 
      (imageSize.x * imageSize.y * imageSize.z));
    _reportIterResults(iter, 0, 1, totalTimer.getSeconds(), 
      delta, squaredError, lastSquaredError, rmsError);
    lastSquaredError = squaredError;

    //
    // optionally write out volumes and/or slices for movies, etc.
    //
    _time("Writing per iteration data (if requested)...");
    _writePerIterationData(iter, 0, 1, rmsError, def, fixed, *h, *h);
    _stopTime();
  }

  _closeSpreadsheetOfstream();

  //
  // report the total time
  //
  totalTimer.stop();
  std::ostringstream totalTimeString;
  totalTimeString << "Total Time: " << totalTimer.getSeconds() 
		  << " (sec)" << std::endl;
  _report(totalTimeString, FW_OUTPUT_TYPE_STANDARD);
}

//
// symmetric version
//
// - compute h1 and h2 field such that 
//   i1(h1(x)) corresponds with i2(h2(x))
//   i1(x) corresponds to i2(h2(h1inv(x)))
//   i2(x) corresponds to i1(h1(h2inv(x)))
//   given initial h1 and h2 fields
// - images i1 and i2 must be the same size
// - h fields h1 and h2 must be the same size as the images
// - h1 and h2, when passed in, should be the identity or the initial h fields
// - at the end of the routine, h1 and h2 will hold the final h fields,
//   the initial h fields will be overwritten
//

void 
FluidWarp
::computeHField2Symmetric(const ImageType& i1,
			  const ImageType& i2,
			  const Parameters& parameters,
			  VectorField& h1,
			  VectorField& h2)
{
  _computeHField2Symmetric(i1, i2, parameters, &h1, &h2, 0, 0);
}

void 
FluidWarp
::computeHField2Symmetric(const ImageType& i1,
			  const ImageType& i2,
			  const Parameters& parameters,
			  VectorField& h1,
			  VectorField& h2,
			  VectorField& h1inv,
			  VectorField& h2inv)
{
  _computeHField2Symmetric(i1, i2, parameters, &h1, &h2, &h1inv, &h2inv);
}

void 
FluidWarp
::_computeHField2Symmetric(const ImageType& i1,
			   const ImageType& i2,
			   const Parameters& parameters,
			   VectorField* h1,
			   VectorField* h2,
			   VectorField* h1inv,
			   VectorField* h2inv)
{
  Timer totalTimer;
  totalTimer.start();

  //
  // make sure images are the same size, and that hs and hinvs are the
  // same size as the images
  //
  Vector3D<unsigned int> imageSize = i1.getSize();
  // we don't actually check spacing, but we probably should..
  Vector3D<CoordinateType> imageSpacing(1,1,1);
  if (imageSize != i2.getSize())
    {
      _report("Incompatible array sizes.\n", FW_OUTPUT_TYPE_ERROR);
      throw std::runtime_error("incompatible image array sizes");
    }
  if (imageSize != h1->getSize())
    {
      _report("Incompatible h and image array sizes.\n", FW_OUTPUT_TYPE_ERROR);
      throw std::runtime_error("incompatible h and image array sizes");
    }
  if (imageSize != h2->getSize())
    {
      _report("Incompatible h and image array sizes.\n", FW_OUTPUT_TYPE_ERROR);
      throw std::runtime_error("incompatible h and image array sizes");
    }
  if (h1inv && imageSize != h1inv->getSize())
    {
      _report("Incompatible hinv and array sizes.\n", FW_OUTPUT_TYPE_ERROR);
      throw std::runtime_error("incompatible hinv and image array sizes");
    }
  if (h2inv && imageSize != h2inv->getSize())
    {
      _report("Incompatible hinv and array sizes.\n", FW_OUTPUT_TYPE_ERROR);
      throw std::runtime_error("incompatible hinv and image array sizes");
    }

  //
  // allocate memory for velocity field and deformed image
  // and average image
  //
  _time("allocating memory...");

  DiffOper op1(imageSize, imageSpacing);
  op1.SetAlpha(parameters.alpha);
  op1.SetBeta(parameters.beta);
  op1.SetGamma(parameters.gamma);
  op1.SetUseEigenLUT(true);
  op1.SetFFTWNumberOfThreads(this->_FFTWNumberOfThreads);
  op1.SetFFTWMeasure(this->_FFTWDoMeasure);
  op1.Initialize();

  DiffOper op2(imageSize, imageSpacing);
  op2.SetAlpha(parameters.alpha);
  op2.SetBeta(parameters.beta);
  op2.SetGamma(parameters.gamma);
  op2.SetUseEigenLUT(true);
  op2.SetFFTWNumberOfThreads(this->_FFTWNumberOfThreads);
  op2.SetFFTWMeasure(this->_FFTWDoMeasure);
  op2.Initialize();

  // this multipurpose array is used to hold
  // gradient, body force, and velocity arrays (in that order)
  // this saves LOTS of memory
  //
  // VERY IMPORTANT: however, because it is used to hold the result of an fft
  // it needs to be padded just a little bit, see www.fftw.org
  // this wont affect its use as gradient and body force arrays
  // as long as logical size (image size) is used for this array
  // and access into this array is done **ONLY** via the (x, y, z) operator
  // and not via incremented pointers.  I REPEAT, don't access the vector
  // field via incremented pointers unless you *know* what you are doing.
  VectorField *vf1 = op1.GetInternalFFTWVectorField();
  VectorField *vf2 = op2.GetInternalFFTWVectorField();

  // allocate deformed image (Ii(hik(x))) and average image (Ihat)
  ImageType def1(imageSize);
  ImageType def2(imageSize);
  VectorField h1h1inv(imageSize);
  VectorField h2h2inv(imageSize);
  _stopTime();

  //
  // main loop
  //
  float delta = 0;
  bool haveDelta = false;
  float lastSquaredError = FLT_MAX;
  _openSpreadsheetOfstream();
  for (unsigned int iter = 0; iter < parameters.numIterations; ++iter)
    {
      //
      // deform the images
      //
      _time("deforming images...");
      HField3DUtils::apply(i1, *h1, def1);
      HField3DUtils::apply(i2, *h2, def2);
      _stopTime();

      //
      // compute the gradient of the deformed images
      //
      _time("computing gradients...");
      Array3DUtils::computeGradient(def1, *vf1);
      Array3DUtils::computeGradient(def2, *vf2);
      _stopTime();

      //
      // generate the body forces
      //
      _time("generating body force...");
      double squaredError;
      // squared error gets set here
      _generateBodyForce(def1, def2, *vf1, *vf2, squaredError);
      _stopTime();

      _time("computing velocity fields...");
      op1.ApplyInverseOperator();
      op2.ApplyInverseOperator();
      _stopTime();

      //
      // compute delta (currently only once at the beginning)
      //
      if (!haveDelta)
	{
	  _time("finding delta...");
	  float delta1 = _computeDelta(*vf1, imageSize, 
				       parameters.maxPerturbation);
	  float delta2 = _computeDelta(*vf2, imageSize, 
				       parameters.maxPerturbation);
	  delta = delta1 < delta2 ? delta1 : delta2;
	  _stopTime();

	  std::ostringstream deltaString;
	  deltaString << "Computed delta: " << delta << std::endl;
	  _report(deltaString, FW_OUTPUT_TYPE_VERBOSE);
	  haveDelta = true;
	}

      //
      // update current h and hinv fields
      //
      _time("updating h fields...");
      if (h1inv)
	{
	  _updateHField(*h1, *h1inv, *vf1, delta);
	  _updateHField(*h2, *h2inv, *vf2, delta);
	}
      else
	{
	  _updateHField(*h1, *vf1, delta);
	  _updateHField(*h2, *vf2, delta);
	}
      _stopTime();

      //
      // output iter results
      //
      float rmsError = sqrt(squaredError / 
			    (imageSize.x * imageSize.y * imageSize.z));
      _reportIterResults(iter, 0, 1, totalTimer.getSeconds(), 
			 delta, squaredError, lastSquaredError, rmsError);
      lastSquaredError = squaredError;
    }
  _closeSpreadsheetOfstream();

  //
  // report the total time
  //
  totalTimer.stop();
  std::ostringstream totalTimeString;
  totalTimeString << "Total Time: " << totalTimer.getSeconds() 
		  << " (sec)" << std::endl;
  _report(totalTimeString, FW_OUTPUT_TYPE_STANDARD);
}

void 
FluidWarp
::computeHFieldNSymmetric(unsigned int numImages,
			  const ImageType* const* images,
			  const Parameters& parameters,
			  ImageType& iHat,
			  VectorField** h)
{
  _computeHFieldNSymmetric(numImages, images, 0, parameters, iHat, h, 0);
}

void 
FluidWarp
::computeHFieldNSymmetric(unsigned int numImages,
			  const ImageType* const* images,
                          const double* imageWeights,
			  const Parameters& parameters,
			  ImageType& iHat,
			  VectorField** h)
{
  _computeHFieldNSymmetric(numImages, images, imageWeights, 
                           parameters, iHat, h, 0);
}

void 
FluidWarp
::computeHFieldNSymmetric(unsigned int numImages,
			  const ImageType* const* images,
			  const Parameters& parameters,
			  ImageType& iHat,
			  VectorField** h,
			  VectorField** hinv)
{
  _computeHFieldNSymmetric(numImages, images, 0, parameters, iHat, h, hinv);
}

void 
FluidWarp
::computeHFieldNSymmetric(unsigned int numImages,
			  const ImageType* const* images,
                          const double* imageWeights,
			  const Parameters& parameters,
			  ImageType& iHat,
			  VectorField** h,
			  VectorField** hinv)
{
  _computeHFieldNSymmetric(numImages, images, imageWeights, 
                           parameters, iHat, h, hinv);
}

void 
FluidWarp
::_computeHFieldNSymmetric(unsigned int numImages,
			   const ImageType* const* image,
                           const double* imageWeights,
			   const Parameters& parameters,
			   ImageType& iHat,
			   VectorField** h,
			   VectorField** hinv)
{
  if (numImages == 0) return;

  Timer totalTimer;
  totalTimer.start();

  //
  // make sure images are the same size, and that hs and hinvs are the
  // same size as the images
  //
  Vector3D<unsigned int> imageSize = image[0]->getSize();
  // we don't actually check spacing, but we probably should..
  Vector3D<CoordinateType> imageSpacing(1,1,1);
  unsigned int i;
  for (i = 0; i < numImages; ++i)
    {
      if (imageSize != image[i]->getSize())
	{
	  _report("Incompatible array sizes.\n", FW_OUTPUT_TYPE_ERROR);
	  throw std::runtime_error("incompatible image array sizes");
	}
      if (imageSize != h[i]->getSize())
	{
	  _report("Incompatible h and image array sizes.\n", FW_OUTPUT_TYPE_ERROR);
	  throw std::runtime_error("incompatible h and image array sizes");
	}
      if (hinv && imageSize != hinv[i]->getSize())
	{
	  _report("Incompatible hinv and array sizes.\n", FW_OUTPUT_TYPE_ERROR);
	  throw std::runtime_error("incompatible hinv and image array sizes");
	}
    }

  //
  // allocate memory for velocity field deformed image
  // and average image
  //
  _time("allocating memory...");

  DiffOper op(imageSize, imageSpacing);
  op.SetAlpha(parameters.alpha);
  op.SetBeta(parameters.beta);
  op.SetGamma(parameters.gamma);
  op.SetUseEigenLUT(true);
  op.SetFFTWNumberOfThreads(this->_FFTWNumberOfThreads);
  op.SetFFTWMeasure(this->_FFTWDoMeasure);
  op.Initialize();

  // this multipurpose array is used to hold
  // gradient, body force, and velocity arrays (in that order)
  // this saves LOTS of memory
  //
  // VERY IMPORTANT: however, because it is used to hold the result of an fft
  // it needs to be padded just a little bit, see www.fftw.org
  // this wont affect its use as gradient and body force arrays
  // as long as logical size (image size) is used for this array
  // and access into this array is done **ONLY** via the (x, y, z) operator
  // and not via incremented pointers.  I REPEAT, don't access the vector
  // field via incremented pointers unless you *know* what you are doing.
  VectorField *vf = op.GetInternalFFTWVectorField();

  // allocate deformed image (Ii(hik(x))) and average image (Ihat)
  ImageType** def = new ImageType*[numImages];
  for (i = 0; i < numImages; ++i)
    {
      def[i] = new ImageType(image[i]->getSize());
      HField3DUtils::apply(*image[i], *h[i], *def[i]);
    }
  iHat.resize(imageSize);
  _stopTime();

  //
  // compute iHat (average image)
  //
  _time("computing initial iHat...");
  if (imageWeights)
  {
    Array3DUtils::weightedArithmeticMean(numImages,
                                         (const Array3D<float>** const)def, 
                                         imageWeights,
                                         iHat);
  }
  else
  {
    Array3DUtils::trimmedMean(numImages, 
                              (const Array3D<float>** const)def, 
                              iHat);
  }
  _stopTime();

  //
  // main loop
  //
  float delta = 0;
  float avgDelta = 0;
  std::vector<float> lastSquaredError(numImages, FLT_MAX);
  _openSpreadsheetOfstream();
  for (unsigned int iter = 0; iter < parameters.numIterations; ++iter) {
    for (unsigned int imageIndex = 0; imageIndex < numImages; ++imageIndex) {
      //
      // compute the gradient of the deformed image
      //
      _time("computing gradient...");
      Array3DUtils::computeGradient(*def[imageIndex], *vf);
      _stopTime();

      //
      // generate body force
      //
      _time("generating body force...");
      double squaredError;                 // squared error gets set here
      _generateBodyForce(iHat, *def[imageIndex], *vf, squaredError);
      _stopTime();

      //
      // compute the velocity field
      //
      _time("computing velocity field...");
      op.ApplyInverseOperator();
      _stopTime();
      
      //
      // compute delta (currently only once at the beginning)
      // use delta for image 0 for the first iteration
      // use the average delta for subsiquent iterations
      //
      if (iter == 0)
	{
	  _time("finding delta...");
	  double currDelta = 
	    _computeDelta(*vf, imageSize, parameters.maxPerturbation);
	  _stopTime();
	  
	  avgDelta += currDelta;
	  
	  if (imageIndex == 0)
	    {
	      delta = currDelta;
	    }
	  
	  std::ostringstream deltaString;
	  deltaString << "Image " << imageIndex 
		      << " delta: " << currDelta << std::endl;
	  _report(deltaString, FW_OUTPUT_TYPE_VERBOSE);
	}
      else 
	{
	  delta = avgDelta / numImages;
	  std::ostringstream deltaString;
	  deltaString << "Average " << " delta: " << delta << std::endl;
	  _report(deltaString, FW_OUTPUT_TYPE_VERBOSE);
	}

      //
      // update current h and hinv fields
      //
      _time("updating h fields...");
      if (hinv)
	{
	  _updateHField(*h[imageIndex], *hinv[imageIndex], *vf, delta);
	}
      else
	{
	  _updateHField(*h[imageIndex], *vf, delta);
	}
      _stopTime();

      //
      // deform the moving image
      //
      _time("deforming image...");
      HField3DUtils::apply(*image[imageIndex], 
			   *h[imageIndex], 
			   *def[imageIndex]);
      _stopTime();

      //
      // update average image (iHat)
      //
      if (this->_updateAverageAfterEverySubIteration || 
          imageIndex == numImages)
      {
        _time("updating Ihat...");
        if (imageWeights)
        {
          Array3DUtils::
            weightedArithmeticMean(numImages,
                                   (const Array3D<float>** const)def, 
                                   imageWeights,
                                   iHat);
        }
        else
        {
          Array3DUtils::trimmedMean(numImages, 
                                    (const Array3D<float>** const)def, 
                                    iHat);
        }
        _stopTime();
      }

      //
      // compute integrated pointwise sample variance
      //
      _time("computing variance...");
      Array3D<float> var(iHat);
      Array3DUtils::sampleVariance(numImages, 
                                   (const Array3D<float>** const) def, 
                                   var);
      double totalVar = 0;
      for (unsigned int element = 0; element<var.getNumElements(); ++element) {
        totalVar += var(element);
      }
      _stopTime();

      //
      // output iter results
      //
      float rmsError = sqrt(squaredError / 
			    (imageSize.x * imageSize.y * imageSize.z));
      _reportIterResults(iter, imageIndex, numImages,
			 totalTimer.getSeconds(), 
			 delta, squaredError, lastSquaredError[imageIndex],
			 rmsError);
      lastSquaredError[imageIndex] = squaredError;
      std::cerr << "total variance: " << totalVar << std::endl;

      //
      // optionally write out volumes and/or slices for movies, etc.
      //
      _time("Writing per iteration data (if requested)...");
      if (hinv)
	{
	  _writePerIterationData(iter, imageIndex, numImages, rmsError, 
				 *def[imageIndex], iHat, 
				 *h[imageIndex], *hinv[imageIndex]);
	}
      else
	{
	  _writePerIterationData(iter, imageIndex, numImages, rmsError, 
				 *def[imageIndex], iHat, 
				 *h[imageIndex], *h[imageIndex]);
	}
      _stopTime();
    }
  }
  _closeSpreadsheetOfstream();

  //
  // clean up memory
  //
  _time("Cleaning up memory...");
  for (i = 0; i < numImages; ++i)
    {
      delete def[i];
    }
  delete [] def;
  _stopTime();

  //
  // report the total time
  //
  totalTimer.stop();
  std::ostringstream totalTimeString;
  totalTimeString << "Total Time: " << totalTimer.getSeconds() 
		  << " (sec)" << std::endl;
  _report(totalTimeString, FW_OUTPUT_TYPE_STANDARD);
}


//
// assymetric version with Jacobian
//
// currently used for asymetric and n-symmetric versions
//

void
FluidWarp
::_generateBodyForceJacobianScale(const ImageType& fixed,
                     const ImageType& def,
                     VectorField& gradToBodyForce,
                     double& squaredError)
{
    Vector3D<unsigned int> size = fixed.getSize();
    ImageType di(size);
    squaredError = 0;

    unsigned int z, y, x;
    for (z = 0; z < size.z; ++z) {
        for (y = 0; y < size.y; ++y) {
            for (x = 0; x < size.x; ++x) {
                double tmp=0.0;
                if (def(x,y,z) != -1.0) {
                    tmp = fixed(x, y, z) - def(x, y, z);
                }
                di(x,y,z) = tmp; 
                squaredError += tmp*tmp;
            }
        }
    }
    Array3DUtils::computeGradient(di, gradToBodyForce);
    for (z = 0; z < size.z; ++z) {
        for (y = 0; y < size.y; ++y) {
            for (x = 0; x < size.x; ++x) {
                gradToBodyForce(x, y, z) *= -def(x,y,z);
            }
        }
    }
}



//
// assymetric version
//
// currently used for asymetric and n-symmetric versions
//

void 
FluidWarp
::_generateBodyForce(const ImageType& fixed,
                     const ImageType& def,
                     VectorField& gradToBodyForce,
                     double& squaredError)
{

  Vector3D<unsigned int> size = fixed.getSize();
  double di;
  squaredError = 0;
  for (unsigned int z = 0; z < size.z; ++z) {
    for (unsigned int y = 0; y < size.y; ++y) {
      for (unsigned int x = 0; x < size.x; ++x) {
        di = 0;
        di = fixed(x, y, z) - def(x, y, z);
        // if (def(x,y,z) != -1.0) 
        // {
        //   di = fixed(x, y, z) - def(x, y, z);
        // }
        squaredError += di * di;
        gradToBodyForce(x, y, z) *= di;
      }
    }
  }  
}

//
// 2-image warp symmetric version
//
void 
FluidWarp
::_generateBodyForce(const ImageType& def1,
		     const ImageType& def2,
		     VectorField& grad1ToBodyForce1,
		     VectorField& grad2ToBodyForce2,
		     double& squaredError)
{
  Vector3D<unsigned int> size = def1.getSize();
  double di;
  squaredError = 0;

  for (unsigned int z = 0; z < size.z; ++z)
    {
      for (unsigned int y = 0; y < size.y; ++y)
	{
	  for (unsigned int x = 0; x < size.x; ++x)
	    {
	      di = def1(x, y, z) - def2(x, y, z); 
	      squaredError += di * di;
	      grad1ToBodyForce1(x, y, z) *= -di;
	      grad2ToBodyForce2(x, y, z) *= di;
	    }
	}
    }  
}

double 
FluidWarp
::_computeDelta(const VectorField& v,
                const Vector3D<unsigned int>& logicalSize,
                const double& maxPerturbation)
{
  // find max length
  double maxLengthSq = 0;
  double currLengthSq;
  Vector3D<unsigned int> maxPosition(0,0,0);
  for (unsigned int z = 0; z < logicalSize.z; ++z)
  {
    for (unsigned int y = 0; y < logicalSize.y; ++y)
    {
      for (unsigned int x = 0; x < logicalSize.x; ++x)
      {
        currLengthSq = v(x, y, z).lengthSquared();
        if (maxLengthSq < currLengthSq)
        {
          maxPosition.set(x, y, z);
          maxLengthSq = currLengthSq;
        }
      }
    }
  }  
  std::ostringstream oss;
  oss << "max velocity of " << sqrt(maxLengthSq) 
    << " at " << maxPosition << std::endl;
  _report(oss, FW_OUTPUT_TYPE_VERBOSE);
  return maxPerturbation / sqrt(maxLengthSq);
}

//
// update h field by composition with current
// incremental step
//
// h_incremental(x) = x + delta * velocity(x)
//
// h_new(x) = h_old(h_incremental(x))
//
// thus: 
// h_final(x) = h1(h2(h3(h4(...hk(x)))))
//
// note that we interpolate into the old h field 
//
// this function destroys the contents of velocity
//
void 
FluidWarp
::_updateHField(VectorField& h,
		VectorField& velocity,
		const float& delta)
{
  Vector3D<unsigned int> size = h.getSize();
  unsigned int z;
  for (z = 0; z < size.z; ++z) {
    for (unsigned int y = 0; y < size.y; ++y) {
      for (unsigned int x = 0; x < size.x; ++x) {
	// use velocity to hold h field composition
	// copy it back into h afterword
	HField3DUtils::trilerp(h, 
			       x + (delta * velocity(x, y, z).x),
			       y + (delta * velocity(x, y, z).y),
			       z + (delta * velocity(x, y, z).z),
			       velocity(x, y, z).x,
			       velocity(x, y, z).y,
			       velocity(x, y, z).z,
			       HField3DUtils::BACKGROUND_STRATEGY_PARTIAL_ID);
      }
    }
  }  

  // copy composed h field back into h array
  for (z = 0; z < size.z; ++z) {
    for (unsigned int y = 0; y < size.y; ++y) {
      for (unsigned int x = 0; x < size.x; ++x) {
	h(x, y, z) = velocity(x, y, z);
      }
    }
  }
}

//
// update h and hinv fields by composition with current
// incremental h field (x + velocity * delta)
//
// h_incremental(x) = x + delta * velocity(x)
// h_inverse_incremental(x) = x - delta * velocity(x)
//
// h_new(x) = h_old(h_incremental(x))
//
// h_inverse_new(x) = h_inverse_incremental(h_inverse_old(x))
//
// thus: 
// h_final(x) = h1(h2(h3(h4(...hk(x)...))))
//
// and
//
// h_inverse_final(x) = hkinv(hk-1inv(...h2(h1(x))...))
//
// this function destroys the contents of velocity
//
void 
FluidWarp
::_updateHField(VectorField& h,
		VectorField& hinv,
		VectorField& velocity,
		const float& delta)
{
  Vector3D<unsigned int> size = h.getSize();

  //
  // compute hIncremental(x) = x + velocity(x) * delta
  //
  VectorField hIncremental(size);
  unsigned int z; // stupid microsoft
  for (z = 0; z < size.z; ++z) {
    for (unsigned int y = 0; y < size.y; ++y) {
      for (unsigned int x = 0; x < size.x; ++x) {
	hIncremental(x,y,z).x = x + velocity(x,y,z).x * delta;
	hIncremental(x,y,z).y = y + velocity(x,y,z).y * delta;
	hIncremental(x,y,z).z = z + velocity(x,y,z).z * delta;
      }
    }
  }

  //
  // compute h(x) = h(hIncremental(x))
  //
  VectorField oldH(h);
  HField3DUtils::compose(oldH, hIncremental, h, HField3DUtils::BACKGROUND_STRATEGY_PARTIAL_ID);

  //
  // compute 
  // hIncrementalInv(x) = x + x - hIncremental(x)
  //
  VectorField& hIncrementalInv = oldH; // reuse memory here
  HField3DUtils::computeInverseZerothOrder(hIncremental, hIncrementalInv);

  //
  // compute hInv(x) = hIncrementalInv(hInv(x))
  //
  HField3DUtils::compose(hIncrementalInv, hinv, hinv, HField3DUtils::BACKGROUND_STRATEGY_PARTIAL_ID);
}



//
// h(x) = h(x) + delta * bodyForce(x) + alpha * laplacian(d)

//
void 
FluidWarp
::_updateHFieldElastic(VectorField& h,
		VectorField& bodyForce,
    VectorField& laplacian,
    const float& alpha,
		const float& delta)
{
  Vector3D<unsigned int> size = h.getSize();

  //
  // compute h(x) = h(x) + bodyForce(x) * delta + alpha*laplacian(x)
  //

  unsigned int x,y, z;
  for (z = 0; z < size.z; ++z) {
    for (y = 0; y < size.y; ++y) {
      for (x = 0; x < size.x; ++x) {
          h(x,y,z).x += bodyForce(x,y,z).x * delta + alpha * laplacian(x, y, z).x;
          h(x,y,z).y += bodyForce(x,y,z).y * delta + alpha * laplacian(x, y, z).y;
          h(x,y,z).z += bodyForce(x,y,z).z * delta + alpha * laplacian(x, y, z).z;
      }
    }
  }
}


//
// h(x) = h(x) + delta * bodyForce(x) + alpha * laplacian(d)
//
void 
FluidWarp
::_updateHFieldElasticWithMask(VectorField& h,
		VectorField& bodyForce,
    MaskType& mask,
    VectorField& laplacian,
    const float& alpha,
		const float& delta)
{
  Vector3D<unsigned int> size = h.getSize();

  //
  // compute h(x) = h(x) + bodyForce(x) * delta + alpha*laplacian(x)
  //

  unsigned int x,y, z;
  for (z = 0; z < size.z; ++z) {
    for (y = 0; y < size.y; ++y) {
      for (x = 0; x < size.x; ++x) {
        if (mask.get(x,y,z) < 0.001)
        {
          h(x,y,z).x += bodyForce(x,y,z).x * delta + alpha * laplacian(x, y, z).x;
          h(x,y,z).y += bodyForce(x,y,z).y * delta + alpha * laplacian(x, y, z).y;
          h(x,y,z).z += bodyForce(x,y,z).z * delta + alpha * laplacian(x, y, z).z;
        }
        else
        {
        h(x,y,z).x = x;
        h(x,y,z).y = y;
        h(x,y,z).z = z;
        }
      }
    }
  }
}


//
// _time and _stopTime
//
// convience methods to start and stop a local timer with specific
// output message
//
void
FluidWarp
::_time(const char* message)
{
  _report(message, FW_OUTPUT_TYPE_TIMING);
  _localTimer.reset();
  _localTimer.start();
}

void
FluidWarp
::_stopTime()
{
  _localTimer.stop();
  std::ostringstream oss;
  oss << " " << _localTimer.getMilliseconds() << " (msec)" << std::endl;
  _report(oss, FW_OUTPUT_TYPE_TIMING);
}

//
// _reportIterResults
//
// output information for current iteration
//
void
FluidWarp
::_reportIterResults(unsigned int iter, 
		     unsigned int imageIndex,
		     unsigned int numImages,
		     unsigned int totalSeconds, 
		     double delta, 
		     double squaredError, 
		     double lastSquaredError,
		     double rmsError)
{

  //
  // write iteration results to console
  //
  std::ostringstream oss;
  oss << "[" << iter << ", " << imageIndex << "] "
      << totalSeconds << " "
      << std::setprecision(5)
      << delta << " "
      << squaredError << " "
      << rmsError;
  _report(oss, FW_OUTPUT_TYPE_STANDARD);

  //
  // flag increases in squaredError
  //
  if (iter != 0 && squaredError > lastSquaredError)
    {
      _report("  <<---<<---<<---<<---<<", FW_OUTPUT_TYPE_STANDARD);
    }
  _report("\n", FW_OUTPUT_TYPE_STANDARD);

  //
  // write to error spreadsheet if desired
  //
  if (_writeErrorSpreadsheet)
    {
      // write header if this is the first image and first iteration
      if (imageIndex == 0 && iter == 0)
	{
	  _errorSpreadsheetOfstream << "Iteration";
	  for (unsigned int i = 0; i < numImages; ++i)
	    {
	      _errorSpreadsheetOfstream 
		<< "\t[" << i << "] delta\t"
		<< "\t[" << i << "] Squared Error\t"
		<< "[" << i << "] RMS Error";
	    }
	  _errorSpreadsheetOfstream << std::endl;
	}

      // write iteration number for the first image
      if (imageIndex == 0)
	{
	  _errorSpreadsheetOfstream << iter;
	}

      // write error for this iteration and this image
      _errorSpreadsheetOfstream << '\t' << delta 
				<< '\t' << squaredError 
				<< '\t' << rmsError;

      // start a new line if this is the last image for this iteration
      if (imageIndex == numImages - 1)
	{
	  _errorSpreadsheetOfstream << std::endl;
	}
    }
}

//
// _report
//
// output a message
//
void
FluidWarp
::_report(const char* message, OutputType messageType)
{
  if (_outputMode == FW_OUTPUT_MODE_SILENT) return;

  if (messageType == FW_OUTPUT_TYPE_ERROR)
    {
      std::cerr << "ERROR: " << message;
      return;
    }
  else if (messageType == FW_OUTPUT_TYPE_STANDARD)
    {
      std::cerr << message;
      return;
    }
  else if (messageType == FW_OUTPUT_TYPE_TIMING)
    {
      if (_outputMode == FW_OUTPUT_MODE_VERBOSE)
	{
	  std::cerr << message;
	  return;
	}
    }
  else if (messageType == FW_OUTPUT_TYPE_VERBOSE)
    {
      if (_outputMode == FW_OUTPUT_MODE_VERBOSE)
	{
	  std::cerr << message;
	  return;
	}
    }
}

void
FluidWarp
::_report(const std::ostringstream& message, OutputType messageType)
{
  _report(message.str().c_str(), messageType);
}

void
FluidWarp
::_writePerIterationData(unsigned int iter,
			 unsigned int imageIndex,
			 unsigned int numImages,
			 double rmsError,
			 const ImageType& deformedImage, 
			 const ImageType& atlas, 
			 const VectorField& h,
			 const VectorField& hinv)
{
  //Write log for each iteration
  if (_writeLogFile)
    {
      char* logtext = new char[500];
      sprintf(logtext,"[Image #%i] - Iteration #%i finished",imageIndex,iter);
      _writeLog(logtext);	   
      delete logtext;
    }

  // decide when to actually write something
  if (iter == 0 ||
      (_writePerIter && iter >= (_lastWrittenIter + _writePerIterSize)) ||
      (_writePerRMSDecline && 
       rmsError <= (_lastWrittenRMSError - _writePerRMSDeclineSize)))
    {
      _lastWrittenRMSError = rmsError;
      if (imageIndex == (numImages - 1))
		_lastWrittenIter = iter;

      if (_writeDeformedImageFiles)
	{
	  _writeVolume("deformedImage", iter, imageIndex, deformedImage);
	  _writeSlices("deformedImage", iter, imageIndex, deformedImage);
	}
      if (_writeAtlasFiles)
	{
	  if ((_writeOneAtlasPerIter && imageIndex == numImages - 1) ||
	      !_writeOneAtlasPerIter)
	    {
	      _writeVolume("atlas", iter, imageIndex, atlas);
	      _writeSlices("atlas", iter, imageIndex, atlas);
	    }
	}
      if (_writeCurrentHFieldFiles)
	{
	  _writeVolume("h", iter, imageIndex, h);
	  _writeSlices("h", iter, imageIndex, h);	  
	}

      if (_writeCurrentHInvFieldFiles)
 	{
 	  _writeVolume("hinv", iter, imageIndex, hinv);
 	  _writeSlices("hinv", iter, imageIndex, hinv);	  
 	}

      if (_writeJacobianFiles)
	{
	  if (_writeVolumes || _writeXSlices || _writeYSlices || _writeZSlices)
	    {
	      ImageType jacobian;
	      HField3DUtils::jacobian(h, jacobian);
	      _writeVolume("jacobian", iter, imageIndex, jacobian);
	      _writeSlices("jacobian", iter, imageIndex, jacobian);	  
	    }
	}
//       if (_writeDivergenceFiles)
// 	{
//	  if (_writeVolumes || _writeXSlices || _writeYSlices || _writeZSlices)
// 	    {
// 	      ImageType divergence;
// 	      HField3DUtils::divergence(h, divergence);
// 	      _writeVolume("divergence", iter, imageIndex, divergence);
// 	      _writeSlices("divergence", iter, imageIndex, divergence);	  
// 	    }
// 	}
    }
}



void
FluidWarp
::_openSpreadsheetOfstream()
{
  if (_writeErrorSpreadsheet)
    {
      std::ostringstream filename;
      filename << _filePrefix << "ErrorSpreadsheet.txt";
      _errorSpreadsheetOfstream.open(filename.str().c_str());
      if (_errorSpreadsheetOfstream.bad())
	{
	  _report("Error opening spreadsheet file.", FW_OUTPUT_TYPE_ERROR);
	  throw std::runtime_error("error opening file");
	}
    }
}

void
FluidWarp
::_closeSpreadsheetOfstream()
{
  if (_writeErrorSpreadsheet)
    {
      _errorSpreadsheetOfstream.close();
    }
}


void 
FluidWarp
::_writeLog(char* logtext)
{
 time_t ltime;
 time(&ltime);
 FILE* file;
 char* text;
 text = (char*) malloc(2000);
 char filename[200];
 sprintf(filename,"%sLogFile.txt",_filePrefix.c_str());
 file = fopen(filename,"ab");
 char* date = asctime(gmtime(&ltime));
 date[strlen(date)-1] ='\0';
 sprintf(text,"[ %s ] : %s\n",date,logtext);
 fwrite(text,1,strlen(text),file);
 delete text;
 fclose(file);
}
