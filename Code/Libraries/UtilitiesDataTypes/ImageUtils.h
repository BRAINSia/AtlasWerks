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

#ifndef ImageUtils_h
#define ImageUtils_h


#ifndef SWIG

#include <float.h>
#include <limits>
#include <fftw3.h>

#include "Array3DUtils.h"
#include "Array3DIO.h"
#include "AffineTransform3D.h"
#include "DataTypes/Image.h"

#endif // SWIG

class ImageUtils
{
public:

	/**
   * threshold the image at the given level
   * 
   */
  template <class T>
  static
  void
  threshold(const Array3D<T>& image, Array3D<T>& Dimage, const T& thValue, float maxVal = 2000 );
  
  /**
   * return the squared L2 norm appropriately scaled by the image
   * spacing
   */
  template <class T>
  static
  double 
  l2NormSqr(const Image<T> &image);
  
#ifdef SWIG
  %template(l2NormSqr) l2NormSqr< float >;
#endif // SWIG

  /**
   * return the dot product, appropriately scaled by spacing
   */
  template <class T>
  static
  double 
  l2DotProd(const Image<T> &i1, const Image<T> &i2);
  
#ifdef SWIG
  %template(l2DotProd) l2DotProd< float >;
#endif // SWIG

  /**  
   *  return the intensity value for this image at a given position in
   *  world coordinates
   *  
   *  bcd 2004
   */  
  template <class T, 
	    Array3DUtils::ScalarBackgroundStrategyT BackgroundStrategy,
	    Array3DUtils::InterpT InterpMethod>
  static
  T
  interp(const Image<T>& image,
	 const Vector3D<double>& worldCoordinates,
	 const T& background);

  template<class T>
  static
  T
  interp(const Image<T>& image,
	 const Vector3D<double>& worldCoordinates,
	 const T& background)
  {
    interp<T, 
      DEFAULT_SCALAR_BACKGROUND_STRATEGY, 
      DEFAULT_SCALAR_INTERP>
      (image, worldCoordinates, background);
  }

#ifdef SWIG
  %template(interp) interp< float, DEFAULT_SCALAR_BACKGROUND_STRATEGY, DEFAULT_SCALAR_INTERP>;
#endif // SWIG

  /**  
   *  translate this image.  note: this just changes the origin
   *  
   *  bcd 2004
   */  
  template <class T>
  static
  void
  translate(Image<T>& image,
	    const double& tx, 
	    const double& ty, 
	    const double& tz);

  /**  
   *  translate this image.  note: this just changes the origin
   *  
   *  bcd 2004
   */  
  template <class T>
  static
  void
  translate(Image<T>& image,
	    const Vector3D<double>& t);

#ifdef SWIG
  %template(translate) translate< float >;
#endif // SWIG

  /**
   *
   * Upsample via sinc interpolation.  Currently using full complex
   * FFT, may want to go to r2c/c2r in the future to save
   * memory. Also, this version only works for images with an even
   * number of pixels in each dimension.
   *
   * jsp 2009
   */
  template <class T>
  static
  void
  sincUpsample(Image<T>& image,
	       unsigned int factor);

  /**
   *
   * Upsample via sinc interpolation.  Currently using full complex
   * FFT, may want to go to r2c/c2r in the future to save
   * memory. Also, this version only works for images with an even
   * number of pixels in each dimension.
   *
   * jsp 2009
   */
  template <class T>
  static
  void
  sincUpsample(Image<T>& image,
	       Vector3D<unsigned int> &newSize);

#ifdef SWIG
  %template(sincUpsample) sincUpsample< float >;
#endif // SWIG

  /**
   * Non-integer downsampling method.  Much slower than integer
   * version, but uses resampleNew so that positions in world space
   * are correctly preserverd. input and output can be the same
   * object. 
   */
  template <class T, 
	    Array3DUtils::ScalarBackgroundStrategyT BackgroundStrategy,
	    Array3DUtils::InterpT InterpMethod>
  static
  void gaussianDownsample(const Image<T>& input,
			  Image<T>& output, 
			  const Vector3D<unsigned int>& newSize);


  template <class T>
  static
  void gaussianDownsample(const Image<T>& input,
			  Image<T>& output, 
			  const Vector3D<unsigned int>& newSize)
  {
    gaussianDownsample<T, 
      DEFAULT_SCALAR_BACKGROUND_STRATEGY, 
      DEFAULT_SCALAR_INTERP>
      (input, output, newSize);
  }

  /**
   * Modified version of resample that correctly centers image.
   * make this image have the given origin, spacing, and dimensions.
   * intensities should stay in the same place in world coordinates.
   *
   * jsp 2009
   */
  template <class T, 
	    Array3DUtils::ScalarBackgroundStrategyT BackgroundStrategy,
	    Array3DUtils::InterpT InterpMethod>
  static
  void
  resampleNew(Image<T>& image,
	      const Vector3D<double>& newOrigin,
	      const Vector3D<double>& newSpacing,
	      const Vector3D<unsigned int>& newDimensions,
	      T bgVal = static_cast<T>(0));

  template <class T>
  static
  void
  resampleNew(Image<T>& image,
	      const Vector3D<double>& newOrigin,
	      const Vector3D<double>& newSpacing,
	      const Vector3D<unsigned int>& newDimensions,
	      T bgVal = static_cast<T>(0))
  {
    resampleNew<T,
      Array3DUtils::BACKGROUND_STRATEGY_CLAMP,
      DEFAULT_SCALAR_INTERP>
      (image,
       newOrigin,
       newSpacing,
       newDimensions,
       bgVal);
  }

  /**
   * Modified version of resample that correctly centers image.
   * fill in the destination image from the source image, taking
   * spacing, origin, and dimensions into account.  where overlapping,
   * source and dest will be the same in world coordinates.
   *
   * jsp 2009
   */
  template <class T, 
	    Array3DUtils::ScalarBackgroundStrategyT BackgroundStrategy,
	    Array3DUtils::InterpT InterpMethod>
  static
  void
  resampleNew(const Image<T>& sourceImage,
	      Image<T>& destImage,
	      T bgVal = static_cast<T>(0));  

  template <class T>
  static
  void
  resampleNew(const Image<T>& sourceImage,
	      Image<T>& destImage,
	      T bgVal = static_cast<T>(0))
  {
    resampleNew<T, 
      Array3DUtils::BACKGROUND_STRATEGY_CLAMP,
      DEFAULT_SCALAR_INTERP>
      (sourceImage,
       destImage,
       bgVal);
  }

#ifdef SWIG
  %template(resampleNew) resampleNew< float, Array3DUtils::BACKGROUND_STRATEGY_CLAMP, DEFAULT_SCALAR_INTERP>;
#endif // SWIG

  /**
   * make this image have the given origin, spacing, and dimensions.
   * intensities should stay in the same place in world coordinates.
   *
   * bcd 2004
   */
  template <class T, 
	    Array3DUtils::ScalarBackgroundStrategyT BackgroundStrategy,
	    Array3DUtils::InterpT InterpMethod>
  static
  void
  resample(Image<T>& image,
           const Vector3D<double>& newOrigin,
           const Vector3D<double>& newSpacing,
           const Vector3D<unsigned int>& newDimensions);

  template <class T>
  static
  void
  resample(Image<T>& image,
           const Vector3D<double>& newOrigin,
           const Vector3D<double>& newSpacing,
           const Vector3D<unsigned int>& newDimensions)
  {
    resample<T, 
      DEFAULT_SCALAR_BACKGROUND_STRATEGY,
      DEFAULT_SCALAR_INTERP>
      (image, newOrigin, newSpacing, newDimensions);
  }

  /**
   * fill in the destination image from the source image, taking
   * spacing, origin, and dimensions into account.  where overlapping,
   * source and dest will be the same in world coordinates.
   *
   * bcd 2004
   */
  template <class T, 
	    Array3DUtils::ScalarBackgroundStrategyT BackgroundStrategy,
	    Array3DUtils::InterpT InterpMethod>
  static
  void
  resample(const Image<T>& sourceImage,
	   Image<T>& destImage);

  template <class T>
  static
  void
  resample(const Image<T>& sourceImage,
	   Image<T>& destImage)
  {
    resample<T, 
      Array3DUtils::BACKGROUND_STRATEGY_CLAMP,
      DEFAULT_SCALAR_INTERP>(sourceImage, destImage);
  }

#ifdef SWIG
  %template(resample) resample< float, Array3DUtils::BACKGROUND_STRATEGY_CLAMP, DEFAULT_SCALAR_INTERP>;
#endif // SWIG

  /**  
   *  Fill in the destination image from the source image, taking
   *  spacing, origin, and dimensions into account.  Where overlapping,
   *  source and dest will be the same in world coordinates.  Where not
   *  overlapping, destImage is left unchanged.  This requires that
   *  sourceImage have only nonnegative intensities.
   *  
   *  foskey 2005
   */  
  template <class T, 
	    Array3DUtils::ScalarBackgroundStrategyT BackgroundStrategy,
	    Array3DUtils::InterpT InterpMethod>
  static
  void
  resampleWithTransparency(const Image<T>& sourceImage,
                           Image<T>& destImage);
  
  template <class T>
  static
  void
  resampleWithTransparency(const Image<T>& sourceImage,
                           Image<T>& destImage)
  {
    resampleWithTransparency<T, 
      DEFAULT_SCALAR_BACKGROUND_STRATEGY, 
      DEFAULT_SCALAR_INTERP>
      (sourceImage, destImage);
  }
  

#ifdef SWIG
  %template(resampleWithTransparency) resampleWithTransparency< float, DEFAULT_SCALAR_BACKGROUND_STRATEGY, DEFAULT_SCALAR_INTERP>;
#endif // SWIG

  /**  
   *  Take a transform in world coordinates and return the corresponding
   *  transform expressed in units of voxels.  
   *  
   *  For the input transform, the Spacing parameters give the
   *  distances in world units between successive voxel centers, and
   *  the Origin parameters give the location of the (0,0,0) voxel in
   *  world coordinates.
   *  
   *  For the returned transform, the coordinate systems of the two
   *  images have units of voxels, and the origins are the centers of
   *  the (0,0,0) voxels.
   */  
  static
  AffineTransform3D<double>
  transformInIndexCoordinates(
			      const Vector3D<double>& fixedImageOrigin,
			      const Vector3D<double>& fixedImageSpacing,
			      const Vector3D<double>& movingImageOrigin,
			      const Vector3D<double>& movingImageSpacing,
			      const AffineTransform3D<double>& transformInWorldCoordinates );

  template <class T, 
	    Array3DUtils::ScalarBackgroundStrategyT BackgroundStrategy,
	    Array3DUtils::InterpT InterpMethod>
  static
  void
  applyAffine(Image<T>& image,
              const Vector3D<double>& newOrigin,
              const Vector3D<double>& newSpacing,
              const Vector3D<unsigned int>& newDimensions,
              const AffineTransform3D<double>& transformInWorldCoordinates, 
              const float& backgroundValue = 0);

  template <class T>
  static
  void
  applyAffine(Image<T>& image,
              const Vector3D<double>& newOrigin,
              const Vector3D<double>& newSpacing,
              const Vector3D<unsigned int>& newDimensions,
              const AffineTransform3D<double>& transformInWorldCoordinates, 
              const float& backgroundValue = 0)
  {
    applyAffine<T, 
      DEFAULT_SCALAR_BACKGROUND_STRATEGY, 
      DEFAULT_SCALAR_INTERP>
      (image, newOrigin, newSpacing, newDimensions, transformInWorldCoordinates, backgroundValue);
  }

  /**  Transform sourceImage into the coordinate system of destImage.
   *  After the call, destImage has data from sourceImage, transformed
   *  by the inverse of 'transform'.  To find the intensity of a point
   *  p in destImage, one looks to the point transform(p) in sourceImage.
   *  If transform(p) is outside of sourceImage, backgroundValue is
   *  used.  The origins of the respective coordinate systems are those
   *  specified by getOrigin() for the images.
   */
  template <class T, 
	    Array3DUtils::ScalarBackgroundStrategyT BackgroundStrategy,
	    Array3DUtils::InterpT InterpMethod>
  static
  void
  applyAffine(const Image<T>& sourceImage,
              Image<T>& destImage,
              const AffineTransform3D<double>& transformInWorldCoordinates,
              const float& backgroundValue = 0);

  template <class T>
  static
  void
  applyAffine(const Image<T>& sourceImage,
              Image<T>& destImage,
              const AffineTransform3D<double>& transformInWorldCoordinates,
              const float& backgroundValue = 0)
  {
    applyAffine<T, 
      DEFAULT_SCALAR_BACKGROUND_STRATEGY, 
      DEFAULT_SCALAR_INTERP>
      (sourceImage, destImage, transformInWorldCoordinates, backgroundValue);
  }


#ifdef SWIG
  %template(applyAffine) applyAffine< float, DEFAULT_SCALAR_BACKGROUND_STRATEGY, DEFAULT_SCALAR_INTERP>;
#endif // SWIG

  /**  
   *  make all voxel spacing the same.  the smallest current voxel
   *  spacing is chosen as the spacing to use.
   *  
   *  bcd 2004
   */  
  template <class T, 
	    Array3DUtils::ScalarBackgroundStrategyT BackgroundStrategy,
	    Array3DUtils::InterpT InterpMethod>
  static
  void
  makeVoxelsIsotropic(Image<T>& image);

  template <class T>
  static
  void
  makeVoxelsIsotropic(Image<T>& image){
    makeVoxelsIsotropic<T, 
      DEFAULT_SCALAR_BACKGROUND_STRATEGY, 
      DEFAULT_SCALAR_INTERP>
      (image);
  }

#ifdef SWIG
  %template(makeVoxelsIsotropic) makeVoxelsIsotropic< float, DEFAULT_SCALAR_BACKGROUND_STRATEGY, DEFAULT_SCALAR_INTERP>;
#endif // SWIG

  /**  
   *  bcd 2004
   */  
  template <class T, 
	    Array3DUtils::ScalarBackgroundStrategyT BackgroundStrategy,
	    Array3DUtils::InterpT InterpMethod>
  static
  void
  resliceZMakeIsotropic(Image<T>& image);

  template <class T>
  static
  void
  resliceZMakeIsotropic(Image<T>& image){
    resliceZMakeIsotropic<T, 
      DEFAULT_SCALAR_BACKGROUND_STRATEGY, 
      DEFAULT_SCALAR_INTERP>
      (image);
  }

#ifdef SWIG
  %template(resliceZMakeIsotropic) resliceZMakeIsotropic< float, DEFAULT_SCALAR_BACKGROUND_STRATEGY, DEFAULT_SCALAR_INTERP>;
#endif // SWIG

  /**
   * gaussian downsample an image.  spacing is updated appropriatly.
   * \see Array3DUtils::gaussianDownsample(const Array3D<T>& input,
   * Array3D<T>& output, const Vector3D<int>& factors, const
   * Vector3D<double>& sigma, const Vector3D<int>& kernelSize).
   * imageIn and imageOut can be the same object.
   *
   * bcd 2004
   */
  template <class T>
  static
  void
  gaussianDownsample(const Image<T>& imageIn,
                     Image<T>& imageOut,
                     const Vector3D<double>& factors,
                     const Vector3D<double>& sigma,
                     const Vector3D<double>& kernelSize);

#ifdef SWIG
  %template(gaussianDownsample) gaussianDownsample< float >;
#endif // SWIG

  /**
   * \param image1 first input image
   * \param image2 second input image
   * \return the sum of voxelwise squared intensity difference between
   * two images.  a difference is calculated for each voxel of image1.
   *
   * bcd 2004
   */
  template <class T, 
	    Array3DUtils::ScalarBackgroundStrategyT BackgroundStrategy,
	    Array3DUtils::InterpT InterpMethod>
  static
  double
  squaredError(const Image<T>& image1,
               const Image<T>& image2);

  template <class T>
  static
  double
  squaredError(const Image<T>& image1,
               const Image<T>& image2)
  {
    return squaredError<T, 
      DEFAULT_SCALAR_BACKGROUND_STRATEGY, 
      DEFAULT_SCALAR_INTERP>
      (image1, image2);
  }

#ifdef SWIG
  %template(squaredError) squaredError< float, DEFAULT_SCALAR_BACKGROUND_STRATEGY, DEFAULT_SCALAR_INTERP>;
#endif // SWIG

  // Centroid of the image, in world coordinates.
  template <class T>
  static
  Vector3D<double> 
  computeCentroid(const Image<T>& image,
		  const ROI<int, unsigned int> voxelROI);

  template <class T>
  static
  Vector3D<double> 
  computeCentroid(const Image<T>& image);

#ifdef SWIG
  %template(computeCentroid) computeCentroid< float >;
#endif // SWIG

  /**  
   *  write the META header file and data file for this image .mhd and
   *  .raw are automatically appended to the filenamePrefix
   *  
   *  bcd 2004
   */  
  template <class T>
  static
  void 
  writeMETA(const Image<T>& image,
	    const char* filenamePrefix);
  
#ifdef SWIG
  %template(writeMETA) writeMETA< float >;
#endif // SWIG

  /**  
   *  extract an roi (specified in voxel coordinates) from an image
   *  
   *  bcd 2004
   */  
  template <class T>
  static
  void 
  extractROIVoxelCoordinates(const Image<T>& image,
			     Image<T>& roiImage,
			     const ROI<int, unsigned int>& voxelROI);
  
#ifdef SWIG
  %template(extractROIVoxelCoordinates) extractROIVoxelCoordinates< float >;
#endif // SWIG

  /**  
   *  extract an roi (specified in world coordinates) from an image 
   *  
   *  bcd 2004
   */  
  template <class T>
  static
  void 
  extractROIWorldCoordinates(const Image<T>& image,
			     Image<T>& roiImage,
			     const ROI<double, double>& worldROI);

#ifdef SWIG
  %template(extractROIWorldCoordinates) extractROIWorldCoordinates< float >;
#endif // SWIG

  /**  
   *  This is a specialized function to handle CT images. These images
   *  are often stored with a minimum pixel value of -1024 (consistent
   *  with the definition of Hounsfield units).  Other times they are
   *  shifted so that the minimum intensity is 0, which is what is
   *  expected by ImMap and BeamLock.  This function performs that
   *  shift.  (Note that PLUNC uses the other convention.)
   */  
  template <class T>
  static 
  void
  makeImageUnsigned(Image<T>& image);

#ifdef SWIG
  %template(makeImageUnsigned) makeImageUnsigned< float >;
#endif // SWIG
  
};

#ifndef SWIG
#include "ImageUtils.txx"
#endif // SWIG

#endif
