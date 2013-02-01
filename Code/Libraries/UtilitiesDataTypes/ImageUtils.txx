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


//#include "ImageUtils.h"
#include <iostream>

template <class T>
inline
void
ImageUtils::threshold(const Array3D<T>& image, Array3D<T>& Dimage , const T& thValue, float maxVal)
{
  using namespace std;
  
  Vector3D<unsigned int> arraySize = image.getSize();

  Dimage.resize(arraySize);

  T vVal;

 

  for (unsigned int z = 0; z < arraySize.z; ++z) {
    for (unsigned int y = 0; y < arraySize.y; ++y) {
      for (unsigned int x = 0; x < arraySize.x; ++x) {
	
	vVal = image(x,y,z);
	
	if(vVal > thValue){
	  Dimage(x,y,z) = maxVal; 
	}
	else{
	  Dimage(x,y,z) = 0;
	}
	
      }
    }
  }   
}

template <class T>
inline
double 
ImageUtils::
l2NormSqr(const Image<T> &image){
  return image.getSpacing().productOfElements() * 
    Array3DUtils::sumOfSquaredElements(image);
}

template <class T>
inline
double 
ImageUtils::
l2DotProd(const Image<T> &i1, const Image<T> &i2){
  Image<T> i = i1;
  i.pointwiseMultiplyBy(i2);
  return i1.getSpacing().productOfElements()*Array3DUtils::sumElements(i);
}
  
template <class T, 
	  Array3DUtils::ScalarBackgroundStrategyT BackgroundStrategy,
	  Array3DUtils::InterpT InterpMethod>
inline
T
ImageUtils::
interp(const Image<T>& image,
       const Vector3D<double>& worldCoordinates,
       const T& background)
{
  Vector3D<double> voxelCoordinates;
  image.worldToImageIndexCoordinates(worldCoordinates,
				     voxelCoordinates);
  return Array3DUtils::
    interp<T, BackgroundStrategy, InterpMethod>(image, voxelCoordinates, background);
}

template <class T>
inline
void
ImageUtils::
translate(Image<T>& image,
	  const double& tx, 
	  const double& ty, 
	  const double& tz)
{
  typename Image<T>::
    ContinuousPointType origin = image.getOrigin();
  image.setOrigin(origin.x + tx, origin.y + ty, origin.z + tz);
}

template <class T>
inline
void
ImageUtils::
translate(Image<T>& image,
	  const Vector3D<double>& t)
{
  typename Image<T>::ContinuousPointType origin = image.getOrigin();
  image.setOrigin(origin.x + t.x, origin.y + t.y, origin.z + t.z);
}

template <class T>
inline
void
ImageUtils::
sincUpsample(Image<T>& image,
	     unsigned int factor)
{

  Vector3D<unsigned int> oldSize = image.getSize();
  Vector3D<unsigned int> newSize = oldSize*factor;
  sincUpsample(image, newSize);
}

template <class T>
inline
void
ImageUtils::
sincUpsample(Image<T>& image,
	     Vector3D<unsigned int> &newSize)
{

  Vector3D<unsigned int> oldSize = image.getSize();

  // set up the complex data arrays.  We pad by one pixel to make
  // the image dimensions odd.
  Vector3D<unsigned int> oldCSize = oldSize+1;
  Vector3D<unsigned int> newCSize = newSize+1;
  unsigned int cDataOldSize = 2*static_cast<unsigned int>(oldCSize.productOfElements());
  float *cDataOld = new float[cDataOldSize];
  unsigned int cDataNewSize = 2*static_cast<unsigned int>(newCSize.productOfElements());
  float *cDataNew = new float[cDataNewSize];
  // zero out the complex arrays
  for(unsigned int i = 0; i < cDataOldSize; i++){
    cDataOld[i] = static_cast<float>(0);
  }
  for(unsigned int i = 0; i < cDataNewSize; i++){
    cDataNew[i] = static_cast<float>(0);
  }

  // set up the fftw plans
  fftwf_plan fftwForwardPlan;
  fftwf_plan fftwBackwardPlan;
  fftwf_plan_with_nthreads(2);
  int dims[3];
  dims[0] = oldCSize.x;
  dims[1] = oldCSize.y;
  dims[2] = oldCSize.z;
  int newDims[3];
  newDims[0] = newCSize.x;
  newDims[1] = newCSize.y;
  newDims[2] = newCSize.z;
  // in-place transforms
  fftwForwardPlan = fftwf_plan_dft(3,dims ,(fftwf_complex *)cDataOld, (fftwf_complex *)(cDataOld),-1, FFTW_ESTIMATE);
  fftwBackwardPlan = fftwf_plan_dft(3,newDims, (fftwf_complex *)cDataNew, (fftwf_complex *)cDataNew,+1, FFTW_ESTIMATE);

  if(!fftwForwardPlan){
    std::cerr << "fftw forward plan could not be initialized!" << std::endl;
    return;
  }
  if(!fftwBackwardPlan){
    std::cerr << "fftw backward plan could not be initialized!" << std::endl;
    return;
  }

  // copy the data into the complex array
  for (unsigned int z = 0; z < oldSize.z; ++z) {
    for (unsigned int y = 0; y < oldSize.y; ++y) {
      for (unsigned int x = 0; x < oldSize.x; ++x) {
	unsigned int cIndex = 2*((x+1) + ((y+1) + (z+1)*oldCSize.y)*oldCSize.x);
	cDataOld[cIndex] = static_cast<float>(image(x,y,z));
	if(!(cDataOld[cIndex] == cDataOld[cIndex])){
	  std::cout << "input data NaN" << std::endl;
	}
      }
    }
  }

  // execute the fft
  fftwf_execute(fftwForwardPlan);

  // copy the data to the larger array.
  // Data goes in four corners.
  Vector3D<unsigned int> freqSplitIdx;
  for(int i=0;i<3;i++){
    freqSplitIdx[i] = oldCSize[i]/2 + 1;
  }

  Vector3D<unsigned int> newIdx;
  for (unsigned int z = 0; z < oldCSize.z; ++z) {
    if(z < freqSplitIdx.z){
      newIdx.z = z;
    }else{
      newIdx.z = newCSize.z+(z-oldCSize.z);
    }
    for (unsigned int y = 0; y < oldCSize.y; ++y) {
      if(y < freqSplitIdx.y){
	newIdx.y = y;
      }else{
	newIdx.y = newCSize.y+(y-oldCSize.y);
      }
      for (unsigned int x = 0; x < oldCSize.x; ++x) {
	if(x < freqSplitIdx.x){
	  newIdx.x = x;
	}else{
	  newIdx.x = newCSize.x+(x-oldCSize.x);
	}
	unsigned int cIndexOld = 2*(x + (y + z*oldCSize.y)*oldCSize.x);
	unsigned int cIndexNew = 2*(newIdx.x + (newIdx.y + newIdx.z*newCSize.y)*newCSize.x);
	cDataNew[cIndexNew] = cDataOld[cIndexOld];
	cDataNew[cIndexNew+1] = cDataOld[cIndexOld+1];
	if(!(cDataNew[cIndexNew] == cDataNew[cIndexNew])){
	  std::cout << "fourier data NaN" << std::endl;
	}
      }
    }
  }
    
  // execute the ifft
  fftwf_execute(fftwBackwardPlan);

  // resize the image
  image.resize(newSize);
  Vector3D<double> factor = newSize/oldSize;
  image.setSpacing(image.getSpacing()/factor);

  // copy out the data to the output image
  for (unsigned int z = 0; z < newSize.z; ++z) {
    for (unsigned int y = 0; y < newSize.y; ++y) {
      for (unsigned int x = 0; x < newSize.x; ++x) {
	unsigned int cIndex = 2*((x+1) + ((y+1) + (z+1)*newCSize.y)*newCSize.x);
	image(x,y,z) = static_cast<T>(cDataNew[cIndex]);
	if(!(cDataNew[cIndex] == cDataNew[cIndex])){
	  std::cout << "output data NaN" << std::endl;
	}
      }
    }
  }

  image.scale(1.0/static_cast<T>(oldSize.productOfElements()));

}

template <class T, 
	  Array3DUtils::ScalarBackgroundStrategyT BackgroundStrategy,
	  Array3DUtils::InterpT InterpMethod>
inline
void ImageUtils::
gaussianDownsample (const Image<T>& input,
		    Image<T>& output, 
		    const Vector3D<unsigned int>& newSize)
{
  /** Downsample the image */
  Vector3D<unsigned int> oldSize = input.getSize();
  Vector3D<double> factor(((double)oldSize.x)/newSize.x,
			  ((double)oldSize.y)/newSize.y,
			  ((double)oldSize.z)/newSize.z);
  Vector3D<double> sigma(sqrt(factor.x/2.0),
			 sqrt(factor.y/2.0),
			 sqrt(factor.z/2.0));
  Vector3D<unsigned int> kernelSize(2*static_cast<unsigned int>(std::ceil(sigma.x)),
				    2*static_cast<unsigned int>(std::ceil(sigma.y)),
				    2*static_cast<unsigned int>(std::ceil(sigma.z)));

  std::cout << "Using non-integer downsampling" << std::endl;
  // otherwise compute full gaussian-filtered image
  GaussianFilter3D filter;
  filter.SetInput(input);
  filter.setSigma(sigma.x, sigma.y, sigma.z);
  filter.setKernelSize(kernelSize.x, kernelSize.y, kernelSize.z);
  filter.Update();
    
  // Create new downsampled image
  Vector3D<double> oldSpacing = input.getSpacing();
  Vector3D<double> origin = input.getOrigin();
  output.resize(newSize);
  output.setSpacing(oldSpacing.x * factor.x,
		    oldSpacing.y * factor.y,
		    oldSpacing.z * factor.z);
  output.setOrigin(origin);

  // resampleNew requires an Image, not an Array3D
  Image<T> gaussianOutput(filter.GetOutput());
  gaussianOutput.setSpacing(oldSpacing);
  gaussianOutput.setOrigin(origin);

  resampleNew<T, BackgroundStrategy, InterpMethod>(gaussianOutput, output);
}

template <class T, 
	  Array3DUtils::ScalarBackgroundStrategyT BackgroundStrategy,
	  Array3DUtils::InterpT InterpMethod>
inline
void
ImageUtils::
resampleNew(Image<T>& image,
	    const Vector3D<double>& newOrigin,
	    const Vector3D<double>& newSpacing,
	    const Vector3D<unsigned int>& newDimensions,
	    T bgVal)
{
  Image<T> tmp(image);
  image.resize(newDimensions);
  image.setOrigin(newOrigin);
  image.setSpacing(newSpacing);
  resampleNew<T, BackgroundStrategy, InterpMethod>(tmp, image, bgVal);
}

template <class T, 
	  Array3DUtils::ScalarBackgroundStrategyT BackgroundStrategy,
	  Array3DUtils::InterpT InterpMethod>
inline
void
ImageUtils::
resampleNew(const Image<T>& sourceImage,
	    Image<T>& destImage,
	    T bgVal)
{
  Vector3D<unsigned int> size = destImage.getSize();
  Vector3D<double> world;
  Vector3D<double> voxel;
  Vector3D<double> samplingOrigin = (destImage.getSpacing() - sourceImage.getSpacing())/2.0;
  for (unsigned int z = 0; z < size.z; ++z) {
    for (unsigned int y = 0; y < size.y; ++y) {
      for (unsigned int x = 0; x < size.x; ++x) {
	destImage.imageIndexToWorldCoordinates(x, y, z, 
					       world.x, world.y, world.z);

	// offset for proper centering
	world = world + samplingOrigin;

	sourceImage.worldToImageIndexCoordinates(world, voxel);

	destImage(x,y,z) = Array3DUtils::
	  interp<T, BackgroundStrategy, InterpMethod>(sourceImage, voxel, bgVal);
      }
    }
  }
}  

template <class T, 
	  Array3DUtils::ScalarBackgroundStrategyT BackgroundStrategy,
	  Array3DUtils::InterpT InterpMethod>
inline
void
ImageUtils::
resample(Image<T>& image,
	 const Vector3D<double>& newOrigin,
	 const Vector3D<double>& newSpacing,
	 const Vector3D<unsigned int>& newDimensions)
{
  Image<T> tmp(image);
  image.resize(newDimensions);
  image.setOrigin(newOrigin);
  image.setSpacing(newSpacing);
  resample<T, BackgroundStrategy, InterpMethod>(tmp, image);
}

template <class T, 
	  Array3DUtils::ScalarBackgroundStrategyT BackgroundStrategy,
	  Array3DUtils::InterpT InterpMethod>
inline
void
ImageUtils::
resample(const Image<T>& sourceImage,
	 Image<T>& destImage)
{
	Vector3D<unsigned int> size = destImage.getSize();
  Vector3D<double> world;
  Vector3D<double> voxel;
  for (unsigned int z = 0; z < size.z; ++z) {
    for (unsigned int y = 0; y < size.y; ++y) {
      for (unsigned int x = 0; x < size.x; ++x) {
	destImage.imageIndexToWorldCoordinates(x, y, z, 
					       world.x, world.y, world.z);
	
	sourceImage.worldToImageIndexCoordinates(world, voxel);	
	destImage(x,y,z) = Array3DUtils::
	  interp<T, BackgroundStrategy, InterpMethod>(sourceImage, voxel, 0.0F);
      }
    }
  }
}

template <class T, 
	  Array3DUtils::ScalarBackgroundStrategyT BackgroundStrategy,
	  Array3DUtils::InterpT InterpMethod>
inline
void
ImageUtils::
resampleWithTransparency(const Image<T>& sourceImage,
			 Image<T>& destImage)
{
  Vector3D<unsigned int> size = destImage.getSize();
  Vector3D<double> world;
  Vector3D<double> voxel;
  Vector3D<double> samplingOrigin = (destImage.getSpacing() - sourceImage.getSpacing())/2.0;
  for (unsigned int z = 0; z < size.z; ++z) {
    for (unsigned int y = 0; y < size.y; ++y) {
      for (unsigned int x = 0; x < size.x; ++x) {
	destImage.imageIndexToWorldCoordinates(x, y, z, 
					       world.x, world.y, world.z);

	sourceImage.worldToImageIndexCoordinates(world, voxel);

	float intensity = Array3DUtils::
	  interp<T, BackgroundStrategy, InterpMethod>(sourceImage, 
						      voxel,
						      -FLT_MAX);
	if (intensity >= 0.0f) destImage(x,y,z) = intensity;
          
      }
    }
  }
}  

inline
AffineTransform3D<double>
ImageUtils::
transformInIndexCoordinates(const Vector3D<double>& fixedImageOrigin,
			    const Vector3D<double>& fixedImageSpacing,
			    const Vector3D<double>& movingImageOrigin,
			    const Vector3D<double>& movingImageSpacing,
			    const AffineTransform3D<double>& transformInWorldCoordinates )
{
  Matrix3D<double> S1inv;
  Matrix3D<double> S2;
  for (unsigned int i = 0; i < 3; ++i) {
    S1inv(i,i) = 1.0 / movingImageSpacing[i];
    S2(i,i) = fixedImageSpacing[i];
  }
  AffineTransform3D<double> result;
  const Matrix3D<double>& wM = transformInWorldCoordinates.matrix;
  const Vector3D<double>& wV = transformInWorldCoordinates.vector;
  result.matrix = S1inv * wM * S2;
  result.vector = S1inv * (wV - movingImageOrigin + wM * fixedImageOrigin);
  return result;
}


template <class T, 
	  Array3DUtils::ScalarBackgroundStrategyT BackgroundStrategy,
	  Array3DUtils::InterpT InterpMethod>
inline
void
ImageUtils::
applyAffine(Image<T>& image,
	    const Vector3D<double>& newOrigin,
	    const Vector3D<double>& newSpacing,
	    const Vector3D<unsigned int>& newDimensions,
	    const AffineTransform3D<double>& transformInWorldCoordinates, 
	    const float& backgroundValue)
{
  Image<T> tmp(image);
  image.resize(newDimensions);
  image.setOrigin(newOrigin);
  image.setSpacing(newSpacing);
  applyAffine<T, BackgroundStrategy, InterpMethod>(tmp, image, transformInWorldCoordinates, backgroundValue);
}

template <class T, 
	  Array3DUtils::ScalarBackgroundStrategyT BackgroundStrategy,
	  Array3DUtils::InterpT InterpMethod>
inline
void
ImageUtils::
applyAffine(const Image<T>& sourceImage,
	    Image<T>& destImage,
	    const AffineTransform3D<double>& transformInWorldCoordinates,
	    const float& backgroundValue)
{

  // Compute transform in index coordinates
  AffineTransform3D<double> newTransform = 
    transformInIndexCoordinates( destImage.getOrigin(),
				 destImage.getSpacing(),
				 sourceImage.getOrigin(),
				 sourceImage.getSpacing(),
				 transformInWorldCoordinates );

  // Get data pointer to write to quickly
  T* newImageDataPtr = destImage.getDataPointer();

  // Determine iteration bounds
  size_t zSize = destImage.getSizeZ();
  size_t ySize = destImage.getSizeY();
  size_t xSize = destImage.getSizeX();

  // Iterate through and interpolate the data
  double x, y, z;
  for( unsigned int zPos = 0; zPos < zSize; zPos++ ) {
    for( unsigned int yPos = 0; yPos < ySize; yPos++ ) {
      for( unsigned int xPos = 0; xPos < xSize; xPos++ ) {

	newTransform.transformCoordinates( xPos, yPos, zPos, x, y, z );

	// Store interpolated value
	*newImageDataPtr++ = 
	  Array3DUtils::
	  interp<T, BackgroundStrategy, InterpMethod>(sourceImage, x, y, z, backgroundValue);
      }
    }
  }
}

template <class T, 
	  Array3DUtils::ScalarBackgroundStrategyT BackgroundStrategy,
	  Array3DUtils::InterpT InterpMethod>
inline
void
ImageUtils::
makeVoxelsIsotropic(Image<T>& image)
{
  Vector3D<double> spacingOld = image.getSpacing();    
  Vector3D<unsigned int> sizeOld = image.getSize();
    
  if (spacingOld.x == spacingOld.y && spacingOld.x == spacingOld.z)
    {
      return;
    }

  // 
  // set new spacing in all dimensions to min spacing
  //
  double minSpacing = std::min(spacingOld.x, 
			       std::min(spacingOld.y, spacingOld.z));
  Vector3D<double> 
    spacingNew(minSpacing * spacingOld.x < 0 ? -minSpacing : minSpacing,
	       minSpacing * spacingOld.y < 0 ? -minSpacing : minSpacing,
	       minSpacing * spacingOld.z < 0 ? -minSpacing : minSpacing);
    
  //
  // origin stays same
  //
  Vector3D<double> originNew(image.getOrigin());

  //
  // determine new dimensions
  //
  Vector3D<unsigned int> 
    sizeNew((unsigned int)floor(double(sizeOld.x) * 
				spacingOld.x / spacingNew.x),
	    (unsigned int)floor(double(sizeOld.y) * 
				spacingOld.y / spacingNew.y),
	    (unsigned int)floor(double(sizeOld.z) * 
				spacingOld.z / spacingNew.z));
              
  //
  // resample
  //
  resample<T, BackgroundStrategy, InterpMethod>(image, originNew, spacingNew, sizeNew); 
}

template <class T, 
	  Array3DUtils::ScalarBackgroundStrategyT BackgroundStrategy,
	  Array3DUtils::InterpT InterpMethod>
inline
void
ImageUtils::
resliceZMakeIsotropic(Image<T>& image)
{
  Vector3D<double> spacingOld = image.getSpacing();    
  Vector3D<unsigned int> sizeOld = image.getSize();
    
  if (spacingOld.x == spacingOld.z)
    {
      return;
    }

  // 
  // set new z spacing to x spacing (maintain sign)
  //
  Vector3D<double> 
    spacingNew(spacingOld.x, spacingOld.y,
	       spacingOld.x * spacingOld.z < 0 
	       ? -spacingOld.x 
	       : spacingOld.x);
    
  //
  // origin stays same
  //
  Vector3D<double> originNew(image.getOrigin());

  //
  // determine new dimensions
  //
  Vector3D<unsigned int> 
    sizeNew(sizeOld.x, sizeOld.y,
	    (unsigned int)floor(double(sizeOld.z) * 
				spacingOld.z / spacingNew.z) - 1);
              
  //
  // resample
  //
  resample<T, BackgroundStrategy, InterpMethod>(image, originNew, spacingNew, sizeNew); 
}

template <class T>
inline
void
ImageUtils::
gaussianDownsample(const Image<T>& imageIn,
		   Image<T>& imageOut,
		   const Vector3D<double>& factors,
		   const Vector3D<double>& sigma,
		   const Vector3D<double>& kernelSize)
{
  Array3DUtils::gaussianDownsample<T>(imageIn, 
				      imageOut,
				      factors, 
				      sigma,
				      kernelSize);
  Vector3D<double> spacingOld = imageIn.getSpacing();
  imageOut.setSpacing(spacingOld.x * factors.x,
		      spacingOld.y * factors.y,
		      spacingOld.z * factors.z);
  imageOut.setOrigin(imageIn.getOrigin());
}

template <class T, 
	  Array3DUtils::ScalarBackgroundStrategyT BackgroundStrategy,
	  Array3DUtils::InterpT InterpMethod>
inline
double
ImageUtils::
squaredError(const Image<T>& image1,
	     const Image<T>& image2)
{
  Vector3D<unsigned int> size = image1.getSize();
  double squaredDifference = 0;

  T val1, val2;
  double d;
  Vector3D<double> scale(image1.getSpacing() / image2.getSpacing());
  Vector3D<double> offset((image1.getOrigin() - image2.getOrigin())
			  / image2.getSpacing());

  for (unsigned int z = 0; z < size.z; ++z) {
    for (unsigned int y = 0; y < size.y; ++y) {
      for (unsigned int x = 0; x < size.x; ++x) {
	val1 = image1(x,y,z);
	val2 = Array3DUtils::
	  interp<T, BackgroundStrategy, InterpMethod>(image2,
						      x * scale.x + offset.x,
						      y * scale.y + offset.y,
						      z * scale.z + offset.z,
						      (T)0.0F);
	d = ((double) val1) - ((double) val2); 
	squaredDifference += d * d;
      }
    }
  }  
  return squaredDifference;    
}

template <class T>
inline
Vector3D<double> 
ImageUtils::
computeCentroid(const Image<T>& image,
		const ROI<int, unsigned int> voxelROI)
{
#if 0
  Vector3D<double> centroid = 
    Array3DUtils::computeCentroid( image, voxelROI.getStart(),
				   voxelROI.getSpacing() );
  centroid *= image.getSpacing();
  centroid += image.getOrigin();
  return centroid;
#endif
  std::cerr << "[ImageUtils.h] INTERNAL ERROR: no definition of spacing "
	    << " in ROI class." << std::endl;
  Vector3D<double> centroid(0,0,0);
  return centroid;
}

template <class T>
inline
Vector3D<double> 
ImageUtils::
computeCentroid(const Image<T>& image)
{
  Vector3D<double> centroid = 
    Array3DUtils::computeCentroid( image, typename Array3D<T>::IndexType(0,0,0),
				   image.getSize() );
  centroid *= image.getSpacing();
  centroid += image.getOrigin();
  return centroid;
}

template <class T>
inline
void 
ImageUtils::
writeMETA(const Image<T>& image,
	  const char* filenamePrefix)
{
  Array3DIO::writeMETAVolume(image, 
			     image.getOrigin(),
			     image.getSpacing(),
			     filenamePrefix);
}  

template <class T>
inline
void 
ImageUtils::
extractROIVoxelCoordinates(const Image<T>& image,
			   Image<T>& roiImage,
			   const ROI<int, unsigned int>& voxelROI)
{
  Array3DUtils::extractROI(image, roiImage, voxelROI);
  roiImage.setSpacing(image.getSpacing());
  roiImage.setOrigin(image.getOrigin().x
		     + voxelROI.getStart().x * image.getSpacing().x,
		     image.getOrigin().y
		     + voxelROI.getStart().y * image.getSpacing().y,
		     image.getOrigin().z
		     + voxelROI.getStart().z * image.getSpacing().z);
}

template <class T>
inline
void 
ImageUtils::
extractROIWorldCoordinates(const Image<T>& image,
			   Image<T>& roiImage,
			   const ROI<double, double>& worldROI)
{
  Vector3D<double> newStart = 
    (worldROI.getStart() - image.getOrigin()) / image.getSpacing();
  Vector3D<double> newSize = worldROI.getSize() / image.getSpacing();
  // !! this should round instead of cast ??
  ROI<int, unsigned int> voxelROI((Vector3D<int>)newStart, 
				  (Vector3D<unsigned int>)newSize );
  extractROIVoxelCoordinates(image, roiImage, voxelROI);
}

template <class T>
inline 
void
ImageUtils::
makeImageUnsigned(Image<T>& image)
{
  T min, max;
  min = max = 0; // make compiler happy
  Array3DUtils::getMinMax(image, min, max);
  if (min < 0) 
    {
      for (unsigned int i = 0; i < image.getNumElements(); ++i)
	{
	  image(i) += 1024;
	}
    }
}

