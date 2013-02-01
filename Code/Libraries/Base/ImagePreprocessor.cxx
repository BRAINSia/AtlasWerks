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

#include "ImagePreprocessor.h"

#include "Array3DUtils.h"
#include "StringUtils.h"
#include "ApplicationUtils.h"

/***** BEGIN IntensityWindow  ****/

IntensityWindow::
IntensityWindow()
  : mVerbose(true)
{
  // use IntensityWindowParam defaults
  IntensityWindowParam defaultParam;
  SetParams(defaultParam);
}

IntensityWindow::
IntensityWindow(const IntensityWindowParam &param)
  : mVerbose(true)
{
  SetParams(param);
}
  
void 
IntensityWindow::
Rescale(RealImage &image) const
{
  if(!mRescaleIntensities){
    if(mVerbose) std::cout << "Not rescaling image" << std::endl;
    return;
  }
  
  Real iMin, iMax;
  iMin=iMax=0; // make compiler happy
  Array3DUtils::getMinMax(image, iMin, iMax);
  Real iwMin = iMin;
  Real iwMax = iMax;
  if(mUseInputIntensityWindow){
    iwMin = mInputWindowMin;
    iwMax = mInputWindowMax;
  }
  Real owMin = mOutputWindowMin;
  Real owMax = mOutputWindowMax;
  if(mVerbose) std::cout 
		 << "   Rescaling from [" << iwMin << "," << iwMax 
		 << "] to [" << owMin << "," << owMax << "]...";
  Array3DUtils::rescaleElements(image, iwMin, iwMax, owMin, owMax);
  if(mVerbose) std::cout << "DONE" << std::endl;
  Real oMin, oMax;
  oMin = oMax = 0; // make compiler happy
  Array3DUtils::getMinMax(image, oMin, oMax);
  if(mVerbose) std::cout << "   New Intensity Range: " << oMin << "-" << oMax << std::endl;

}

void 
IntensityWindow::
SetParams(const IntensityWindowParam &param)
{
  mRescaleIntensities = param.RescaleIntensities();
  mUseInputIntensityWindow = param.UseInputIntensityWindow();
  mInputWindowMin = param.InputWindowMin();
  mInputWindowMax = param.InputWindowMax();
  mOutputWindowMin = param.OutputWindowMin();
  mOutputWindowMax = param.OutputWindowMax();
}

void 
IntensityWindow::
GetParams(IntensityWindowParam &param) const
{
  param.RescaleIntensities() = mRescaleIntensities;
  param.UseInputIntensityWindow() = mUseInputIntensityWindow;
  param.InputWindowMin() = mInputWindowMin;
  param.InputWindowMax() = mInputWindowMax;
  param.OutputWindowMin() = mOutputWindowMin;
  param.OutputWindowMax() = mOutputWindowMax;
}

/***** END IntensityWindow  ****/

/***** BEGIN TukeyWindow  ****/

TukeyWindow::
TukeyWindow()
  : mApplyWindow(true),
    mWidth(0),
    mBorderMask(NULL)
{
}

TukeyWindow::
TukeyWindow(const TukeyWindowParam &param)
  : mApplyWindow(true),
    mWidth(0),
    mBorderMask(NULL)
{
  SetParams(param);
};

void
TukeyWindow::
SetParams(const TukeyWindowParam &param)
{
  mApplyWindow = param.DoWindowing();
  mWidth = param.Width();
  if(mBorderMask){
    delete [] mBorderMask;
  }
  mBorderMask = new Real[mWidth];
  for(unsigned int i=0;i<mWidth;i++){
    double v = M_PI*((double)i)/((double)mWidth);
    mBorderMask[i] = 0.5*(1.0-cos(v));
  }
}

void
TukeyWindow::
GetParams(TukeyWindowParam &param) const
{
  param.DoWindowing() = mApplyWindow;
  param.Width() = mWidth;
}

void
TukeyWindow::
ApplyWindow(RealImage &image)
{

  if(!mApplyWindow){
    return;
  }

  SizeType size = image.getSize();
  for(unsigned int x = 0; x < size.x; x++){
    for(unsigned int y = 0; y < size.y; y++){
      for(unsigned int z = 0; z < size.z; z++){
	// x border
	if(x < mWidth){
	  image(x,y,z) = image(x,y,z)*mBorderMask[x];
	}
	if(x >= size.x-mWidth){
	  image(x,y,z) = image(x,y,z)*mBorderMask[(size.x-1)-x];
	}
	// y border
	if(y < mWidth){
	  image(x,y,z) = image(x,y,z)*mBorderMask[y];
	}
	if(y >= size.y-mWidth){
	  image(x,y,z) = image(x,y,z)*mBorderMask[(size.y-1)-y];
	}
	// z border
	if(z < mWidth){
	  image(x,y,z) = image(x,y,z)*mBorderMask[z];
	}
	if(z >= size.z-mWidth){
	  image(x,y,z) = image(x,y,z)*mBorderMask[(size.z-1)-z];
	}
      }
    }
  }
}

/***** END TukeyWindow  ****/

/***** BEGIN GaussianBlur  ****/

GaussianBlur::
GaussianBlur()
  : mSigma(0.0)
{
}

GaussianBlur::
GaussianBlur(const GaussianBlurParam &param)
  : mSigma(0.0)
{
  SetParams(param);
};

void
GaussianBlur::
SetParams(const GaussianBlurParam &param)
{
  mSigma = param.Sigma();
}

void
GaussianBlur::
GetParams(GaussianBlurParam &param) const
{
  param.Sigma() = mSigma;
}

void
GaussianBlur::
Blur(RealImage &image)
{
  if(mSigma > 0.f){
    Vector3D<double> sigma(mSigma, mSigma, mSigma);
    Vector3D<int> kernel(static_cast<int>(std::ceil(sigma.x*3)),
			 static_cast<int>(std::ceil(sigma.x*3)), 
			 static_cast<int>(std::ceil(sigma.x*3)));
    Array3DUtils::gaussianBlur(image, image, sigma, kernel);
  }
}

/***** END GaussianBlur  ****/

/***** BEGIN ImagePreprocessor  ****/

ImagePreprocessor::
ImagePreprocessor()
  : mSize(NULL),
    mOrigin(NULL),
    mSpacing(NULL)
{
  ImagePreprocessorParam defaultParam;
  SetParams(defaultParam);
}

ImagePreprocessor::
ImagePreprocessor(const ImagePreprocessorParam &param)
  : mSize(NULL),
    mOrigin(NULL),
    mSpacing(NULL)
{
  SetParams(param);
}

void 
ImagePreprocessor::
SetParams(const ImagePreprocessorParam &param)
{
  mIntensityWindow.SetParams(param.IntensityWindow());
  mTukeyWindow.SetParams(param.TukeyWindow());
  mGaussianBlur.SetParams(param.GaussianBlur());
  mSetUnitSpacing = param.SetUnitSpacing();
  mSetZeroOrigin = param.SetZeroOrigin();
}

void 
ImagePreprocessor::
GetParams(ImagePreprocessorParam &param) const
{
  mIntensityWindow.GetParams(param.IntensityWindow());
  mTukeyWindow.GetParams(param.TukeyWindow());
  mGaussianBlur.GetParams(param.GaussianBlur());
  param.SetUnitSpacing() = mSetUnitSpacing;
  param.SetZeroOrigin() = mSetZeroOrigin;
}

void 
ImagePreprocessor::
SetImageSize(const Vector3D<unsigned int> size){
  if(mSize != NULL){
    if((*mSize) == size){
      return;
    }
    std::ostringstream sstr;
    sstr << "Error, image size already set to " << mSize << ", trying to set to " << size;
    throw AtlasWerksException(__FILE__, __LINE__, sstr.str());
  }else{
    mSize = new Vector3D<unsigned int>(size);
  }
}
  
void 
ImagePreprocessor::
SetImageOrigin(const Vector3D<Real> origin){
  if(mOrigin != NULL){
    if((*mOrigin) == origin){
      return;
    }
    std::ostringstream sstr;
    sstr << "Error, image origin already set to " << mOrigin << ", trying to set to " << origin;
    throw AtlasWerksException(__FILE__, __LINE__, sstr.str());
  }else{
    mOrigin = new Vector3D<Real>(origin);
  }
}

void 
ImagePreprocessor::
SetImageSpacing(const Vector3D<Real> spacing){
  if(mSpacing != NULL){
    SpacingType diff = *mSpacing-spacing;
    if(diff.maxElement() <= ATLASWERKS_EPS){
      return;
    }
    std::ostringstream sstr;
    sstr << "Error, image spacing already set to " << mSpacing << ", trying to set to " << spacing;
    throw AtlasWerksException(__FILE__, __LINE__, sstr.str());
  }else{
    mSpacing = new Vector3D<Real>(spacing);
  }
}

const Vector3D<unsigned int>&
ImagePreprocessor::
GetImageSize() const{
  if(mSize == NULL){
    throw AtlasWerksException(__FILE__, __LINE__, 
			   "Size not set yet (probably means no images processed?)");
  }
  return *mSize;
}

const Vector3D<Real>&
ImagePreprocessor::
GetImageOrigin() const{
  if(mOrigin == NULL){
    throw AtlasWerksException(__FILE__, __LINE__, 
			   "Origin not set yet (probably means no images processed?)");
  }
  return *mOrigin;
}

const Vector3D<Real>&
ImagePreprocessor::
GetImageSpacing() const{
  if(mSpacing == NULL){
    throw AtlasWerksException(__FILE__, __LINE__, "Spacing not set yet (probably means no images processed?)");
  }
  return *mSpacing;
}

void 
ImagePreprocessor::
Process(RealImage &image, std::string imageID)
{

  //
  // Do intensity windowing
  // 
  mIntensityWindow.Rescale(image);

  //
  // Do border windowing
  //
  mTukeyWindow.ApplyWindow(image);

  //
  // Do blurring 
  //
  mGaussianBlur.Blur(image);

  //
  // Change origin or spacing if requested
  //
  if(mSetZeroOrigin){
    std::cout << "Setting zero origin " << imageID << std::endl;
    image.setOrigin(Vector3D<Real>(0.0, 0.0, 0.0));
  }
  if(mSetUnitSpacing){
    std::cout << "Setting unit spacing " << imageID << std::endl;
    image.setSpacing(Vector3D<Real>(1.0, 1.0, 1.0));
  }
  
  //
  // Test image size, origin, and spacing
  // 
  Vector3D<unsigned int> size = image.getSize();
  Vector3D<Real> origin = image.getOrigin();
  Vector3D<Real> spacing = image.getSpacing();
  
  if(mSize == NULL){
    this->SetImageSize(size);
  }else if(size != (*mSize)){
    throw AtlasWerksException(__FILE__, __LINE__, "Error, image sizes not the same! " + imageID);
  }
  
  if(mOrigin == NULL){
    this->SetImageOrigin(origin);
  }else if(origin != (*mOrigin)){
    throw AtlasWerksException(__FILE__, __LINE__, "Error, image origins not the same!" + imageID);
  }
  
  if(mSpacing == NULL){
    this->SetImageSpacing(spacing);
  }else{
    SpacingType diff = *mSpacing-spacing;
    if(diff.maxElement() > ATLASWERKS_EPS){
      throw AtlasWerksException(__FILE__, __LINE__, "Error, image spacings not the same!" + imageID);
    }
  }
  
}

void 
ImagePreprocessor::
Process(std::vector<RealImage*> &imageVec, std::vector<std::string> imageIDVec)
{
  for(unsigned int i=0;i<imageVec.size();i++){
    std::string imageID = "";
    if(i < imageIDVec.size()) imageID = imageIDVec[i];
    this->Process(*imageVec[i], imageID);
  }
}

/***** END ImagePreprocessor  ****/

