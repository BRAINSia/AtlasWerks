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


#include <sstream>

#include "WeightedImageSet.h"
#include "ApplicationUtils.h"
#include "Array3DUtils.h"

#include "itkFixedArray.h"
#include "itkMatrix.h"

WeightedImageSet::
WeightedImageSet(const WeightedImageSetParam &param)
  : mParam(param)
{

  // Load all the filenames etc.
  
  // first get the names of images specified by the format string
  std::string format = mParam.InputImageFormatString().FormatString();
  if(format.size() > 0){
    int base = mParam.InputImageFormatString().Base();
    int numImages = mParam.InputImageFormatString().NumFiles();
    Real weight = mParam.InputImageFormatString().Weight();
    for(int i=base;i<base+numImages;i++){
      mImageNames.push_back(StringUtils::strPrintf(format.c_str(), i));
      mImageWeights.push_back(weight);
      std::cout << "Image "<< mImageNames.size() << ": " << mImageNames.back() << std::endl;
    }
  }
  
  // get the image names of the individually specified images
  unsigned int nIndividualImages = mParam.WeightedImage().size();
  for (int i = 0; i < (int) nIndividualImages; ++i){
    const WeightedImageParam &curImageParam = mParam.WeightedImage()[i];
    mImageNames.push_back(curImageParam.Filename());
    mImageWeights.push_back(curImageParam.Weight());
    std::cout << "Image "<< mImageNames.size() << ": " << mImageNames.back() << std::endl;
    if(curImageParam.Transform().length() > 0){
      if(mTransformNames.size() == mImageNames.size()-1){
	mTransformNames.push_back(curImageParam.Transform());
	mItkStyleTransforms.push_back(curImageParam.ItkTransform());
      }else{
	throw AtlasWerksException(__FILE__, __LINE__, 
			       "If any image transforms are specified, "
			       "transforms must be specified for all inputs");
      }
    }
  }

}

void 
WeightedImageSet::
Load(bool verbose, int base, int num)
{

  if(num < 0){
    num = mImageNames.size() - base;
  }

  if(base < 0 || base > (int)mImageNames.size() || 
     num+base == 0 || num+base > (int)mImageNames.size())
    {
      std::stringstream ss;
      ss << "Illegal subset of images specified -- base: "
	 << base << ", num: " << num;
      throw AtlasWerksException(__FILE__, __LINE__, ss.str());
    }

  if(mImageNames.size() == 0){
    throw AtlasWerksException(__FILE__, __LINE__, "No input images found.");
  }
  
  //
  // Scale the weights if requested
  //
  Real imageWeightSum = 0.0;
  for(unsigned int i=0;i<mImageWeights.size();i++){
    imageWeightSum += mImageWeights[i];
  }
  if(verbose) std::cout << "Sum of weights is " << imageWeightSum << std::endl;
  if(mParam.ScaleImageWeights()){
    if(verbose) std::cout << "Scaling weights to sum to 1.0" << std::endl;
    for(unsigned int i=0;i<mImageNames.size();i++){
      mImageWeights[i] /= imageWeightSum;
      std::cout << "Weight " << i << " = " << mImageWeights[i] << std::endl;
    }
  }

  // get the subset of images, weights, and transforms we're going to use
  std::vector<std::string> tmpImageNames;  
  std::vector<Real> tmpImageWeights;  
  std::vector<std::string> tmpTransformNames;  
  std::vector<bool> tmpItkStyleTransforms;
  for (int i = base; i < base+num; ++i){
    tmpImageNames.push_back(mImageNames[i]);
    tmpImageWeights.push_back(mImageWeights[i]);
    if(mTransformNames.size()){
      tmpTransformNames.push_back(mTransformNames[i]);
      tmpItkStyleTransforms.push_back(mItkStyleTransforms[i]);
    }
  }
  mImageNames = tmpImageNames;
  mImageWeights = tmpImageWeights;
  mTransformNames = tmpTransformNames;
  mItkStyleTransforms = tmpItkStyleTransforms;

  //
  // load images
  //
  if(verbose) std::cout << "Loading Images..." << std::endl;

  for (unsigned int i = 0; i < mImageNames.size(); ++i){
    RealImage *curImage = new RealImage;
    ApplicationUtils::LoadImageITK(mImageNames[i].c_str(), *curImage);
    mImages.push_back(curImage);
    Vector3D<unsigned int> size = curImage->getSize();
    Vector3D<Real> origin = curImage->getOrigin();
    Vector3D<Real> spacing = curImage->getSpacing();
    Real iMin, iMax;
    iMin = iMax = 0; // make compiler happy
    Array3DUtils::getMinMax(*curImage, iMin, iMax);
    if(verbose){
      std::cout << "   Loaded: " << mImageNames[i] << std::endl;
      std::cout << "   Dimensions: " << size << std::endl;
      std::cout << "   Origin: " << origin << std::endl;
      std::cout << "   Spacing: " << spacing << std::endl;
      std::cout << "   Intensity Range: " << iMin << "-" << iMax << std::endl;
    }
  }


  //
  // See if we have affine transform files, load them if so
  //
  if(mTransformNames.size() > 0){
    for(unsigned int i=0;i<mTransformNames.size();i++){
      RealAffineTransform *transform = new RealAffineTransform();
      if(mItkStyleTransforms[i]){
	// read ITK-style transform
	transform->readITKStyle(mTransformNames[i]);
      }else{
	// read RealAffineTransform PLUNC-style transform
	transform->readPLUNCStyle(mTransformNames[i]);
      }
      // add the transform
      mTransforms.push_back(transform);
    } // end loop over transforms
    
  } // end test if we have transforms
  
}

void 
WeightedImageSet::
Clear()
{
  mImageNames.clear();
  mTransformNames.clear();
  mImageWeights.clear();
  for(unsigned int i=0;i<mImages.size();i++){
    delete mImages[i];
  }
  mImages.clear();
  mTransforms.clear();
}

RealImage*
WeightedImageSet::
GetImage(int i)
{
  if(i >= 0 && i < (int)mImages.size()){
    return mImages[i];
  }else{
    throw AtlasWerksException(__FILE__, __LINE__, 
			   "image index out of bounds"); 
  }
}

const RealImage*
WeightedImageSet::
GetImage(int i) const
{
  if(i >= 0 && i < (int)mImages.size()){
    return mImages[i];
  }else{
    throw AtlasWerksException(__FILE__, __LINE__, 
			   "image index out of bounds"); 
  }
}

std::vector<RealImage*> 
WeightedImageSet::
GetImageVec()
{
  std::vector<RealImage*> rtn = mImages;
  return rtn;
}

std::vector<const RealImage*> 
WeightedImageSet::
GetImageVec() const
{
  std::vector<const RealImage*> rtn;
  rtn.assign(mImages.begin(), mImages.end());
  return rtn;
}

std::string
WeightedImageSet::
GetImageName(int i) const
{
  if(i >= 0 && i < (int)mImageNames.size()){
    return mImageNames[i];
  }else{
    throw AtlasWerksException(__FILE__, __LINE__, 
			   "image index out of bounds");
  }
}

std::vector<std::string> 
WeightedImageSet::
GetImageNameVec() const
{
  std::vector<std::string> rtn = mImageNames;
  return rtn;
}

Real 
WeightedImageSet::
GetWeight(int i) const
{
  if(i >= 0 && i < (int)mImageWeights.size()){
    return mImageWeights[i];
  }else{
    throw AtlasWerksException(__FILE__, __LINE__, 
			   "weight index out of bounds");
  }
}

std::vector<Real> 
WeightedImageSet::
GetWeightVec() const
{
  std::vector<Real> rtn = mImageWeights;
  return rtn;
}

RealAffineTransform*
WeightedImageSet::
GetTransform(int i)
{
  if(i >= 0 && i < (int)mTransforms.size()){
    return mTransforms[i];
  }else{
    throw AtlasWerksException(__FILE__, __LINE__, 
			   "transform index out of bounds");
  }
}

const RealAffineTransform*
WeightedImageSet::
GetTransform(int i) const
{
  if(i >= 0 && i < (int)mTransforms.size()){
    return mTransforms[i];
  }else{
    throw AtlasWerksException(__FILE__, __LINE__, 
			   "transform index out of bounds");
  }
}

std::vector<RealAffineTransform*> 
WeightedImageSet::
GetTransformVec()
{
  std::vector<RealAffineTransform*> rtn = mTransforms;
  return rtn;
}

std::vector<const RealAffineTransform*> 
WeightedImageSet::
GetTransformVec() const
{
  std::vector<const RealAffineTransform*> rtn;
  rtn.assign(mTransforms.begin(), mTransforms.end());
  return rtn;
}

SizeType 
WeightedImageSet::
GetImageSize() const
{
  unsigned int nImages = mImages.size();
  if(nImages < 1){
    throw AtlasWerksException(__FILE__, __LINE__, 
			   "GetImageSize: No images loaded!");
  }
  SizeType size = mImages[0]->getSize();
  SizeType curSize;
  for(unsigned int i=1;i<mImages.size();i++){
    curSize = mImages[i]->getSize();
    if(size != curSize){
      throw AtlasWerksException(__FILE__, __LINE__, 
			     "GetImageSize: Size different in image " + mImageNames[i]);
    }
  }
  return size;
}

SpacingType 
WeightedImageSet::
GetImageSpacing() const 
{
  unsigned int nImages = mImages.size();
  if(nImages < 1){
    throw AtlasWerksException(__FILE__, __LINE__, 
			   "GetImageSpacing: No images loaded!");
  }
  SpacingType spacing = mImages[0]->getSpacing();
  SpacingType curSpacing;
  for(unsigned int i=1;i<mImages.size();i++){
    curSpacing = mImages[i]->getSpacing();
    SpacingType diff = spacing-curSpacing;
    if(diff.maxElement() > ATLASWERKS_EPS){
      throw AtlasWerksException(__FILE__, __LINE__, 
			     "GetImageSpacing: Spacing different in image " + mImageNames[i]);
    }
  }
  return spacing;
}

OriginType 
WeightedImageSet::
GetImageOrigin() const
{
  unsigned int nImages = mImages.size();
  if(nImages < 1){
    throw AtlasWerksException(__FILE__, __LINE__, 
			   "GetImageOrigin: No images loaded!");
  }
  OriginType origin = mImages[0]->getOrigin();
  OriginType curOrigin;
  for(unsigned int i=1;i<mImages.size();i++){
    curOrigin = mImages[i]->getOrigin();
    if(origin != curOrigin){
      throw AtlasWerksException(__FILE__, __LINE__, 
			     "GetImageOrigin: Origin different in image " + mImageNames[i]);
    }
  }
  return origin;
}
