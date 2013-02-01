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

#include "MultiscaleManager.h"

MultiscaleManager::
MultiscaleManager(const Vector3D<unsigned int> &origSize, 
		  const Vector3D<Real> &origSpacing,
		  const Vector3D<Real> &origin) :
  mOrigScale(origSize, origSpacing, 1),
  mOrigin(origin),
  mCurScaleLevel(-1),
  mScaleVectorFields(false),
  mUseSincImageUpsample(false),
  mInitialScaleLevel(0)
{
}

MultiscaleManager::
MultiscaleManager(const Vector3D<unsigned int> &origSize, 
		  const Vector3D<Real> &origSpacing,
		  const Vector3D<Real> &origin,
		  const MultiscaleSettingsParam &param) :
  mOrigScale(origSize, origSpacing, 1),
  mOrigin(origin),
  mCurScaleLevel(-1),
  mScaleVectorFields(false),
  mUseSincImageUpsample(param.UseSincImageUpsample()),
  mInitialScaleLevel(0)
{
}

MultiscaleManager::
MultiscaleManager(const Vector3D<unsigned int> &origSize, 
		  const Vector3D<Real> &origSpacing,
		  const Vector3D<Real> &origin,
		  const MultiscaleParamInterface &param) :
  mOrigScale(origSize, origSpacing, 1),
  mOrigin(origin),
  mCurScaleLevel(-1),
  mScaleVectorFields(false),
  mUseSincImageUpsample(param.MultiscaleSettings().UseSincImageUpsample()),
  mInitialScaleLevel(0)
{
  unsigned int numLevels = param.GetNumberOfScaleLevels();
  for(unsigned int scaleLevel = 0; scaleLevel < numLevels; scaleLevel++){
    unsigned int downsampleFactor = param.GetScaleLevelBase(scaleLevel).ScaleLevel().DownsampleFactor();
    this->AddScaleLevel(downsampleFactor);
  }
}

MultiscaleManager::
~MultiscaleManager()
{
  for(unsigned int i=0;i<mOrigImages.size();i++){
    delete mOrigImages[i];
  }
  for(unsigned int i=0;i<mImagesFromOrig.size();i++){
    delete mImagesFromOrig[i];
  }
  for(unsigned int i=0;i<mOrigFields.size();i++){
    delete mOrigFields[i];
  }
  for(unsigned int i=0;i<mFieldsFromOrig.size();i++){
    delete mFieldsFromOrig[i];
  }
  for(unsigned int i=0;i<mImagesFromMinScale.size();i++){
    delete mImagesFromMinScale[i];
  }
  for(unsigned int i=0;i<mFieldsFromMinScale.size();i++){
    delete mFieldsFromMinScale[i];
  }
}

void 
MultiscaleManager::
GenerateScaleLevels(unsigned int nScaleLevels, 
		    unsigned int downsampleFactor)
{
  if(NumberOfScaleLevels() != 0){
    std::cerr << "Error, scale level generation requested with "
	      << NumberOfScaleLevels() << " scale levels already added"
	      << std::endl;
  }

  unsigned int curFactor = 1;
  for(unsigned int i=0; i<nScaleLevels; i++){
    AddScaleLevel(curFactor);
    curFactor *= downsampleFactor;
  }
}

void 
MultiscaleManager::
AddScaleLevel(unsigned int downsampleFactor)
{
  // create the new scale
  ScaleLevel scale(mOrigScale);
  scale.DownsampleFactor = downsampleFactor;
  scale.Size /= downsampleFactor;
  scale.Spacing *= downsampleFactor;

  // insert the new scale
  std::vector<ScaleLevel>::iterator it = mScaleLevels.begin();
  for(;it!=mScaleLevels.end();++it){

    if(it->DownsampleFactor < scale.DownsampleFactor) break;

    if(it->DownsampleFactor == scale.DownsampleFactor){
      throw MultiscaleException(__FILE__, __LINE__, "Error, cannot add multiple "
				"scale levels with the same downsample factor");
    }
    
  }
  mScaleLevels.insert(it,scale);
  
  mCurScaleLevel = mInitialScaleLevel;
}

RealImage*
MultiscaleManager::
GenerateBaseLevelImage()
{
  RealImage *image = new RealImage(mScaleLevels[0].Size,
				   mOrigin,
				   mScaleLevels[0].Spacing);
  mImagesFromMinScale.push_back(image);
  image->fill(0.0);
  return image;
}

RealImage*
MultiscaleManager::
GenerateBaseLevelImage(const RealImage *origImage)
{
  RealImage *orig = new RealImage(*origImage);
  mOrigImages.push_back(orig);
  RealImage *image = new RealImage();
  DownsampleToLevel(*orig, 0, *image);
  mImagesFromOrig.push_back(image);
  return image;
}

VectorField*
MultiscaleManager::
GenerateBaseLevelVectorField()
{
  VectorField *vf = new VectorField(mScaleLevels[0].Size);
  mFieldsFromMinScale.push_back(vf);
  vf->fill(Vector3D<Real>(0,0,0));
  return vf;
}

VectorField*
MultiscaleManager::
GenerateBaseLevelVectorField(const VectorField *origVectorField)
{
  VectorField *orig = new VectorField(*origVectorField);
  mOrigFields.push_back(orig);
  VectorField *vf = new VectorField();
  DownsampleToLevel(*orig, 0, *vf);
  mFieldsFromOrig.push_back(vf);
  return vf;
}

// void 
// MultiscaleManager::
// AttachAtCurrentScale(RealImage *imageToUpsample)
// {
//   if(imageToUpsample.getSize() != CurScaleSize()){
//     throw(MultiscaleException("Cannot attach image, size does not match current size", 
// 			      "MultiscaleManager::AttachAtCurrentScaleLevel"));
//   }

//   if(imageToUpsample.getSpacing() != CurScaleSpacing()){
//     throw(MultiscaleException("Cannot attach image, spacing does not match current spacing", 
// 			      "MultiscaleManager::AttachAtCurrentScaleLevel"));
//   }

//   mImagesFromMinScale.push_back(imageToUpsample);
// }

// void
// MultiscaleManager::
// AttachAtCurrentScale(VectorField *fieldToUpsample)
// {
//   if(fieldToUpsample.getSize() != CurScaleSize()){
//     throw(MultiscaleException("Cannot attach vector field, size does not match current size", 
// 			      "MultiscaleManager::AttachAtCurrentScaleLevel"));
//   }

//   mFieldsFromMinScale.push_back(fieldToUpsample);
// }

int 
MultiscaleManager::
NextScaleLevel(){
  SetScaleLevel(mCurScaleLevel+1);
  return mCurScaleLevel;
}

void
MultiscaleManager::
SetScaleLevel(unsigned int scaleLevel){

  if(scaleLevel < 0){
    throw MultiscaleException(__FILE__, __LINE__, "Requesting negative scale level");
    
  }
  if(scaleLevel >= mScaleLevels.size()){
    throw MultiscaleException(__FILE__, __LINE__, "Requesting scale level greater than lowest level");
  }

  // don't have to do anything if we're already at this scale level
  if((int)scaleLevel == mCurScaleLevel) return;

  mCurScaleLevel = scaleLevel;

  for(unsigned int i=0;i<mImagesFromOrig.size();i++){
    DownsampleToLevel(*mOrigImages[i],mCurScaleLevel,*mImagesFromOrig[i]);
  }
  for(unsigned int i=0;i<mFieldsFromOrig.size();i++){
    DownsampleToLevel(*mOrigFields[i],mCurScaleLevel,*mFieldsFromOrig[i]);
  }
  for(unsigned int i=0;i<mImagesFromMinScale.size();i++){
    UpsampleToLevel(*mImagesFromMinScale[i], mCurScaleLevel);
  }
  for(unsigned int i=0;i<mFieldsFromMinScale.size();i++){
    UpsampleToLevel(*mFieldsFromMinScale[i], mCurScaleLevel);
  }

}

bool
MultiscaleManager::
Detach(const RealImage *toDetach)
{
  // search through images downsampled from originals
  std::vector<RealImage *>::iterator it = mImagesFromOrig.begin();
  std::vector<const RealImage *>::iterator it_orig = mOrigImages.begin();
  for(;it!=mImagesFromOrig.end();++it,++it_orig){
    // pointer comparison
    if(toDetach == *it){
      mImagesFromOrig.erase(it);
      mOrigImages.erase(it_orig);
      return true;
    }
  }
  // search through images upsampled from a min-scale image
  it = mImagesFromMinScale.begin();
  for(;it!=mImagesFromMinScale.end();++it){
    // pointer comparison
    if(toDetach == *it){
      mImagesFromMinScale.erase(it);
      return true;
    }
  }
  return false;
}

bool
MultiscaleManager::
Detach(const VectorField *toDetach)
{
  // search through fields downsampled from originals.
  // have to detach both downsampled image and original
  std::vector<VectorField *>::iterator it = mFieldsFromOrig.begin();
  std::vector<const VectorField *>::iterator it_orig = mOrigFields.begin();
  for(;it!=mFieldsFromOrig.end();++it,++it_orig){
    // pointer comparison
    if(toDetach == *it){
      mFieldsFromOrig.erase(it);
      mOrigFields.erase(it_orig);
      return true;
    }
  }
  // search through images upsampled from a min-scale image
  it = mFieldsFromMinScale.begin();
  for(;it!=mFieldsFromMinScale.end();++it){
    // pointer comparison
    if(toDetach == *it){
      mFieldsFromMinScale.erase(it);
      return true;
    }
  }
  return false;

}

void 
MultiscaleManager::
UpsampleToLevel(RealImage &image, 
		unsigned int scaleLevel) const
{

  if(scaleLevel <= 0 || scaleLevel >= NumberOfScaleLevels()){
    throw AtlasWerksException(__FILE__, __LINE__, "Error, invalid scale level to upsample to: " + scaleLevel);
  }
  Vector3D<unsigned int> newSize = mScaleLevels[scaleLevel].Size;
  Vector3D<Real> newSpacing = mScaleLevels[scaleLevel].Spacing;

  std::cerr << "Upsampling image to " << newSize << ", spacing = " << newSpacing << std::endl;
  if(mUseSincImageUpsample){
    std::cerr << "Using sinc upsampling" << std::endl;
    ImageUtils::sincUpsample(image,
			     newSize);
  }else{
    ImageUtils::resampleNew(image,
			    mOrigin,
			    newSpacing,
			    newSize);
  }
}

void
MultiscaleManager::
UpsampleToLevel(VectorField &vf, 
		unsigned int scaleLevel) const
{
  if(scaleLevel <= 0 || scaleLevel >= NumberOfScaleLevels()){
    throw AtlasWerksException(__FILE__, __LINE__, "Error, invalid scale level to upsample to: " + scaleLevel);
  }
  Vector3D<unsigned int> origSize = vf.getSize();
  Vector3D<unsigned int> newSize = mScaleLevels[scaleLevel].Size;
  std::cerr << "Upsampling hField to " << newSize << std::endl;
  VectorField tmp(newSize);
  // don't rescale the vectors on upsample if we don't take spacing
  // into account when deforming images
  HField3DUtils::resampleNew(vf, tmp, newSize, HField3DUtils::BACKGROUND_STRATEGY_CLAMP, mScaleVectorFields);
  vf = tmp;
}

void 
MultiscaleManager::
DownsampleToLevel(const RealImage &orig, 
		  unsigned int scaleLevel,
		  RealImage &downsampled) const
{
  if(scaleLevel < 0 || scaleLevel >= NumberOfScaleLevels()){
    std::cerr << "Error, invalid scale level to upsample to:" << scaleLevel << std::endl;
    return;
  }

  Vector3D<unsigned int> newSize = mScaleLevels[scaleLevel].Size;
  Vector3D<Real> newSpacing = mScaleLevels[scaleLevel].Spacing;
  std::cerr << "Downsampling imge to " << newSize << std::endl;
  if(newSize == mOrigScale.Size){
    downsampled = orig;
  }else{
    ImageUtils::gaussianDownsample(orig,
				   downsampled,
				   newSize);
    downsampled.setSpacing(newSpacing);
  }
  
}

// UNIMPLEMENTED  
void 
MultiscaleManager::
DownsampleToLevel(const VectorField &orig, 
		  unsigned int scaleLevel,
		  VectorField &downsampled) const
{
  std::cerr << "DownsampleToLevel unimplemented for VectorFields" << std::endl;
  std::exit(-1);
}

// Just for testing, this is the old downsample routine (box filter style)
void BoxFilterDownsample(const RealImage &origImage,
			 RealImage &newImage,
			 const Vector3D<unsigned int> &newSz)
{

  Vector3D<unsigned int> origSz = origImage.getSize();
  Vector3D<unsigned int> factor(static_cast<unsigned int>(std::ceil(static_cast<Real>(origSz.x)/static_cast<Real>(newSz.x))),
				static_cast<unsigned int>(std::ceil(static_cast<Real>(origSz.y)/static_cast<Real>(newSz.y))),
				static_cast<unsigned int>(std::ceil(static_cast<Real>(origSz.z)/static_cast<Real>(newSz.z))));
  newImage.resize(newSz.x, newSz.y, newSz.z);

  unsigned int i, j, k;
  
  newImage.fill(0.0);

  unsigned int index = 0;
  unsigned int newIndex;
  Real divFactor = static_cast<Real>(factor.productOfElements());
  for(k = 0; k < origSz.z; k++)
    {
      for(j = 0; j < origSz.y; j++)
	{
	  for(i = 0; i < origSz.x; i++)
	    {
	      newIndex = i / factor.x + newSz.x * ((j / factor.y) + newSz.y * (k / factor.z));
	      newImage(newIndex) +=  origImage(index) / divFactor;
	      index++;
	    }
	}
    }
}


