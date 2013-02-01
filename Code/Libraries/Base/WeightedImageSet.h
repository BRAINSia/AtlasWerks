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


#ifndef __WeightedImageParam_H__
#define __WeightedImageParam_H__


#include "AtlasWerksTypes.h"
#include "CompoundParam.h"
#include "ValueParam.h"
#include "MultiParam.h"

#ifndef SWIG

#include <vector>
#include "AffineTransform3D.h"
#include <itkTransformFileReader.h>
#include <itkTransformBase.h>
#include <itkAffineTransform.h>

typedef itk::AffineTransform<float, 3> ItkAffineTransform;
typedef AffineTransform3D<Real> RealAffineTransform;

#endif // !SWIG

/** Weighted image parameter class */
class WeightedImageParam : public CompoundParam {
public:
  WeightedImageParam(const std::string& name = "WeightedImage",
		     const std::string& desc = "A weighted input image file",
		     ParamLevel level = PARAM_COMMON)
    : CompoundParam(name, desc, level)
  {
    this->AddChild(ValueParam<std::string>("Filename", "input image filename", PARAM_REQUIRED, ""));
    this->AddChild(ValueParam<Real>("Weight", "input image weight", PARAM_COMMON, 1.0));
    this->AddChild(ValueParam<std::string>("Transform", "filename of affine transform image", PARAM_COMMON, ""));
    this->AddChild(ValueParam<bool>("ItkTransform", "is this an ITK-style transform file vs. an AffineTransform3D-style file?", PARAM_COMMON, false));
  }
  
  ValueParamAccessorMacro(std::string, Filename)
  ValueParamAccessorMacro(Real, Weight)
  ValueParamAccessorMacro(std::string, Transform)
  ValueParamAccessorMacro(bool, ItkTransform)
  
  CopyFunctionMacro(WeightedImageParam)
};

class WeightedFileFormatParam : public CompoundParam {
public:
  WeightedFileFormatParam(const std::string& name = "FileFormatString", 
			const std::string& desc = "printf-style format string", 
			ParamLevel level = PARAM_COMMON)
    : CompoundParam(name, desc, level)
  {
    this->AddChild(ValueParam<std::string>("FormatString", "filename format, expects single integer format (%d or %0d)", PARAM_REQUIRED, ""));
    this->AddChild(ValueParam<unsigned int>("Base", "Minimum image index", PARAM_COMMON, 0));
    this->AddChild(ValueParam<unsigned int>("NumFiles", "Number of files to read in (filnames from Base to NumImages-1)", PARAM_REQUIRED, 0));
    this->AddChild(ValueParam<Real>("Weight", "Weight given to each of the input images", PARAM_RARE, 1.0));
  }
  
  ValueParamAccessorMacro(std::string, FormatString)
  ValueParamAccessorMacro(unsigned int, Base)
  ValueParamAccessorMacro(unsigned int, NumFiles)
  ValueParamAccessorMacro(Real, Weight)
  
  CopyFunctionMacro(WeightedFileFormatParam)
  
};

class WeightedImageSetParam : public CompoundParam {
public:
  WeightedImageSetParam(const std::string& name = "WeightedImageSet", 
			const std::string& desc = "Specify a set of input images, possibly weighted", 
			ParamLevel level = PARAM_COMMON)
    : CompoundParam(name, desc, level)
  {
    this->AddChild(WeightedFileFormatParam("InputImageFormatString"));
    this->AddChild(MultiParam<WeightedImageParam>(WeightedImageParam("WeightedImage")));
    this->AddChild(ValueParam<bool>("ScaleImageWeights", "If true, scale the image weights to 1.0", PARAM_COMMON, true));
  }
  
  ParamAccessorMacro(WeightedFileFormatParam, InputImageFormatString)
  ParamAccessorMacro(MultiParam<WeightedImageParam>, WeightedImage)
  ValueParamAccessorMacro(bool, ScaleImageWeights)
  
  CopyFunctionMacro(WeightedImageSetParam)
  
};

class WeightedImageSet {
public:
  WeightedImageSet(const WeightedImageSetParam &param);
  
  /**
   * Load the images and transforms.  If base and num are unspecified,
   * load all images.  Otherwise load a subset, from image 'base'
   * through image 'base+num-1'.
   */
  void Load(bool verbose=false, int base=0, int num=-1);
  void Clear();
  /**
   * Before Load(...) is called, returns the total number of images
   * specified in parameter file.  After Load(...), returns the number
   * of loaded images (may be a subset of the images specified in
   * parameter file)
   */
  unsigned int NumImages() const { return mImageNames.size(); }
  bool HasTransforms() const { return (mTransforms.size() > 0); }
  std::string GetImageName(int i) const;
  std::vector<std::string> GetImageNameVec() const;
  RealImage *GetImage(int i);
  const RealImage *GetImage(int i) const;
  std::vector<RealImage*> GetImageVec();
  std::vector<const RealImage*> GetImageVec() const;
  Real GetWeight(int i) const;
  std::vector<Real> GetWeightVec() const;
  RealAffineTransform* GetTransform(int i);
  const RealAffineTransform* GetTransform(int i) const;
  std::vector<RealAffineTransform*> GetTransformVec();
  std::vector<const RealAffineTransform*> GetTransformVec() const;

  /**
   * Loop over all images and get the size.  Return the size if all
   * images are equal size, throw an exception otherwise.  'Load()'
   * must have been called prior to this call.
   */
  SizeType GetImageSize() const;

  /**
   * Loop over all images and get the spacing.  Return the spacing if all
   * images have the same spacing, throw an exception otherwise.  'Load()'
   * must have been called prior to this call.
   */
  SpacingType GetImageSpacing() const;

  /**
   * Loop over all images and get the origin.  Return the origin if all
   * images have the same origin, throw an exception otherwise.  'Load()'
   * must have been called prior to this call.
   */
  OriginType GetImageOrigin() const;

  
protected:
  
  WeightedImageSetParam mParam;
  std::vector<std::string> mImageNames;
  std::vector<std::string> mTransformNames;
  std::vector<Real> mImageWeights;
  std::vector<RealImage*> mImages;
  std::vector<RealAffineTransform*> mTransforms;
  std::vector<bool> mItkStyleTransforms;

};

#endif // __WeightedImageParam_H__
