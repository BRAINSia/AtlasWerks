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

#ifndef __INPUT_IMAGE_PREPROCESSOR_H__
#define __INPUT_IMAGE_PREPROCESSOR_H__

#include <iostream>
#include <sys/stat.h>

#include "AtlasWerksTypes.h"
#include "CompoundParam.h"
#include "ValueParam.h"
#include "MultiParam.h"

/**
 * Parameter settings for intensity window
 */
class IntensityWindowParam : public CompoundParam {
public:
  IntensityWindowParam(const std::string& name = "IntensityWindow",
		       const std::string& desc =
		       "Intensity window used for rescaling (image min/max used "
		       "if no intensity window specified)", 
		       ParamLevel level=PARAM_COMMON,
		       bool defaultToRescale = true)
    : CompoundParam(name, desc, level)
  {
    this->AddChild(ValueParam<bool>("RescaleIntensities", "Perform intensity rescaling?", PARAM_COMMON, defaultToRescale));
    this->AddChild(ValueParam<bool>("UseInputIntensityWindow", "Use this intensity window instead of image min/max intensity", PARAM_COMMON, false));
    this->AddChild(ValueParam<Real>("InputWindowMin", "input window min", PARAM_COMMON, 0.0));
    this->AddChild(ValueParam<Real>("InputWindowMax", "input window max", PARAM_COMMON, 1.0));
    this->AddChild(ValueParam<Real>("OutputWindowMin", "output window min", PARAM_COMMON, 0.0));
    this->AddChild(ValueParam<Real>("OutputWindowMax", "output window max", PARAM_COMMON, 1.0));
  }

  ValueParamAccessorMacro(bool, RescaleIntensities)
  ValueParamAccessorMacro(bool, UseInputIntensityWindow)
  ValueParamAccessorMacro(Real, InputWindowMin)
  ValueParamAccessorMacro(Real, InputWindowMax)
  ValueParamAccessorMacro(Real, OutputWindowMin)
  ValueParamAccessorMacro(Real, OutputWindowMax)

  CopyFunctionMacro(IntensityWindowParam)
};

//
// ################ IntensityWindow ################
//

/**
 * Class for performing intensity windowing on input images.  If input
 * window is not specified, the min,max range of the image to be
 * processed is used.
 */
class IntensityWindow {
public:
  IntensityWindow();
  IntensityWindow(const IntensityWindowParam &param);
  
  void Rescale(RealImage &image) const;
  
  void SetParams(const IntensityWindowParam &param);
  void GetParams(IntensityWindowParam &param) const;

  bool GetRescaleIntensities() const { return mRescaleIntensities; }
  void SetRescaleIntensities(bool rescale) { mRescaleIntensities = rescale; }

  bool GetVerbose() const { return mVerbose; }
  void SetVerbose(bool verbose) { mVerbose = verbose; }

  bool GetUseIntensityWindow() const { return mUseInputIntensityWindow; }
  void SetUseIntensityWindow(bool useInputWindow) { mUseInputIntensityWindow = useInputWindow; }

  void GetInputWindow(Real &min, Real &max) const { 
    min = mInputWindowMin;
    max = mInputWindowMax;
  }
  void SetInputWindow(Real min, Real max){ 
    mInputWindowMin = min;
    mInputWindowMax = max;
  }
  void GetOutputWindow(Real &min, Real &max) const { 
    min = mOutputWindowMin;
    max = mOutputWindowMax;
  }
  void SetOutputWindow(Real min, Real max){ 
    mOutputWindowMin = min;
    mOutputWindowMax = max;
  }


protected:
  bool mVerbose;
  bool mRescaleIntensities;
  bool mUseInputIntensityWindow;
  Real mInputWindowMin;
  Real mInputWindowMax;
  Real mOutputWindowMin;
  Real mOutputWindowMax;
};

//
// ################ TukeyWindowParam ################
//

class TukeyWindowParam : public CompoundParam {
public:
  TukeyWindowParam(const std::string& name = "TukeyWindow",
		   const std::string& desc = "Window used to suppress border of image", 
		   ParamLevel level = PARAM_RARE,
		   bool defaultToWindow = true)
    : CompoundParam(name, desc, level)
  {
    this->AddChild(ValueParam<bool>("DoWindowing", "Apply the Tukey window?", PARAM_COMMON, defaultToWindow));
    this->AddChild(ValueParam<unsigned int>("Width", "Width of the border region of the filter, in pixels", PARAM_COMMON, 5));
  }
  
  ValueParamAccessorMacro(bool, DoWindowing)
  ValueParamAccessorMacro(unsigned int, Width)

  CopyFunctionMacro(TukeyWindowParam)
};

//
// ################ TukeyWindow ################
//

/**
 * Class to apply a Tukey window to an image, suppresses border.
 */
class TukeyWindow {

public:
  TukeyWindow();
  TukeyWindow(const TukeyWindowParam &param);
  void SetParams(const TukeyWindowParam &param);
  void GetParams(TukeyWindowParam &param) const;
  void ApplyWindow(RealImage &image);
protected:
  bool mApplyWindow;
  unsigned int mWidth;
  Real *mBorderMask;
};

//
// ################ GaussianBlurParam ################
//

class GaussianBlurParam : public CompoundParam {
public:
  GaussianBlurParam(const std::string& name = "GaussianBlur",
		    const std::string& desc = "Blur input image", 
		    ParamLevel level = PARAM_RARE)
    : CompoundParam(name, desc, level)
  {
    this->AddChild(ValueParam<float>("Sigma", "Std. dev. of gaussian, 0.0 for no blurring", level, 0.0));
  }
  
  ValueParamAccessorMacro(float, Sigma)

  CopyFunctionMacro(GaussianBlurParam)
};

class GaussianBlur {
public:
  GaussianBlur();
  GaussianBlur(const GaussianBlurParam &param);
  void SetParams(const GaussianBlurParam &param);
  void GetParams(GaussianBlurParam &param) const;
  void Blur(RealImage &image);
protected:
  Real mSigma;
};

//
// ################ ImagePreprocessorParam ################
//

class ImagePreprocessorParam : public CompoundParam {
public:
  ImagePreprocessorParam(const std::string& name = "ImagePreprocessing",
			 const std::string& desc = "A weighted input image file",
			 ParamLevel level = PARAM_COMMON)
    : CompoundParam(name, desc, level)
  {
    this->AddChild(IntensityWindowParam("IntensityWindow"));
    this->AddChild(TukeyWindowParam("TukeyWindow", "Settings for Tukey Window (border supression)", PARAM_RARE, false));
    this->AddChild(GaussianBlurParam("GaussianBlur"));
    this->AddChild(ValueParam<bool>("SetUnitSpacing", "Set the spacing of input images to (1,1,1) (no resampling)", PARAM_COMMON, false));
    this->AddChild(ValueParam<bool>("SetZeroOrigin", "Set the origin of input images to (0,0,0)", PARAM_COMMON, false));
  }
  ParamAccessorMacro(IntensityWindowParam, IntensityWindow)
  ParamAccessorMacro(TukeyWindowParam, TukeyWindow)
  ParamAccessorMacro(GaussianBlurParam, GaussianBlur)
  ValueParamAccessorMacro(bool, SetUnitSpacing)
  ValueParamAccessorMacro(bool, SetZeroOrigin)

  CopyFunctionMacro(ImagePreprocessorParam)
};

//
// ################ ImagePreprocessor ################
//

/**
 * Class to preprocess input images using intensity windowing, tukey
 * window, and other options.
 */
class ImagePreprocessor {
public:
  ImagePreprocessor();
  ImagePreprocessor(const ImagePreprocessorParam &param);
  void SetParams(const ImagePreprocessorParam &param);
  void GetParams(ImagePreprocessorParam &param) const;
  void Process(RealImage &image, std::string imageID = "");
  void Process(std::vector<RealImage*> &imageVec, std::vector<std::string> imageIDVec = std::vector<std::string>());

  void SetImageSize(const Vector3D<unsigned int> size);
  void SetImageOrigin(const Vector3D<Real> origin);
  void SetImageSpacing(const Vector3D<Real> spacing);
  const Vector3D<unsigned int> &GetImageSize() const;
  const Vector3D<Real> &GetImageOrigin() const;
  const Vector3D<Real> &GetImageSpacing() const;

protected:
  IntensityWindow mIntensityWindow;
  TukeyWindow mTukeyWindow;
  GaussianBlur mGaussianBlur;
  bool mSetUnitSpacing;
  bool mSetZeroOrigin;
  Vector3D<unsigned int> *mSize;
  Vector3D<Real> *mOrigin;
  Vector3D<Real> *mSpacing;
  
};

#endif
