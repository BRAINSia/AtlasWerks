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

#ifndef __MULTISCALE_MANAGER_H__
#define __MULTISCALE_MANAGER_H__

#ifndef SWIG

#include "Array3D.h"
#include "Array3DUtils.h"
#include "HField3DUtils.h"
#include "DataTypes/Image.h"
#include "ImageUtils.h"
#include "ApplicationUtils.h"

#include "ValueParam.h"
#include "CompoundParam.h"
#include "MultiParam.h"

#include "AtlasWerksTypes.h"

#include "ScaleLevelParamOrderingConstraint.h"

#endif // !SWIG

/**
 * Class defining exceptions thrown by the MultiscalseManager
 */
class MultiscaleException : public AtlasWerksException
{
public:
  MultiscaleException(const char *file="unknown file", int line=0,
		      const std::string& text = "undefined exception",
		      const std::exception *cause = NULL) :
    AtlasWerksException(file, line, text, cause)
  {
    mTypeDescription = "MultiscaleException";
  }
  
  virtual ~MultiscaleException() throw() {}
};

/**
 * Defines basic settings for up/downsampling images
 */
class MultiscaleSettingsParam : public CompoundParam {
public:
  MultiscaleSettingsParam(const std::string& name = "MultiscaleSettings", 
			  const std::string& desc = "General settings for multiscale manager", 
			  ParamLevel level = PARAM_COMMON)
    : CompoundParam(name, desc, level)
  {
    this->
      AddChild(ValueParam<bool>("UseSincImageUpsample",
				"Use sinc upsampling for images? (trilinear interpolation is the default)",
				PARAM_RARE,
				false));
  }
  
  ValueParamAccessorMacro(bool, UseSincImageUpsample)
  
  CopyFunctionMacro(MultiscaleSettingsParam)
  
};

/**
 * Just one parameter right now, the downsample factor
 */
class ScaleLevelParam : public CompoundParam {
public:
  ScaleLevelParam(const std::string& name = "ScaleLevel", 
		  const std::string& desc = "setting for a single scale level", 
		  ParamLevel level = PARAM_COMMON)
    : CompoundParam(name, desc, level)
  {
    this->
      AddChild(ValueParam<unsigned int>("DownsampleFactor",
					"factor by which to downsample images",
					PARAM_COMMON,
					1));
  }
  
  ValueParamAccessorMacro(unsigned int, DownsampleFactor)
  
  CopyFunctionMacro(ScaleLevelParam)

};

/**
 * Extend (subclass) this to hold per-scale-level settings, used by
 * MultiscaleParamBase
 */
class ScaleLevelSettingsParam : public CompoundParam {
public:
  ScaleLevelSettingsParam(const std::string& name = "ScaleLevelSettings", 
			  const std::string& desc = "setting for a single scale level", 
			  ParamLevel level = PARAM_COMMON)
    : CompoundParam(name, desc, level)
  {
    this->AddChild(ScaleLevelParam("ScaleLevel"));
  }

  ParamAccessorMacro(ScaleLevelParam, ScaleLevel)

  CopyFunctionMacro(ScaleLevelSettingsParam)

};

class MultiscaleParamInterface {
public:
  virtual ~MultiscaleParamInterface(){}
  virtual unsigned int GetNumberOfScaleLevels() const = 0;
  virtual MultiscaleSettingsParam& MultiscaleSettings() = 0;
  virtual const MultiscaleSettingsParam& MultiscaleSettings() const = 0;
  virtual ScaleLevelSettingsParam &GetScaleLevelBase(int i) = 0;
  virtual const ScaleLevelSettingsParam &GetScaleLevelBase(int i) const = 0;
};

/**
 * Base class for per-scale-level settings 
 */  
template <class ScaleLevelSettingsType>
class MultiscaleParamBase : 
  public CompoundParam, 
  public MultiscaleParamInterface 
{
  
public:
  /**
   * Constructor, pass it the template version of your ScaleLevelSettingsParam
   */
  MultiscaleParamBase(ScaleLevelSettingsType param,
		  const std::string& name = "Multiscale", 
		  const std::string& desc = "Parameter containing multiple scale levels", 
		  ParamLevel level = PARAM_COMMON)
    : CompoundParam(name, desc, level)
  {
    mScaleLevelName = param.GetName();
    this->
      AddChild(MultiscaleSettingsParam());
    this->AddChild(MultiParam<ScaleLevelSettingsType>(param));
    this->ScaleLevel().SetConstraint(new ScaleLevelParamOrderingConstraint<ScaleLevelSettingsType>());
  }

  /** Copy constructor, necessary for copying mScaleLevelName */
  MultiscaleParamBase(const MultiscaleParamBase &other)
    : CompoundParam(other),
      mScaleLevelName(other.mScaleLevelName)
  {
  }

  ParamAccessorMacro(MultiscaleSettingsParam, MultiscaleSettings)

  /** Accessor very similar to ParamAccessorMacro, but uses
      mScaleLevelName as child name to retrieve */
  MultiParam<ScaleLevelSettingsType> &ScaleLevel(){
    ParamBase *baseParam = GetChild(mScaleLevelName);
    if(!baseParam){
      throw ParamException(__FILE__, __LINE__,
			   "ParamAccessorException : No Child Named " + mScaleLevelName);
    } 
    MultiParam<ScaleLevelSettingsType> *param = dynamic_cast<MultiParam<ScaleLevelSettingsType>*>(baseParam);
    if(!param){
      throw ParamException(__FILE__, __LINE__,
			   "ParamAccessorException : Cannot cast " + mScaleLevelName 
			   + " to template type ScaleLevelSettingsType*");
    }
    return *param;
  }
  
  /** const accessor very similar to ParamAccessorMacro, but uses
      mScaleLevelName as child name to retrieve */
  const MultiParam<ScaleLevelSettingsType> &ScaleLevel() const {
    const ParamBase *baseParam = GetChild(mScaleLevelName);
    if(!baseParam){
      throw ParamException(__FILE__, __LINE__,
			   "ParamAccessorException : No Child Named " + mScaleLevelName);
    } 
    const MultiParam<ScaleLevelSettingsType> *param = dynamic_cast<const MultiParam<ScaleLevelSettingsType>*>(baseParam);
    if(!param){
      throw ParamException(__FILE__, __LINE__,
			   "ParamAccessorException : Cannot cast " + mScaleLevelName 
			   + " to template type const ScaleLevelSettingsType*");
    }
    return *param;
  }
  
  unsigned int GetNumberOfScaleLevels() const { return ScaleLevel().GetNumberOfParsedParams(); }
  ScaleLevelSettingsType &GetScaleLevel(int i){ return ScaleLevel().GetParsedParam(i); }
  const ScaleLevelSettingsType &GetScaleLevel(int i) const { return ScaleLevel().GetParsedParam(i); }
  ScaleLevelSettingsParam &GetScaleLevelBase(int i){ return ScaleLevel().GetParsedParam(i); }
  const ScaleLevelSettingsParam &GetScaleLevelBase(int i) const { return ScaleLevel().GetParsedParam(i); }
  
  MultiscaleParamBase&
  operator=(const MultiscaleParamBase &other)
  {
    if(this != &other){
      this->CompoundParam::operator=(other);
      this->mScaleLevelName = other.mScaleLevelName;
    }
    return *this;
  }

  CopyFunctionMacro(MultiscaleParamBase)
  
protected:
  
  std::string mScaleLevelName;
};

/**
 * MultiscaleManager handles up/downsampling of images and vector
 * fields necessary for multiscale optimization
 */
class MultiscaleManager{

protected:
  
  class ScaleLevel {
  public:
    ScaleLevel(const Vector3D<unsigned int> inSize, const Vector3D<Real> inSpacing, unsigned int inFactor)
      : Size(inSize),
	Spacing(inSpacing),
	DownsampleFactor(inFactor)
    {}
    
    SizeType Size;
    SpacingType Spacing;
    unsigned int DownsampleFactor;
  };
    
public:

  /**
   * Create a new MultiscaleManager for images/fields with the given
   * native size and spacing.
   */
  MultiscaleManager(const Vector3D<unsigned int> &origSize, 
		    const Vector3D<Real> &origSpacing,
		    const Vector3D<Real> &origin = Vector3D<Real>(0,0,0));

  /**
   * Create a new MultiscaleManager for images/fields with the given
   * native size and spacing. Settings are initialized from
   * MultiscaleSettingsParam
   */
  MultiscaleManager(const Vector3D<unsigned int> &origSize, 
		    const Vector3D<Real> &origSpacing,
		    const Vector3D<Real> &origin,
		    const MultiscaleSettingsParam &param);

  /**
   * Create a new MultiscaleManager for images/fields with the given
   * native size and spacing. Settings are initialized from
   * MultiscaleParamInterface, including scale levels.
   */
  MultiscaleManager(const Vector3D<unsigned int> &origSize, 
		    const Vector3D<Real> &origSpacing,
		    const Vector3D<Real> &origin,
		    const MultiscaleParamInterface &param);
  
  virtual ~MultiscaleManager();

  /** 
   * ScaleVectorFields controls whether vector fields are scaled on
   * up/downsample so that vector*spacing maintains the same
   * magnitude before and afer scaling.  False by default.
   */
  void SetScaleVectorFields(bool scale){ mScaleVectorFields = scale; }
  bool GetScaleVectorFields() const { return mScaleVectorFields; }

  /** 
   * Should we use sinc interpolation for upsampling images?  False by
   * default.
   */
  void SetUseSincImageUpsample(bool sinc){ mUseSincImageUpsample = sinc; }
  bool GetUseSincImageUpsample() const { return mUseSincImageUpsample; }

  /** 
   * Used to automatically generate nScaleLevels scale levels, with
   * the given downsample factor between levels. 
   */
  void GenerateScaleLevels(unsigned int nScaleLevels, unsigned int downsampleFactor = 2);

  /** 
   * Add a scale level with uniform downsampling
   */
  void AddScaleLevel(unsigned int downsampleFactor);

  /** generate an empty image at the base level size, which will be
      upsampled to higher scale levels */
  RealImage *GenerateBaseLevelImage();
  /** generate a downsampled version of the image given.  When the
      scale level is changed, the image will again be downsampled to
      the new level from the original image. */
  RealImage *GenerateBaseLevelImage(const RealImage *origImage);
  /** generate an empty vector field at the base level size, which
      will be upsampled to higher scale levels */
  VectorField *GenerateBaseLevelVectorField();
  /** UNIMPLEMENTED */
  VectorField *GenerateBaseLevelVectorField(const VectorField *origVectorField);
  
  /** Get the initial scale level index, zero unless otherwise set */
  unsigned int GetInitialScaleLevel() const { return mInitialScaleLevel; }
  /** Set the initial scale level index, zero by default */
  void SetInitialScaleLevel(unsigned int init) { mInitialScaleLevel = init; }

  /** Get the number of scale levels */
  unsigned int NumberOfScaleLevels() const { return mScaleLevels.size(); }

  /** Get the current scale level.  Scale levels are ordered from
      min-resolution (scale level 0) to full-resolution (scale level
      NumberOfScaleLevels()-1) 
  */
  int CurScaleLevel() const { return mCurScaleLevel; }

  /** Is this the initial (min-resolution) scale level? */
  bool InitialScaleLevel() const { return (static_cast<unsigned int>(mCurScaleLevel) == mInitialScaleLevel); }

  /** Is this the final (full-resolution) scale level? */
  bool FinalScaleLevel() const { return (mCurScaleLevel == ((int)NumberOfScaleLevels())-1); }

  /** Get the image/vector field dimensions at the current scale level */
  SizeType CurScaleSize() const { 
    return mScaleLevels[mCurScaleLevel].Size ;
  }

  /** Get the image/vector field spacing at the current scale level */
  SpacingType CurScaleSpacing() const { 
    return mScaleLevels[mCurScaleLevel].Spacing; 
  }

  /** 
   * Resample all images/vector fields to the next scale level.
   * Returns the new scale level (where scale level 0 is the
   * full-sized image).
   */
  int NextScaleLevel();

  /**
   * Resample all images/vector fields to the given scale level.
   */
  virtual void SetScaleLevel(unsigned int scaleLevel);

  /** 
   * Normally images/vector fields generated by the MultiscaleManager
   * are automatically deleted by the destructor.  These methods
   * remove the structure from management (future changes to scale
   * level will not affect it) and removes it from deletion when the
   * MultiscaleManager is destroyed.
   */
  bool Detach(const RealImage *toDetach);
  bool Detach(const VectorField *toDetach);
  
  /**
   * Upsample the given image to the given scale level
   */
  void UpsampleToLevel(RealImage &image, 
		       unsigned int scaleLevel) const;

  /**
   * Upsample the given vector field to the given scale level
   */
  void UpsampleToLevel(VectorField &vf, 
		       unsigned int scaleLevel) const;

  /**
   * Downsample the given image to the given scale level
   */
  void DownsampleToLevel(const RealImage &orig, 
			 unsigned int scaleLevel,
			 RealImage &downsampled) const;
  /**
   * UNIMPLEMENTED
   */
  void DownsampleToLevel(const VectorField &orig, 
			 unsigned int scaleLevel,
			 VectorField &downsampled) const;

protected:

  ScaleLevel mOrigScale;
  Vector3D<Real> mOrigin;

  int mCurScaleLevel;
  /**
   * Controls whether vector fields are scaled so that vector*spacing
   * remains constant.  False by default.
   */
  bool mScaleVectorFields;
  
  /** 
   * Specifies whether we should use sinc interpolation for upsampling
   * images.  False by default.
   */
  bool mUseSincImageUpsample;

  unsigned int mInitialScaleLevel;
  
  std::vector<ScaleLevel > mScaleLevels;
  
  std::vector<const RealImage *> mOrigImages;
  std::vector<RealImage *> mImagesFromOrig;
  std::vector<const VectorField *> mOrigFields;
  std::vector<VectorField *> mFieldsFromOrig;
  std::vector<RealImage *> mImagesFromMinScale;
  std::vector<VectorField *> mFieldsFromMinScale;
  
};

#endif
