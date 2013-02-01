#ifndef MultiScaleFluidWarp_h
#define MultiScaleFluidWarp_h

// 'identifier' : identifier was truncated to 'number' characters in the
// debug information
#ifdef WIN32
#pragma warning ( disable : 4786 )
#endif

#include "FluidWarp.h"
#include "FluidWarpParameters.h"
#include <cmath>
#include <list>
#include <string>
#include "Surface.h"

#include <ImageIO.h>
#include <ROI.h>

class MultiScaleFluidWarp
{
public:
  typedef float                  VoxelType;
  typedef Image<VoxelType>       ImageType;
  typedef ImageType*             ImagePointer;
  typedef ROI<int, unsigned int> ImageRegionType;
  typedef Vector3D<double>       ImageSizeType;
  typedef Vector3D<double>       ImageIndexType;
  typedef Array3D<float>         MaskType;

  MultiScaleFluidWarp();
  ~MultiScaleFluidWarp();

  void createWarp(ImagePointer& atlas,
		  ImagePointer& subject,
		  const char *atlasName,
		  const char *subjectName,
		  const VoxelType& atlasRescaleMinThreshold,
		  const VoxelType& atlasRescaleMaxThreshold,
		  const VoxelType& subjectRescaleMinThreshold,
		  const VoxelType& subjectRescaleMaxThreshold,
		  const ImageRegionType&  roi,
		  unsigned int numScaleLevels,
		  FluidWarpParameters fluidParameters[],
		  int reslice,
                  int deformationType,
                  MaskType mask);
  void shrink(ImagePointer& atlas,
              const char *atlasName,
              const VoxelType& atlasRescaleMinThreshold,
              const VoxelType& atlasRescaleMaxThreshold,
              const ImageRegionType&  roi,
              unsigned int numScaleLevels,
              FluidWarpParameters fluidParameters[],
              int reslice,
              int deformationType,
              MaskType mask);
  ImagePointer apply(ImagePointer& image);
  void apply(Surface& surface);
  void applyInv(Surface& surface);

  void loadHField(const char* fileName);
  void loadHField(const std::string& fileName) {loadHField(fileName.c_str());}
  void writeHField(const char* fileName);
  void writeHField(const std::string& fileName) {loadHField(fileName.c_str());}

  ImageRegionType 
  loadHFieldMETA(const char* fileName,
                 ImagePointer atlas, ImagePointer subject,
                 const char *atlasName = 0, const char *subjectName = 0);
  ImageRegionType 
  loadHFieldMETA(const std::string& fileName,
                 ImagePointer atlas, ImagePointer subject)
  { return loadHFieldMETA(fileName.c_str(), atlas, subject); }

  void writeHFieldMETA(const std::string& filenamePrefix);

  Vector3D<double> getAtlasOrigin() const
  { return _atlasOrigin; }

  Vector3D<double> getAtlasSpacing() const
  { return _atlasSpacing; }

  Vector3D<double> getHFieldOrigin() const
  { return _hFieldOrigin; }

  Vector3D<double> getHFieldSpacing() const
  { return _hFieldSpacing; }

  void setHField(const Array3D<Vector3D<float> >& hField,
		 const ImageRegionType& roi) {
    _hField = hField;
    _roi = roi;
  }

  const ImageRegionType& getROI() const
  { return _roi; }

  const Array3D< Vector3D<float> >& getHField() const { return _hField; }

  void getHField(Array3D<Vector3D<float> >& hField) const {
    hField = _hField;
  }

  //debug output
  void setAtlasROIChunk(ImagePointer* atlasROIChunk)
  {_atlasROIChunk = atlasROIChunk;}

  void setSubjectROIChunk(ImagePointer* subjectROIChunk)
  {_subjectROIChunk = subjectROIChunk;}

  ImagePointer* getAtlasROIChunk()
  {return _atlasROIChunk;}
  
  ImagePointer* getSubjectROIChunk()
  {return _subjectROIChunk;}
  //debug output

private:

  //debug
  ImagePointer* _atlasROIChunk;
  ImagePointer* _subjectROIChunk;
  //debug

  bool                        _verbose;
  bool                        _haveValidHField;
  bool                        _useOriginAndSpacing;

  std::string                 _atlasName;
  std::string                 _subjectName;
  std::vector<std::pair<int, FluidWarpParameters> >  
  _fluidParameters;
  ImageRegionType             _roi;
  Array3D<Vector3D<float> >   _hField, _hFieldInv;

  Vector3D<double>            _atlasOrigin;
  Vector3D<double>            _atlasSpacing;
  Vector3D<double>            _subjectOrigin;
  Vector3D<double>            _subjectSpacing;
  Vector3D<double>            _hFieldOrigin;
  Vector3D<double>            _hFieldSpacing;

  static void _convertImageToArray3DUChar(ImagePointer Image,
                                          Array3D<float>& array3DImage,
                                          const VoxelType& rescaleThresholdMin,
                                          const VoxelType& rescaleThresholdMax);

  static void _convertArray3DUCharToImage(Array3D<float>& array3DImage,
                                          ImagePointer Image);

  static void _downsampleImageBy2WithGaussianFilter(ImagePointer& image, ImagePointer& shrinkImage);


  static bool _checkImageCompatibility(ImagePointer atlas,
				       ImagePointer subject);

  static void _saveImage(ImagePointer image, const std::string& fileName);
  static void _resliceImage(ImagePointer& image, float newSpacing);

  static void _resliceMask(MaskType& mask, unsigned int newNb_Slice);

  static std::string  _parseValue(const std::string& str);

  void _createImageFromROI(const ImagePointer& inImage, 
                           ImagePointer& outImage);

#ifdef WIN32
  inline static double _pow(const double& a, const double& b) 
  {
    return pow(a,b);
  }
#else
  inline static double _pow(const double& a, const double& b) 
  {
    return std::pow(a,b);
  }
#endif
};

#endif
