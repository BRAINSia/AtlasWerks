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

#ifndef SymmetricFluidWarp_h
#define SymmetricFluidWarp_h

// 'identifier' : identifier was truncated to 'number' characters in the
// debug information
#ifdef WIN32
#pragma warning ( disable : 4786 )
#endif

#include <vector>
#include <sstream>
#include "FluidWarpParameters.h"
#include "Array3D.h"
#include "Array3DUtils.h"
#include "Array3DIO.h"
#include "Timer.h"
#include "fftw3.h"

/**
 * Single-threaded implementations of many fluid warping algorithms.
 * Some of these have been superceded by other multithreaded
 * implementations.
 *
 * Note: These functions take and produce transformations h(x) such that
 * the deformed image Idef = I(h(x)), not Idef = I(x+u(x)).
 */
class FluidWarp
{
public:
  typedef FluidWarpParameters                    Parameters;
  typedef float                                  VoxelType;
  typedef float                                  CoordinateType;
  typedef Array3D<VoxelType>                     ImageType;
  typedef Array3D<Vector3D<CoordinateType> >     VectorField;
  typedef Array3D<float>                         MaskType;

  enum OutputMode {FW_OUTPUT_MODE_NORMAL, 
		   FW_OUTPUT_MODE_VERBOSE, 
		   FW_OUTPUT_MODE_SILENT};
  enum OutputType {FW_OUTPUT_TYPE_TIMING, 
		   FW_OUTPUT_TYPE_STANDARD, 
		   FW_OUTPUT_TYPE_VERBOSE,
		   FW_OUTPUT_TYPE_ERROR};

  FluidWarp();
  ~FluidWarp();
  FluidWarp& operator=(const FluidWarp& rhs);

  //
  // set up algorithm reporting output
  // 

  // what to output to the console
  void setOutputMode(OutputMode mode);

  // what type of data to write
  void setFilePrefix(const char* prefix);  
  void setWriteVolumes(bool shouldWrite);
  void setWriteXSlices(bool shouldWrite);
  void setWriteYSlices(bool shouldWrite);
  void setWriteZSlices(bool shouldWrite);
  void setWriteErrorSpreadsheet(bool shouldWrite);

  // when to write (for volumes and slices)
  void setWritePerIter(unsigned int numIters);
  void setWritePerRMSDecline(double rmsDecline);
  void setWriteOneAtlasPerIter(bool shouldWriteOnlyOne);
  void setXSlice(unsigned int sliceNumber);
  void setYSlice(unsigned int sliceNumber);
  void setZSlice(unsigned int sliceNumber);

  // what to write
  void setWriteDeformedImageFiles(bool shouldWrite);
  void setWriteAtlasFiles(bool shouldWrite);
  void setWriteJacobianFiles(bool shouldWrite);
  void setWriteDivergenceFiles(bool shouldWrite);
  void setWriteCurrentHFieldFiles(bool shouldWrite);
  void setWriteCurrentHInvFieldFiles(bool shouldWrite);
  void setWriteLogFile(bool shouldWrite);

  void setFFTWMeasure(bool shouldMeasure);
  void setFFTWNumberOfThreads(int numThreads);
  int  getFFTWNumberOfThreads() { return this->_FFTWNumberOfThreads; }

  //
  // set image origin and spacing
  //
  void setImageOrigin(const Vector3D<double>& origin);
  void setImageSpacing(const Vector3D<double>& spacing);

  void setUpdateAverageAfterEverySubIteration(bool);
  bool getUpdateAverageAfterEverySubIteration() const 
    { return this->_updateAverageAfterEverySubIteration; }

  //
  // generate h fields
  //

  /**
   * Generate a deformation that shrinks dark regions of the image.
   * Alternately shrink light regions if shrinkLightRegions = true.  h
   * and hinv fields must initialized to the same size as the image.
   */
  void shrinkRegion(const ImageType& image,
		    const Parameters& parameters,
		    VectorField& h,
		    bool shrinkLightRegions = false);

  /**
   * Generate a deformation that shrinks dark regions of the image.
   * Alternately shrink light regions if shrinkLightRegions = true.  h
   * and hinv fields must initialized to the same size as the image.
   */
  void shrinkRegion(const ImageType& image,
		    const Parameters& parameters,
		    VectorField& h,
		    VectorField& hinv,
		    bool shrinkLightRegions = false);

  void elasticShrinkRegionWithMask(const ImageType& image,
                                   const Parameters& parameters,
                                   MaskType& mask,
                                   VectorField& h,
                                   bool shrinkLightRegions = true);

  void shrinkRegionForward(const ImageType& image,
                           const Parameters& parameters,
                           VectorField& h,
                           bool shrinkLightRegions = false);

  /**
   * The 'computeHField*' functions compute a vector field
   * asymmetric version
   *
   * - compute h field such that fixed(x) corresponds with moving(h(x)) given
   *   initial h and hinv fields
   * - fixed and moving images must be the same size
   * - h and hinv fields must be the same size as the images
   * - h and hinv, when passed in, should either both be the identity 
   *   or be specialized initial h and hinv fields
   * - at the end of the routine, h and hinv will hold the final h and hinv
   *   fields; the initial h field will be overwritten
   *
   */
  void computeHFieldAsymmetric(const ImageType& fixed,
			       const ImageType& moving,
			       const Parameters& parameters,
			       VectorField& h);
  
  void computeHFieldAsymmetric(const ImageType& fixed,
			       const ImageType& moving,
			       const Parameters& parameters,
			       VectorField& h,
			       VectorField& hinv);

  void computeHFieldElastic(const ImageType& fixed,
                            const ImageType& moving,
                            const Parameters& parameters,
                            VectorField& h);
  

  void computeHFieldElasticWithMask(const ImageType& fixed,
                                    const ImageType& moving,
                                    MaskType& mask,
                                    const Parameters& parameters,
                                    VectorField& h);

  void computeHField2Symmetric(const ImageType& i1,
			       const ImageType& i2,
			       const Parameters& parameters,
			       VectorField& h1,
			       VectorField& h2);

  void computeHField2Symmetric(const ImageType& i1,
			       const ImageType& i2,
			       const Parameters& parameters,
			       VectorField& h1,
			       VectorField& h2,
			       VectorField& h1inv,
			       VectorField& h2inv);

  void computeHFieldNSymmetric(unsigned int numImages,
			       const ImageType* const* images,
			       const Parameters& parameters,
			       ImageType& iHat,
			       VectorField** h);

  void computeHFieldNSymmetric(unsigned int numImages,
			       const ImageType* const* images,
                               const double* imageWeights,
			       const Parameters& parameters,
			       ImageType& iHat,
			       VectorField** h);

  void computeHFieldNSymmetric(unsigned int numImages,
			       const ImageType* const* images,
			       const Parameters& parameters,
			       ImageType& iHat,
			       VectorField** h,
			       VectorField** hinv);

  void computeHFieldNSymmetric(unsigned int numImages,
			       const ImageType* const* images,
                               const double* imageWeights,
			       const Parameters& parameters,
			       ImageType& iHat,
			       VectorField** h,
			       VectorField** hinv);
  
private:
  bool _updateAverageAfterEverySubIteration;
  
  //
  // member vars for reporting/output
  //
  OutputMode   _outputMode;
  std::string  _filePrefix;
  bool         _writeVolumes;
  bool         _writeXSlices;
  bool         _writeYSlices;
  bool         _writeZSlices;
  bool         _writeErrorSpreadsheet;
  bool         _writePerIter;
  bool         _writePerRMSDecline;
  unsigned int _writePerIterSize;
  double       _writePerRMSDeclineSize;
  bool         _writeOneAtlasPerIter;
  unsigned int _xSliceNumber;
  unsigned int _ySliceNumber;
  unsigned int _zSliceNumber;
  bool         _writeDeformedImageFiles;
  bool         _writeAtlasFiles;
  bool         _writeJacobianFiles;
  bool         _writeDivergenceFiles;
  bool         _writeCurrentHFieldFiles;
  bool         _writeCurrentHInvFieldFiles;
  bool         _writeLogFile;

  Vector3D<double> _imageOrigin;
  Vector3D<double> _imageSpacing;

  bool         _FFTWDoMeasure;
  int          _FFTWNumberOfThreads;

  std::ofstream _errorSpreadsheetOfstream;
  unsigned int  _lastWrittenIter;
  double        _lastWrittenRMSError;

  Timer         _localTimer;

  //
  // look up table to for Linv computation
  //
  struct LUT
  {
    std::vector<CoordinateType> cosWX, cosWY, cosWZ;
    std::vector<CoordinateType> sinWX, sinWY, sinWZ;

    LUT()
      : cosWX(0), cosWY(0), cosWZ(0),
	sinWX(0), sinWY(0), sinWZ(0)
    {}

    LUT(unsigned int xSize, 
	unsigned int ySize, 
	unsigned int zSize)
      : cosWX(xSize / 2 + 1), cosWY(ySize), cosWZ(zSize),
	sinWX(xSize / 2 + 1), sinWY(ySize), sinWZ(zSize)
    {}
  };

  void _shrinkRegion(const ImageType& moving,
		     const Parameters& parameters,
		     VectorField* h,
		     VectorField* hinv,
		     bool shrinkLightRegions = true);

  void _shrinkRegionForward(const ImageType& moving,
                            const Parameters& parameters,
                            VectorField* h,
                            bool shrinkLightRegions = true);

  void _elasticShrinkRegionWithMask(const ImageType& moving,
                                    const Parameters& parameters,
                                    MaskType& mask,
                                    VectorField* h,
                                    bool shrinkLightRegions = true);

  void _computeHFieldAsymmetric(const ImageType& fixed,
				const ImageType& moving,
				const Parameters& params,
				VectorField* h,
				VectorField* hinv);

  void _computeHFieldElastic(const ImageType& fixed,
				const ImageType& moving,
				const Parameters& params,
				VectorField* h);

  void _computeHFieldElasticWithMask(const ImageType& fixed,
				const ImageType& moving,
        MaskType& mask,
				const Parameters& params,
				VectorField* h);

  void _computeHField2Symmetric(const ImageType& i1,
				const ImageType& i2,
				const Parameters& parameters,
				VectorField* h1,
				VectorField* h2,
				VectorField* h1inv,
				VectorField* h2inv);

  void _computeHFieldNSymmetric(unsigned int numImages,
				const ImageType* const* image,
                                const double* imageWeights,
				const Parameters& parameters,
				ImageType& iHat,
				VectorField** h,
				VectorField** hinv);

  // asymmetric
  static void _generateBodyForce(const ImageType& fixed,
				 const ImageType& def,
				 VectorField& grad,
				 double& squaredError);
 // asymetric with Jacobian Scale 
  static void _generateBodyForceJacobianScale(const ImageType& fixed,
				 const ImageType& def,
				 VectorField& grad,
				 double& squaredError);
  // symmetric
  static void _generateBodyForce(const ImageType& def1,
				 const ImageType& def2,
				 VectorField& grad1,
				 VectorField& grad2,
				 double& squaredError);

  static void _computeVelocityField(VectorField& v,
				    const Vector3D<unsigned int>& logicalSize,
				    const Parameters& params,
				    const LUT& lut,
				    fftwf_plan& fftwForwardPlan,
				    fftwf_plan& fftwBackwardPlan,
            bool divergenceFree = false);

  double _computeDelta(const VectorField& v,
		       const Vector3D<unsigned int>& logicalSize,
		       const double& maxPerturbation);

  static void _updateHField(VectorField& h,
			    VectorField& v,
			    const CoordinateType& delta);

  static void _updateHField(VectorField& h,
			    VectorField& hinv,
			    VectorField& v,
			    const CoordinateType& delta);

  static void _updateHFieldElastic(VectorField& h,
			    VectorField& b,
          VectorField& l,
          const CoordinateType& alpha,
			    const CoordinateType& delta);

  static void _updateHFieldElasticWithMask(VectorField& h,
			    VectorField& b,
          MaskType& m,
          VectorField& l,
          const CoordinateType& alpha,
			    const CoordinateType& delta);

  void _createFFTWPlans(VectorField& v,
                        const Vector3D<unsigned int>& logicalSize,
                        fftwf_plan& fftwForwardPlan,
                        fftwf_plan& fftwBackwardPlan);

  static void _inverseLMultiply(CoordinateType *complexPtr,
				CoordinateType& L00, 
				CoordinateType& L10, 
				CoordinateType& L11, 
				CoordinateType& L20, 
				CoordinateType& L21, 
				CoordinateType& L22);

  void _time(const char* message);
  void _stopTime();
  void _reportIterResults(unsigned int iter, 
			  unsigned int imageIndex,
			  unsigned int numImages,
			  unsigned int totalSeconds, 
			  double delta, 
			  double squaredError, 
			  double lastSquaredError,
			  double rmsError);
  void _report(const char* message, OutputType messageType);
  void _report(const std::ostringstream& message, OutputType messageType);

  void _writePerIterationData(unsigned int iter,
			      unsigned int imageIndex,
			      unsigned int numImages,
			      double rmsError,
			      const ImageType& deformedImage, 
			      const ImageType& atlas, 
			      const VectorField& h,
				  const VectorField& hinv);

//
// convert an int in [0,999] to a 3 digit right justified string
//
std::string
itos(const int& i)
{
  std::ostringstream oss;
  if (i < 10) oss << "0";
  if (i < 100) oss << "0";
  oss << i;
  return oss.str();
}

  template <class T>
std::string
_buildXSliceFileName(const char* name,
		       unsigned int iter,
		       unsigned int imageIndex,
		       unsigned int sliceIndex,
		       const Array3D<T>& array)
{
  std::ostringstream filename;
  filename << _filePrefix << name << "-xSlice" << itos(sliceIndex) << "-image" << itos(imageIndex)
	   << "-iter" << itos(iter)
	   << "-dims" << itos(array.getSizeY()) << "x" 
	   << itos(array.getSizeZ())
	   << "-eleSize" << sizeof(T);  
  return filename.str();
}

template <class T>
std::string
_buildYSliceFileName(const char* name,
		       unsigned int iter,
		       unsigned int imageIndex,
		       unsigned int sliceIndex,
		       const Array3D<T>& array)
{
  std::ostringstream filename;
  filename << _filePrefix << name << "-ySlice" << itos(sliceIndex) << "-image" << itos(imageIndex)
	   << "-iter" << itos(iter) 
	   << "-dims" << itos(array.getSizeX()) << "x" 
	   << itos(array.getSizeZ())
	   << "-eleSize" << sizeof(T);  
  return filename.str();
}

template <class T>
std::string
_buildZSliceFileName(const char* name,
		       unsigned int iter,
		       unsigned int imageIndex,
		       unsigned int sliceIndex,
		       const Array3D<T>& array)
{
  std::ostringstream filename;
  filename << _filePrefix << name << "-zSlice" << itos(sliceIndex) << "-image" << itos(imageIndex)
	   << "-iter" << itos(iter) 
	   << "-dims" << itos(array.getSizeX()) << "x" 
	   << itos(array.getSizeY())
	   << "-eleSize" << sizeof(T);  
  return filename.str();
}

template <class T>
std::string
_buildVolumeFileName(const char* name, 
		       unsigned int iter, 
		       unsigned int imageIndex,
		       const Array3D<T>& array)
{
  std::ostringstream filename;
  filename << _filePrefix << name << "-volume-image" << itos(imageIndex)
	   << "-iter" << itos(iter) 
	   << "-dims" << itos(array.getSizeX()) << "x" 
	   << itos(array.getSizeY()) << "x"
	   << itos(array.getSizeZ())
	   << "-eleSize" << sizeof(T);  
  return filename.str();
}

template <class T>
void
_writeVolume(const char* label,
	       unsigned int iter,
	       unsigned int imageIndex,
	       const Array3D<T>& array)
{
  if (_writeVolumes)
    {
      std::string name = 
	_buildVolumeFileName(label, iter, imageIndex, array);
      try
	{
          std::cout << "writing volume..." << name;
	  Array3DIO::writeMETAVolume(array, 
                                     _imageOrigin, _imageSpacing,
                                     name.c_str());
          std::cout << "DONE" << std::endl;
	}
      catch(...)
	{
	  std::ostringstream oss;
	  oss << "Could not open file: " << name;
	  _report(oss, FW_OUTPUT_TYPE_ERROR);
	}
    }
}

template <class T>
void
_writeSlices(const char* label,
	       unsigned int iter,
	       unsigned int imageIndex,
	       const Array3D<T>& array)
{
  if (_writeXSlices)
    {
      std::string name = 
	_buildXSliceFileName(label, iter, imageIndex, _xSliceNumber, array);
      try
	{
          std::cout << "writing x slice " << _xSliceNumber << "..." << name;
	  Array3DIO::writeMETASliceX(array, _xSliceNumber, name.c_str());
          std::cout << "DONE" << std::endl;
	}
      catch(...)
	{
	  std::ostringstream oss;
	  oss << "Could not open file: " << name;
	  _report(oss, FW_OUTPUT_TYPE_ERROR);
	}
    }
  if (_writeYSlices)
    {
      std::string name = 
	_buildYSliceFileName(label, iter, imageIndex, _ySliceNumber, array);
      try
	{
          std::cout << "writing y slice " << _ySliceNumber << "..." << name;
	  Array3DIO::writeMETASliceY(array, _ySliceNumber, name.c_str());
          std::cout << "DONE" << std::endl;
	}
      catch(...)
	{
	  std::ostringstream oss;
	  oss << "Could not open file: " << name;
	  _report(oss, FW_OUTPUT_TYPE_ERROR);
	}
    }
  if (_writeZSlices)
    {
      std::string name = 
	_buildZSliceFileName(label, iter, imageIndex, _zSliceNumber, array);
      try
	{
          std::cout << "writing z slice " << _zSliceNumber << "..." << name;
	  Array3DIO::writeMETASliceZ(array, _zSliceNumber, name.c_str());
          std::cout << "DONE" << std::endl;
	}
      catch(...)
	{
	  std::ostringstream oss;
	  oss << "Could not open file: " << name;
	  _report(oss, FW_OUTPUT_TYPE_ERROR);
	}
    }
}
  
  void _openSpreadsheetOfstream();
  void _closeSpreadsheetOfstream();

  void _writeLog(char* logtext);
};

#endif
