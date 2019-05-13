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

#ifndef ApplicationUtils_h
#define ApplicationUtils_h

#ifndef SWIG

#include <string>
#include "DataTypes/Image.h"
#include "Image2D.h"
#include "AtlasWerksException.h"
#include "AtlasWerksTypes.h"

#endif // SWIG

/**
 * Class containing various I/O and support functions.
 */
class ApplicationUtils
{
 public:

#ifndef SWIG

  static void        ParseOptionVoid  (int& argc, char**& argv);
  static int         ParseOptionInt   (int& argc, char**& argv);
  static double      ParseOptionDouble(int& argc, char**& argv);
  static std::string ParseOptionString(int& argc, char**& argv);
  static bool        ParseOptionBool  (int& argc, char**& argv);

#endif // SWIG
  
  /** 
   * Does ITK have a file reader that can read the given file?
   */
  static bool ITKHasImageFileReader(const char* fileName);

  /** 
   * Test that the given filename exists
   */
  static bool FileExists(const char *filename);

  /** 
   * Parse a pathname into path, name base, and name extension (not
   * including period)
   */
  static void SplitName(const char *fullPathIn, 
			std::string &path,
			std::string &nameBase,
			std::string &nameExt);
  
  /**
   * Read an image header without reading image data.
   */
  static void        
  ReadHeaderITK(const char* fileName,
		SizeType &size,
		OriginType &origin,
		SpacingType &spacing);

  /**
   * Load an Image with the given filename.  Uses ITK functions
   * internally, so formats know to ITK are supported.  Filetype
   * determined by extension.
   */
  template <class VoxelType>
  static void        
  LoadImageITK(const char* fileName, Image<VoxelType>& image);
  
#ifdef SWIG
  %template(LoadImageITK) LoadImageITK< float >;
#endif // SWIG

  /**
   * Save an Image with the given filename.  Uses ITK functions
   * internally, so formats know to ITK are supported.  Filetype
   * determined by extension.
   */
  template <class VoxelType>
  static void        
  SaveImageITK(const char* fileName, const Image<VoxelType>& image);



  /**
  * Save an Image Slice with the given filename, 3-D Image,
  * slice plane type(AXIAL, CORONAL, SAGITTAL) and slice number
  * It supports ITK 2D image formats like png, jpg, etc.
  */
  template <class VoxelType>
  static int
  SaveImageSlice(const char* fileName, const Image<VoxelType>& image, const char* slicePlaneType, unsigned int sliceNumber);
  
#ifdef SWIG
  %template(SaveImageITK) SaveImageITK< float >;
#endif // SWIG

  /**
   * Load an HField saved as three scalar arrays of the form
   * HFieldName_{xyz}.ext, where HFieldName is the fileNamePrefix and
   * ext is the extension.
   */
  template <class VoxelType>
  static void        
  LoadHFieldITK(const char* fileNamePrefix,
		const char* extension, 
		Array3D<Vector3D<VoxelType> >& h);

  /**
   * Load an HField saved in three scalar arrays.  File type
   * determined by extension.
   */
  template <class VoxelType>
  static void        
  LoadHFieldITK(const char* fileNameX,
		const char* fileNameY, 
		const char* fileNameZ, 
		Array3D<Vector3D<VoxelType> >& h);

  /**
   * Load an HField from the given filename.  The extension determines
   * the file type.  Uses ITK libraries internally, so formats known
   * to ITK are supported.
   */
  template <class VoxelType>
  static void        
  LoadHFieldITK(const char* filename,
		Array3D<Vector3D<VoxelType> >& h);

  /**
   * Load an HField from the given filename, as well as retrieve the
   * origin and spacing from the file.  The extension determines the
   * file type.  Uses ITK libraries internally, so formats known to
   * ITK are supported.
   */
  template <class VoxelType>
  static void        
  LoadHFieldITK(const char* filename,
		OriginType &hOrigin,
		SpacingType &hSpacing,
		Array3D<Vector3D<VoxelType> >& h);

#ifdef SWIG
  %template(LoadHFieldITK) LoadHFieldITK< float >;
#endif // SWIG

  /**
   * Save an HField to the given filename.  The extension determines
   * the file type.  Uses ITK libraries internally, so formats known
   * to ITK are supported.
   */
  template <class VoxelType>
  static void        
  SaveHFieldITK(const char* fileNamePrefix,
		const char* extension, 
		 const Array3D<Vector3D<VoxelType> >& h);

  /**
   * Save an HField to the given filename, writing the given origin
   * and spacing to the file.  The extension determines the file type.
   * Uses ITK libraries internally, so formats known to ITK are
   * supported.
   */
  template <class VoxelType>
  static void        
  SaveHFieldITK(const char *filename,
		const Array3D<Vector3D<VoxelType> > &h,
		const OriginType &origin = 
		Vector3D<VoxelType>(0,0,0),
		const SpacingType &spacing = 
		Vector3D<VoxelType>(1,1,1));

#ifdef SWIG
  %template(SaveHFieldITK) SaveHFieldITK< float >;
#endif // SWIG

  /**
   * distribute numTotalElements among numBins bins.  Returns
   * beginning element ID and number of elements in bin binNum.
   * Assumes 0-based binNum.
   */
  static void Distribute(int numTotalElements, int numBins, int binNum, int &beginId, int &numElements);
};

#ifndef SWIG
#include "ApplicationUtils.txx"
#endif // SWIG

#endif // ApplicationUtils_h
