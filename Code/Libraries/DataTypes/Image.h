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

#ifndef IMAGE_H
#define IMAGE_H

#ifndef SWIG

#include "Vector3D.h"
#include "Array3D.h"
#include <iosfwd>
#include <string.h>

#endif // SWIG

template <class VoxelType = unsigned short>
class Image : public Array3D<VoxelType>
{
public:
  typedef double                    CoordinateType;
  typedef Vector3D<CoordinateType>  SpacingType;
  typedef Vector3D<CoordinateType>  ContinuousPointType;
  typedef Vector3D<unsigned int>    SizeType;

  Image();
  Image(const SizeType& size,
	const ContinuousPointType& origin = ContinuousPointType(0, 0, 0),
	const SpacingType& spacing = ContinuousPointType(1, 1, 1));
  Image(const Image<VoxelType>& rhs);
  Image(const Array3D<VoxelType>& rhs);

  // Location, in physical coordinates, of the center of the (0,0,0) voxel
  const ContinuousPointType& getOrigin() const;
  CoordinateType getOriginX() const;
  CoordinateType getOriginY() const;
  CoordinateType getOriginZ() const;

  void setOrigin(const ContinuousPointType& origin);
  void setOrigin(const CoordinateType& originX,
		 const CoordinateType& originY,
		 const CoordinateType& originZ);
  
  //  Physical units per voxel in each dimension.
  const SpacingType& getSpacing() const;
  CoordinateType getSpacingX() const;
  CoordinateType getSpacingY() const;
  CoordinateType getSpacingZ() const;

  void setSpacing(const SpacingType& spacing);
  void setSpacing(const CoordinateType& spacingX,
		  const CoordinateType& spacingY,
		  const CoordinateType& spacingZ);
  void setSpacingX(const CoordinateType& spacingX);
  void setSpacingY(const CoordinateType& spacingY);
  void setSpacingZ(const CoordinateType& spacingZ);

  // orientation
  
  enum ImageOrientation {
    LAI,LAS,LPI,LPS,RAI,RAS,RPI,RPS,
    RSP,RSA,RIP,RIA,LSP,LSA,LIP,LIA,
    AIR,AIL,ASR,ASL,PIR,PIL,PSR,PSL
  };
  
  void setOrientation(ImageOrientation orientation) {orient=orientation;}

  ImageOrientation getOrientation() const {return orient;}


  std::string getOrientationStr() const;

  
  static ImageOrientation strToImageOrientation(const char strorient[3])
  {
    const char *orientation_options[] = { 
      "LAI","LAS","LPI","LPS","RAI","RAS","RPI","RPS",
      "RSP","RSA","RIP","RIA","LSP","LSA","LIP","LIA",
      "AIR","AIL","ASR","ASL","PIR","PIL","PSR","PSL"
    };
    
    for(int i=0;i<24;i++)
      if(!strcmp(strorient,orientation_options[i]))
	return ((ImageOrientation)i);

    std::cerr << "[Image::strToImageOrientation] Error: internal" << 
      std::endl;
    return( (ImageOrientation)0 );
  }

  // convert image to a given orientation
  void toOrientation(ImageOrientation orientation);

  // This is generally not the same type as the template parameter T.
  // Instead, it indicates the type of data to which the image would
  // be saved.  When an image is read in, the data is converted to the
  // template type T (typically float), and the original type of
  // the data in the file is saved via setDataType().
  enum ImageDataType { Char, UnsignedChar, 
                       Short, UnsignedShort,
                       Int, UnsignedInt,
		       Float, Double,
                       VectorDataType,
                       UnknownDataType };
  
  void setDataType(ImageDataType dataType){data_type=dataType;}
  
  ImageDataType getDataType() const {return data_type;}
  
  std::string getDataTypeStr() const;

  template <class T, class U>
  void imageIndexToWorldCoordinates(const Vector3D<T>& index,
				    Vector3D<U>& world) const
  {
    world.x = (U) (_spacing.x * index.x + _origin.x);
    world.y = (U) (_spacing.y * index.y + _origin.y);
    world.z = (U) (_spacing.z * index.z + _origin.z);
  }

  template <class T, class U>
  void imageIndexToWorldCoordinates(const T& indexX,
				    const T& indexY,
				    const T& indexZ,
				    U& worldX,
				    U& worldY,
				    U& worldZ) const
  {
    worldX = (U) (_spacing.x * indexX + _origin.x);
    worldY = (U) (_spacing.y * indexY + _origin.y);
    worldZ = (U) (_spacing.z * indexZ + _origin.z);
  }


  template <class T, class U>
  void worldToImageIndexCoordinates(const Vector3D<T>& world,
				    Vector3D<U>& index) const
  {
    index.x = (U) ((world.x - _origin.x) / _spacing.x);
    index.y = (U) ((world.y - _origin.y) / _spacing.y);
    index.z = (U) ((world.z - _origin.z) / _spacing.z);    
  }

  template <class T, class U>
  void worldToImageIndexCoordinates(const T& worldX,
				    const T& worldY,
				    const T& worldZ,
				    U& indexX,
				    U& indexY,
				    U& indexZ) const
  {
    indexX = (U) ((worldX - _origin.x) / _spacing.x);
    indexY = (U) ((worldY - _origin.y) / _spacing.y);
    indexZ = (U) ((worldZ - _origin.z) / _spacing.z);
  }

  std::ostream& writeInfoASCII(std::ostream& output = std::cerr) const;

private:
  ContinuousPointType   _origin; 
  SpacingType           _spacing;

  ImageDataType data_type;
  ImageOrientation orient;

};

#ifndef SWIG
#include "Image.txx"
#endif // SWIG

#endif
