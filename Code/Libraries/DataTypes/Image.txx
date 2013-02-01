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

#ifndef IMAGE_TXX
#define IMAGE_TXX


/////////////////
// constructor //
/////////////////

template <class VoxelType> 
Image<VoxelType>
::Image()
  : Array3D<VoxelType>(),
    _origin(0, 0, 0),
    _spacing(1, 1, 1),
    data_type(Image<VoxelType>::UnknownDataType),
    orient(Image<VoxelType>::LAI)
{}

///////////////////////////////////
// constructor : three parameters//
///////////////////////////////////

template <class VoxelType>
Image<VoxelType>
::Image(const SizeType& size,
	const ContinuousPointType& origin,
	const SpacingType& spacing)
  : Array3D<VoxelType>(size), 
    _origin(origin), 
    _spacing(spacing),
    data_type(Image<VoxelType>::UnknownDataType),
    orient(Image<VoxelType>::LAI)
{}

//////////////////////
// copy constructor //
//////////////////////

template <class VoxelType>
Image<VoxelType>
::Image(const Image<VoxelType>& rhs)
  : Array3D<VoxelType>(rhs),
    _origin(rhs._origin), 
    _spacing(rhs._spacing),
    data_type(rhs.data_type),
    orient(rhs.orient)
{}

/////////////////////////
// Array3D constructor //
/////////////////////////

template <class VoxelType>
Image<VoxelType>::
Image(const Array3D<VoxelType>& rhs)
  : Array3D<VoxelType>(rhs),
    _origin(0, 0, 0),
    _spacing(1, 1, 1),
    data_type(Image<VoxelType>::UnknownDataType),
    orient(Image<VoxelType>::LAI)
{}


///////////////
// getOrigin //
///////////////  

template <class VoxelType>
inline
const typename Image<VoxelType>::ContinuousPointType&
Image<VoxelType>
::getOrigin() const 
{ 
  return _origin; 
}

template <class VoxelType>
inline
typename Image<VoxelType>::CoordinateType
Image<VoxelType>
::getOriginX() const 
{ 
  return _origin.x; 
}

template <class VoxelType>
inline
typename Image<VoxelType>::CoordinateType
Image<VoxelType>
::getOriginY() const 
{ 
  return _origin.y; 
}

template <class VoxelType>
inline
typename Image<VoxelType>::CoordinateType
Image<VoxelType>
::getOriginZ() const 
{ 
  return _origin.z; 
}

//////////////////////////
// setOrigin : vector3D //
//////////////////////////

template <class VoxelType>
inline
void
Image<VoxelType>
::setOrigin(const ContinuousPointType& origin) 
{ 
  _origin = origin; 
}

/////////////////////////////////
// setOrigin : three parameters//
/////////////////////////////////

template <class VoxelType>
inline
void
Image<VoxelType>
::setOrigin(const CoordinateType& originX,
	    const CoordinateType& originY,
	    const CoordinateType& originZ) 
{ 
  _origin.set(originX, originY, originZ); 
}

////////////////
// getSpacing //
////////////////

template <class VoxelType>
inline
const typename Image<VoxelType>::SpacingType&
Image<VoxelType>
::getSpacing() const 
{ 
  return _spacing; 
}

template <class VoxelType>
inline
typename Image<VoxelType>::CoordinateType
Image<VoxelType>
::getSpacingX() const 
{ 
  return _spacing.x; 
}

template <class VoxelType>
inline
typename Image<VoxelType>::CoordinateType
Image<VoxelType>
::getSpacingY() const 
{ 
  return _spacing.y; 
}

template <class VoxelType>
inline
typename Image<VoxelType>::CoordinateType
Image<VoxelType>
::getSpacingZ() const 
{ 
  return _spacing.z; 
}

///////////////////////////
// setSpacing : vector3D //
///////////////////////////

template <class VoxelType>
inline
void
Image<VoxelType>
::setSpacing(const SpacingType& spacing) 
{ 
  _spacing = spacing;
}

///////////////////////////////////
// setSpacing : three parameters //
///////////////////////////////////

template <class VoxelType>
inline
void
Image<VoxelType>
::setSpacing(const CoordinateType& spacingX,
	     const CoordinateType& spacingY,
	     const CoordinateType& spacingZ) 

{ 
  _spacing.set(spacingX, spacingY, spacingZ); 
}

template <class VoxelType>
inline
void
Image<VoxelType>
::setSpacingX(const CoordinateType& spacingX)
{ 
  _spacing.x = spacingX;
}

template <class VoxelType>
inline
void
Image<VoxelType>
::setSpacingY(const CoordinateType& spacingY)
{ 
  _spacing.y = spacingY;
}

template <class VoxelType>
inline
void
Image<VoxelType>
::setSpacingZ(const CoordinateType& spacingZ)
{ 
  _spacing.z = spacingZ;
}

template <class VoxelType>
inline
void
Image<VoxelType>
::toOrientation(const ImageOrientation orientation)
{
  switch (orientation)
  { // what are we converting to
  case RPS:
    if (orient == RAI)
    {
      std::cerr << "Converting from RAI to RPS..." << std::endl;
      SizeType size = Array3D<VoxelType>::getSize();
      VoxelType tmppix;
      for (unsigned int i=0;i<size.x;i++)
      {
        for (unsigned int j=0;j<size.y;j++)
        {
          for (unsigned int k=0;k<size.z/2;k++)
          {
            tmppix = this->get(i,j,k);
            this->set(i,j,k, this->get(i, size.y-j-1, size.z-k-1));
            this->set(i, size.y-j-1, size.z-k-1, tmppix); // invert R and I
          }
        }
      }
    }
    else if (orient != LAS)
    {
      std::cerr << "WARNING: Cannot yet convert from " << orient << " to " << orientation << std::endl;
      return;
    }
    break;
  default:
    std::cerr << "WARNING: Unimplemented orientation, not reordering data..." << std::endl;
    return;
  }
  orient = orientation; // update orient variable
}

///////////////////////
// getOrientationStr //
///////////////////////
template <class VoxelType>
std::string
Image<VoxelType>
::getOrientationStr() const
{
  std::string orientationStr;

  switch(orient) {
  case Image<VoxelType>::LAI: orientationStr = std::string("LAI"); break;
  case Image<VoxelType>::LAS: orientationStr = std::string("LAS"); break;
  case Image<VoxelType>::LPI: orientationStr = std::string("LPI"); break;
  case Image<VoxelType>::LPS: orientationStr = std::string("LPS"); break;
  case Image<VoxelType>::RAI: orientationStr = std::string("RAI"); break;
  case Image<VoxelType>::RAS: orientationStr = std::string("RAS"); break;
  case Image<VoxelType>::RPI: orientationStr = std::string("RPI"); break;
  case Image<VoxelType>::RPS: orientationStr = std::string("RPS"); break;
  case Image<VoxelType>::RSP: orientationStr = std::string("RSP"); break;
  case Image<VoxelType>::RSA: orientationStr = std::string("RSA"); break;
  case Image<VoxelType>::RIP: orientationStr = std::string("RIP"); break;
  case Image<VoxelType>::RIA: orientationStr = std::string("RAI"); break;
  case Image<VoxelType>::LSP: orientationStr = std::string("LSP"); break;
  case Image<VoxelType>::LSA: orientationStr = std::string("LSA"); break;
  case Image<VoxelType>::LIP: orientationStr = std::string("LIP"); break;
  case Image<VoxelType>::LIA: orientationStr = std::string("LIA"); break;
  case Image<VoxelType>::AIR: orientationStr = std::string("AIR"); break;
  case Image<VoxelType>::AIL: orientationStr = std::string("AIL"); break;
  case Image<VoxelType>::ASR: orientationStr = std::string("ASR"); break;
  case Image<VoxelType>::ASL: orientationStr = std::string("ASL"); break;
  case Image<VoxelType>::PIR: orientationStr = std::string("PIR"); break;
  case Image<VoxelType>::PIL: orientationStr = std::string("PIL"); break;
  case Image<VoxelType>::PSR: orientationStr = std::string("PSR"); break;
  case Image<VoxelType>::PSL: orientationStr = std::string("PSL"); break;
  default:
    std::cerr << "[Image<VoxelType>::getOrientation] internal errpr" 
	      << std::endl;
    orientationStr = std::string("<internal error>");
    break;
  }

  return(orientationStr);
}

////////////////////
// getDataTypeStr //
////////////////////
template <class VoxelType>
std::string
Image<VoxelType> 
::getDataTypeStr() const
{
  std::string dataTypeStr;

  switch(data_type) {
  case Image<VoxelType>::UnsignedChar:
    dataTypeStr = std::string("unsigned char");
    break;
  case Image<VoxelType>::UnsignedShort:
    dataTypeStr = std::string("unsigned short");
    break;
  case Image<VoxelType>::Short:
    dataTypeStr = std::string("short");
    break;
  case Image<VoxelType>::Float:
    dataTypeStr = std::string("float");
    break;
  case Image<VoxelType>::UnknownDataType:
    dataTypeStr = std::string("<unknown>");
    break;
  default:
    std::cerr << "[Image<VoxelType>::getDataType] internal errpr" 
	      << std::endl;
    dataTypeStr = std::string("<internal error>");
    break;
  }

  return(dataTypeStr);
}

////////////////////
// writeInfoASCII //
////////////////////
template <class VoxelType>
std::ostream&
Image<VoxelType>
::writeInfoASCII(std::ostream& output) const
{
  output << "### Image Information ###" << std::endl;
  output << "-------------------------" << std::endl;
  output << "Data Type:   " << getDataTypeStr() << std::endl;
  output << "Size:        " << Array3D<VoxelType>::getSize() << std::endl;
  output << "Spacing:     " << getSpacing() << std::endl;
  output << "Origin:      " << getOrigin() << std::endl;
  output << "Orientation: " << getOrientationStr() << std::endl;
  output << std::endl;

  return(output);
}
#endif
