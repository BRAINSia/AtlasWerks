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

#ifndef IMAGE2D_TXX
#define IMAGE2D_TXX


/////////////////
// constructor //
/////////////////

template <class VoxelType> 
Image2D<VoxelType>
::Image2D()
  : Array2D<VoxelType>(),
    _origin(0, 0),
    _spacing(1, 1),
    data_type(Image2D<VoxelType>::UnknownDataType),
    orient(Image2D<VoxelType>::LAI)
{}

//////////////////////
// copy constructor //
//////////////////////

template <class VoxelType>
Image2D<VoxelType>
::Image2D(const Image2D<VoxelType>& rhs)
  : Array2D<VoxelType>(rhs),
    _origin(rhs._origin), 
    _spacing(rhs._spacing),
    data_type(rhs.data_type),
    orient(rhs.orient)
{}

/////////////////////////
// Array2D constructor //
/////////////////////////

template <class VoxelType>
Image2D<VoxelType>::
Image2D(const Array2D<VoxelType>& rhs)
  : Array2D<VoxelType>(rhs),
    _origin(0, 0),
    _spacing(1, 1),
    data_type(Image2D<VoxelType>::UnknownDataType),
    orient(Image2D<VoxelType>::LAI)
{}


///////////////
// getOrigin //
///////////////  

template <class VoxelType>
inline
const typename Image2D<VoxelType>::ContinuousPointType&
Image2D<VoxelType>
::getOrigin() const 
{ 
  return _origin; 
}

template <class VoxelType>
inline
typename Image2D<VoxelType>::CoordinateType
Image2D<VoxelType>
::getOriginX() const 
{ 
  return _origin.x; 
}

template <class VoxelType>
inline
typename Image2D<VoxelType>::CoordinateType
Image2D<VoxelType>
::getOriginY() const 
{ 
  return _origin.y; 
}

//////////////////////////
// setOrigin : vector2D //
//////////////////////////

template <class VoxelType>
inline
void
Image2D<VoxelType>
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
Image2D<VoxelType>
::setOrigin(const CoordinateType& originX,
	    const CoordinateType& originY)
{ 
  _origin.set(originX, originY); 
}

////////////////
// getSpacing //
////////////////

template <class VoxelType>
inline
const typename Image2D<VoxelType>::SpacingType&
Image2D<VoxelType>
::getSpacing() const 
{ 
  return _spacing; 
}

template <class VoxelType>
inline
typename Image2D<VoxelType>::CoordinateType
Image2D<VoxelType>
::getSpacingX() const 
{ 
  return _spacing.x; 
}

template <class VoxelType>
inline
typename Image2D<VoxelType>::CoordinateType
Image2D<VoxelType>
::getSpacingY() const 
{ 
  return _spacing.y; 
}

///////////////////////////
// setSpacing : vector2D //
///////////////////////////

template <class VoxelType>
inline
void
Image2D<VoxelType>
::setSpacing(const SpacingType& spacing) 
{ 
  _spacing = spacing;
}

///////////////////////////////////
// setSpacing : two parameters //
///////////////////////////////////

template <class VoxelType>
inline
void
Image2D<VoxelType>
::setSpacing(const CoordinateType& spacingX,
	     const CoordinateType& spacingY)

{ 
  _spacing.set(spacingX, spacingY); 
}

template <class VoxelType>
inline
void
Image2D<VoxelType>
::setSpacingX(const CoordinateType& spacingX)
{ 
  _spacing.x = spacingX;
}

template <class VoxelType>
inline
void
Image2D<VoxelType>
::setSpacingY(const CoordinateType& spacingY)
{ 
  _spacing.y = spacingY;
}

template <class VoxelType>
inline
void
Image2D<VoxelType>
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
            tmppix = this->get(i,j);
            this->set(i,j, this->get(i, size.y-j-1));
            this->set(i, size.y-j-1, tmppix); // invert R and I
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
Image2D<VoxelType>
::getOrientationStr() const
{
  std::string orientationStr;

  switch(orient) {
  case Image2D<VoxelType>::LAI: orientationStr = std::string("LAI"); break;
  case Image2D<VoxelType>::LAS: orientationStr = std::string("LAS"); break;
  case Image2D<VoxelType>::LPI: orientationStr = std::string("LPI"); break;
  case Image2D<VoxelType>::LPS: orientationStr = std::string("LPS"); break;
  case Image2D<VoxelType>::RAI: orientationStr = std::string("RAI"); break;
  case Image2D<VoxelType>::RAS: orientationStr = std::string("RAS"); break;
  case Image2D<VoxelType>::RPI: orientationStr = std::string("RPI"); break;
  case Image2D<VoxelType>::RPS: orientationStr = std::string("RPS"); break;
  case Image2D<VoxelType>::RSP: orientationStr = std::string("RSP"); break;
  case Image2D<VoxelType>::RSA: orientationStr = std::string("RSA"); break;
  case Image2D<VoxelType>::RIP: orientationStr = std::string("RIP"); break;
  case Image2D<VoxelType>::RIA: orientationStr = std::string("RAI"); break;
  case Image2D<VoxelType>::LSP: orientationStr = std::string("LSP"); break;
  case Image2D<VoxelType>::LSA: orientationStr = std::string("LSA"); break;
  case Image2D<VoxelType>::LIP: orientationStr = std::string("LIP"); break;
  case Image2D<VoxelType>::LIA: orientationStr = std::string("LIA"); break;
  case Image2D<VoxelType>::AIR: orientationStr = std::string("AIR"); break;
  case Image2D<VoxelType>::AIL: orientationStr = std::string("AIL"); break;
  case Image2D<VoxelType>::ASR: orientationStr = std::string("ASR"); break;
  case Image2D<VoxelType>::ASL: orientationStr = std::string("ASL"); break;
  case Image2D<VoxelType>::PIR: orientationStr = std::string("PIR"); break;
  case Image2D<VoxelType>::PIL: orientationStr = std::string("PIL"); break;
  case Image2D<VoxelType>::PSR: orientationStr = std::string("PSR"); break;
  case Image2D<VoxelType>::PSL: orientationStr = std::string("PSL"); break;
  default:
    std::cerr << "[Image2D<VoxelType>::getOrientation] internal errpr" 
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
Image2D<VoxelType> 
::getDataTypeStr() const
{
  std::string dataTypeStr;

  switch(data_type) {
  case Image2D<VoxelType>::UnsignedChar:
    dataTypeStr = std::string("unsigned char");
    break;
  case Image2D<VoxelType>::UnsignedShort:
    dataTypeStr = std::string("unsigned short");
    break;
  case Image2D<VoxelType>::Short:
    dataTypeStr = std::string("short");
    break;
  case Image2D<VoxelType>::Float:
    dataTypeStr = std::string("float");
    break;
  case Image2D<VoxelType>::UnknownDataType:
    dataTypeStr = std::string("<unknown>");
    break;
  default:
    std::cerr << "[Image2D<VoxelType>::getDataType] internal errpr" 
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
Image2D<VoxelType>
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
