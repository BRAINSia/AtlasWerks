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

//////////////////////////////////////////////////////////////////////
//
// File: ROI.txx
//
// Implementation of ROI.h (see that file for further details)
//
//////////////////////////////////////////////////////////////////////
#ifndef ROI_TXX
#define ROI_TXX

template <class IndexType, class SizeType>
ROI<IndexType, SizeType>
::ROI()
  : _start(0,0,0),
    _size(0,0,0)
{}

template <class IndexType, class SizeType>
ROI<IndexType, SizeType>
::ROI(const ROIIndexType& start,
      const ROISizeType& size )
  : _start(start.x,start.y,start.z),
    _size(size.x,size.y,size.z)
{}

template <class IndexType, class SizeType>
ROI<IndexType, SizeType>
::ROI(const ROI<IndexType, SizeType>& rhs)
  : _start(rhs._start),
    _size(rhs._size)
{}

template <class IndexType, class SizeType>
ROI<IndexType, SizeType>&
ROI<IndexType, SizeType>
::operator=(const ROI<IndexType, SizeType>& rhs)
{
  if (this == &rhs) return *this;

  _size  = rhs._size;
  _start = rhs._start;
  return *this;
}

template <class IndexType, class SizeType>
inline
typename ROI<IndexType, SizeType>::ROIIndexType
ROI<IndexType, SizeType>
::getStart() const
{
  return _start; 
}

template <class IndexType, class SizeType>
inline
IndexType
ROI<IndexType, SizeType>
::getStartX() const
{
  return _start.x; 
}

template <class IndexType, class SizeType>
inline
IndexType
ROI<IndexType, SizeType>
::getStartY() const
{
  return _start.y; 
}

template <class IndexType, class SizeType>
inline
IndexType
ROI<IndexType, SizeType>
::getStartZ() const
{
  return _start.z; 
}

template <class IndexType, class SizeType>
inline
void
ROI<IndexType, SizeType>
::setStart(const ROIIndexType& start)
{
  _start = start;
}

template <class IndexType, class SizeType>
inline
void
ROI<IndexType, SizeType>
::setStart(const IndexType& x, const IndexType& y, const IndexType& z)
{
  _start.set(x,y,z);
}

template <class IndexType, class SizeType>
inline
void
ROI<IndexType, SizeType>
::setStartX(const IndexType& x)
{
  _start.x = x;
}

template <class IndexType, class SizeType>
inline
void
ROI<IndexType, SizeType>
::setStartY(const IndexType& y)
{
  _start.y = y;
}

template <class IndexType, class SizeType>
inline
void
ROI<IndexType, SizeType>
::setStartZ(const IndexType& z)
{
  _start.z = z;
}

template <class IndexType, class SizeType>
inline
typename ROI<IndexType, SizeType>::ROISizeType
ROI<IndexType, SizeType>
::getSize() const
{
  return _size; 
}

template <class IndexType, class SizeType>
inline
SizeType
ROI<IndexType, SizeType>
::getSizeX() const
{
  return _size.x; 
}

template <class IndexType, class SizeType>
inline
SizeType
ROI<IndexType, SizeType>
::getSizeY() const
{
  return _size.y; 
}

template <class IndexType, class SizeType>
inline
SizeType
ROI<IndexType, SizeType>
::getSizeZ() const
{
  return _size.z; 
}

template <class IndexType, class SizeType>
inline
void
ROI<IndexType, SizeType>
::setSize(const ROISizeType& size)
{
  _size = size;
}

template <class IndexType, class SizeType>
inline
void
ROI<IndexType, SizeType>
::setSize(const SizeType& x, const SizeType& y, const SizeType& z)
{
  _size.set(x,y,z);
}

template <class IndexType, class SizeType>
inline
void
ROI<IndexType, SizeType>
::setSizeX(const SizeType& x)
{
  _size.x = x;
}

template <class IndexType, class SizeType>
inline
void
ROI<IndexType, SizeType>
::setSizeY(const SizeType& y)
{
  _size.y = y;
}

template <class IndexType, class SizeType>
inline
void
ROI<IndexType, SizeType>
::setSizeZ(const SizeType& z)
{
  _size.z = z;
}

template <class IndexType, class SizeType>
inline
typename ROI<IndexType, SizeType>::ROIIndexType
ROI<IndexType, SizeType>
::getStop() const
{
  return(_start + _size - 1);
}

template <class IndexType, class SizeType>
inline
IndexType
ROI<IndexType, SizeType>
::getStopX() const
{
  return(_start.x + _size.x - 1); 
}

template <class IndexType, class SizeType>
inline
IndexType
ROI<IndexType, SizeType>
::getStopY() const
{
  return(_start.y + _size.y - 1); 
}

template <class IndexType, class SizeType>
inline
IndexType
ROI<IndexType, SizeType>
::getStopZ() const
{
  return(_start.z + _size.z - 1); 
}

template <class IndexType, class SizeType>
inline
void
ROI<IndexType, SizeType>
::setStop(const ROIIndexType& stop)
{
  _size = stop - _start + 1;
}

template <class IndexType, class SizeType>
inline
void
ROI<IndexType, SizeType>
::setStop(const IndexType& x, const IndexType& y, const IndexType& z)
{
  _size.x = x - _start.x + 1;
  _size.y = y - _start.y + 1;
  _size.z = z - _start.z + 1;
}

template <class IndexType, class SizeType>
inline
void
ROI<IndexType, SizeType>
::setStopX(const IndexType& x)
{
  _size.x = x - _start.x + 1;
}

template <class IndexType, class SizeType>
inline
void
ROI<IndexType, SizeType>
::setStopY(const IndexType& y)
{
  _size.y = y - _start.y + 1;
}

template <class IndexType, class SizeType>
inline
void
ROI<IndexType, SizeType>
::setStopZ(const IndexType& z)
{
  _size.z = z - _start.z + 1;
}
#endif
