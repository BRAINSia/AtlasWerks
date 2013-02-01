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

#ifndef ARRAY3D_TXX
#define ARRAY3D_TXX

#include <iostream>
#include <fstream>
#include <cstring>

#include <memory.h>

//////////////////
// constructor  //
//////////////////

template <class T>
Array3D<T>
::Array3D()
  : _size(0, 0, 0),
    _xySize(0),
    _yzSize(0),
    _zxSize(0),
    _xyzSize(0),
    _slicePtrs(0),
    _rowPtrs(0),
    _dataPtr(0)
{}

////////////////////////////////////
// constructor : three parameters //
////////////////////////////////////

template <class T>
Array3D<T>
::Array3D(unsigned int xSize,
	  unsigned int ySize,
	  unsigned int zSize)
  : _size(xSize, ySize, zSize),
    _xySize(xSize * ySize),
    _yzSize(ySize * zSize),
    _zxSize(zSize * xSize),
    _xyzSize(xSize * ySize * zSize),
    _slicePtrs(0),
    _rowPtrs(0),
    _dataPtr(0)
{
  _allocateData();
}

////////////////////////////
// constructor : vector3D //
////////////////////////////

template <class T>
Array3D<T>
::Array3D(const SizeType& size)
  : _size(size.x, size.y, size.z),
    _xySize(size.x * size.y),
    _yzSize(size.y * size.z),
    _zxSize(size.z * size.x),
    _xyzSize(size.x * size.y * size.z),
    _slicePtrs(0),
    _rowPtrs(0),
    _dataPtr(0)
{
  _allocateData();
}

//////////////////////
// copy constructor //
//////////////////////


template <class T>
Array3D<T>
::Array3D(const Array3D<T>& rhs)
  : _size(rhs._size),
    _xySize(rhs._xySize),
    _yzSize(rhs._yzSize),
    _zxSize(rhs._zxSize),
    _xyzSize(rhs._xyzSize),
    _slicePtrs(0),
    _rowPtrs(0),
    _dataPtr(0)
{
  _allocateData();
  //memcpy(_dataPtr, rhs._dataPtr, _xyzSize * sizeof(T));
  for (unsigned int i = 0; i < _xyzSize; i++)
    _dataPtr[i] = rhs._dataPtr[i];
}

////////////////
// destructor //
////////////////

template <class T>
Array3D<T>
::~Array3D()
{
  _deallocateData();
}

///////////////
// operator= //
///////////////

template <class T>
Array3D<T>&
Array3D<T>
::operator=(const Array3D<T>& rhs)
{
  if (this == &rhs) return *this;

  if (_size != rhs._size)
    {
      _size    = rhs._size;
      _xySize  = rhs._xySize;
      _yzSize  = rhs._yzSize;
      _zxSize  = rhs._zxSize;
      _xyzSize = rhs._xyzSize;
      _deallocateData();
      _allocateData();
    }
  //memcpy(_dataPtr, rhs._dataPtr, _xyzSize * sizeof(T));
  for (unsigned int i = 0; i < _xyzSize; i++)
    _dataPtr[i] = rhs._dataPtr[i];
  return *this;
}

/////////////
// getSize //
/////////////

template <class T>
inline
const typename Array3D<T>::SizeType&
Array3D<T>
::getSize() const 
{ 
  return _size; 
}

//////////////////////
// get avarage value//
//////////////////////

template <class T>
inline 
const T Array3D<T>::getAverage() const
{
  return getSum() / _xyzSize;
}

//////////////////////
// get sum value//
//////////////////////

template <class T>
inline 
const T Array3D<T>::getSum() const
{
  long i;
  T sum, correction, corrected_next_term, new_sum;

  sum = _dataPtr[0];
  correction = 0.0;
  for (i = 1; i < _xyzSize; i++)
  {
    corrected_next_term = _dataPtr[i] - correction;
    new_sum = sum + corrected_next_term;
    correction = (new_sum - sum) - corrected_next_term;
    sum = new_sum;
  }
  return sum;
}

// getSlice: extract a slice and return an Array2D
// sliceNormal is one of 0,1,2 = X,Y,Z
// jdh 2010
template <class T>
Array2D<T>
Array3D<T>
::getSlice(unsigned int sliceNormal, unsigned int w) const
{
  Array2D<T> slice;
  switch (sliceNormal)
    {
    case 0:
      slice.resize(_size.y,_size.z);
      for (unsigned int z=0;z < _size.z;++z)
        for (unsigned int y=0;y < _size.y;++y)
          slice.set(y,z,_slicePtrs[z][y][w]);
      break;
    case 1:
      slice.resize(_size.x,_size.z);
      for (unsigned int z=0;z < _size.z;++z)
        for (unsigned int x=0;x < _size.x;++x)
          slice.set(x,z,_slicePtrs[z][w][x]);
      break;
    case 2:
      slice.resize(_size.x,_size.y);
      for (unsigned int y=0;y < _size.y;++y)
        for (unsigned int x=0;x < _size.x;++x)
          slice.set(x,y,_slicePtrs[w][y][x]);
      break;
    default:
      throw std::runtime_error("Array2D::getSlice: sliceNormal must be <= 2");
    }
  return slice;
}


//////////////
// getSizeX //
//////////////

template <class T>
inline
unsigned int
Array3D<T>
::getSizeX() const 
{ 
  return _size.x; 
}

//////////////
// getSizeY //
//////////////

template <class T>
inline
unsigned int
Array3D<T>
::getSizeY() const 
{ 
  return _size.y; 
}

//////////////
// getSizeZ //
//////////////

template <class T>
inline
unsigned int
Array3D<T>
::getSizeZ() const 
{ 
  return _size.z; 
}

/////////////
// isEmpty //
/////////////

template <class T>
inline
bool
Array3D<T>
::isEmpty() const 
{
  return _slicePtrs == 0;
}


////////////
// resize //
////////////

template <class T>
inline
void 
Array3D<T>
::resize(const SizeType& size) 
{ 
  resize(size.x, size.y, size.z); 
}


////////////
// resize //
////////////

template <class T>
void
Array3D<T>
::resize(unsigned int xSize, unsigned int ySize, unsigned int zSize)
{
  if (_size.x == xSize && _size.y == ySize && _size.z == zSize) return;

  _size.set(xSize, ySize, zSize);
  _xySize  = xSize * ySize;
  _yzSize  = ySize * zSize;
  _zxSize  = zSize * xSize;
  _xyzSize = xSize * ySize * zSize;
  
  _deallocateData();
  _allocateData();
}


////////////////////
// getNumElements //
////////////////////

template <class T>
inline
unsigned int 
Array3D<T>
::getNumElements() const 
{ 
  return _xyzSize; 
}


//////////////////
// getSizeBytes //
//////////////////

template <class T>
inline
unsigned int 
Array3D<T>
::getSizeBytes() const 
{ 
  return _xyzSize * sizeof(T); 
}

/////////
// set //
/////////

template <class T>
inline
void
Array3D<T>
::set(unsigned int xIndex,
      unsigned int yIndex,
      unsigned int zIndex, 
      const T& item)
{
  if(_slicePtrs == 0)
    {
      throw std::runtime_error("Data is NULL");
    } 
  // note unsigned int's can't be less than zero
 if(xIndex < _size.x &&
     yIndex < _size.y &&
     zIndex < _size.z)
    {
      _slicePtrs[zIndex][yIndex][xIndex] = item;
    }
  else
    {
      throw std::out_of_range("Index is out of range");
    }
}

/////////
// get //
/////////

template <class T>
inline
T&
Array3D<T>
::get(unsigned int xIndex,unsigned int yIndex,unsigned int zIndex) const
{
  if (_slicePtrs == 0)
    {
      throw std::runtime_error("Data is NULL");
    }
  // note unsigned int's can't be less than zero
  if(xIndex < _size.x &&
     yIndex < _size.y &&
     zIndex < _size.z)
    {
      return _slicePtrs[zIndex][yIndex][xIndex];
    }
  else
    {
      throw std::out_of_range("Index is out of range");
    }
}

template <class T>
inline
T&
Array3D<T>
::get(unsigned int elementIndex) const
{
  if (_dataPtr == NULL)
    {
      throw std::runtime_error("Data is NULL");
    }
  // note unsigned int's can't be less than zero
  if(elementIndex < _size.productOfElements())
    {
      return _dataPtr[elementIndex];
    }
  else
    {
      throw std::out_of_range("Index is out of range");
    }
}

////////////////
// operator() //
////////////////

template <class T>
inline
T&
Array3D<T>
::operator()(unsigned int xIndex, 
	     unsigned int yIndex,
	     unsigned int zIndex)
{
  return _slicePtrs[zIndex][yIndex][xIndex];
}


//////////////////////
// const operator() //
//////////////////////

template <class T>
inline
const T&
Array3D<T>
::operator()(unsigned int xIndex, 
	     unsigned int yIndex,
	     unsigned int zIndex) const
{
  return _slicePtrs[zIndex][yIndex][xIndex];
}


///////////////////////////
// operator() : Vector3D //
///////////////////////////

template <class T>
inline
T&
Array3D<T>
::operator()(const IndexType& index)
{
  return _slicePtrs[index.z][index.y][index.x];
}


////////////////////////////////
// const operator() : vector3D//
////////////////////////////////

template <class T>
inline
const T&
Array3D<T>
::operator()(const IndexType& index) const
{
  return _slicePtrs[index.z][index.y][index.x];
}

template <class T>
inline
T&
Array3D<T>
::operator()(unsigned int elementIndex)
{
  return _dataPtr[elementIndex];
}

template <class T>
inline
const T&
Array3D<T>::
operator()(unsigned int elementIndex) const
{
  return _dataPtr[elementIndex];
}

template <class T>
inline
bool 
Array3D<T>::
isValidIndex(const IndexType& index)
{
  return (index.x < getSizeX() && index.y < getSizeY() && 
          index.z < getSizeZ());
}

template <class T>
void
Array3D<T>
::fill(const T& fillValue)
{
  for (unsigned int i = 0; i < _xyzSize; ++i)
    {
      _dataPtr[i] = fillValue;
    }
}


template <class T>
void
Array3D<T>
::scale(const T& v)
{
  for (unsigned int i = 0; i < _xyzSize; ++i)
    {
      _dataPtr[i] = static_cast<T>(_dataPtr[i] * v);
    }
}

template <class T>
void
Array3D<T>
::add(const T& c)
{
  for (unsigned int i = 0; i < _xyzSize; ++i)
    {
      _dataPtr[i] = _dataPtr[i] + c;
    }
}

template <class T>
void
Array3D<T>
::addScalar(const double& d)
{
  for (unsigned int i = 0; i < _xyzSize; ++i)
    {
      _dataPtr[i] = T(_dataPtr[i] + d);
    }
}

template <class T> template <class U>
void
Array3D<T>
::pointwiseMultiplyBy(const Array3D<U>& rhs)
{
  for (unsigned int i = 0; i < _xyzSize; ++i)
    {
      _dataPtr[i] = T(_dataPtr[i] * rhs(i));
    }
}

template <class T> template <class U>
void
Array3D<T>
::pointwiseDivideBy(const Array3D<U>& rhs)
{
  for (unsigned int i = 0; i < _xyzSize; ++i)
    {
      _dataPtr[i] = T(_dataPtr[i] / rhs(i));
    }
}

template <class T>
void
Array3D<T>
::pointwiseAdd(const Array3D<T>& rhs)
{
  for (unsigned int i = 0; i < _xyzSize; ++i)
    {
      _dataPtr[i] = T(_dataPtr[i] + rhs._dataPtr[i]);
    }
}

template <class T>
void
Array3D<T>
::pointwiseSubtract(const Array3D<T>& rhs)
{
  for (unsigned int i = 0; i < _xyzSize; ++i)
    {
      _dataPtr[i] = T(_dataPtr[i] - rhs._dataPtr[i]);
    }
}

template <class T>
bool
Array3D<T>
::operator==(const Array3D<T>& rhs) const
{
/*
  return (_xySize == rhs._xySize) &&
    (_yzSize == rhs._yzSize) &&
    (_zxSize == rhs._zxSize) &&
    memcmp(_dataPtr, rhs._dataPtr, _xyzSize * sizeof(T)) ==  0;
*/

  if (_xySize != rhs._xySize
      || _yzSize != rhs._yzSize || _zxSize != rhs._zxSize)
    return false;

  bool sameEl = true;
  for (unsigned int i = 0; i < _xyzSize; i++)
    if (_dataPtr[i] != rhs._dataPtr[i])
    {
      sameEl = false;
      break;
    }

  if (sameEl)
    return true;
  else
    return false;
}


////////////////
// operator!= //
////////////////

template <class T>
bool
Array3D<T>
::operator!=(const Array3D<T>& rhs) const
{
/*
  return (_xySize != rhs._xySize) ||
    (_yzSize != rhs._yzSize) ||
    (_zxSize != rhs._zxSize) ||
    memcmp(_dataPtr, rhs._dataPtr, _xyzSize * sizeof(T)) != 0;
*/

  if (_xySize != rhs._xySize
      || _yzSize != rhs._yzSize || _zxSize != rhs._zxSize)
    return true;

  bool sameEl = true;
  for (unsigned int i = 0; i < _xyzSize; i++)
    if (_dataPtr[i] != rhs._dataPtr[i])
    {
      sameEl = false;
      break;
    }

  if (sameEl)
    return false;
  else
    return true;
}

////////////////////
// getDataPointer //
////////////////////

template <class T>
inline
T* 
Array3D<T>
::getDataPointer(unsigned int xIndex, 
		 unsigned int yIndex, 
		 unsigned int zIndex)
{
  return &_slicePtrs[zIndex][yIndex][xIndex];
}

///////////////////////////////
// getDataPointer : vector3D //
///////////////////////////////

template <class T>
inline
T* 
Array3D<T>
::getDataPointer(const IndexType& index)
{
  return &_slicePtrs[index.z][index.y][index.x];
}

//////////////////////////
// const getDataPointer //
//////////////////////////

template <class T>
inline
const T* 
Array3D<T>
::getDataPointer(unsigned int xIndex, 
		 unsigned int yIndex, 
		 unsigned int zIndex) const
{
  return &_slicePtrs[zIndex][yIndex][xIndex];
}

/////////////////////////////////////
// const getDataPointer : vector3D //
/////////////////////////////////////

template <class T>
inline
const T* 
Array3D<T>
::getDataPointer(const IndexType& index) const
{
  return &_slicePtrs[index.z][index.y][index.x];
}

///////////////////////
// setData : array3D //
///////////////////////
template <class T>
inline
void 
Array3D<T>
::setData(const Array3D<T>& rhs)
{
  if (_xyzSize != rhs.getNumElements())
    {
      std::cerr<<" different size: "<< _xyzSize << " vs " << rhs.getNumElements() << std::endl;
      return;
    }
  //memcpy(_dataPtr, rhs._dataPtr, _xyzSize * sizeof(T));
  for (unsigned int i = 0; i < _xyzSize; i++)
    _dataPtr[i] = rhs._dataPtr[i];
}

template <class T>
inline
void 
Array3D<T>
::copyData(const void* const dataPtr)
{
  //memcpy(_dataPtr, dataPtr, _xyzSize * sizeof(T));
  for (unsigned int i = 0; i < _xyzSize; i++)
    _dataPtr[i] = ((T*)dataPtr)[i];
}

///////////////////
// _allocateData //
///////////////////

template <class T>
void
Array3D<T>
::_allocateData()
{
  _slicePtrs = new T **[_size.z];
  _rowPtrs   = new T  *[_yzSize];
  _dataPtr   = new T   [_xyzSize];

  for (unsigned int rowIndex = 0; rowIndex < _yzSize; ++rowIndex) 
    {
      _rowPtrs[rowIndex] = &_dataPtr[rowIndex * _size.x];
    }

  for (unsigned int sliceIndex = 0; sliceIndex < _size.z; ++sliceIndex)
    {
      _slicePtrs[sliceIndex] = &_rowPtrs[sliceIndex * _size.y];
    }
}

/////////////////////
// _deallocateData //
/////////////////////

template <class T>
void
Array3D<T>
::_deallocateData()
{
  if (_slicePtrs != 0) delete [] _slicePtrs;
  _slicePtrs = 0;

  if (_rowPtrs != 0)   delete [] _rowPtrs;
  _rowPtrs = 0;

// BUG w/ itk::VariableLengthVector?
//std::cout << "PP BUG delete dataPtr BEGIN" << std::endl;
  if (_dataPtr != 0)   delete [] _dataPtr;
   _dataPtr = 0;
//std::cout << "PP BUG delete dataPtr END" << std::endl;

}

#endif
