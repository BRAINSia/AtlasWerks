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

#ifndef ARRAY2D_TXX
#define ARRAY2D_TXX

#ifndef SWIG

#include <iostream>
#include <fstream>
#include <limits>
#include <memory.h>

#endif // SWIG

//////////////////
// constructor  //
//////////////////

template <class T>
Array2D<T>
::Array2D()
  : _size(0, 0),
    _xySize(0),
    _rowPtrs(0),
    _dataPtr(0)
{}

////////////////////////////////////
// constructor : two parameters   //
////////////////////////////////////

template <class T>
Array2D<T>
::Array2D(unsigned int xSize,
	      unsigned int ySize)
  : _size(xSize, ySize),
    _xySize(xSize * ySize),
    _rowPtrs(0),
    _dataPtr(0)
{
  _allocateData();
}

////////////////////////////
// constructor : vector2D //
////////////////////////////

template <class T>
Array2D<T>
::Array2D(const SizeType& size)
  : _size(size.x, size.y),
    _xySize(size.x * size.y),
    _rowPtrs(0),
    _dataPtr(0)
{
  _allocateData();
}

//////////////////////
// copy constructor //
//////////////////////


template <class T>
Array2D<T>
::Array2D(const Array2D<T>& rhs)
  : _size(rhs._size),
    _xySize(rhs._xySize),
    _rowPtrs(0),
    _dataPtr(0)
{
  _allocateData();
  memcpy(_dataPtr, rhs._dataPtr, _xySize * sizeof(T));
}

////////////////
// destructor //
////////////////

template <class T>
Array2D<T>
::~Array2D()
{
  _deallocateData();
}

///////////////
// operator= //
///////////////

template <class T>
Array2D<T>&
Array2D<T>
::operator=(const Array2D<T>& rhs)
{
  if (this == &rhs) return *this;

  if (_size != rhs._size)
    {
      _size    = rhs._size;
      _xySize  = rhs._xySize;
      _deallocateData();
      _allocateData();
    }
  memcpy(_dataPtr, rhs._dataPtr, _xySize * sizeof(T));
  return *this;
}

/////////////
// getSize //
/////////////

template <class T>
inline
typename Array2D<T>::SizeType
Array2D<T>
::getSize() const 
{ 
  return _size; 
}


//////////////
// getSizeX //
//////////////

template <class T>
inline
unsigned int
Array2D<T>
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
Array2D<T>
::getSizeY() const 
{ 
  return _size.y; 
}


/////////////
// isEmpty //
/////////////

template <class T>
inline
bool
Array2D<T>
::isEmpty() const 
{
return ( _rowPtrs== NULL);
}


////////////
// resize //
////////////

template <class T>
inline
void 
Array2D<T>
::resize(const SizeType& size) 
{ 
  resize(size.x, size.y); 
}


////////////
// resize //
////////////

template <class T>
void
Array2D<T>
::resize(unsigned int xSize, unsigned int ySize)
{
  if (_size.x == xSize && _size.y == ySize) return;

  _size.set(xSize, ySize);
  _xySize  = xSize * ySize;
  _deallocateData();
  _allocateData();
}


////////////////////
// getNumElements //
////////////////////

template <class T>
inline
unsigned int 
Array2D<T>
::getNumElements() const 
{ 
  return _xySize; 
}


//////////////////
// getSizeBytes //
//////////////////

template <class T>
inline
unsigned int 
Array2D<T>
::getSizeBytes() const 
{ 
  return _xySize * sizeof(T); 
}

/////////
// set //
/////////

template <class T>
inline
void
Array2D<T>
::set(unsigned int xIndex,unsigned int yIndex, T item)
{
  if(_rowPtrs == NULL){
	  throw std::runtime_error("Data is NULL");

    }
    // note unsigned int's can't be less than zero
    if(xIndex < _size.x &&
       yIndex < _size.y ){
        _rowPtrs[yIndex][xIndex] = item;
    }else{
        throw std::out_of_range("Index is out of range");
    }
 }

/////////
// get //
/////////

template <class T>
inline
T&
Array2D<T>
::get(unsigned int xIndex,unsigned int yIndex) const
{
  if(_rowPtrs == NULL){
        throw std::runtime_error("Data is NULL");

    }
    // note unsigned int's can't be less than zero
    if(xIndex < _size.x &&
       yIndex < _size.y){
        return _rowPtrs[yIndex][xIndex];
    }else{
        throw std::out_of_range("Index is out of range");
    }
 }




////////////////
// operator() //
////////////////

template <class T>
inline
T&
Array2D<T>
::operator()(unsigned int xIndex, 
	     unsigned int yIndex)
{
  return _rowPtrs[yIndex][xIndex];
}


//////////////////////
// const operator() //
//////////////////////

template <class T>
inline
const T&
Array2D<T>
::operator()(unsigned int xIndex, 
	     unsigned int yIndex) const
{
  return _rowPtrs[yIndex][xIndex];
}


///////////////////////////
// operator() : Vector2D //
///////////////////////////

template <class T>
inline
T&
Array2D<T>
::operator()(const IndexType& index)
{
  return _rowPtrs[index.y][index.x];
}


////////////////////////////////
// const operator() : vector2D//
////////////////////////////////

template <class T>
inline
const T&
Array2D<T>
::operator()(const IndexType& index) const
{
  return _rowPtrs[index.y][index.x];
}


//////////
// fill //
//////////

template <class T>
inline
T&
Array2D<T>
::operator()(unsigned int elementIndex)
{
  return _dataPtr[elementIndex];
}

template <class T>
inline
const T&
Array2D<T>
::operator()(unsigned int elementIndex) const
{
  return _dataPtr[elementIndex];
}

template <class T>
void
Array2D<T>
::fill(const T& fillValue)
{
  for (unsigned int i = 0; i < _xySize; ++i)
    {
      _dataPtr[i] = fillValue;
    }
}

///////////////////////
// setData : array3D //
///////////////////////
template <class T>
inline
void 
Array2D<T>
::setData(const Array2D<T>& rhs)
{
  if (_xySize != rhs.getNumElements())
    {
      std::cerr<<" different size "<< std::endl;
      return;
    }
  memcpy(_dataPtr, rhs._dataPtr, _xySize * sizeof(T));
}

////////////////
// operator== //
////////////////

template <class T>
void
Array2D<T>
::scale(const double& d)
{
  for (unsigned int i = 0; i < _xySize; ++i)
    {
      _dataPtr[i] *= d;
    }
}

template <class T>
bool
Array2D<T>
::operator==(const Array2D<T>& rhs) const
{
  return (_size == rhs._size) &&
    memcmp(_dataPtr, rhs._dataPtr, _xySize * sizeof(T));
}

////////////////
// operator!= //
////////////////

template <class T>
bool
Array2D<T>
::operator!=(const Array2D<T>& rhs) const
{
  return !(*this == rhs);
}

////////////////////
// getDataPointer //
////////////////////

template <class T>
inline
T* 
Array2D<T>
::getDataPointer(unsigned int xIndex, 
		 unsigned int yIndex)
{
  return &_rowPtrs[yIndex][xIndex];
}

///////////////////////////////
// getDataPointer : vector2D //
///////////////////////////////

template <class T>
inline
T* 
Array2D<T>
::getDataPointer(const IndexType& index)
{
  return &_rowPtrs[index.y][index.x];
}

//////////////////////////
// const getDataPointer //
//////////////////////////

template <class T>
inline
const T* 
Array2D<T>
::getDataPointer(unsigned int xIndex, 
		 unsigned int yIndex) const
{
  return &_rowPtrs[yIndex][xIndex];
}

/////////////////////////////////////
// const getDataPointer : vector2D //
/////////////////////////////////////

template <class T>
inline
const T* 
Array2D<T>
::getDataPointer(const IndexType& index) const
{
  return &_rowPtrs[index.y][index.x];
}

////////////////////////////////
// operator<<() [text output] //
////////////////////////////////

template <class T>
std::ostream& 
operator<<(std::ostream& output, const Array2D<T>& array)
{
  for (unsigned int y = 0; y < array.getSizeY(); ++y) {
    for (unsigned int x = 0; x < array.getSizeX(); ++x) {
      output << array(x, y) << " ";
    }
    output << '\n';
  }
  return output;
}

///////////////////
// _allocateData //
///////////////////

template <class T>
void
Array2D<T>
::_allocateData()
{
  _rowPtrs   = new T  *[_size.y];
  _dataPtr   = new T   [_xySize];

  for (unsigned int rowIndex = 0; rowIndex < _size.y; ++rowIndex) 
    {
      _rowPtrs[rowIndex] = &_dataPtr[rowIndex * _size.x];
    }
}

/////////////////////
// _deallocateData //
/////////////////////

template <class T>
void
Array2D<T>
::_deallocateData()
{
  if (!_rowPtrs)   delete [] _rowPtrs;
  if (!_dataPtr)   delete [] _dataPtr;

  _rowPtrs   = 0;
  _dataPtr   = 0;
}

/////////////////
// TukeyWindow //
/////////////////
// jdh 2010, copied jsam's code in Base for this
template <class T>
inline
void
Array2D<T>
::TukeyWindow(unsigned int width)
{
  // create a 1D profile beforehand of size width
  double profile[width];

  for(unsigned int i=0;i<width;i++){
    double v = M_PI*((double)i)/((double)width);
    profile[i] = 0.5*(1.0-cos(v));
  }

  // visit each pixel, and if it's within width of the border in each
  // dimension then multiply by the profile
  for (unsigned int y = 0; y < this->_size.y; ++y)
    for (unsigned int x = 0; x < this->_size.x; ++x)
      {
      // X border
      if (x < width) this->set(x,y,this->get(x,y)*profile[x]);
      if (x >= this->_size.x-width) this->set(x,y,this->get(x,y)*profile[this->_size.x-1-x]);
      // Y border
      if (y < width) this->set(x,y,this->get(x,y)*profile[y]);
      if (y >= this->_size.y-width) this->set(x,y,this->get(x,y)*profile[this->_size.y-1-y]);
      }
}

///////////////
// outputPGM //
///////////////

template <class T>
inline
void
Array2D<T>
::outputPGM(const std::string filename, float iwmin, float iwmax)
{
  // This needs to be specialized always
  throw std::runtime_error("Array2D::outputPGM : not specialized for this template type!");
}

// copied in by jdh 2009
template <>
inline
void
Array2D<float>
::outputPGM(const std::string filename, float iwmin, float iwmax)
  {
    std::ofstream out(filename.c_str(), std::ios::binary);
    if (out.bad())
    {
      throw std::runtime_error("error opening file");
    }
    out << "P5\n" 
        << this->_size.x << "\n" 
        << this->_size.y << "\n"
        << 255 << "\n";

    if (iwmax < iwmin)
      { // Default case: scale to min/max of array
      // Find brightest pixel so we can normalize correctly
      iwmin = std::numeric_limits<float>::infinity();
      iwmax = -iwmin;
      for (unsigned int y = 0; y < this->_size.y; ++y)
        for (unsigned int x = 0; x < this->_size.x; ++x)
          {
          if (this->get(x,y) > iwmax) iwmax = this->get(x,y);
          if (this->get(x,y) < iwmin) iwmin = this->get(x,y);
          }
      }

    std::cout << "Writing PGM with intensity window: " << iwmin << "-" << iwmax << std::endl;

    unsigned char pixel;
    for (unsigned int y = 0; y < this->_size.y; ++y)
      for (unsigned int x = 0; x < this->_size.x; ++x)
        {
        float val = this->get(x,y);

        // Apply intensity window
        if (val >= iwmax) pixel = 255;
        else if (val <= iwmin) pixel = 0;
        else pixel = (unsigned char) ((val-iwmin)*255.0f/(iwmax-iwmin));

        out.write(reinterpret_cast<char*>(&pixel), 1);
        }
    out.close();	      
  }

///////////////
// outputRAW //
///////////////
// jdh 2009
template <class T>
void
Array2D<T>
::outputRAW(const std::string filename)
  {
    std::ofstream out(filename.c_str(), std::ios::binary);
    if (out.bad())
    {
      throw std::runtime_error("error opening file");
    }

    T buffer;
    for (unsigned int y = 0; y < this->_size.y; ++y)
      for (unsigned int x = 0; x < this->_size.x; ++x)
        {
        buffer = this->get(x,y);
	out.write(reinterpret_cast<char*>(&buffer), sizeof(T));
        }
    out.close();	      
  }

/////////////
// readRAW //
/////////////
// jdh 2009
template <class T>
void
Array2D<T>
::readRAW(const std::string filename, unsigned int Nx, unsigned int Ny)
  {
    std::ifstream rawin(filename.c_str(), std::ios::in | std::ios::binary);
    if (rawin.bad())
    {
      throw std::runtime_error("error opening file");
    }

    T buffer;
    for (unsigned int y = 0; y < Ny; ++y)
      for (unsigned int x = 0; x < Nx; ++x)
        {
        rawin.read(reinterpret_cast<char*>(&buffer),sizeof(T));
        this->set(x,y,buffer);
        }
    rawin.close();	      
  }

////////////
// bilerp //
////////////
// jdh 2009
template <class T>
T
Array2D<T>
::bilerp(const float x, const float y, const T& background)
{
    // a faster version of the floor function
    int floorX = static_cast<int>(x);
    int floorY = static_cast<int>(y);

    if (x < 0 && x != static_cast<int>(x)) --floorX;
    if (y < 0 && y != static_cast<int>(y)) --floorY;


    // this is not truly ceiling, but floor + 1, which is usually ceiling
    int ceilX = floorX + 1;
    int ceilY = floorY + 1;

    //
    // ^
    // |  v3   v2       -z->        v4   v5
    // y           --next slice-->      
    // |  v0   v1                   v7   v6
    //
    //      -x->
    //     
    T v0, v1, v2, v3;

    int sizeX = this->getSizeX();
    int sizeY = this->getSizeY();

    if (floorX >= 0 && ceilX < sizeX &&
	floorY >= 0 && ceilY < sizeY)
    {
      // this is the fast path
      v0 = this->get(floorX, floorY);
      v1 = this->get(ceilX, floorY);
      v2 = this->get(ceilX, ceilY);
      v3 = this->get(floorX, ceilY);
    }
    else
    {
      bool floorXIn = floorX >= 0 && floorX < sizeX;
      bool floorYIn = floorY >= 0 && floorY < sizeY;
      
      bool ceilXIn = ceilX >= 0 && ceilX < sizeX;
      bool ceilYIn = ceilY >= 0 && ceilY < sizeY;
     
      v0 = (floorXIn && floorYIn) 
           ? this->get(floorX, floorY) : background;
      v1 = (ceilXIn && floorYIn)  
           ? this->get(ceilX, floorY)  : background;
      v2 = (ceilXIn && ceilYIn)   
           ? this->get(ceilX, ceilY)   : background;
      v3 = (floorXIn && ceilYIn)  
           ? this->get(floorX, ceilY)  : background;
    }

    const double t = x - floorX;
    const double u = y - floorY;

    const double oneMinusT = 1.0 - t;
    const double oneMinusU = 1.0 - u;
  
    //
    // this is the basic bilerp function...
    //
    //     val = 
    //       v0 * (1 - t) * (1 - u)
    //       v1 * t       * (1 - u)
    //       v2 * t       * u      
    //       v3 * (1 - t) * u      
    //
    // the following saves some computation...
    //

    return      oneMinusT * (oneMinusU * v0 + u * v3) +
                   t      * (oneMinusU * v1 + u * v2);
}



#endif
