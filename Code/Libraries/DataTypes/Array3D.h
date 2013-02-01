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

#ifndef ARRAY3D_H
#define ARRAY3D_H

#include "Array2D.h"
#include "Vector3D.h"

#ifndef SWIG

#include <iosfwd>
#include <stdexcept>
#include <float.h>

#endif // SWIG

// NOTE: The data layout is such that the X coordinate is the one that
// changes most rapidly, and Z the least.  So, when iterating via 3
// nested loops, the innermost loop should be controlled by the
// changing X index.

template <class T>
class Array3D
{
public:
  typedef Vector3D<unsigned int>   IndexType;
  typedef Vector3D<unsigned int>   SizeType;

  Array3D();
  Array3D(unsigned int xSize,
	  unsigned int ySize, 
	  unsigned int zSize);

  Array3D(const SizeType& size);
  Array3D(const Array3D<T>& rhs);
  virtual ~Array3D();

#ifndef SWIG
  // Assignment operators not handled by python
  Array3D<T>& operator=(const Array3D<T>& rhs);
#endif // SWIG

  const SizeType& getSize() const;
  unsigned int getSizeX() const;
  unsigned int getSizeY() const;
  unsigned int getSizeZ() const;

  bool isEmpty() const;

  // does not maintain data
  void resize(const SizeType& size);
  void resize(unsigned int xSize, 
	      unsigned int ySize,
	      unsigned int zSize);
  
 
  unsigned int getNumElements() const;
  unsigned int getSizeBytes() const;

  void set(unsigned int xIndex,
	   unsigned int yIndex,
	   unsigned int zIndex, 
	   const T& item);

#ifndef SWIG
  // this is changed to return a value, not a reference, in DataTypes.i
  
  T& get(unsigned int xIndex,
	 unsigned int yIndex,
	 unsigned int zIndex) const;

  T& get(unsigned int elementIndex) const;
  
#endif // SWIG
  
  T& operator()(const IndexType& index);
  T& operator()(unsigned int xIndex, 
		unsigned int yIndex,
		unsigned int zIndex);
  T& operator()(unsigned int elementIndex);

  bool isValidIndex(const IndexType& index);

  const T& operator()(const IndexType& index) const;
  const T& operator()(unsigned int xIndex, 
		      unsigned int yIndex,
		      unsigned int zIndex) const;
  const T& operator()(unsigned int elementIndex) const;

  void fill(const T& fillValue);

  // Multiply all values by d
  void scale(const T& v);
  void add(const T& c);
  void addScalar(const double& scalar);

  template <class U> void pointwiseMultiplyBy(const Array3D<U>& rhs);

#ifdef SWIG
  %template(pointwiseMultiplyBy) pointwiseMultiplyBy<float>;
#endif // SWIG

  template <class U> void pointwiseDivideBy(const Array3D<U>& rhs);

#ifdef SWIG
  %template(pointwiseDivideBy) pointwiseDivideBy<float>;
#endif // SWIG

  void pointwiseAdd(const Array3D<T>& rhs);
  void pointwiseSubtract(const Array3D<T>& rhs);

  bool operator==(const Array3D<T>& rhs) const;
  bool operator!=(const Array3D<T>& rhs) const;

  T* getDataPointer(const IndexType& index);
  T* getDataPointer(unsigned int xIndex = 0, 
		    unsigned int yIndex = 0, 
		    unsigned int zIndex = 0);

  const T* getDataPointer(const IndexType& index) const;
  const T* getDataPointer(unsigned int xIndex = 0, 
			  unsigned int yIndex = 0, 
			  unsigned int zIndex = 0) const;
  
  void setData(const Array3D<T>& rhs);
  void copyData(const void* const dataPtr);

  const T getAverage() const;
  const T getSum() const;
    
  Array2D<T> getSlice(unsigned int sliceNormal, unsigned int sliceNum) const;

protected:

#ifndef SWIG

  SizeType     _size;
  unsigned int _xySize;
  unsigned int _yzSize;
  unsigned int _zxSize;
  unsigned int _xyzSize;
  T ***_slicePtrs;
  T  **_rowPtrs;
  T   *_dataPtr;

  void _allocateData();
  void _deallocateData();

#endif // SWIG

};

#ifndef SWIG
#include "Array3D.txx"
#endif // SWIG

#ifndef SWIG

inline Vector3D<float> getMax(Array3D<Vector3D<float> >&a){
    Vector3D<float> s(-FLT_MAX, -FLT_MAX, -FLT_MAX);
    for (unsigned int i=0; i< a.getNumElements();++i){
        s.x = (s.x > a(i).x) ? s.x : a(i).x;
        s.y = (s.y > a(i).y) ? s.y : a(i).y;
        s.z = (s.z > a(i).z) ? s.z : a(i).z;
    }
    return s;
}

inline float getMax(Array3D<float >&a){
    float s(-FLT_MAX);
    for (unsigned int i=0; i< a.getNumElements();++i){
        s = (s > a(i)) ? s : a(i);
    }
    return s;
}

inline float getSum(Array3D<float >&a){
    double sum = 0.0;
    for (unsigned int i=0; i< a.getNumElements();++i)
        sum += a(i);
    return sum;
}

inline Vector3D<float> getSum(Array3D<Vector3D<float> >&a){
    double sumx = 0.0, sumy = 0.0, sumz = 0.0;
    for (unsigned int i=0; i< a.getNumElements();++i){
        sumx += a(i).x;
        sumy += a(i).y;
        sumz += a(i).z;
    }
    return Vector3D<float>(sumx, sumy, sumz);
}

#endif // SWIG

#endif
