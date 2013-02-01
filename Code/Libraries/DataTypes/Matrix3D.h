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

#ifndef MATRIX3D_H
#define MATRIX3D_H

#ifndef SWIG

#include <iosfwd>
#include "Vector3D.h"

typedef long int integer;
typedef double   doublereal;
typedef long int longint;

extern "C"
{
  int dgetrf_(const integer *m, const integer *n, doublereal *a, integer *
	      lda, integer *ipiv, integer *info);

  int dgetri_(const integer *n, doublereal *a, integer *lda, integer 
	      *ipiv, doublereal *work, const integer *lwork, integer *info);

  int dgesvd_(char *jobu, char *jobvt, const integer *m, const integer *n,
              doublereal *a, const integer *lda, doublereal *s,
              doublereal *u, const integer * ldu, doublereal *vt,
              const integer *ldvt, doublereal *work, const integer *lwork,
              integer *info);
  int dsyev_(char *jobz, char *uplo, integer *n, doublereal *a, 
	     integer *lda, doublereal *w, doublereal *work, integer *lwork, 
	     integer *info);
}

#endif // !SWIG

template <typename T>
class Matrix3D
{
public:
  T a[9];

  Matrix3D();

  template <typename U>
  Matrix3D(const Matrix3D<U>& rhs)
  {
    a[0] = static_cast<T>(rhs.a[0]);
    a[1] = static_cast<T>(rhs.a[1]);
    a[2] = static_cast<T>(rhs.a[2]);
    a[3] = static_cast<T>(rhs.a[3]);
    a[4] = static_cast<T>(rhs.a[4]);
    a[5] = static_cast<T>(rhs.a[5]);
    a[6] = static_cast<T>(rhs.a[6]);
    a[7] = static_cast<T>(rhs.a[7]);
    a[8] = static_cast<T>(rhs.a[8]);
  };

  template <typename U>
  Matrix3D(const U *in)
  {
    a[0] = static_cast<T>(in[0]);
    a[1] = static_cast<T>(in[1]);
    a[2] = static_cast<T>(in[2]);
    a[3] = static_cast<T>(in[3]);
    a[4] = static_cast<T>(in[4]);
    a[5] = static_cast<T>(in[5]);
    a[6] = static_cast<T>(in[6]);
    a[7] = static_cast<T>(in[7]);
    a[8] = static_cast<T>(in[8]);
  }


#ifndef SWIG
  // Do not wrap assignment operator, python doesn't support it
  template <typename U>
  Matrix3D<T>& operator=(const Matrix3D<U>& rhs)
  {
    if (this != &rhs) 
      {
	a[0] = static_cast<T>(rhs.a[0]);
	a[1] = static_cast<T>(rhs.a[1]);
	a[2] = static_cast<T>(rhs.a[2]);
	a[3] = static_cast<T>(rhs.a[3]);
	a[4] = static_cast<T>(rhs.a[4]);
	a[5] = static_cast<T>(rhs.a[5]);
	a[6] = static_cast<T>(rhs.a[6]);
	a[7] = static_cast<T>(rhs.a[7]);
	a[8] = static_cast<T>(rhs.a[8]);    
      }
    return *this;
  };

#endif // !SWIG

  T& operator()(const unsigned int rowIndex,
		const unsigned int columnIndex);
  const T& operator()(const unsigned int rowIndex,
                      const unsigned int columnIndex) const;

  T& operator()(const unsigned int elementIndex);
  const T& operator()(const unsigned int elementIndex) const;

  bool operator==( const Matrix3D<T> &other ) const;

  const T& get(const unsigned int rowIndex,
	       const unsigned int columnIndex);

  void set(const unsigned int rowIndex,
	   const unsigned int columnIndex,
	   const T& val);

  // set to the identity
  void eye();

  Matrix3D<T> Transpose();

  std::ostream& writeASCII(std::ostream& output = std::cerr) const;

  bool invert();
  static bool computeInverse(const T* const a, T* const ainv);

  static bool computeSVD(const T* const A, T* const U, T* const Vt, T* const sv);

  static bool factorEVsym(const T* const A, T* const eVals, T* const eVecs, bool upper=true);

private:

  template<typename U, typename V>
  static void _transposeMatrixData(const U* const a, V* const at)
  {
    at[0] = static_cast<V>(a[0]); 
    at[1] = static_cast<V>(a[3]); 
    at[2] = static_cast<V>(a[6]);
    at[3] = static_cast<V>(a[1]); 
    at[4] = static_cast<V>(a[4]); 
    at[5] = static_cast<V>(a[7]);
    at[6] = static_cast<V>(a[2]); 
    at[7] = static_cast<V>(a[5]); 
    at[8] = static_cast<V>(a[8]);  
  }

};

template <typename T, typename U>
Vector3D<T> operator*(const Matrix3D<T>& m, 
		      const Vector3D<U>& v);

template <typename T, typename U>
Matrix3D<T>& operator*=(Matrix3D<T>& lhs, const Matrix3D<U>& rhs);

template <typename T, typename U>
Matrix3D<T>  operator* (const Matrix3D<T>& lhs, const Matrix3D<U>& rhs);

template <typename T>
std::ostream& operator<<(std::ostream& output, const Matrix3D<T>& matrix);

#ifndef SWIG
#include "Matrix3D.txx"
#endif // !SWIG
#endif
