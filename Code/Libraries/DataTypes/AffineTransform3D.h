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

// Mark Foskey 5/04

#ifndef AFFINETRANSFORM_H
#define AFFINETRANSFORM_H

#ifndef SWIG

#include <stdexcept>

#include "Matrix3D.h"
#include "Vector3D.h"

#endif // !SWIG

template< class T >
class AffineTransform3D
{
public:

  typedef T CoordinateType;
  typedef AffineTransform3D< CoordinateType > Self;
  typedef Matrix3D< CoordinateType > MatrixType;
  typedef Vector3D< CoordinateType > VectorType; 

  MatrixType matrix;
  VectorType vector;

  AffineTransform3D() : matrix(), vector() {} // Identity.
  AffineTransform3D(const MatrixType& m, const VectorType& v ) 
    : matrix(m), vector(v) {}
  explicit AffineTransform3D( const MatrixType& m ) : matrix(m), vector() {}
  explicit AffineTransform3D( const VectorType& v ) : matrix(), vector(v) {}

  template< class U >
  AffineTransform3D( const AffineTransform3D<U>& t )
    : matrix( t.matrix ), vector( t.vector ) {}

#ifndef SWIG
  // Do not wrap assignment operator, python doesn't support it
  template< class U >
  Self& operator=( const AffineTransform3D<U>& rhs )
  { 
    this->matrix = rhs.matrix; 
    this->vector = rhs.vector; 
    return *this; 
  }
#endif // !SWIG

  void eye() { *this = AffineTransform3D<T>(); } // Sets to identity
  bool invert();
  Self& operator*=( const Self& rhs );
    
  Self operator*( const Self& other ) const;

template< class U >
  void applyTransform(const AffineTransform3D<U>& rhs)
  {
	this->matrix = this->matrix * rhs.matrix;
	this->vector = this->vector + rhs.vector;
  }

  VectorType operator*( const VectorType& v ) const;
  void transformVector( const VectorType& in, VectorType& out ) const;
  void transformVector( VectorType& v ) const;

  template <class U, class V>
  void transformCoordinates( const U& xIn, const U& yIn, const U& zIn, 
                             V& xOut, V& yOut, V& zOut ) const
  {
    xOut =(V)(xIn*matrix.a[0] + yIn*matrix.a[1] + zIn*matrix.a[2] + vector[0]);
    yOut =(V)(xIn*matrix.a[3] + yIn*matrix.a[4] + zIn*matrix.a[5] + vector[1]);
    zOut =(V)(xIn*matrix.a[6] + yIn*matrix.a[7] + zIn*matrix.a[8] + vector[2]);
  }

  bool operator==( const Self t ) const;

  // No other operators; e.g., '+' isn't really a sensible op.

  void writePLUNCStyle(const char* filename) const;
  void writePLUNCStyle(const std::string& filename) const
    { writePLUNCStyle(filename.c_str()); }
  void readPLUNCStyle(const char* filename);
  void readPLUNCStyle(const std::string& filename)
    { readPLUNCStyle(filename.c_str()); }

  void readITKStyle(const char* filename);
  void readITKStyle(const std::string& filename)
    { readITKStyle(filename.c_str()); }

double determinant()
{
double a = this->matrix.a[0];
double b = this->matrix.a[1];
double c = this->matrix.a[2];
double d = this->matrix.a[3];
double e = this->matrix.a[4];
double f = this->matrix.a[5];
double g = this->matrix.a[6];
double h = this->matrix.a[7];
double i = this->matrix.a[8];
double tp1 = (a*e*i) + (b*f*g) + (c*d*h);
double tp2 = (a*f*h) + (b*d*i) + (c*e*g);
double det = tp1 - tp2;
return det;
}
private:

  AffineTransform3D( const Self& first, const Self& second );

};

template <typename T>
std::ostream& 
operator<<(std::ostream& output, const AffineTransform3D<T>& t)
{
  return output << '\n' << t.matrix << "\n\n" << t.vector[0] << " "
                << t.vector[1] << " " << t.vector[2];
}

#ifndef SWIG
#include "AffineTransform3D.txx"
#endif // !SWIG

#endif  // ndef AFFINETRANSFORM_H
