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

#ifndef __ATLASWERKS_TYPES_H__
#define __ATLASWERKS_TYPES_H__

#include "Vector3D.h"
#include "Array3D.h"
//#include "AffineTransform3D.h"
#include "DataTypes/Image.h"
#include "AtlasWerksException.h"

#ifndef SWIG
#include <complex>
#endif // SWIG

#define ATLASWERKS_EPS 0.0001

typedef float Real;

typedef Image<Real> RealImage;
typedef Array3D<Vector3D<Real> > VectorField;
//typedef AffineTransform3D<Real> Affine3D;

typedef Real VectorFieldCoordType;

typedef RealImage::SizeType SizeType;
typedef RealImage::ContinuousPointType OriginType;
typedef RealImage::SpacingType SpacingType;

// This just clears up some inadequacies with std::complex that would
// otherwise cause stuff to go haywire when doing ordinary arithmetic
// with complex numbers..
template<typename Tp, typename T>
std::complex<Tp> operator/(const std::complex<Tp>& c, const T& t)
{
  return std::complex<Tp>(c.real()/t, c.imag()/t);
}
template<typename Tp, typename T>
std::complex<Tp> operator*(const T& c, const std::complex<Tp>& t)
{
  return std::complex<Tp>(t.real()*c, t.imag()*c);
}
template<typename Tp, typename T>
std::complex<Tp> operator*(const std::complex<Tp>& c, const T& t)
{
  return std::complex<Tp>(c.real()*t, c.imag()*t);
}

typedef std::complex<Real> Complex;

// Real part
inline Real real(Real r){ return r;}
inline Real real(Complex c){ return c.real();}

// Imaginary part
inline Real imag(Real r){ return 0.0f;}
inline Real imag(Complex c){ return c.imag();}

// Complex conjugate
inline Real conj(Real r){ return r;}
inline Complex conj(Complex c){ return std::conj(c);}

// Check if either part is NaN
// inline bool isnan(Real r){ return std::isnan(r);}
inline bool isnan(Complex c){ return std::isnan(c.real()) || std::isnan(c.imag());}

// Check if either part is infinity
// inline bool isinf(Real r){ return std::isinf(r);}
inline bool isinf(Complex c){ return std::isinf(c.real()) || std::isinf(c.imag());}

#endif // __ATLASWERKS_TYPES_H__
