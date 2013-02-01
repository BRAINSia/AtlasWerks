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

#ifndef Vector3D_txx
#define Vector3D_txx

#include <iostream>
#include <cmath>

template <class T>
inline
Vector3D<T>
::Vector3D()
  : x(0), y(0), z(0)
{}

template <class T>
inline
Vector3D<T>
::Vector3D(const T& xIn, 
	   const T& yIn, 
	   const T& zIn)
  :x(xIn),y(yIn), z(zIn)
{}

template <class T>
inline
T&
Vector3D<T>
::operator[](unsigned int index) 
{
  return (&x)[index];
}
  
template <class T>
inline
const T&
Vector3D<T>
::operator[](unsigned int index) const
{
  return (&x)[index];
}

template <class T>
inline
double
Vector3D<T>
::length() const
{
  return sqrt(x*x + y*y + z*z);
}

template <class T>
inline
double
Vector3D<T>
::lengthSquared() const
{
  return x*x + y*y + z*z;
}

template <class T>
inline
double
Vector3D<T>
::normL2() const
{
  return sqrt(x*x + y*y + z*z);
}

template <class T>
inline
double
Vector3D<T>
::normL1() const
{
  return (x > 0 ? x : -x) + (y > 0 ? y : -y) + (z > 0 ? z : -z);
}

template <class T>
inline
T
Vector3D<T>
::productOfElements() const
{
  return x * y * z;
}

template <class T>
inline
T
Vector3D<T>
::sumOfElements() const
{
  return x + y + z;
}

template <class T>
inline
T
Vector3D<T>
::maxElement() const
{
  return std::max(x, std::max(y, z));
}

template <class T>
inline
T
Vector3D<T>
::minElement() const
{
  return std::min(x, std::min(y, z));
}

template <class T>
inline
void
Vector3D<T>
::set(const T& x,
      const T& y,
      const T& z)
{ 
  this->x = x;
  this->y = y;
  this->z = z;
}

template <class T>
inline
void
Vector3D<T>
::scale(const T& sx,
	const T& sy,
	const T& sz)
{
  x *= sx;
  y *= sy;
  z *= sz;
}

template <class T>
inline
void
Vector3D<T>
::translate(const T& tx,
	    const T& ty,
	    const T& tz)
{
  x += tx;
  y += ty;
  z += tz;
}

template <class T>
inline
void
Vector3D<T>
::invert()
{
  x = -x;
  y = -y;
  z = -z;
}  

template <class T>
inline
void
Vector3D<T>
::normalize()	
{
  double len = 1.0 / length();
  x *= len;
  y *= len;
  z *= len;
}

template <class T>
inline
double
Vector3D<T>
::distance(const Vector3D<T>& rhs) const
{
  return sqrt(distanceSquared(rhs));
}

template <class T>
inline
double
Vector3D<T>
::distanceSquared(const Vector3D<T>& rhs) const
{
  return 
    (x - rhs.x) * (x - rhs.x) +
    (y - rhs.y) * (y - rhs.y) +
    (z - rhs.z) * (z - rhs.z);
}

template <class T>
inline
double
Vector3D<T>
::dot(const Vector3D<T>& rhs) const
{
  return (x * rhs.x + y * rhs.y + z * rhs.z);
}

template <class T>
inline
Vector3D<T>
Vector3D<T>
::cross(const Vector3D<T>& rhs) const
{
  return Vector3D<T>(y * rhs.z - z * rhs.y,
		     z * rhs.x - x * rhs.z,
		     x * rhs.y - y * rhs.x);
}

template <class T>
inline
std::ostream& 
Vector3D<T>
::writeASCII(std::ostream& output) const
{
  output << "(" << x << ", " << y << ", " << z << ")";
  return output;  
}

template <class T>
inline
std::istream& 
Vector3D<T>
::readASCII(std::istream& input)
{
  char paren, comma;
  input >> paren
	>> x >> comma
	>> y >> comma
	>> z >> paren;
  return input;
} 

template <class T>
inline
std::ostream& 
Vector3D<T>
::writeBinary(std::ostream& output) const
{
  output.write((char*)(&x), 3 * sizeof(T));
  return output;
}
  
template <class T>
inline
std::istream& 
Vector3D<T>
::readBinary(std::istream& input)
{
  input.read((char*)(&x), 3 * sizeof(T));
  return input;
}

//
// comparison operators
//
template <class T>
inline
bool 
operator<(const Vector3D<T>& lhs, const Vector3D<T>& rhs)
{
  if (lhs.x < rhs.x) return true;
  if (lhs.x > rhs.x) return false;
  if (lhs.y < rhs.y) return true;
  if (lhs.y > rhs.y) return false;
  return (lhs.z < rhs.z);
}

template <class T>
inline
bool 
operator==(const Vector3D<T>& lhs, const Vector3D<T>& rhs)
{
  return ((lhs.x == rhs.x) && 
	  (lhs.y == rhs.y) &&
	  (lhs.z == rhs.z));
}

template <class T>
inline
bool 
operator!=(const Vector3D<T>& lhs, const Vector3D<T>& rhs)
{
  return ((lhs.x != rhs.x) || 
	  (lhs.y != rhs.y) ||
	  (lhs.z != rhs.z));
}

/**
 * invert operator (unary minus)
 */
template <class T>
inline
Vector3D<T> 
operator-(const Vector3D<T>& rhs)
{
  return Vector3D<T>(-rhs.x, -rhs.y, -rhs.z);
}

/**
 * Element-wise addition
 */
template <class T, class U>
inline
Vector3D<T>& 
operator+=(Vector3D<T>& lhs, const Vector3D<U>& rhs)
{
  lhs.x += rhs.x;
  lhs.y += rhs.y;
  lhs.z += rhs.z;
  return lhs;
}

/**
 * Element-wise subtraction
 */
template <class T, class U>
inline
Vector3D<T>& 
operator-=(Vector3D<T>& lhs, const Vector3D<U>& rhs)
{
  lhs.x -= rhs.x;
  lhs.y -= rhs.y;
  lhs.z -= rhs.z;
  return lhs;
}

/**
 * Element-wise multiplication
 */
template <class T, class U>
inline
Vector3D<T>& 
operator*=(Vector3D<T>& lhs, const Vector3D<U>& rhs)
{
  lhs.x *= rhs.x;
  lhs.y *= rhs.y;
  lhs.z *= rhs.z;
  return lhs;
}

/**
 * Element-wise division
 */
template <class T, class U>
inline
Vector3D<T>& 
operator/=(Vector3D<T>& lhs, const Vector3D<U>& rhs)
{
  lhs.x /= rhs.x;
  lhs.y /= rhs.y;
  lhs.z /= rhs.z;
  return lhs;
}

/**
 * Element-wise addition
 */
template <class T, class U>
inline
Vector3D<T> 
operator+(const Vector3D<T>& lhs, const Vector3D<U>& rhs)
{
  return Vector3D<T>(static_cast<T>(lhs.x + rhs.x), 
		     static_cast<T>(lhs.y + rhs.y), 
		     static_cast<T>(lhs.z + rhs.z));
}

/**
 * Element-wise subtraction
 */
template <class T, class U>
inline
Vector3D<T> 
operator-(const Vector3D<T>& lhs, const Vector3D<U>& rhs)
{
  return Vector3D<T>(static_cast<T>(lhs.x - rhs.x), 
		     static_cast<T>(lhs.y - rhs.y), 
		     static_cast<T>(lhs.z - rhs.z));
}

/**
 * Element-wise multiply
 */
template <class T, class U>
inline
Vector3D<T> 
operator*(const Vector3D<T>& lhs, const Vector3D<U>& rhs)
{
  return Vector3D<T>(static_cast<T>(lhs.x * rhs.x), 
		     static_cast<T>(lhs.y * rhs.y), 
		     static_cast<T>(lhs.z * rhs.z));
}

/**
 * Element-wise division
 */
template <class T, class U>
inline
Vector3D<T> 
operator/(const Vector3D<T>& lhs, const Vector3D<U>& rhs)
{
  return Vector3D<T>(static_cast<T>(lhs.x / rhs.x), 
		     static_cast<T>(lhs.y / rhs.y), 
		     static_cast<T>(lhs.z / rhs.z));
}

/**
 * Add constant to each element
 */
template <class T, class U>
inline
Vector3D<T>& 
operator+=(Vector3D<T>& v, const U &u)
{
  v.x += u;
  v.y += u;
  v.z += u;
  return v;
}

/**
 * Subtract constant from each element
 */
template <class T, class U>
inline
Vector3D<T>& 
operator-=(Vector3D<T>& v, const U &u)
{
  v.x -= u;
  v.y -= u;
  v.z -= u;
  return v;
}

/**
 * Scale by a constant
 */
template <class T, class U>
inline
Vector3D<T>& 
operator*=(Vector3D<T>& v, const U &u)
{
  v.x *= u;
  v.y *= u;
  v.z *= u;
  return v;
}

/**
 * Divide by a constant
 */
template <class T, class U>
inline
Vector3D<T>& 
operator/=(Vector3D<T>& v, const U &u)
{
  v.x /= u;
  v.y /= u;
  v.z /= u;
  return v;
}

/**
 * Add a constant to each element
 */
template <class T, class U>
inline
Vector3D<T> 
operator+(const Vector3D<T>& v, const U &u)
{
  return Vector3D<T>(static_cast<T>(v.x + u), 
		     static_cast<T>(v.y + u), 
		     static_cast<T>(v.z + u)); 
}

/**
 * Add constant to each element
 */
template <class T, class U>
inline
Vector3D<T> 
operator+(const U &u, const Vector3D<T>& v)
{
  return Vector3D<T>(static_cast<T>(v.x + u),
		     static_cast<T>(v.y + u), 
		     static_cast<T>(v.z + u)); 
}

/**
 * Subtract constant from each element
 */
template <class T, class U>
inline
Vector3D<T> 
operator-(const Vector3D<T>& v, const U &u)
{
  return Vector3D<T>(static_cast<T>(v.x - u), 
		     static_cast<T>(v.y - u), 
		     static_cast<T>(v.z - u)); 
}

/**
 * Negate vector and add constant to each element
 */
template <class T, class U>
inline
Vector3D<T> 
operator-(const U &u, const Vector3D<T>& v)
{
  return Vector3D<T>(static_cast<T>(u - v.x), 
		     static_cast<T>(u - v.y),
		     static_cast<T>(u - v.z)); 
}

/**
 * Scale by a constant
 */
template <class T, class U>
inline
Vector3D<T> 
operator*(const Vector3D<T>& v, const U &u)
{
  return Vector3D<T>(static_cast<T>(v.x * u), 
		     static_cast<T>(v.y * u),
		     static_cast<T>(v.z * u)); 
}

/**
 * Scale by a constant
 */
template <class T, class U>
inline
Vector3D<T> 
operator*(const U &u, const Vector3D<T>& v)
{
  return Vector3D<T>(static_cast<T>(v.x * u), 
		     static_cast<T>(v.y * u), 
		     static_cast<T>(v.z * u)); 
}

/**
 * Divide by a constant
 */
template <class T, class U>
inline
Vector3D<T> 
operator/(const Vector3D<T>& v, const U &u)
{
  return Vector3D<T>(static_cast<T>(v.x / u), 
		     static_cast<T>(v.y / u), 
		     static_cast<T>(v.z / u)); 
}

/**
 * Divide constant by a vector (element-wise)
 */
template <class T, class U>
inline
Vector3D<T> 
operator/(const U &u, const Vector3D<T>& v)
{
  return Vector3D<T>(static_cast<T>(u / v.x), 
		     static_cast<T>(u / v.y), 
		     static_cast<T>(u / v.z)); 
}

//
// ======== input/output ========
//
template <class T>
std::ostream& 
operator<<(std::ostream& output, const Vector3D<T>& v)
{
  return v.writeASCII(output);
}

template <class T>
std::istream& 
operator>>(std::istream& input, Vector3D<T>& v)
{
  return v.readASCII(input);
}
  

template <class T>
inline Vector3D<T> max(const Vector3D<T>& a, Vector3D<T>& b)
{
   return Vector3D<T>(std::max(a.x, b.x),std::max(a.y, b.y),std::max(a.z, b.z));
}

template <class T>
inline Vector3D<T> min(const Vector3D<T>& a, Vector3D<T>& b)
{
   return Vector3D<T>(std::min(a.x, b.x),std::min(a.y, b.y),std::min(a.z, b.z));
}

#endif
