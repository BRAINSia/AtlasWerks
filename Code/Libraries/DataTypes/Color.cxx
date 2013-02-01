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

#include "Color.h"
#include <iostream>

Color
::Color()
  : r(), g(), b(), a()
{}

Color
::Color(const Color& rhs)
  : r(rhs.r), g(rhs.g), b(rhs.b), a(rhs.a)
{}

Color
::Color(const double& rIn, 
	const double& gIn, 
	const double& bIn,
	const double& aIn)
  : r(rIn), g(gIn), b(bIn), a(aIn) 
{}

std::ostream& 
Color
::writeASCII(std::ostream& output) const
{
  output 
    << "[r = " << r
    << ", g = " << g
    << ", b = " << b 
    << ", a = " << a 
    << "]";
  return output;
}

std::ostream& 
operator<<(std::ostream& output, const Color& color)
{
  return color.writeASCII(output);
}

void
Color
::set(const double& rIn, 
      const double& gIn, 
      const double& bIn, 
      const double& aIn)
{
  r = rIn;
  g = gIn;
  b = bIn;
  a = aIn;
}

void
Color
::get(double& rOut, 
      double& gOut, 
      double& bOut, 
      double& aOut) const
{
  rOut = r;
  gOut = g;
  bOut = b;
  aOut = a;
}

void
Color
::setR255(unsigned int rIn) 
{
  r = double(rIn) / 255.0;
}

void
Color
::setG255(unsigned int gIn) 
{
  g = double(gIn) / 255.0;
}

void
Color
::setB255(unsigned int bIn) 
{
  b = double(bIn) / 255.0;
}

void
Color
::setA255(unsigned int aIn) 
{
  a = double(aIn) / 255.0;
}

void
Color
::set255(unsigned int rIn, 
         unsigned int gIn, 
         unsigned int bIn, 
         unsigned int aIn)
{
  r = double(rIn) / 255.0;
  g = double(gIn) / 255.0;
  b = double(bIn) / 255.0;
  a = double(aIn) / 255.0;
}

unsigned int 
Color
::getR255() const
{
  return (unsigned int)(r * 255.0);
}

unsigned int 
Color
::getG255() const
{
  return (unsigned int)(g * 255.0);
}

unsigned int 
Color
::getB255() const
{
  return (unsigned int)(b * 255.0);
}

unsigned int 
Color
::getA255() const
{
  return (unsigned int)(a * 255.0);
}

void
Color
::get255(unsigned int& rOut, 
         unsigned int& gOut, 
         unsigned int& bOut, 
         unsigned int& aOut) const
{
  rOut = (unsigned int)(r * 255.0);
  gOut = (unsigned int)(g * 255.0);
  bOut = (unsigned int)(b * 255.0);
  aOut = (unsigned int)(a * 255.0);
}
