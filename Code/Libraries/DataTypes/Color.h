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

#ifndef COLOR_H
#define COLOR_H

#include <iostream>

class Color
{
public:
  double r, g, b, a;
  
  Color();
  Color(const Color& rhs);
  Color(const double& rIn, 
	const double& gIn, 
	const double& bIn,
	const double& aIn);
  std::ostream& writeASCII(std::ostream& output = std::cerr) const;
  
  void set(const double& rIn, 
           const double& gIn, 
           const double& bIn, 
           const double& aIn);
  void SetR(const double& r) { this->r = r; }
  void SetG(const double& g) { this->g = g; }
  void SetB(const double& b) { this->b = b; }
  void SetA(const double& a) { this->a = a; }

  void get(double& rOut,
           double& gOut,
           double& bOut,
           double& aOut) const;
  double getR() const { return r; }
  double getG() const { return g; }
  double getB() const { return b; }
  double getA() const { return a; }

  void set255(unsigned int rIn,
              unsigned int gIn,
              unsigned int bIn,
              unsigned int aIn);           
  void setR255(unsigned int rIn);
  void setG255(unsigned int gIn);
  void setB255(unsigned int bIn);
  void setA255(unsigned int aIn);

  void get255(unsigned int& rOut,
              unsigned int& gOut,
              unsigned int& bOut,
              unsigned int& aOut) const;
  unsigned int getR255() const;
  unsigned int getG255() const;
  unsigned int getB255() const;
  unsigned int getA255() const;

};

std::ostream& operator<<(std::ostream& output, const Color& color);

#endif
