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

#ifndef LDMMParameters_h
#define LDMMParameters_h

#include <iosfwd>

class LDMMParameters
{
public:
  unsigned int    numIterations;
  double          epsilon;
  double          sigma;

  LDMMParameters();
  void writeASCII(std::ostream& output) const;
  void readASCII(std::istream& input);
};

std::ostream& operator<<(std::ostream& output, 
			 const LDMMParameters& parameters);
std::istream& operator>>(std::istream& input, 
			 LDMMParameters& parameters);
#endif
