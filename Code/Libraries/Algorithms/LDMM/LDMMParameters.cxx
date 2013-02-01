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

#include <iostream>
#include <string>
#include "LDMMParameters.h"

LDMMParameters
::LDMMParameters()
  : numIterations(0),
    epsilon(0),
    sigma(0)
{}

void
LDMMParameters
::writeASCII(std::ostream& output) const
{
  output << "Num Iterations:   " << numIterations   << std::endl
	 << "Epsilon:          " << epsilon         << std::endl
	 << "Sigma:            " << sigma           << std::endl;
}

void 
LDMMParameters
::readASCII(std::istream& input)
{
}

std::ostream& 
operator<<(std::ostream& output, 
	   const LDMMParameters& parameters)
{
  parameters.writeASCII(output);
  return output;
}

std::istream&
operator>>(std::istream& input,
           LDMMParameters& parameters)
{
  parameters.readASCII(input);
  return input;
}
           
