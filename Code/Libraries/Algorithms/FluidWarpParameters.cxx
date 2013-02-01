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
#include <string.h>
#include <stdlib.h>
#include "FluidWarpParameters.h"
#include <cstdlib>

FluidWarpParameters
::FluidWarpParameters()
  : numIterations(0),
    alpha(0),
    beta(0),
    gamma(0),
    maxPerturbation(0),
    numBasis(0),
    jacobianScale(false),
    divergenceFree(false)
{}

void
FluidWarpParameters
::writeASCII(std::ostream& output) const
{
  output << "Num Iterations:   " << numIterations   << std::endl
	 << "Alpha:            " << alpha           << std::endl
	 << "Beta:             " << beta            << std::endl
	 << "Gamma:            " << gamma           << std::endl
	 << "Max Perturbation: " << maxPerturbation << std::endl
	 << "Num Basis:        " << numBasis        << std::endl
         << "Incompressible:   " << (divergenceFree ? "Yes" : "No") << std::endl;
}

void 
FluidWarpParameters
::readASCII(std::istream& input)
{
  // Skip '#' comment lines
  std::string lineStr;
  std::getline(input, lineStr);
  while(lineStr[0] == '#') {
    std::getline(input, lineStr);
  }
  
  // Read parameters
  std::string valueStr;
  valueStr = lineStr.substr(lineStr.find_last_of(":") + 1);
  numIterations = atoi(valueStr.c_str());

  std::getline(input, lineStr);
  valueStr = lineStr.substr(lineStr.find_last_of(":") + 1);
  alpha = atof(valueStr.c_str());

  std::getline(input, lineStr);
  valueStr = lineStr.substr(lineStr.find_last_of(":") + 1);
  beta = atof(valueStr.c_str());

  std::getline(input, lineStr);
  valueStr = lineStr.substr(lineStr.find_last_of(":") + 1);
  gamma = atof(valueStr.c_str());

  std::getline(input, lineStr);
  valueStr = lineStr.substr(lineStr.find_last_of(":") + 1);
  maxPerturbation = atof(valueStr.c_str());

  std::getline(input, lineStr);
  valueStr = lineStr.substr(lineStr.find_last_of(":") + 1);
  numBasis = atoi(valueStr.c_str());

  std::getline(input, lineStr);
  valueStr = lineStr.substr(lineStr.find_last_of(":") + 1);
  if (valueStr == "Yes") divergenceFree = true;
  else divergenceFree = false;
}

std::ostream& 
operator<<(std::ostream& output, 
	   const FluidWarpParameters& parameters)
{
  parameters.writeASCII(output);
  return output;
}

std::istream&
operator>>(std::istream& input,
           FluidWarpParameters& parameters)
{
  parameters.readASCII(input);
  return input;
}
           
