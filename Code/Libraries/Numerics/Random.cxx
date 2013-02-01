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

#include "Random.h"

// NOTE: We cannot use cstdlib and cmath because MSVC implements them
// incorrectly, failing to put the functions in the std:: namespace.
#include <stdlib.h>
#include <math.h>

// produce a sample from a uniform distribution [0...1]
double Random::sampleUniform01() 
{
  return static_cast<double>(rand()) / static_cast<double>(RAND_MAX);
}

// produce a sample from a uniform distribution [min...max]
double Random::sampleUniform(const double& a, const double& b) {
  double min = a, max = b;
  if (min == max) {
    return min;
  }
  else if (min > max) {
    min = b;
    max = a;
  }
  return min + sampleUniform01() * (max - min);
}

// select a random integer in range [min...max]
int Random::sampleUniformInt(const int& a, const int& b) {
  int min = a, max = b;
  if (min == max) {
    return min;
  }
  else if (min > max) {
    min = b;
    max = a;
  }
  return min + (int) (sampleUniform01() * (max - min + 1));
}

//
// Produce a sample from a unit normal distn.  Algorithm based on
// Polar Method (Ross: A First Course in Probability).
//
double Random::sampleUnitNormal()
{
  double v1, v2, s;
  do
    {
      v1 = 2 * sampleUniform01() - 1;
      v2 = 2 * sampleUniform01() - 1;
      s  = v1*v1 + v2*v2;
    } while (s >= 1);
  return sqrt(-2 * log(s) / s) * v1;
}

//
// Produce a sample from a normal distn with given mean and variance.
//
double Random::sampleNormal(const double& mean, const double& var)
{
  return sampleUnitNormal() * sqrt(var) + mean;
}
