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

#ifndef RANDOM_H
#define RANDOM_H

class Random {
 public:

  // produce a sample from a uniform distribution [0...1]
  static double sampleUniform01();

  // produce a sample from a uniform distribution [min...max]
  static double sampleUniform(const double& min, const double& max);

  // select a random integer in range [min...max]
  static int sampleUniformInt(const int& min, const int& max);

  // produce a sample from a unit normal distribution
  static double sampleUnitNormal();

  // produce a sample from a normal distribution with given mean and variance
  static double sampleNormal(const double& mean, const double& var);
};

#endif
