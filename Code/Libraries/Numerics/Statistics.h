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

#ifndef STATISTICS_H
#define STATISTICS_H

#include <vector>

class Statistics
{
 public:
  // sum of elements divided by count of elements
  template <class T>
    static
    T arithmeticMean(const std::vector<T>& vals);
  
  // (val1 * weight1 + ... + valN * weightN) / (weight1 + ... + weightN)
  template <class T, class U>
    static
    T weightedArithmeticMean(const std::vector<T>& vals, 
                             const std::vector<U>& weights);
  
  // arithmetic mean except that outliers are ignored
  template <class T>
    static
    T trimmedMean(const std::vector<T>& vals, 
                  unsigned int numTrimLeft, 
                  unsigned int numTrimRight);

  // arithmetic mean except that outliers are moved in (each outlier
  // takes the value of a non-outlier---it is stacked in the bin for
  // that value)
  template <class T>
    static
    T winsorizedMean(const std::vector<T>& vals, 
                     unsigned int numStackLeft, 
                     unsigned int numStackRight);

  // unbiased sample standard deviation
  template <class T>
    static
    T stdDev(const std::vector<T>& vals);

  // compute the standard deviation when a weight is given for
  // each value
  template <class T, class U>
    static
    T weightedStdDev(const std::vector<T>& vals, 
                     const std::vector<U>& weights);

  // syntactic sugar---this will compute and then throw away the
  // sample mean.  the unbiased sample variance is returned.
  template <class T>
    static
    T variance(const std::vector<T>& vals);

  // compute the unbiased sample variance given the sample mean
  template <class T>
    static
    T variance(const std::vector<T>& vals, 
               const T& mean);

  // compute variance given a weight for each value
  template <class T, class U>
    static
    T weightedVariance(const std::vector<T>& vals, 
                       const std::vector<U>& weights,
                       const T& mean);
};

#include "Statistics.txx"

#endif
