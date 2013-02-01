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

#include "Statistics.h"
#include <algorithm>
#include <numeric>
#include <vector>
#include <cmath>

template <class T>
T 
Statistics::
arithmeticMean(const std::vector<T>& vals)
{
  double sum = std::accumulate(vals.begin(), vals.end(), 0.0);
  return static_cast<T>(sum / static_cast<double>(vals.size()));
}

template <class T, class U>
T 
Statistics::
weightedArithmeticMean(const std::vector<T>& vals, 
                       const std::vector<U>& weights)
{
  double weightSum = std::accumulate(weights.begin(), weights.end(), 0.0);
  double sum = std::inner_product(vals.begin(), vals.end(), 
                                  weights.begin(), 0.0);
  return static_cast<T>(sum / weightSum);
}

template <class T>
T 
Statistics::
trimmedMean(const std::vector<T>& vals, 
            unsigned int numTrimLeft, 
            unsigned int numTrimRight)
{
  std::vector<T> tmp(vals);
  std::sort(tmp.begin(), tmp.end());
  double sum = std::accumulate(tmp.begin() + numTrimLeft, 
                               tmp.end() - numTrimRight, 0.0);
  return static_cast<T>(sum / static_cast<double>(tmp.size() 
                                                  - numTrimLeft 
                                                  - numTrimRight));
}

template <class T>
T 
Statistics::
winsorizedMean(const std::vector<T>& vals, 
               unsigned int numStackLeft, 
               unsigned int numStackRight)
{
  std::vector<T> tmp(vals);
  std::sort(tmp.begin(), tmp.end());    
  
  // stack outliers
  unsigned int i;
  int size = tmp.size();
  for (i = 0; i < numStackLeft; ++i)
  {
	tmp[i] = tmp[numStackLeft];
  }
  for (i = 0; i < numStackRight; ++i)
  {
    tmp[size - 1 - i] = tmp[size - 1 - numStackRight];
  }
  
  double sum = std::accumulate(tmp.begin(),tmp.end(), 0.0);
  return static_cast<T>(sum / static_cast<double>(tmp.size())); 
}

template <class T>
T 
Statistics::
stdDev(const std::vector<T>& vals)
{
  return static_cast<T>(sqrt(variance(vals, arithmeticMean(vals))));
}

template <class T, class U>
T 
Statistics::
weightedStdDev(const std::vector<T>& vals, const std::vector<U>& weights)
{
  return static_cast<T>(sqrt(weightedVariance(vals, weights, 
                                              arithmeticMean(vals))));
}

template <class T>
T 
Statistics::
variance(const std::vector<T>& vals)
{
  return variance(vals, arithmeticMean(vals));
}

template <class T>
T 
Statistics::
variance(const std::vector<T>& vals, 
         const T& mean)
{
  unsigned int size = vals.size();
  double sum = 0;
  for (unsigned int i = 0; i < size; ++i)
  {
    sum += (vals[i] - mean) * (vals[i] - mean);
  }
  return static_cast<T>(sum / (static_cast<double>(size) - 1));
}

template <class T, class U>
T 
Statistics::
weightedVariance(const std::vector<T>& vals, 
                 const std::vector<U>& weights,
                 const T& mean)
{
  unsigned int size = vals.size();
  double sum = 0;
  double weightSum = 0;
  for (unsigned int i = 0; i < size; ++i)
  {
    sum += weights[i] * (vals[i] - mean) * (vals[i] - mean);
    weightSum += weights[i];
  }
  return static_cast<T>(sum / weightSum);
}

