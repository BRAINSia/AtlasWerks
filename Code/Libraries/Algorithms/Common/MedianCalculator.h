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

#ifndef __MEDIAN_CALCULATOR_H__
#define __MEDIAN_CALCULATOR_H__

#include <iostream>
#include <vector>

#include "AtlasWerksTypes.h"

#ifdef MPI_ENABLED
#include <mpi.h>
#endif // MPI_ENABLD

class MedianCalculator {

public:

  /**
   * Selects the k'th smallest element from the elements in buff,
   * across MPI nodes if necessary.  If there is only one node,
   * nTotalSz should be equal to buffSz.  Otherwise, nTotalSize is the
   * total number of elements across all nodes.  This method collects
   * all elements on a single node and then computes the median by
   * sorting the total list (not efficient)
   */
  static Real Select(Real *buff, 
		     unsigned int buffSz, 
		     unsigned int k);
  
  /**
   * Selects the k'th smallest element from the elements in buff, for
   * each entry in array k, across MPI nodes if necessary.  If there
   * is only one node, nTotalSz should be equal to buffSz.  Otherwise,
   * nTotalSize is the total number of elements across all nodes.
   * This method collects all elements on a single node and then
   * computes the median by sorting the total list (not efficient).
   * Element values are returned in rtn (must already be allocated) in
   * the same order as k.
   */
  static void Select(Real *buff, 
		     unsigned int buffSz, 
		     unsigned int *k,
		     unsigned int kSz,
		     Real *rtn);

  static void Select(std::vector<RealImage*> input,
		     unsigned int k,
		     RealImage &rtn);

private:

  static std::vector<Real> Sort(Real *buff, 
				unsigned int buffSz);

  static void MinMax(std::vector<RealImage*> input,
		     RealImage &min,
		     RealImage &max);
  
  static void GetIndex(std::vector<RealImage*> input,
		       const RealImage &pim,
		       const RealImage &gmin,
		       const RealImage &gmax,
		       unsigned int *idx,
		       Real *min,
		       Real *max);

};

#endif // __MEDIAN_CALCULATOR_H__
