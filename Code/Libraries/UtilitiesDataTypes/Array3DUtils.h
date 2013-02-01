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

#ifndef ARRAY3D_UTILS_H
#define ARRAY3D_UTILS_H

#ifndef SWIG

#include <deque>
#include <assert.h>
#include <limits>
#include "Vector3D.h"
#include "Array3D.h"
#include "DownsampleFilter3D.h"
#include <algorithm>
#include <limits>
#include <string>
#include <numeric>
#include <vector>
#include <iostream>
#include <fstream>
#include "ROI.h"
#include "Timer.h"
#include <typeinfo>

#endif // SWIG

#ifndef DEFAULT_INTERP_METHOD
#define DEFAULT_INTERP_METHOD INTERP_LINEAR
#endif // !DEFAULT_INTERP_METHOD

// scoped version
#define DEFAULT_SCALAR_INTERP Array3DUtils::DEFAULT_INTERP_METHOD

#ifndef DEFAULT_BACKGROUND_STRATEGY
#define DEFAULT_BACKGROUND_STRATEGY BACKGROUND_STRATEGY_VAL
#endif // !DEFAULT_INTERP_METHOD

// scoped version
#define DEFAULT_SCALAR_BACKGROUND_STRATEGY Array3DUtils::DEFAULT_BACKGROUND_STRATEGY

class Array3DUtils
{
public:
  /**
   * Strategies for functions that do trilinear interpolation -- what
   * to do if the value falls outside the image region.
   */
  enum ScalarBackgroundStrategyT { 
    BACKGROUND_STRATEGY_CLAMP, /// Clamp to the closest pixel value
    BACKGROUND_STRATEGY_WRAP, /// Wrap around the image
    BACKGROUND_STRATEGY_VAL  /// Use a background value, zero if not specified
  };

  /**
   * selection of interpolation strategy
   */
  enum InterpT {INTERP_NN, INTERP_LINEAR, INTERP_CUBIC};

  /**
   * Compile-time choice of interp method and background strategy
   */
  template <class T, 
	    ScalarBackgroundStrategyT BackgroundStrategy,
	    InterpT InterpMethod>
  static
  inline
  T 
  interp(const Array3D<T>& array,
	 const double& x,
	 const double& y,
	 const double& z,
	 const T& background=(T)0.0)
  {
    if(InterpMethod == INTERP_NN){
      return Array3DUtils::nearestNeighbor(array, x, y, z, background);
    }else if(InterpMethod == INTERP_LINEAR){
      return Array3DUtils::trilerp<T, BackgroundStrategy>(array, x, y, z, background); 
    }else if(InterpMethod == INTERP_CUBIC){
      return Array3DUtils::cubicSplineinterp<T, BackgroundStrategy>(array, x, y, z, background);	
    }
  }; 

  template <class T>
  static
  inline
  T 
  interp(const Array3D<T>& array,
  	 const double& x,
  	 const double& y,
  	 const double& z,
  	 const T& background=(T)0.0)
  {
    return interp<T, DEFAULT_BACKGROUND_STRATEGY, DEFAULT_INTERP_METHOD>(array, x, y, z, background);
  }; 

  template <class T, 
	    ScalarBackgroundStrategyT BackgroundStrategy,
	    InterpT InterpMethod>
  static
  inline
  T 
  interp(const Array3D<T>& array,
	 const Vector3D<double>& index,
	 const T& background=(T)0.0)
  {
    return interp<T, BackgroundStrategy, InterpMethod>(array, index.x, index.y, index.z, background);
  }; 

  template<class T>
  static
  inline
  T 
  interp(const Array3D<T>& array,
	 const Vector3D<double>& index,
	 const T& background=(T)0.0)
  {
    return interp<T, DEFAULT_BACKGROUND_STRATEGY, DEFAULT_INTERP_METHOD>(array, index.x, index.y, index.z, background);
  }; 
  
  /**  
   *  return array(round(x,y,z))
   *  if round(x,y,z) falls outside array, return background
   *  
   *  bcd 2003
   */
  template <class T>
  static
  T nearestNeighbor(const Array3D<T>& array,
		    const double& x,
		    const double& y,
		    const double& z,
		    const T& background);

  template <class T>
  static
  T nearestNeighbor(const Array3D<T>& array, 
		    const Vector3D<double>& index,
		    const T& background);

#ifdef SWIG
  %template(nearestNeighbor) nearestNeighbor< float >;
  %template(nearestNeighbor) nearestNeighbor< Vector3D< float > >;
#endif // SWIG

  /**  
   *  trilinear interpolation into array at position x,y,z
   *  
   *  value is interpolated from corners of the cube...
   *  
   *  ^
   *  |  v3   v2       -z->        v4   v5
   *  y           --next slice-->      
   *  |  v0   v1                   v7   v6
   *  
   *       -x->
   *      
   *  where v0 is floor(x), floor(y), floor(z)
   *  and v5 is floor(x)+1, floor(y)+1, floor(z)+1
   *  and so on.
   *  
   *  if any corner of the cube is outside of the volume, that corner
   *  is set to background before performing the interpolation.  If 
   *  background is set to numeric_limits<T>::max(), then clamping
   *  is performed.
   *  
   *  
   *  a fair amount of optimization work has gone into this function
   *  
   *  bcd 2003
   */  
  template <class T, ScalarBackgroundStrategyT BackgroundStrategy>
  static
  T trilerp(const Array3D<T>& array,
	    const double& x,
	    const double& y,
	    const double& z,
	    const T& background = 0.0F);

  template <class T>
  static
  T trilerp(const Array3D<T>& array,
	    const double& x,
	    const double& y,
	    const double& z,
	    const T& background = 0.0F)
  {
    return trilerp<T, DEFAULT_SCALAR_BACKGROUND_STRATEGY>(array, x, y, z, background);
  }


  template <class T, ScalarBackgroundStrategyT BackgroundStrategy>
  static
  T trilerp(const Array3D<T>& array, 
	    const Vector3D<double>& position,
	    const T& background=0.0F);

  template <class T>
  static
  T trilerp(const Array3D<T>& array, 
	    const Vector3D<double>& position,
	    const T& background=0.0F)
  {
    return trilerp<T, DEFAULT_SCALAR_BACKGROUND_STRATEGY>(array, position, background);
  }

#ifdef SWIG
  %template(trilerp) trilerp< float, DEFAULT_SCALAR_BACKGROUND_STRATEGY >;
#endif // SWIG
  
  template <class T, ScalarBackgroundStrategyT BackgroundStrategy>
  static
  T cubicSplineinterp(const Array3D<T>& array, 
		      const Vector3D<double>& position,
		      const T& background);

  template <class T, ScalarBackgroundStrategyT BackgroundStrategy>
  static
  T cubicSplineinterp(const Array3D<T>& array,
		      const double& x,
		      const double& y,
		      const double& z,
		      const T& background = 0.0F);
  
  template <class T> static T cubicsplineinterp(T*,double t,double u,double v);
  
  /**  ------------------------------------------------------------------------
   *  Trilinear interpolation split into two functions.  The first
   *  computes 8 indices into an array and corresponding coefficients.
   *  The second simply applies them.  This makes sense if you have to
   *  trilerp the same location in several images.
   */

  template <class T>
  static
  bool computeTrilerpCoeffs(const Array3D<T>& array,
			    const double& x, const double& y, const double& z,
			    size_t& i0, size_t& i1, size_t& i2, size_t& i3,
			    size_t& i4, size_t& i5, size_t& i6, size_t& i7,
			    double& a0, double& a1, double& a2, double& a3,
			    double& a4, double& a5, double& a6, double& a7);
  
#ifdef SWIG
  %template(computeTrilerpCoeffs) computeTrilerpCoeffs< float >;
#endif // SWIG

  // If the appropriate trilinear interpolation coefficients and
  // integer indices have been computed, apply them to a given array3D
  template <class T>
  static
  T weightedSumOfArrayValues(
			     const Array3D<T>& array,
			     const size_t& i0, const size_t& i1, const size_t& i2, const size_t& i3,
			     const size_t& i4, const size_t& i5, const size_t& i6, const size_t& i7,
			     const double& a0, const double& a1, const double& a2, const double& a3,
			     const double& a4, const double& a5, const double& a6, const double& a7);

#ifdef SWIG
  %template(weightedSumOfArrayValues) weightedSumOfArrayValues< float >;
#endif // SWIG

  template <class T>
  static
  void extractROI(const Array3D<T>& array,
                  Array3D<T>& roi,
                  const Vector3D<int>& roiStart,
                  const Vector3D<unsigned int>& roiSize);

  template <class T>
  static
  void extractROI(const Array3D<T>& array,
                  Array3D<T>& roiArray,
                  const ROI<int, unsigned int>& roi);

#ifdef SWIG
  %template(extractROI) extractROI< float >;
  %template(extractROI) extractROI< Vector3D< float > >;
#endif // SWIG

  /**  
   *  fill a region with a given value
   *  
   *  foskey 2005
   */  
  template <class T>
  static
  void fillRegion(Array3D<T>& array,
                  const ROI<int, unsigned int>& roi,
                  const T& value);

#ifdef SWIG
  %template(fillRegion) fillRegion< float >;
  %template(fillRegion) fillRegion< Vector3D< float > >;
#endif // SWIG

  /**  
   *  copy a region from one Array3D<T> to another of the same type.
   *  the region is specified by an origin and extent(size), these are
   *  specified in array index coordinates.
   *  
   *  bcd 2003
   */  
  template <class T>
  static
  void copyRegion(const Array3D<T>& inArray,
		  Array3D<T>& outArray,
		  const typename Array3D<T>::IndexType& inOrigin,
		  const typename Array3D<T>::IndexType& outOrigin,
		  const typename Array3D<T>::SizeType& regionSize);

#ifdef SWIG
  %template(copyRegion) copyRegion< float >;
  %template(copyRegion) copyRegion< Vector3D< float > >;
#endif // SWIG

  /**  
   *  copy a region from one Array3D<T> to another of type U.  each
   *  element is cast from type T to type U.  the region is specified
   *  by an origin and extent(size), these are specified in array index
   *  coordinates.
   *  
   *  bcd 2003
   */  
  template <class T, class U>
  static
  void copyRegionCast(const Array3D<T>& inArray,
		      Array3D<U>& outArray,
		      const typename Array3D<T>::IndexType& inOrigin,
		      const typename Array3D<U>::IndexType& outOrigin,
		      const typename Array3D<T>::SizeType& regionSize);

#ifdef SWIG
  // since SWIG only supports float arrays, no casting really makes sense
#endif // SWIG
  
  template <class T, class U>
  static
  void copyCast(const Array3D<T>& inArray,
		Array3D<U>& outArray);

#ifdef SWIG
  // since SWIG only supports float arrays, no casting really makes sense
#endif // SWIG
  
  /**  
   *  copy a region from one Array3D<T> to another of type U.  each
   *  element is rounded to the nearest integer and then cast to type
   *  U.  the region is specified by an origin and extent(size), these
   *  are specified in array index coordinates.
   *  
   *  bcd 2003
   */  
  template <class T, class U>
  static
  void copyRegionRound(const Array3D<T>& inArray,
		       Array3D<U>& outArray,
		       const typename Array3D<T>::IndexType& inOrigin,
		       const typename Array3D<U>::IndexType& outOrigin,
		       const typename Array3D<T>::SizeType& regionSize);

#ifdef SWIG
  %template(copyRegionRound) copyRegionRound< float, float >;
#endif // SWIG

  /**  
   *  compute the centroid of a region in an image.  the centroid is
   *  based on the origin of the array, not the origin of the region of
   *  interest.
   *  
   *  bcd/lorenzen 2003
   */  
  template <class T>
  static
  Vector3D<double> computeCentroid(const Array3D<T>& array,
				   const typename Array3D<T>::IndexType& origin,
				   const typename Array3D<T>::SizeType& size);

#ifdef SWIG
  %template(computeCentroid) computeCentroid< float >;
#endif // SWIG

  /**
   * a cheap way to downsample an image.
   *  
   * if n is the downsampleFactor, take every nth element of an
   * Array3D in each dimension (starting with the 0th element),
   * producing an Array3D of size max(floor((x,y,z)/n, 1)
   *
   * bcd 2003
   */
  template <class T>
  static
  void downsampleByInt(const Array3D<T>& input,
		       Array3D<T>& output,
		       unsigned int downsampleFactor);

#ifdef SWIG
  %template(downsampleByInt) downsampleByInt< float >;
#endif // SWIG

  /**  
   *  a cheap way to downsample an image.
   *   
   *  if n is the downsampleFactor, take every nth element of an
   *  Array3D in each dimension (starting with the 0th element),
   *  producing an Array3D of size max(floor((x,y,z)/n, 1)
   *  
   *  bcd 2003
   */  
  template <class T>
  static
  void downsampleByInts(const Array3D<T>& input,
                        Array3D<T>& output,
                        const Vector3D<unsigned int>& downsampleFactors);

#ifdef SWIG
  %template(downsampleByInts) downsampleByInts< float >;
#endif // SWIG

  template <class T>
  static
  void downsampleByTwo(const Array3D<T>& input,
		       Array3D<T>& output);

#ifdef SWIG
  %template(downsampleByTwo) downsampleByTwo< float >;
#endif // SWIG

  /**
   * output can be the same array as input
   *
   * jsp 2010
   */
  template <class T>
  static
  void gaussianBlur (const Array3D<T>& input,
		     Array3D<T>& output, 
		     const Vector3D<double>& sigma,
		     const Vector3D<int>& kernelSize);
  
#ifdef SWIG
  %template(gaussianBlur) gaussianBlur< float >;
#endif // SWIG
  
  /**
   * output can be the same array as input
   *
   * prigent 2004
   */
  template <class T>
  static
  void gaussianDownsample (const Array3D<T>& input,
			   Array3D<T>& output, 
			   const Vector3D<int>& factors,
			   const Vector3D<double>& sigma,
			   const Vector3D<int>& kernelSize);

#ifdef SWIG
  %template(gaussianDownsample) gaussianDownsample< float >;
#endif // SWIG
  
  /**
   * compute the gradient of an Array3D
   *
   * symmetric difference is used except on the boundaries where
   * foreward and reverse difference are used.  If 'wrap' is true,
   * use wrapped symmetric differences throughout
   *
   * pjl 2004 (jdh 2008)
   */
  template <class T, class U>
  static
  void computeGradient(const Array3D<T>& array, 
		       Array3D<Vector3D<U> >& grad, 
		       Vector3D<double> spacing = Vector3D<double>(1.0,1.0,1.0),
		       bool wrap=false);

#ifdef SWIG
  %template(computeGradient) computeGradient< float, float >;
#endif // SWIG
  
  /**  
   *  
   *  compute the laplacian of an Array3D
   *  
   *  
   *  dp 2004
   */
  template <class T, class U>
  static
  void computeLaplacian(const Array3D<Vector3D<T> >& array,
			Array3D<Vector3D<U> >& laplacian);

#ifdef SWIG
  %template(computeLaplacian) computeLaplacian< float, float >;
#endif // SWIG

  /**  
   *  return the elementwise squared difference between two arrays
   *  
   *  sum (a1(x) - a2(x))^2
   *   x
   *  
   *  bcd 2003
   */  
  template <class T>
  static
  T squaredDifference(const Array3D<T>& array1,
		      const Array3D<T>& array2);

#ifdef SWIG
  %template(squaredDifference) squaredDifference< float >;
#endif // SWIG

  /**  
   *  compute the minimum and maximum values in an array
   *  
   *  if array is of size zero, min/max are not changed
   *  
   *  bcd 2003
   *
   *  Note: Do not change the names MIN_OUTPUT, MAX_OUTPUT -- these
   *  are used by SWIG for generating python wrapping.  The python
   *  function operates as:
   * min, max = getMinMax(array)
   */  
  template <class T>
  static 
  void getMinMax(const Array3D<T>& array, T& MIN_OUTPUT, T& MAX_OUTPUT);

#ifdef SWIG
  %template(getMinMax) getMinMax< float >;
#endif // SWIG

  /**  
   *  compute the elementwise arithmetic mean of a group of Array3Ds
   *  
   *  avg(x) = 1/numArrays *     sum     arrays[i](x) 
   *                          numArrays
   *  
   *  avg must be initialized to the proper size prior to this function
   *  call
   *  
   *  bcd 2003
   */  
  template <class T>
  static
  void arithmeticMean(unsigned int numArrays,
		      const Array3D<T>* const* arrays,
		      Array3D<T>& avg);

#ifdef SWIG
  %template(arithmeticMean) arithmeticMean< float >;
#endif // SWIG

  /**
   *  compute the elementwise sum of a group of Array3Ds
   *
   *  sum(x) = 			sum     arrays[i](x)
   *                          numArrays
   *
   *  sum must be initialized to the proper size prior to this function
   *  call
   *
   *  
   */
  template <class T>
  static
  void arithmeticSum(unsigned int numArrays,
		     const Array3D<T>* const* arrays,
		     Array3D<T>& sum);

#ifdef SWIG
  %template(arithmeticSum) arithmeticSum< float >;
#endif // SWIG

  /**
   * Sum all elements in array.
   *
   * jsp 2009
   */
  template <class T>
  static
  T sumElements(const Array3D<T>& inArray);

#ifdef SWIG
  %template(sumElements) sumElements< float >;
  %template(sumElements) sumElements< Vector3D< float > >;
#endif // SWIG

  /**  
   *  compute the elementwise arithmetic mean of a group of Array3Ds
   *  
   *  avg(x) = 1/numArrays *     sum     arrays[i](x) 
   *                          numArrays
   *  
   *  avg must be initialized to the proper size prior to this function
   *  call
   *  
   *  bcd 2003
   */  
  template <class T>
  static
  void weightedArithmeticMean(unsigned int numArrays,
                              const Array3D<T>** const arrays,
                              const double* weights,
                              Array3D<T>& avg);

#ifdef SWIG
  %template(weightedArithmeticMean) weightedArithmeticMean< float >;
#endif // SWIG

  /**  
   *  compute the elementwise sample variance of a group of Array3Ds
   *  
   *  var must be initialized to the proper size prior to this function
   *  call
   *  
   *  bcd 2006
   */  
  template <class T>
  static
  void sampleVariance(unsigned int numArrays,
		      const Array3D<T>** const arrays,
		      Array3D<T>& var);

#ifdef SWIG
  %template(sampleVariance) sampleVariance< float >;
#endif // SWIG

  /**  
   *  compute the elementwise trimmed arithmetic mean of a group of
   *  Array3Ds, choose reasonable trim parameters
   *  
   *  avg must be initialized to the proper size prior to this function
   *  call
   *  
   *  bcd 2004
   */  
  template <class T>
  static
  void trimmedMean(unsigned int numArrays,
		   const Array3D<T>** const arrays,
		   Array3D<T>& avg);

  /**  
   *  compute the elementwise trimmed arithmetic mean of a group of
   *  Array3Ds
   *  
   *  avg must be initialized to the proper size prior to this function
   *  call
   *  
   *  bcd 2004
   */  
  template <class T>
  static
  void trimmedMean(unsigned int numArrays,
		   const Array3D<T>** const arrays,
		   Array3D<T>& avg,
		   unsigned int numTrimLeft,
		   unsigned int numTrimRight);

#ifdef SWIG
  %template(trimmedMean) trimmedMean< float >;
#endif // SWIG

  /**  
   *  recompute the elementwise arithmetic mean of a group of Array3Ds
   *  replacing one Array3D from the original mean computation with
   *  another
   *  
   *  this just saves time over recomputing the mean from all arrays
   *  
   *  bcd 2003
   */  
  template <class T>
  static
  void updateArithmeticMean(unsigned int numArrays,
			    const Array3D<T>& oldInput,
			    const Array3D<T>& newInput,
			    Array3D<T>& avg);

#ifdef SWIG
  %template(updateArithmeticMean) updateArithmeticMean< float >;
#endif // SWIG

  /**  
   *  each element gets min of itself and its neighbors in x
   *  
   *  bcd 2004
   */  
  template <class T>
  static
  void minFilter1DX(Array3D<T>& array);

#ifdef SWIG
  %template(minFilter1DX) minFilter1DX< float >;
#endif // SWIG

  /**  
   *  each element gets min of itself and its neighbors in y
   *  
   *  bcd 2004
   */  
  template <class T>
  static
  void minFilter1DY(Array3D<T>& array);

#ifdef SWIG
  %template(minFilter1DY) minFilter1DY< float >;
#endif // SWIG

  /**  
   *  each element gets min of itself and its neighbors in z
   *  
   *  bcd 2004
   */  
  template <class T>
  static
  void minFilter1DZ(Array3D<T>& array);

#ifdef SWIG
  %template(minFilter1DZ) minFilter1DZ< float >;
#endif // SWIG

  /**  
   *  each element gets min of itself and its neighbors 26 neighbors
   *  
   *  bcd 2004
   */  
  template <class T>
  static
  void minFilter3D(Array3D<T>& array);

#ifdef SWIG
  %template(minFilter3D) minFilter3D< float >;
#endif // SWIG

  /**  
   *  each element gets max of itself and its neighbors in x
   *  
   *  bcd 2004
   */  
  template <class T>
  static
  void maxFilter1DX(Array3D<T>& array);

#ifdef SWIG
  %template(maxFilter1DX) maxFilter1DX< float >;
#endif // SWIG

  /**  
   *  each element gets max of itself and its neighbors in y
   *  
   *  bcd 2004
   */  
  template <class T>
  static
  void maxFilter1DY(Array3D<T>& array);

#ifdef SWIG
  %template(maxFilter1DY) maxFilter1DY< float >;
#endif // SWIG

  /**  
   *  each element gets max of itself and its neighbors in z
   *  
   *  bcd 2004
   */  
  template <class T>
  static
  void maxFilter1DZ(Array3D<T>& array);

#ifdef SWIG
  %template(maxFilter1DZ) maxFilter1DZ< float >;
#endif // SWIG

  /**  
   *  each element gets max of itself and its neighbors 26 neighbors
   *  
   *  bcd 2004
   */  
  template <class T>
  static
  void maxFilter3D(Array3D<T>& array);

#ifdef SWIG
  %template(maxFilter3D) maxFilter3D< float >;
#endif // SWIG

  template <class T>
  static
  void rescaleElements(Array3D<T>& array,
		       const T& minValueOut,
		       const T& maxValueOut);

  template <class T>
  static
  void rescaleElements(Array3D<T>& array,
		       const T& minThreshold,
		       const T& maxThreshold,
		       const T& minValueOut,
		       const T& maxValueOut);

#ifdef SWIG
  %template(rescaleElements) rescaleElements< float >;
#endif // SWIG

  template <class T>
  static
  void diffuse1DX(Array3D<T>& array);

#ifdef SWIG
  %template(diffuse1DX) diffuse1DX< float >;
#endif // SWIG

  template <class T>
  static
  void diffuse1DY(Array3D<T>& array);

#ifdef SWIG
  %template(diffuse1DY) diffuse1DY< float >;
#endif // SWIG

  template <class T>
  static
  void diffuse1DZ(Array3D<T>& array);

#ifdef SWIG
  %template(diffuse1DZ) diffuse1DZ< float >;
#endif // SWIG

  template <class T>
  static
  void diffuse3D(Array3D<T>& array);

#ifdef SWIG
  %template(diffuse3D) diffuse3D< float >;
#endif // SWIG

  template <class T, class U>
  static
  void select(const Array3D<T>& source,
	      Array3D<T>& dest,
	      const Array3D<U>& mask,
	      const U& maskMin, const U& maskMax);

#ifdef SWIG
  %template(select) select< float, float >;
#endif // SWIG

  template <class T>
  static
  void refineZ(Array3D<T>& array,
	       const double& newToOldSlicesRatio);

#ifdef SWIG
  %template(refineZ) refineZ< float >;
#endif // SWIG
  
  template <class T>
  static
  void flipX(Array3D<T>& array);

#ifdef SWIG
  %template(flipX) flipX< float >;
  %template(flipX) flipX< Vector3D< float > >;
#endif // SWIG

  template <class T>
  static
  void flipY(Array3D<T>& array);

#ifdef SWIG
  %template(flipY) flipY< float >;
  %template(flipY) flipY< Vector3D< float > >;
#endif // SWIG

  template <class T>
  static
  void flipZ(Array3D<T>& array);

#ifdef SWIG
  %template(flipZ) flipZ< float >;
  %template(flipZ) flipZ< Vector3D< float > >;
#endif // SWIG

  // Computes the mass of the array
  template <class T>
  static
  T mass(const Array3D<T>& array);

#ifdef SWIG
  %template(mass) mass< float >;
#endif // SWIG

  // Computes the mass of the array
  template <class T>
  static
  T sumOfSquaredElements(const Array3D<T>& array);

#ifdef SWIG
  %template(sumOfSquaredElements) sumOfSquaredElements< float >;
#endif // SWIG

  /**
   * adds the contents of b, componentwise, to a
   * the arrays must be of the same dimensions
   */
  template <class T>
  static
  void sum(Array3D<T>& a, const Array3D<T>& b);

#ifdef SWIG
  %template(sum) sum< float >;
#endif // SWIG

  /**  
   *  Floodfill the region contiguous with 'seed' bounded by voxels
   *  less than minThresh and higher than maxThresh
   */  
  template< class T >
  static void
  maskRegionDeterminedByThresholds(Array3D<T>& array, 
                                   Array3D<unsigned char>& mask,
                                   const Vector3D<unsigned int>& seed,
                                   const T& minThresh,
                                   const T& maxThresh);

#ifdef SWIG
  %template(maskRegionDeterminedByThresholds) maskRegionDeterminedByThresholds< float >;
#endif // SWIG
  
}; // class Array3DUtils

#ifndef SWIG
#include "Array3DUtils.txx"
#endif // SWIG

#endif
