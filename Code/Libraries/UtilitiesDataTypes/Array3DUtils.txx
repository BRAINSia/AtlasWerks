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


//#include "Array3DUtils.h"
#include "Arithmetic.h"

template <class T>
inline
T 
Array3DUtils::
nearestNeighbor(const Array3D<T>& array,
		const double& x,
		const double& y,
		const double& z,
		const T& background)
{
  int xIndex = static_cast<int>(x + 0.5);
  int yIndex = static_cast<int>(y + 0.5);
  int zIndex = static_cast<int>(z + 0.5);

  if (xIndex >= 0 && xIndex < (int) array.getSizeX() &&
      yIndex >= 0 && yIndex < (int) array.getSizeY() &&
      zIndex >= 0 && zIndex < (int) array.getSizeZ())
    { 
      return array(xIndex, yIndex, zIndex);
    }
  else
    {
      return background;
    }
}

template <class T>
inline
T 
Array3DUtils::
nearestNeighbor(const Array3D<T>& array, 
		const Vector3D<double>& index,
		const T& background)
{
  return nearestNeighbor(array, 
			 index.x, 
			 index.y, 
			 index.z, 
			 background);
}

template <class T, Array3DUtils::ScalarBackgroundStrategyT BackgroundStrategy>
inline
T 
Array3DUtils::
trilerp(const Array3D<T>& array,
	const double& x,
	const double& y,
	const double& z,
	const T& background)
{
  // a faster version of the floor function
  int floorX = static_cast<int>(x);
  int floorY = static_cast<int>(y);
  int floorZ = static_cast<int>(z);
  if (x < 0 && x != static_cast<int>(x)) --floorX;
  if (y < 0 && y != static_cast<int>(y)) --floorY;
  if (z < 0 && z != static_cast<int>(z)) --floorZ;

  // this is not truly ceiling, but floor + 1, which is usually ceiling
  int ceilX = floorX + 1;
  int ceilY = floorY + 1;
  int ceilZ = floorZ + 1;

  const double t = x - floorX;
  const double u = y - floorY;
  const double v = z - floorZ;
  const double oneMinusT = 1.0 - t;
  const double oneMinusU = 1.0 - u;
  const double oneMinusV = 1.0 - v;
  
  //
  // ^
  // |  v3   v2       -z->        v4   v5
  // y           --next slice-->      
  // |  v0   v1                   v7   v6
  //
  //      -x->
  //     
  T v0, v1, v2, v3, v4, v5, v6, v7;

  int sizeX = array.getSizeX();
  int sizeY = array.getSizeY();
  int sizeZ = array.getSizeZ();

  bool inside = true;

  // clamp if necessary
  if(BackgroundStrategy == BACKGROUND_STRATEGY_WRAP){
    // wrap
  floorX = safe_mod(floorX, sizeX);
  floorY = safe_mod(floorY, sizeY);
  floorZ = safe_mod(floorZ, sizeZ);
  ceilX = safe_mod(ceilX, sizeX);
  ceilY = safe_mod(ceilY, sizeY);
  ceilZ = safe_mod(ceilZ, sizeZ);
    // if(floorX < 0) floorX = floorX + sizeX;
    // if(ceilX >= sizeX) ceilX = ceilX % sizeX;
    // if(floorY < 0) floorY = floorY + sizeY;
    // if(ceilY >= sizeY) ceilY = ceilY % sizeY;
    // if(floorZ < 0) floorZ = floorZ + sizeZ;
    // if(ceilZ >= sizeZ) ceilZ = ceilZ % sizeZ;
  }else if(BackgroundStrategy == BACKGROUND_STRATEGY_CLAMP){
    if(floorX < 0){
      floorX = 0;
      if(ceilX < 0) ceilX=0;
    }
    if(ceilX >= sizeX){
      ceilX = sizeX-1;
      if(floorX > sizeX) floorX = sizeX-1;
    }
    if(floorY < 0){
      floorY = 0;
      if(ceilY < 0) ceilY=0;
    }
    if(ceilY >= sizeY){
      ceilY = sizeY-1;
      if(floorY > sizeY) floorY = sizeY-1;
    }
    if(floorZ < 0){
      floorZ = 0;
      if(ceilZ < 0) ceilZ=0;
    }
    if(ceilZ >= sizeZ){
      ceilZ = sizeZ-1;
      if(floorZ > sizeZ) floorZ = sizeZ-1;
    }
  }else{
    inside = (floorX >= 0 && ceilX < sizeX &&
	      floorY >= 0 && ceilY < sizeY &&
	      floorZ >= 0 && ceilZ < sizeZ);
  }

  if (inside)
    {
      // this is the fast path
      v0 = array(floorX, floorY, floorZ);
      v1 = array(ceilX, floorY, floorZ);
      v2 = array(ceilX, ceilY, floorZ);
      v3 = array(floorX, ceilY, floorZ);
      v4 = array(floorX, ceilY, ceilZ);
      v5 = array(ceilX, ceilY, ceilZ);
      v6 = array(ceilX, floorY, ceilZ);
      v7 = array(floorX, floorY, ceilZ);
    }
  else
    {
      bool floorXIn = floorX >= 0 && floorX < sizeX;
      bool floorYIn = floorY >= 0 && floorY < sizeY;
      bool floorZIn = floorZ >= 0 && floorZ < sizeZ;
      
      bool ceilXIn = ceilX >= 0 && ceilX < sizeX;
      bool ceilYIn = ceilY >= 0 && ceilY < sizeY;
      bool ceilZIn = ceilZ >= 0 && ceilZ < sizeZ;
     
      v0 = (floorXIn && floorYIn && floorZIn) 
	? array(floorX, floorY, floorZ) : background;
      v1 = (ceilXIn && floorYIn && floorZIn)  
	? array(ceilX, floorY, floorZ)  : background;
      v2 = (ceilXIn && ceilYIn && floorZIn)   
	? array(ceilX, ceilY, floorZ)   : background;
      v3 = (floorXIn && ceilYIn && floorZIn)  
	? array(floorX, ceilY, floorZ)  : background;
      v4 = (floorXIn && ceilYIn && ceilZIn)   
	? array(floorX, ceilY, ceilZ)   : background;
      v5 = (ceilXIn && ceilYIn && ceilZIn)    
	? array(ceilX, ceilY, ceilZ)    : background;
      v6 = (ceilXIn && floorYIn && ceilZIn)   
	? array(ceilX, floorY, ceilZ)   : background;
      v7 = (floorXIn && floorYIn && ceilZIn)  
	? array(floorX, floorY, ceilZ)  : background; 
    }

  //
  // this is the basic trilerp function...
  //
  //     val = 
  //       v0 * (1 - t) * (1 - u) * (1 - v) +
  //       v1 * t       * (1 - u) * (1 - v) +
  //       v2 * t       * u       * (1 - v) +
  //       v3 * (1 - t) * u       * (1 - v) +
  //       v4 * (1 - t) * u       * v       +
  //       v5 * t       * u       * v       +
  //       v6 * t       * (1 - u) * v       +
  //       v7 * (1 - t) * (1 - u) * v;
  //
  // the following saves some computation...
  //

  T retval = oneMinusT * (oneMinusU * (v0 * oneMinusV + v7 * v)  +
			  u         * (v3 * oneMinusV + v4 * v)) +
    t         * (oneMinusU * (v1 * oneMinusV + v6 * v)  +
		 u         * (v2 * oneMinusV + v5 * v));

  if (retval != retval)
    {
      std::cerr << "Trilerp parameters" << std::endl;
      std::cerr << "  v0:         " << v0 << std::endl;
      std::cerr << "  v1:         " << v1 << std::endl;
      std::cerr << "  v2:         " << v2 << std::endl;
      std::cerr << "  v3:         " << v3 << std::endl;
      std::cerr << "  v4:         " << v4 << std::endl;
      std::cerr << "  v5:         " << v5 << std::endl;
      std::cerr << "  v6:         " << v6 << std::endl;
      std::cerr << "  v7:         " << v7 << std::endl;
      std::cerr << "  u:         " << u << std::endl;
      std::cerr << "  v:         " << v << std::endl;
      std::cerr << "  oneMinusT: " << oneMinusT << std::endl;
      std::cerr << "  oneMinusU: " << oneMinusU << std::endl;
      std::cerr << "  oneMinusV: " << oneMinusV << std::endl;
      throw std::runtime_error("Found a NAN in trilerp.");
    }

  return retval;      
}

template <class T, Array3DUtils::ScalarBackgroundStrategyT BackgroundStrategy>
inline
T 
Array3DUtils::
trilerp(const Array3D<T>& array, 
	const Vector3D<double>& position,
	const T& background)
{
  return trilerp<T, BackgroundStrategy>(array, position.x, position.y, position.z, background);
}

template <class T, Array3DUtils::ScalarBackgroundStrategyT BackgroundStrategy>
inline
T 
Array3DUtils::
cubicSplineinterp(const Array3D<T>& array, 
		  const Vector3D<double>& position,
		  const T& background)
{
  return cubicSplineinterp<T, BackgroundStrategy>(array, position.x, position.y, position.z, background);
}

template <class T, Array3DUtils::ScalarBackgroundStrategyT BackgroundStrategy>
inline
T 
Array3DUtils::
cubicSplineinterp(const Array3D<T>& array,
		  const double& x,
		  const double& y,
		  const double& z,
		  const T& background)
{
  int floorX = static_cast<int>(x);
  int floorY = static_cast<int>(y);
  int floorZ = static_cast<int>(z);
  if (x < 0 && x != static_cast<int>(x)) --floorX;
  if (y < 0 && y != static_cast<int>(y)) --floorY;
  if (z < 0 && z != static_cast<int>(z)) --floorZ;
		
  int sizeX = array.getSizeX();
  int sizeY = array.getSizeY();
  int sizeZ = array.getSizeZ();

  const double t = x - floorX;
  const double u = y - floorY;
  const double v = z - floorZ; 

  T sm_array[4][4][4];

  for(int sm_a=0;sm_a<4;sm_a++){
    for(int sm_b=0;sm_b<4;sm_b++){
      for(int sm_c=0;sm_c<4;sm_c++){
	// current image co-ordinates of an image
	int arrayX=floorX+sm_a-1;
	int arrayY=floorY+sm_b-1;
	int arrayZ=floorZ+sm_c-1;
	// checking boundary condition
	if (BackgroundStrategy == BACKGROUND_STRATEGY_WRAP){
	  if (arrayX < 0) arrayX = arrayX + sizeX;
	  if (arrayX >= sizeX) arrayX = arrayX % sizeX;
	  if (arrayY < 0) arrayY = arrayY + sizeY;
	  if (arrayY >= sizeY) arrayY = arrayY % sizeY;
	  if (arrayZ < 0) arrayZ = arrayZ + sizeZ;
	  if (arrayZ >= sizeZ) arrayZ = arrayZ % sizeZ;
	  sm_array[sm_a][sm_b][sm_c]=array(arrayX,arrayY,arrayZ);
	}else if(BackgroundStrategy == BACKGROUND_STRATEGY_CLAMP){ 
	  if (arrayX < 0) arrayX = 0;
	  if (arrayX > sizeX - 1) arrayX = sizeX - 1;
	  if (arrayY < 0) arrayY = 0;
	  if (arrayY > sizeY - 1) arrayY = sizeY - 1;
	  if (arrayZ < 0) arrayZ = 0;
	  if (arrayZ > sizeZ - 1) arrayZ = sizeZ - 1 ;
	  sm_array[sm_a][sm_b][sm_c]=array(arrayX,arrayY,arrayZ);				
	}else if(BackgroundStrategy == BACKGROUND_STRATEGY_VAL){
	  if (arrayX < 0 ||
	      arrayX > sizeX - 1 ||
	      arrayY < 0 ||
	      arrayY > sizeY - 1 ||
	      arrayZ < 0 || 
	      arrayZ > sizeZ - 1)
	    {
	      sm_array[sm_a][sm_b][sm_c]=background;
	    }
	  else
	    {
	      sm_array[sm_a][sm_b][sm_c]=array(arrayX,arrayY,arrayZ);
	    }
	}else{
	  throw std::runtime_error("Error, unknown background strategy");
	}
      }
    }
  }	

  T retval =  cubicsplineinterp(&sm_array[0][0][0],v,u,t);
  if (retval != retval)
    {
      throw std::runtime_error("Found a NAN in cubicSplineinterp.");
    } 	

  return retval;
}

// Cubic Spline interpolation w.r.to Catmull-Rom
template <class T>
inline
T 
Array3DUtils::
cubicsplineinterp(T *cubestart, double x,double y, double z)
{
	double xcube = x*x*x;
	double xsquare = x*x;
	double ycube = y*y*y;
	double ysquare = y*y;
	double zcube = z*z*z;
	double zsquare = z*z;
// along X-direction	
	T v[4][4];	
	for (int r=0;r<4;r++){
		for (int c=0;c<4;c++){
			T x0 = *(cubestart+16*r+4*c);
			T x1 = *(cubestart+16*r+4*c+1);
			T x2 = *(cubestart+16*r+4*c+2);
               T x3 = *(cubestart+16*r+4*c+3);
			T a0 = x3 - x2 - x0 + x1;
			T a1 = x0 - x1 - a0;
			T a2 = x2 - x0;
			T a3 = x1;
			v[r][c] = a0 * xcube + a1 * xsquare + a2 * x + a3;  
		}
	}
// along Y-direction
	T u[4];	
	for (int r=0;r<4;r++){
		T y0 = v[r][0];
		T y1 = v[r][1];
		T y2 = v[r][2];
		T y3 = v[r][3];
		T a0 = y3 - y2 - y0 + y1;
		T a1 = y0 - y1 - a0;
		T a2 = y2 - y0;
		T a3 = y1;
 		u[r] = a0 * ycube + a1 * ysquare + a2 * y + a3;
	}
// along Z-direction
		T z0 = u[0];
		T z1 = u[1];
		T z2 = u[2];
		T z3 = u[3];
		T a0 = z3 - z2 - z0 + z1;
		T a1 = z0 - z1 - a0;
		T a2 = z2 - z0;
		T a3 = z1;
 		T retval = a0 * zcube + a1 * zsquare + a2 * z + a3;
		//if (retval < 0) retval = 0; // making negative values = 0
	return(retval);
	
}

template <class T>
inline
bool 
Array3DUtils::
computeTrilerpCoeffs(
		     const Array3D<T>& array,
		     const double& x, const double& y, const double& z,
		     size_t& i0, size_t& i1, size_t& i2, size_t& i3,
		     size_t& i4, size_t& i5, size_t& i6, size_t& i7,
		     double& a0, double& a1, double& a2, double& a3,
		     double& a4, double& a5, double& a6, double& a7)
{
  int floorX = static_cast<int>(x);
  int floorY = static_cast<int>(y);
  int floorZ = static_cast<int>(z);

  int sizeX = array.getSizeX();
  int sizeY = array.getSizeY();
  int sizeZ = array.getSizeZ();

  // return if outside image bounds
  if (x < 0.0 || floorX + 1 >= sizeX ||
      y < 0.0 || floorY + 1 >= sizeY ||
      z < 0.0 || floorZ + 1 >= sizeZ) {
    return false;
  }

  // this is not truly ceiling, but floor + 1, which is usually ceiling
  int ceilX = floorX + 1;
  int ceilY = floorY + 1;
  int ceilZ = floorZ + 1;

  const T* base = &array(0, 0, 0);
  i0 = &array(floorX, floorY, floorZ) - base;
  i1 = &array(ceilX,  floorY, floorZ) - base;
  i2 = &array(ceilX,  ceilY,  floorZ) - base;
  i3 = &array(floorX, ceilY,  floorZ) - base;
  i4 = &array(floorX, ceilY,  ceilZ)  - base;
  i5 = &array(ceilX,  ceilY,  ceilZ)  - base;
  i6 = &array(ceilX,  floorY, ceilZ)  - base;
  i7 = &array(floorX, floorY, ceilZ)  - base;

  const double t = x - floorX;
  const double u = y - floorY;
  const double v = z - floorZ;
  const double oneMinusT = 1.0 - t;
  const double oneMinusU = 1.0 - u;
  const double oneMinusV = 1.0 - v;

  // This results in more total multiplies if you only apply it to
  // one image, but for more than two images, it uses fewer.
  a7 = a0 = oneMinusT * oneMinusU;
  a4 = a3 = oneMinusT * u;
  a6 = a1 = t * oneMinusU;
  a5 = a2 = t * u;

  a0 *= oneMinusV;
  a1 *= oneMinusV;
  a2 *= oneMinusV;
  a3 *= oneMinusV;
  a4 *= v;
  a5 *= v;
  a6 *= v;
  a7 *= v;
  
  return true;

}

template <class T>
inline
T 
Array3DUtils::
weightedSumOfArrayValues(
			 const Array3D<T>& array,
			 const size_t& i0, const size_t& i1, const size_t& i2, const size_t& i3,
			 const size_t& i4, const size_t& i5, const size_t& i6, const size_t& i7,
			 const double& a0, const double& a1, const double& a2, const double& a3,
			 const double& a4, const double& a5, const double& a6, const double& a7)
{
  const T* arr = array.getDataPointer();
  return 
    arr[i0] * a0 + arr[i1] * a1 + arr[i2] * a2 + arr[i3] * a3 +
    arr[i4] * a4 + arr[i5] * a5 + arr[i6] * a6 + arr[i7] * a7;
}

template <class T>
inline
void 
Array3DUtils::
extractROI(const Array3D<T>& array,
	   Array3D<T>& roi,
	   const Vector3D<int>& roiStart,
	   const Vector3D<unsigned int>& roiSize)
{
  Vector3D<unsigned int> roiOutStart(0,0,0);
  roi.resize(roiSize);
  Array3DUtils::copyRegion(array, roi, roiStart, roiOutStart, roiSize);
}

template <class T>
inline
void 
Array3DUtils::
extractROI(const Array3D<T>& array,
	   Array3D<T>& roiArray,
	   const ROI<int, unsigned int>& roi)
{
  roiArray.resize(roi.getSize());
  Vector3D<int> roiOutStart(0,0,0);
  Array3DUtils::copyRegion(array, roiArray, 
			   roi.getStart(), roiOutStart, roi.getSize());
}

template <class T>
inline
void 
Array3DUtils::
fillRegion(Array3D<T>& array,
	   const ROI<int, unsigned int>& roi,
	   const T& value)
{

  Vector3D<unsigned int> arraySize = array.getSize();

  Vector3D<int> begin = roi.getStart();
  Vector3D<unsigned int> end = roi.getStop();


  for (unsigned int i = 0; i < 3; ++i) 
    {
      if (end[i] > arraySize[i]) end[i] = arraySize[i];
    }

  for (unsigned int z = begin.z; z < end.z; ++z)
    {
      for (unsigned int y = begin.y; y < end.y; ++y)
	{
	  for (unsigned int x = begin.x; x < end.x; ++x)
	    {
	      array.set(x, y, z, value);
	    }
	}
    }
}

template <class T>
inline
void 
Array3DUtils::
copyRegion(const Array3D<T>& inArray,
	   Array3D<T>& outArray,
	   const typename Array3D<T>::IndexType& inOrigin,
	   const typename Array3D<T>::IndexType& outOrigin,
	   const typename Array3D<T>::SizeType& regionSize)
{

  Vector3D<unsigned int> inLimit = inOrigin + regionSize;
  Vector3D<unsigned int> inSize = inArray.getSize();

  for (unsigned int i = 0; i < 3; ++i) 
    {
      if (inLimit[i] > inSize[i]) inLimit[i] = inSize[i];
    }

  for (unsigned int zIn = inOrigin.z, zOut = outOrigin.z; 
       zIn < inLimit.z;
       ++zIn, ++zOut)
    {
      for (unsigned int yIn = inOrigin.y, yOut = outOrigin.y; 
	   yIn < inLimit.y;
	   ++yIn, ++yOut)
	{
	  memcpy(outArray.getDataPointer(outOrigin.x, yOut, zOut),
		 inArray.getDataPointer(inOrigin.x, yIn, zIn),
		 regionSize.x * sizeof(T));
	}
    }
}

template <class T, class U>
inline
void 
Array3DUtils::
copyRegionCast(const Array3D<T>& inArray,
	       Array3D<U>& outArray,
	       const typename Array3D<T>::IndexType& inOrigin,
	       const typename Array3D<U>::IndexType& outOrigin,
	       const typename Array3D<T>::SizeType& regionSize)
{
  typename Array3D<T>::IndexType inLimit = inOrigin + regionSize;
  for (unsigned int zIn = inOrigin.z, zOut = outOrigin.z; 
       zIn < inLimit.z;
       ++zIn, ++zOut)
    {
      for (unsigned int yIn = inOrigin.y, yOut = outOrigin.y; 
	   yIn < inLimit.y;
	   ++yIn, ++yOut)
	{
	  for (unsigned int xIn = inOrigin.x, xOut = outOrigin.x; 
	       xIn < inLimit.x;
	       ++xIn, ++xOut)
	    {
	      outArray(xOut, yOut, zOut) =
		static_cast<U>(inArray(xIn, yIn, zIn));
	    }
	}
    }
}
  
template <class T, class U>
inline
void 
Array3DUtils::
copyCast(const Array3D<T>& inArray,
	 Array3D<U>& outArray)
{
  typename Array3D<T>::IndexType inOrigin(0,0,0);
  typename Array3D<U>::IndexType outOrigin(0,0,0);
  copyRegionCast(inArray, outArray, inOrigin, outOrigin, inArray.getSize());
}

  
template <class T, class U>
inline
void 
Array3DUtils::
copyRegionRound(const Array3D<T>& inArray,
		Array3D<U>& outArray,
		const typename Array3D<T>::IndexType& inOrigin,
		const typename Array3D<U>::IndexType& outOrigin,
		const typename Array3D<T>::SizeType& regionSize)
{
  typename Array3D<T>::IndexType inLimit = inOrigin + regionSize;
  for (unsigned int zIn = inOrigin.z, zOut = outOrigin.z; 
       zIn < inLimit.z;
       ++zIn, ++zOut)
    {
      for (unsigned int yIn = inOrigin.y, yOut = outOrigin.y; 
           yIn < inLimit.y;
           ++yIn, ++yOut)
	{
	  for (unsigned int xIn = inOrigin.x, xOut = outOrigin.x; 
	       xIn < inLimit.x;
	       ++xIn, ++xOut)
	    {
	      outArray(xOut, yOut, zOut) =
		static_cast<U>(round(inArray(xIn, yIn, zIn)));
	    }
	}
    }
}

template <class T>
inline
Vector3D<double> 
Array3DUtils::
computeCentroid(const Array3D<T>& array,
		const typename Array3D<T>::IndexType& origin,
		const typename Array3D<T>::SizeType& size)
{
  typename Array3D<T>::IndexType regionLimit = origin + size;
  Vector3D<double> centroid(0, 0, 0);
  double mass = 0;
  double value;
  for (unsigned int z = origin.z; z < regionLimit.z; ++z) {
    for (unsigned int y = origin.y; y < regionLimit.y; ++y)	{
      for (unsigned int x = origin.x; x < regionLimit.x; ++x) {
	value = array(x, y, z);
	mass += value;
	centroid.x += x * value;
	centroid.y += y * value;
	centroid.z += z * value;
      }	  
    }
  }
  return centroid / mass;
}

template <class T>
inline
void 
Array3DUtils::
downsampleByInt(const Array3D<T>& input,
		Array3D<T>& output,
		unsigned int downsampleFactor)
{
  Vector3D<unsigned int> factors(downsampleFactor, 
				 downsampleFactor, 
				 downsampleFactor);
  downsampleByInts(input, output, factors);
}

template <class T>
inline
void 
Array3DUtils::
downsampleByInts(const Array3D<T>& input,
		 Array3D<T>& output,
		 const Vector3D<unsigned int>& downsampleFactors)
{
  typename Array3D<T>::SizeType inputSize = input.getSize();
  //ASSERT(inputSize.x > 0 && inputSize.y > 0 && inputSize.z > 0);
  
  double shrinkFactorX = static_cast<double>(downsampleFactors[0]);
  double shrinkFactorY = static_cast<double>(downsampleFactors[1]);
  double shrinkFactorZ = static_cast<double>(downsampleFactors[2]);

  typename Array3D<T>::SizeType
    outputSize((unsigned int) floor(inputSize.x / shrinkFactorX),
	       (unsigned int) floor(inputSize.y / shrinkFactorY),
	       (unsigned int) floor(inputSize.z / shrinkFactorZ));
  if( outputSize[0] < 1 ) outputSize[0] = 1;
  if( outputSize[1] < 1 ) outputSize[1] = 1;
  if( outputSize[2] < 1 ) outputSize[2] = 1;
  output.resize(outputSize);
  
  for (unsigned int outputZ = 0, inputZ = 0; 
       outputZ < outputSize.z; 
       ++outputZ, inputZ += downsampleFactors.z)
    {
      for (unsigned int outputY = 0, inputY = 0; 
           outputY < outputSize.y; 
           ++outputY, inputY += downsampleFactors.y)
	{
	  for (unsigned int outputX = 0, inputX = 0; 
	       outputX < outputSize.x; 
	       ++outputX, inputX += downsampleFactors.x)	    
	    {
	      output(outputX, outputY, outputZ) =
		input(inputX, inputY, inputZ);
	    }
	}
    }
}

template <class T>
inline
void 
Array3DUtils::
downsampleByTwo(const Array3D<T>& input,
		Array3D<T>& output)
{
  downsampleByInt(input, output, 2);
}

template <class T>
inline
void 
Array3DUtils::
gaussianBlur(const Array3D<T>& input,
	     Array3D<T>& output, 
	     const Vector3D<double>& sigma,
	     const Vector3D<int>& kernelSize)
{
  /** blur the image */
  GaussianFilter3D filter;
  filter.SetInput(input);
  filter.setSigma(sigma.x,sigma.y,sigma.z);
  filter.setKernelSize(kernelSize.x,kernelSize.y,kernelSize.z);
  filter.Update();
  
  /** Create new downsampled image */
  output.resize(input.getSize());
  output.setData(filter.GetOutput());
}

template <class T>
inline
void 
Array3DUtils::
gaussianDownsample (const Array3D<T>& input,
		    Array3D<T>& output, 
		    const Vector3D<int>& factors,
		    const Vector3D<double>& sigma,
		    const Vector3D<int>& kernelSize)
{
  /** Downsample the image */
  DownsampleFilter3D filter;
  filter.SetInput(input);
  filter.SetFactor(factors.x,factors.y,factors.z);
  filter.SetSigma(sigma.x,sigma.y,sigma.z);
  filter.SetSize(kernelSize.x,kernelSize.y,kernelSize.z);
  filter.Update();
    
  /** Create new downsampled image */
  output.resize(filter.GetNewSize());
  output.setData(filter.GetOutput());
}

template <class T, class U>
inline
void 
Array3DUtils::
computeGradient(const Array3D<T>& array, 
		Array3D<Vector3D<U> >& grad, 
		Vector3D<double> spacing,
		bool wrap)
{
  unsigned int xIndex, yIndex, zIndex;

  typename Array3D<T>::SizeType size = array.getSize();
  // LORENZEN 02-SEP-2004
  //
  // For some reason this resizing leads to serious memory errors.
  // Perhaps the resizing of a 3D array of 3D vectors should be 
  // tested fist in a simple application using valgrind.
  //
  //grad.resize(size);
    
  // Process interior points
  for(zIndex = 0; zIndex < size.z; zIndex++) {
    for(yIndex = 0; yIndex < size.y; yIndex++) {
      for(xIndex = 1; xIndex < size.x - 1; xIndex++) {
	grad(xIndex, yIndex, zIndex).x =
	  (array(xIndex + 1, yIndex, zIndex) - 
	   array(xIndex - 1, yIndex, zIndex)) / (2.0 * spacing.x);
      }
    }
  }
  for(zIndex = 0; zIndex < size.z; zIndex++) {
    for(yIndex = 1; yIndex < size.y - 1; yIndex++) {
      for(xIndex = 0; xIndex < size.x; xIndex++) {
	grad(xIndex, yIndex, zIndex).y =
	  (array(xIndex, yIndex + 1, zIndex) - 
	   array(xIndex, yIndex - 1, zIndex)) / (2.0 * spacing.y);
      }
    }
  }
  for(zIndex = 1; zIndex < size.z - 1; zIndex++) {
    for(yIndex = 0; yIndex < size.y; yIndex++) {
      for(xIndex = 0; xIndex < size.x; xIndex++) {
	grad(xIndex, yIndex, zIndex).z =
	  (array(xIndex, yIndex, zIndex + 1) - 
	   array(xIndex, yIndex, zIndex - 1)) / (2.0 * spacing.z);
      }
    }
  }
    
  if(wrap){
      
    // Process the voxels on the x-dimensional edge
    for(zIndex = 0; zIndex < size.z; zIndex++) {
      for(yIndex = 0; yIndex < size.y; yIndex++) {
	grad(0, yIndex, zIndex).x = 
	  (array(1, yIndex, zIndex) - 
	   array(size.x - 1, yIndex, zIndex)) / (2.0*spacing.x);
	grad(size.x - 1, yIndex, zIndex).x = 
	  (array(0, yIndex, zIndex) - 
	   array(size.x - 2, yIndex, zIndex)) / (2.0*spacing.x);
      }
    }
      
    // Process the voxel on the y-dimensional edge
    for(zIndex = 0; zIndex < size.z; zIndex++) {
      for(xIndex = 0; xIndex < size.x; xIndex++) {
	grad(xIndex, 0, zIndex).y = 
	  (array(xIndex, 1, zIndex) - 
	   array(xIndex, size.y - 1, zIndex)) / (2.0*spacing.y);
	grad(xIndex, size.y - 1, zIndex).y = 
	  (array(xIndex, 0, zIndex) - 
	   array(xIndex, size.y - 2, zIndex)) / (2.0*spacing.y);
      }
    }
      
    // Process the voxel on the z-dimensional edge
    for(yIndex = 0; yIndex < size.y; yIndex++) {
      for(xIndex = 0; xIndex < size.x; xIndex++) {
	grad(xIndex, yIndex, 0).z = 
	  (array(xIndex, yIndex, 1) - 
	   array(xIndex, yIndex, size.z - 1)) / (2.0*spacing.z);
	grad(xIndex, yIndex, size.z - 1).z = 
	  (array(xIndex, yIndex, 0) - 
	   array(xIndex, yIndex, size.z - 2)) / (2.0*spacing.z);
      }
    }

  }else{
    // Process the voxels on the x-dimensional edge
    for(zIndex = 0; zIndex < size.z; zIndex++) {
      for(yIndex = 0; yIndex < size.y; yIndex++) {
	grad(0, yIndex, zIndex).x = 
	  (array(1, yIndex, zIndex) - array(0, yIndex, zIndex)) / spacing.x;
	grad(size.x - 1, yIndex, zIndex).x = 
	  (array(size.x - 1, yIndex, zIndex) - array(size.x - 2, yIndex, zIndex)) / spacing.x;
      }
    }
      
    // Process the voxel on the y-dimensional edge
    for(zIndex = 0; zIndex < size.z; zIndex++) {
      for(xIndex = 0; xIndex < size.x; xIndex++) {
	grad(xIndex, 0, zIndex).y = 
	  (array(xIndex, 1, zIndex) - array(xIndex, 0, zIndex)) / spacing.y;
	grad(xIndex, size.y - 1, zIndex).y = 
	  (array(xIndex, size.y - 1, zIndex) - array(xIndex, size.y - 2, zIndex)) / spacing.y;
      }
    }
      
    // Process the voxel on the z-dimensional edge
    for(yIndex = 0; yIndex < size.y; yIndex++) {
      for(xIndex = 0; xIndex < size.x; xIndex++) {
	grad(xIndex, yIndex, 0).z = 
	  (array(xIndex, yIndex, 1) - array(xIndex, yIndex, 0)) / spacing.z;
	grad(xIndex, yIndex, size.z - 1).z = 
	  (array(xIndex, yIndex, size.z - 1) - array(xIndex, yIndex, size.z - 2)) / spacing.z;
      }
    }
  }// end if not wrap
}
  
template <class T, class U>
inline
void 
Array3DUtils::
computeLaplacian(const Array3D<Vector3D<T> >& array,
		 Array3D<Vector3D<U> >& laplacian)
{
  typename Array3D<T>::SizeType size = array.getSize();
  
  unsigned int x,y,z;

  //set the laplacian to zero

  for (z = 0; z < (size.z ); ++z) {
    for (y = 0; y < (size.y); ++y) {
      for (x = 0; x < (size.x); ++x) {
        
        laplacian(x, y, z).x = 0;
        
        laplacian(x, y, z).y = 0;
        
        laplacian(x, y, z).z = 0;
      }
    }
  }

  

  for (z = 1; z < (size.z - 1); ++z) {
    for (y = 1; y < (size.y - 1); ++y) {
      for (x = 1; x < (size.x - 1); ++x) {
        
        laplacian(x, y, z).x =  array(x + 1, y, z).x + array(x - 1, y, z ).x +
	  array(x, y + 1, z).x + array(x , y - 1, z).x +
	  array(x, y, z + 1).x + array(x, y, z - 1 ).x -
	  6.0*array(x, y, z).x;
        
        laplacian(x, y, z).y =  array(x + 1, y, z).y + array(x - 1, y, z ).y +
	  array(x, y + 1, z).y + array(x , y - 1, z).y +
	  array(x, y, z + 1).y + array(x, y, z - 1 ).y -
	  6.0*array(x, y, z).y;
        
        laplacian(x, y, z).z =  array(x + 1, y, z).z + array(x - 1, y, z ).z +
	  array(x, y + 1, z).z + array(x , y - 1, z).z +
	  array(x, y, z + 1).z + array(x, y, z - 1 ).z -
	  6.0*array(x, y, z).z;

      }
    }
  }
         
       
          
}

template <class T>
inline
T 
Array3DUtils::
squaredDifference(const Array3D<T>& array1,
		  const Array3D<T>& array2)
{
  Vector3D<unsigned int> size = array1.getSize();
  T squaredDifference = 0;
  T d;
  for (unsigned int z = 0; z < size.z; ++z) {
    for (unsigned int y = 0; y < size.y; ++y) {
      for (unsigned int x = 0; x < size.x; ++x) {
	d = array1(x, y, z) - array2(x, y, z); 
	squaredDifference += d * d;
      }
    }
  }  
  return squaredDifference;
}

template <class T>
inline
void 
Array3DUtils::
getMinMax(const Array3D<T>& array, T& min, T& max)
{
  unsigned int size = array.getNumElements();
  if (size == 0) return;
  min = array(0);
  max = array(0);
  for (unsigned int i = 0; i < size; ++i)
    {
      if (array(i) > max) max = array(i);
      if (array(i) < min) min = array(i);
    }
}

template <class T>
inline
void 
Array3DUtils::
arithmeticMean(unsigned int numArrays,
	       const Array3D<T>* const* arrays,
	       Array3D<T>& avg)
{
  unsigned int numElements = avg.getNumElements();
  for (unsigned int element = 0; element < numElements; ++element)
    {
      avg(element) = 0;
      for (unsigned int array = 0; array < numArrays; ++array)
	{
	  avg(element) += (*arrays[array])(element);
	}
      avg(element) /= static_cast<T>(numArrays);
    }
}



template <class T>
inline
void 
Array3DUtils::
arithmeticSum(unsigned int numArrays,
	      const Array3D<T>* const* arrays,
	      Array3D<T>& sum)
{
  unsigned int numElements = sum.getNumElements();
  for (unsigned int element = 0; element < numElements; ++element)
    {
      sum(element) = 0;
      for (unsigned int array = 0; array < numArrays; ++array)
	{
	  sum(element) += (*arrays[array])(element);
	}
    }
}




template <class T>
inline
T 
Array3DUtils::
sumElements(const Array3D<T>& inArray)
{
    
  unsigned int numElements = inArray.getNumElements();
  T sum = 0;

  for (unsigned int element = 0; element < numElements; ++element)
    sum += inArray(element);

  return sum;

}

// specialization for float
template <>
inline
float
Array3DUtils::
sumElements(const Array3D<float>& inArray)
{
    
  unsigned int numElements = inArray.getNumElements();
  double sum = 0;

  for (unsigned int element = 0; element < numElements; ++element)
    sum += inArray(element);

  return static_cast<float>(sum);

}

template <class T>
inline
void 
Array3DUtils::
weightedArithmeticMean(unsigned int numArrays,
		       const Array3D<T>** const arrays,
		       const double* weights,
		       Array3D<T>& avg)
{
  unsigned int numElements = avg.getNumElements();
  for (unsigned int element = 0; element < numElements; ++element)
    {
      avg(element) = 0;
      for (unsigned int array = 0; array < numArrays; ++array)
	{
	  avg(element) += weights[array]*(*arrays[array])(element);
	}
    }
}

template <class T>
inline
void 
Array3DUtils::
sampleVariance(unsigned int numArrays,
	       const Array3D<T>** const arrays,
	       Array3D<T>& var)
{
  Array3D<T> mu(var);
  Array3DUtils::arithmeticMean(numArrays,arrays,mu);

  unsigned int numElements = var.getNumElements();
  for (unsigned int element = 0; element < numElements; ++element)
    {
      var(element) = 0;
      for (unsigned int array = 0; array < numArrays; ++array)
	{
	  var(element) += 
	    ((*arrays[array])(element)-mu(element))
	    * ((*arrays[array])(element)-mu(element));
	}
      var(element) /= static_cast<T>(numArrays-1);
    }
}

template <class T>
inline
void 
Array3DUtils::
trimmedMean(unsigned int numArrays,
	    const Array3D<T>** const arrays,
	    Array3D<T>& avg)
{
  if (numArrays < 5)
    {
      // don't do any trimming, just compute the mean
      arithmeticMean(numArrays, arrays, avg);
      return;
    }
  else
    {
      unsigned int numTrim = 0;
      if (numArrays < 10)
	{
	  numTrim = 1;
	}
      else
	{
	  numTrim = static_cast<int>(static_cast<float>(numArrays) / 5.0F);
	}
      trimmedMean(numArrays, arrays, avg, numTrim, numTrim);
    }
}

template <class T>
inline
void 
Array3DUtils::
trimmedMean(unsigned int numArrays,
	    const Array3D<T>** const arrays,
	    Array3D<T>& avg,
	    unsigned int numTrimLeft,
	    unsigned int numTrimRight)
{
  unsigned int numElements = avg.getNumElements();
  std::vector<T> vals(numArrays);
  unsigned int element, array;
  for (element = 0; element < numElements; ++element)
    {
      // copy values locally
      for (array = 0; array < numArrays; ++array)
	{
	  vals[array] = (*arrays[array])(element);
	}

      // compute trimmed mean
      std::sort(vals.begin(), vals.end());
      double sum = std::accumulate(vals.begin() + numTrimLeft, 
                                   vals.end() - numTrimRight, 0.0);

      avg(element) = static_cast<T>(sum / 
                                    static_cast<double>(vals.size() 
                                                        - numTrimLeft 
                                                        - numTrimRight));
    }
}

template <class T>
inline
void 
Array3DUtils::
updateArithmeticMean(unsigned int numArrays,
		     const Array3D<T>& oldInput,
		     const Array3D<T>& newInput,
		     Array3D<T>& avg)
{
  unsigned int numElements = avg.getNumElements();
  for (unsigned int element = 0; element < numElements; ++element)
    {
      avg(element) = (numArrays * avg(element) - oldInput(element) 
                      + newInput(element)) / numArrays;
    }
}

template <class T>
inline
void 
Array3DUtils::
minFilter1DX(Array3D<T>& array)
{
  Array3D<T> copy(array);
  Vector3D<unsigned int> size = array.getSize();
  T min;
  for (unsigned int z = 0; z < size.z; ++z) {
    for (unsigned int y = 0; y < size.y; ++y) {
      for (unsigned int x = 0; x < size.x; ++x) {
	min = copy(x,y,z);
	if (x > 0 && copy(x-1,y,z) < min) min = copy(x-1,y,z);
	if (x < size.x - 1 && copy(x+1,y,z) < min) min = copy(x+1,y,z);
	array(x,y,z) = min;
      }
    }
  }
}

template <class T>
inline
void 
Array3DUtils::
minFilter1DY(Array3D<T>& array)
{
  Array3D<T> copy(array);
  Vector3D<unsigned int> size = array.getSize();
  T min;
  for (unsigned int z = 0; z < size.z; ++z) {
    for (unsigned int y = 0; y < size.y; ++y) {
      for (unsigned int x = 0; x < size.x; ++x) {
	min = copy(x,y,z);
	if (y > 0 && copy(x,y-1,z) < min) min = copy(x,y-1,z);
	if (y < size.y - 1 && copy(x,y+1,z) < min) min = copy(x,y+1,z);
	array(x,y,z) = min;
      }
    }
  }
}

template <class T>
inline
void 
Array3DUtils::
minFilter1DZ(Array3D<T>& array)
{
  Array3D<T> copy(array);
  Vector3D<unsigned int> size = array.getSize();
  T min;
  for (unsigned int z = 0; z < size.z; ++z) {
    for (unsigned int y = 0; y < size.y; ++y) {
      for (unsigned int x = 0; x < size.x; ++x) {
	min = copy(x,y,z);
	if (z > 0 && copy(x,y,z-1) < min) min = copy(x,y,z-1);
	if (z < size.z - 1 && copy(x,y,z+1) < min) min = copy(x,y,z+1);
	array(x,y,z) = min;
      }
    }
  }
}

template <class T>
inline
void 
Array3DUtils::
minFilter3D(Array3D<T>& array)
{
  minFilter1DX(array);
  minFilter1DY(array);
  minFilter1DZ(array);
}

template <class T>
inline
void 
Array3DUtils::
maxFilter1DX(Array3D<T>& array)
{
  Array3D<T> copy(array);
  Vector3D<unsigned int> size = array.getSize();
  T max;
  for (unsigned int z = 0; z < size.z; ++z) {
    for (unsigned int y = 0; y < size.y; ++y) {
      for (unsigned int x = 0; x < size.x; ++x) {
	max = copy(x,y,z);
	if (x > 0 && copy(x-1,y,z) > max) max = copy(x-1,y,z);
	if (x < size.x - 1 && copy(x+1,y,z) > max) max = copy(x+1,y,z);
	array(x,y,z) = max;
      }
    }
  }
}

template <class T>
inline
void 
Array3DUtils::
maxFilter1DY(Array3D<T>& array)
{
  Array3D<T> copy(array);
  Vector3D<unsigned int> size = array.getSize();
  T max;
  for (unsigned int z = 0; z < size.z; ++z) {
    for (unsigned int y = 0; y < size.y; ++y) {
      for (unsigned int x = 0; x < size.x; ++x) {
	max = copy(x,y,z);
	if (y > 0 && copy(x,y-1,z) > max) max = copy(x,y-1,z);
	if (y < size.y - 1 && copy(x,y+1,z) > max) max = copy(x,y+1,z);
	array(x,y,z) = max;
      }
    }
  }
}

template <class T>
inline
void 
Array3DUtils::
maxFilter1DZ(Array3D<T>& array)
{
  Array3D<T> copy(array);
  Vector3D<unsigned int> size = array.getSize();
  T max;
  for (unsigned int z = 0; z < size.z; ++z) {
    for (unsigned int y = 0; y < size.y; ++y) {
      for (unsigned int x = 0; x < size.x; ++x) {
	max = copy(x,y,z);
	if (z > 0 && copy(x,y,z-1) > max) max = copy(x,y,z-1);
	if (z < size.z - 1 && copy(x,y,z+1) > max) max = copy(x,y,z+1);
	array(x,y,z) = max;
      }
    }
  }
}

template <class T>
inline
void 
Array3DUtils::
maxFilter3D(Array3D<T>& array)
{
  maxFilter1DX(array);
  maxFilter1DY(array);
  maxFilter1DZ(array);
}

template <class T>
inline
void 
Array3DUtils::
rescaleElements(Array3D<T>& array,
		const T& minValueOut,
		const T& maxValueOut)
{
  T max, min;
  getMinMax(array, min, max);
  rescaleElements(array, min, max, minValueOut, maxValueOut);
}

template <class T>
inline
void 
Array3DUtils::
rescaleElements(Array3D<T>& array,
		const T& minThreshold,
		const T& maxThreshold,
		const T& minValueOut,
		const T& maxValueOut)
{
  unsigned int numElements = array.getNumElements();
  double scale = 
    (double) (maxValueOut - minValueOut) / 
    (double) (maxThreshold - minThreshold);

  for (unsigned int i = 0; i < numElements; ++i)
    {
      if (array(i) < minThreshold) array(i) = minValueOut;
      else if (array(i) > maxThreshold) array(i) = maxValueOut;
      else 
	{
	  array(i) = minValueOut + 
	    (T)(((double)(array(i) - minThreshold)) * scale);
	}
    }
}

template <class T>
inline
void 
Array3DUtils::
diffuse1DX(Array3D<T>& array)
{
  Array3D<T> copy(array);
  Vector3D<unsigned int> size = array.getSize();
  if (size.x < 3) return;

  double weight1 = 1.0/3.0;
  double weight2 = 1.0/3.0;
  double weight3 = 1.0/3.0;
  for (unsigned int z = 0; z < size.z; ++z) {
    for (unsigned int y = 0; y < size.y; ++y) {
      for (unsigned int x = 1; x < size.x-1; ++x) {
	array(x,y,z) = 
	  weight1 * copy(x-1,y,z) + 
	  weight2 * copy(x,y,z) + 
	  weight3 * copy(x+1,y,z);
      }
    }
  }
}

template <class T>
inline
void 
Array3DUtils::
diffuse1DY(Array3D<T>& array)
{
  Array3D<T> copy(array);
  Vector3D<unsigned int> size = array.getSize();
  if (size.y < 3) return;

  double weight1 = 1.0/3.0;
  double weight2 = 1.0/3.0;
  double weight3 = 1.0/3.0;
  for (unsigned int z = 0; z < size.z; ++z) {
    for (unsigned int y = 1; y < size.y-1; ++y) {
      for (unsigned int x = 0; x < size.x; ++x) {
	array(x,y,z) = 
	  weight1 * copy(x,y-1,z) + 
	  weight2 * copy(x,y,z) + 
	  weight3 * copy(x,y+1,z);	  
      }
    }
  }
}

template <class T>
inline
void 
Array3DUtils::
diffuse1DZ(Array3D<T>& array)
{
  Array3D<T> copy(array);
  Vector3D<unsigned int> size = array.getSize();
  if (size.z < 3) return;

  double weight1 = 1.0/3.0;
  double weight2 = 1.0/3.0;
  double weight3 = 1.0/3.0;
  for (unsigned int z = 1; z < size.z-1; ++z) {
    for (unsigned int y = 0; y < size.y; ++y) {
      for (unsigned int x = 0; x < size.x; ++x) {
	array(x,y,z) = 
	  weight1 * copy(x,y,z-1) + 
	  weight2 * copy(x,y,z) + 
	  weight3 * copy(x,y,z+1);	  
      }
    }
  }
}

template <class T>
inline
void 
Array3DUtils::
diffuse3D(Array3D<T>& array)
{
  // this could be more efficient
  diffuse1DX(array);
  diffuse1DY(array);
  diffuse1DZ(array);
}

template <class T, class U>
inline
void 
Array3DUtils::
select(const Array3D<T>& source,
       Array3D<T>& dest,
       const Array3D<U>& mask,
       const U& maskMin, const U& maskMax)
{
  unsigned int numElements = source.getNumElements();
  for (unsigned int i = 0; i < numElements; ++i)
    {
      if (mask(i) >= maskMin && mask(i) <= maskMax)
	{
	  dest(i) = source(i);
	}
    }
}

template <class T>
inline
void 
Array3DUtils::
refineZ(Array3D<T>& array,
	const double& newToOldSlicesRatio)
{
  //
  // IMPORTANT: this function only refines, it is not a general resample
  //
  if (newToOldSlicesRatio <= 1) return;

  //
  // get a copy of array and then resize array
  //
  typename Array3D<T>::SizeType oldSize = array.getSize();
  typename Array3D<T>::SizeType newSize(oldSize);
  newSize.z = (unsigned int) ((oldSize.z-1) * newToOldSlicesRatio) + 1;
  Array3D<T> copy(array);
  array.resize(newSize);

  //
  // use linear interpolation to copy values into new array
  //
  for (unsigned int z = 0; z < newSize.z; ++z) {
    for (unsigned int y = 0; y < newSize.y; ++y) {
      for (unsigned int x = 0; x < newSize.x; ++x) {
	double oldZ = z / newToOldSlicesRatio;
	unsigned int oldZLow  = (unsigned int) floor(oldZ);
	unsigned int oldZHigh = oldZLow + 1;

	if (oldZHigh == oldSize.z)
          {
            //
            // special case: exactly on last slice
            //
            array(x,y,z) = copy(x,y,oldZLow);
          }
	  
	//
	// linear interpolation
	//
	array(x,y,z) = 
	  copy(x,y,oldZLow) * (oldZHigh - oldZ) +
	  copy(x,y,oldZHigh) * (oldZ - oldZLow);
      }
    }
  }
}
  
template <class T>
inline
void 
Array3DUtils::
flipX(Array3D<T>& array)
{
  Vector3D<unsigned int> size = array.getSize();
  T tmp;
  for (unsigned int z = 0; z < size.z; ++z) {
    for (unsigned int y = 0; y < size.y; ++y) {
      for (unsigned int x = 0; x < size.x/2; ++x) {    
	tmp = array(x,y,z);
	array(x,y,z) = array(size.x-x-1,y,z);
	array(size.x-x-1,y,z) = tmp;
      }
    }
  }
}

template <class T>
inline
void 
Array3DUtils::
flipY(Array3D<T>& array)
{
  Vector3D<unsigned int> size = array.getSize();
  T tmp;
  for (unsigned int z = 0; z < size.z; ++z) {
    for (unsigned int x = 0; x < size.x; ++x) {    
      for (unsigned int y = 0; y < size.y/2; ++y) {

	tmp = array(x,y,z);
	array(x,y,z) = array(x,size.y-y-1,z);
	array(x,size.y-y-1,z) = tmp;
      }
    }
  }
}

template <class T>
inline
void 
Array3DUtils::
flipZ(Array3D<T>& array)
{
  Vector3D<unsigned int> size = array.getSize();
  T tmp;
  for (unsigned int y = 0; y < size.y; ++y) {
    for (unsigned int x = 0; x < size.x; ++x) {    
      for (unsigned int z = 0; z < size.z/2; ++z) {
	tmp = array(x,y,z);
	array(x,y,z) = array(x,y,size.z-z-1);
	array(x,y,size.z-z-1) = tmp;
      }
    }
  }
}

template <class T>
inline
T 
Array3DUtils::
mass(const Array3D<T>& array) {
  Vector3D<unsigned int> size = array.getSize();
  T mass = 0;

  unsigned int zIndex, yIndex, xIndex;
  for(zIndex = 0; zIndex < size.z; ++zIndex) {
    for(yIndex = 0; yIndex < size.y; ++yIndex) {
      for(xIndex = 0; xIndex < size.x; ++xIndex) {
	mass += array(xIndex, yIndex, zIndex);
      }	  
    }
  }
  return(mass);
}

template <class T>
inline
T 
Array3DUtils::
sumOfSquaredElements(const Array3D<T>& array) {
  Vector3D<unsigned int> size = array.getSize();
  T ssv = 0;

  unsigned int zIndex, yIndex, xIndex;
  for(zIndex = 0; zIndex < size.z; ++zIndex) {
    for(yIndex = 0; yIndex < size.y; ++yIndex) {
      for(xIndex = 0; xIndex < size.x; ++xIndex) {
	ssv += 
	  array(xIndex, yIndex, zIndex)*array(xIndex, yIndex, zIndex);
      }	  
    }
  }
  return(ssv);
}

template <class T>
inline
void 
Array3DUtils::
sum(Array3D<T>& a, const Array3D<T>& b) {
  unsigned int zIndex, yIndex, xIndex;
  Vector3D<unsigned int> size = a.getSize();
  for(zIndex = 0; zIndex < size.z; ++zIndex) {
    for(yIndex = 0; yIndex < size.y; ++yIndex) {
      for(xIndex = 0; xIndex < size.x; ++xIndex) {
	a(xIndex, yIndex, zIndex) += b(xIndex, yIndex, zIndex);
      }	  
    }
  }
}

template< class T >
inline
void
Array3DUtils::
maskRegionDeterminedByThresholds(Array3D<T>& array, 
				 Array3D<unsigned char>& mask,
				 const Vector3D<unsigned int>& seed,
				 const T& minThresh,
				 const T& maxThresh)
{
  typedef Vector3D<unsigned int> IndexType;

  mask.resize(array.getSize());
  mask.fill(0);

  std::deque< IndexType > Q;
  if (array.isValidIndex(seed) && 
      minThresh < array(seed) && array(seed) < maxThresh) {
    Q.push_back(seed);
  }

  while (!Q.empty()) {

    IndexType index = Q.front();
    Q.pop_front();

    for (unsigned int i = 0; i < 3; ++i) {
      IndexType newIndex = index;
      for (int j = -1; j <= 1; j += 2) {
	newIndex[i] = index[i] + j;
	if (array.isValidIndex(newIndex) && !mask(newIndex) &&
	    minThresh <= array(newIndex) && array(newIndex) <= maxThresh) {
	  mask(newIndex) = 1;
	  Q.push_back(newIndex);
	}
      }
    }

  }
}

