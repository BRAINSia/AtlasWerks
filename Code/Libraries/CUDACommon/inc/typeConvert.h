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

#ifndef TYPECONVERT_H
#define TYPECONVERT_H

#include <float.h>
#include <cutil_math.h>

template <typename T, int N>
struct typeToVector
{
    typedef T Result;
};

template<>
struct typeToVector<int, 4>
{
    typedef int4 Result;
};

template<>
struct typeToVector<unsigned int, 4>
{
    typedef uint4 Result;
};
template<>
struct typeToVector<float, 4>
{
    typedef float4 Result;
};
template<>
struct typeToVector<int, 3>
{
    typedef int3 Result;
};
template<>
struct typeToVector<unsigned int, 3>
{
    typedef uint3 Result;
};
template<>
struct typeToVector<float, 3>
{
    typedef float3 Result;
};
template<>
struct typeToVector<int, 2>
{
    typedef int2 Result;
};

template<>
struct typeToVector<unsigned int, 2>
{
    typedef uint2 Result;
};

template<>
struct typeToVector<float, 2>
{
    typedef float2 Result;
};


/** @brief Returns the maximum value for type \a T.
  * 
  * Implemented using template specialization on \a T.
  */
template <class T> 
__host__ __device__ inline T getMax() { return 0xFFFFFFFF; }
/** @brief Returns the minimum value for type \a T.
* 
* Implemented using template specialization on \a T.
*/
template <class T> 
__host__ __device__ inline T getMin() { return 0; }
// type specializations for the above
// getMax
template <> __host__ __device__ inline unsigned int   getMax() { return UINT_MAX; }
template <> __host__ __device__ inline unsigned short getMax() { return USHRT_MAX; }
template <> __host__ __device__ inline unsigned char  getMax() { return UCHAR_MAX; }

template <> __host__ __device__ inline int getMax()   { return INT_MAX; }
template <> __host__ __device__ inline short getMax() { return SHRT_MAX; }
template <> __host__ __device__ inline char getMax()  { return CHAR_MAX; }
template <> __host__ __device__ inline float getMax() { return FLT_MAX; }

// getMin
template <> __host__ __device__ inline unsigned int   getMin() { return 0; }
template <> __host__ __device__ inline unsigned short getMin() { return 0; }
template <> __host__ __device__ inline unsigned char  getMin() { return 0; }

template <> __host__ __device__ inline int   getMin() { return INT_MIN; }
template <> __host__ __device__ inline short getMin() { return SHRT_MIN; }
template <> __host__ __device__ inline char  getMin() { return CHAR_MIN; }
template <> __host__ __device__ inline float getMin() { return -FLT_MAX; }

#endif
