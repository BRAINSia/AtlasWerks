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

#ifndef __CUDA_TEXFETCH_H
#define __CUDA_TEXFETCH_H

#include <cutil_math.h>
#include <cutil.h>

enum TexType{
    TEX_FLOAT,
    TEX_INT,
    TEX_UINT,
    TEX_FLOAT2,
    TEX_INT2,
    TEX_UINT2,
    TEX_FLOAT4,
    TEX_INT4,
    TEX_UINT4,
};

// Texture cache for array
texture<float, 1> com_tex_float_x;
texture<int, 1> com_tex_int_x;
texture<uint, 1> com_tex_uint_x;

texture<float2, 1> com_tex_float2_x;
texture<int2, 1> com_tex_int2_x;
texture<uint2, 1> com_tex_uint2_x;

texture<float4, 1> com_tex_float4_x;
texture<int4, 1> com_tex_int4_x;
texture<uint4, 1> com_tex_uint4_x;

template<typename T>
inline void cache_bind(const T* d_i){
    cudaBindTexture(NULL, com_tex_float_x, d_i);
}

template<>
inline void cache_bind(const float* d_i){
    cudaBindTexture(NULL, com_tex_float_x, d_i);
}

template<>
inline void cache_bind(const float2* d_i){
    cudaBindTexture(NULL, com_tex_float2_x, d_i);
}

template<>
inline void cache_bind(const float4*d_i){
    cudaBindTexture(NULL, com_tex_float4_x, d_i);
}

template<>
inline void cache_bind(const int* d_i){
    cudaBindTexture(NULL, com_tex_int_x, d_i);
}

template<>
inline void cache_bind(const int2* d_i){
    cudaBindTexture(NULL, com_tex_int2_x, d_i);
}

template<>
inline void cache_bind(const int4*d_i){
    cudaBindTexture(NULL, com_tex_int4_x, d_i);
}

template<>
inline void cache_bind(const uint* d_i){
    cudaBindTexture(NULL, com_tex_uint_x, d_i);
}

template<>
inline void cache_bind(const uint2* d_i){
    cudaBindTexture(NULL, com_tex_uint2_x, d_i);
}

template<>
inline void cache_bind(const uint4*d_i){
    cudaBindTexture(NULL, com_tex_uint4_x, d_i);
}


template<typename T>
__inline__ __device__ T fetch(const uint& i, const T* d_i){
    return (T) tex1Dfetch(com_tex_float_x, i);
}

template<> __inline__ __device__ float fetch(const uint& i, const float* d_i){
    return tex1Dfetch(com_tex_float_x, i);
}

template<> __inline__ __device__ float2 fetch(const uint& i, const float2* d_i){
    return tex1Dfetch(com_tex_float2_x, i);
}

template<> __inline__ __device__ float4 fetch(const uint& i, const float4* d_i){
    return tex1Dfetch(com_tex_float4_x, i);
}

template<> __inline__ __device__ uint fetch(const uint& i, const uint* d_i){
    return tex1Dfetch(com_tex_uint_x, i);
}

template<> __inline__ __device__ uint2 fetch(const uint& i, const uint2* d_i){
    return tex1Dfetch(com_tex_uint2_x, i);
}

template<> __inline__ __device__ uint4 fetch(const uint& i, const uint4* d_i){
    return tex1Dfetch(com_tex_uint4_x, i);
}

template<> __inline__ __device__ int fetch(const uint& i, const int* d_i){
    return tex1Dfetch(com_tex_int_x, i);
}

template<> __inline__ __device__ int2 fetch(const uint& i, const int2* d_i){
    return tex1Dfetch(com_tex_int2_x, i);
}

template<> __inline__ __device__ int4 fetch(const uint& i, const int4* d_i){
    return tex1Dfetch(com_tex_int4_x, i);
}



#endif
