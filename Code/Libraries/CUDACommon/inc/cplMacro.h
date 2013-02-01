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

#ifndef __CPLV_MACRO_H
#define __CPLV_MACRO_H

#define FIX_SCALE_20 1048576.f

////////////////////////////////////////////////////////////////////////////////
//
//  This include file specialy for the function that require CUDA context 
//  and data structure. It SHOULD only be included in the .CU file only
//  since the CUDA specially variable have no meaning in regular C++
//
////////////////////////////////////////////////////////////////////////////////

/**
 * Get the block ID from the config
 */
inline __device__ uint get_blockID(){
    return blockIdx.x + blockIdx.y * gridDim.x;
}

/**
 * Get the thread ID from the current config with current block
 */

inline __device__ uint get_threadID(uint blockId){
    return blockId * blockDim.x + threadIdx.x;
}

inline __host__ __device__ bool isInside3D(int x, int y, int z, 
                                           int w, int h, int l){
    return ((x >= 0) && (x < w) &&
            (y >= 0) && (y < h) &&
            (z >= 0) && (z < l));
}

/**
 * @brief Common function to compute the id of the threads in side a block 
 * for 1D/2D setup 
 */
inline __device__ unsigned int get_x_2D(){
    return blockIdx.x * blockDim.x + threadIdx.x;
}

inline __device__ unsigned int get_y_2D(){
    return blockIdx.y * blockDim.y + threadIdx.y;
}

inline __device__ unsigned int get_id_2D(int w){
    return get_y_2D() * w + get_x_2D();
}



inline __device__ __host__ int S2p20(float a){
    return int(a* FIX_SCALE_20 + 0.5f);
}

inline __device__ __host__ int2 S2p20(float2 a){
    return make_int2((int)(a.x *FIX_SCALE_20 + 0.5f),
                     (int)(a.y *FIX_SCALE_20 + 0.5f));
}

inline __device__ __host__ int3 S2p20(float3 a){
    return make_int3((int)(a.x *FIX_SCALE_20 + 0.5f),
                     (int)(a.y *FIX_SCALE_20 + 0.5f),
                     (int)(a.z *FIX_SCALE_20 + 0.5f));
}

inline __device__ __host__ int4 S2p20(float4 a){
    return make_int4((int)(a.x *FIX_SCALE_20 + 0.5f),
                     (int)(a.y *FIX_SCALE_20 + 0.5f),
                     (int)(a.z *FIX_SCALE_20 + 0.5f),
                     (int)(a.w *FIX_SCALE_20 + 0.5f));
}

inline __device__ __host__ float S2n20(int a){
    return (float)a / FIX_SCALE_20;
}

inline __device__ __host__ float2 S2n20(int2 a){
    return make_float2((float)a.x / FIX_SCALE_20, (float)a.y / FIX_SCALE_20);
}

inline __device__ __host__ float3 S2n20(int3 a){
    return make_float3((float)a.x / FIX_SCALE_20, (float)a.y / FIX_SCALE_20, (float)a.z / FIX_SCALE_20);
}

inline __device__ __host__ float4 S2n20(int4 a){
    return make_float4((float)a.x / FIX_SCALE_20,
                       (float)a.y / FIX_SCALE_20,
                       (float)a.z / FIX_SCALE_20,
                       (float)a.w / FIX_SCALE_20);
}

#endif
