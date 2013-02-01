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

#ifndef __CUDA_DATA_CONVERT_H
#define __CUDA_DATA_CONVERT_H

#include <cutil_math.h>
#include <cuda_runtime.h>
class cplVector3DArray;

template<typename T, typename T2>
void convertXYtoX_Y(T* d_x, T* d_y, T2* d_xy, uint n, cudaStream_t stream=NULL);
template<typename T2, typename T>
void convertX_YtoXY(T2* d_xy, T* d_x, T* d_y, uint n, cudaStream_t stream=NULL);

template<typename T, typename T3>
void convertXYZtoX_Y_Z(T* d_x, T* d_y, T* d_z, T3* d_xyz, uint n, cudaStream_t stream=NULL);
template<typename T3, typename T>
void convertX_Y_ZtoXYZ(T3* d_xyz, T* d_x, T* d_y, T* d_z, uint n, cudaStream_t stream=NULL);

template<typename T, typename T4>
void convertXYZWtoX_Y_Z_W(T* d_x, T* d_y, T* d_z, T* d_w, T4* d_xyzw, uint n, cudaStream_t stream=NULL);
template<typename T4, typename T>
void convertX_Y_Z_WtoXYZW(T4* d_xyzw, T* d_x, T* d_y, T* d_z, T* d_w, uint n, cudaStream_t stream=NULL);

template<typename T4, typename T>
void convertX_Y_ZtoXYZW(T4* d_xyzw, T* d_x, T* d_y, T* d_z, uint n, cudaStream_t stream=NULL);
template<typename T, typename T4>
void convertXYZWtoX_Y_Z(T* d_x, T* d_y, T* d_z, T4* d_xyzw, uint n, cudaStream_t stream=NULL);

void convertXYZtoX_Y_Z(cplVector3DArray& d_o, float3* d_i, uint n, cudaStream_t stream=NULL);
void convertX_Y_ZtoXYZ(float3* d_o, cplVector3DArray& d_i, uint n, cudaStream_t stream=NULL);
void convertX_Y_ZtoXYZW(float4* d_o, cplVector3DArray& d_i, uint n, cudaStream_t stream=NULL);
void convertXYZWtoX_Y_Z(cplVector3DArray& d_o, float4* d_xyzw, uint n, cudaStream_t stream=NULL);

void testDataConvert(int n);
    
#endif
