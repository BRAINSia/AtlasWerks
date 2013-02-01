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

#ifndef __CUDA_TRANSPOSE_H
#define __CUDA_TRANSPOSE_H

#include <Vector3D.h>
#include <cuda_runtime.h>

template<typename T>
void transpose_naive(T* odata, T* idata, int width, int height, cudaStream_t stream=NULL);
template<typename T>
void transpose_x32y8(T *odata, T *idata, int width, int height, cudaStream_t stream=NULL);
template<typename T>
void transpose_x32y8_v2(T *odata, T *idata, int width, int height, cudaStream_t stream=NULL);
template<typename T>
void transpose(T *odata, T *idata, int width, int height, cudaStream_t stream=NULL);

void  cpuShift3DMem(float* h_o, float* h_i, int sizeX, int sizeY, int sizeZ, bool dir);

template<typename T>
void cplShiftCoordinate(T* d_o, T* d_i, int sizeX, int sizeY, int sizeZ, bool dir, cudaStream_t stream=NULL);
template<typename T>
void cplShiftCoordinate(T* d_o, T* d_i, const Vector3Di& size, bool dir, cudaStream_t stream=NULL);
void cplShiftCoordinate_tex(float* d_o, float* d_i, const Vector3Di& size, bool dir, cudaStream_t stream=NULL);

template<typename T>
void cplShiftCoordinate_tex(T* d_o, T* d_i, int sizeX, int sizeY, int sizeZ, bool dir, cudaStream_t stream=NULL);
template<typename T>
void cplShiftCoordinate_tex(float* d_o, float* d_i, const Vector3Di& size, bool dir, cudaStream_t stream=NULL);
template<typename T>
void cplShiftCoordinate_shared(T* d_o, T* d_i, int sizeX, int sizeY, int sizeZ, bool dir, cudaStream_t stream=NULL);

// Segmented memshift function
template<typename T>
void cpuSegShiftMem(T* d_odata, T* d_idata, int w, int h, int l,  int nSeg, int size);
template<typename T>
void cudaSegShiftMem(T* d_odata, T* d_idata, int w, int h, int l, int nSeg, int size, cudaStream_t stream=NULL);
template<typename T>
void cudaSegShiftMem_tex(T* d_odata, T* d_idata, int w, int h, int l, int nSeg, int size, cudaStream_t stream=NULL);

void cudaSegShiftMem_shared(float* d_odata, float* d_idata, int w, int h, int l, int nSeg, int size, cudaStream_t stream=NULL);

//testing function
void testSegShiftMem(int width, int height, int length, int nSeg);
void testSegShiftMemComplex(int width, int height, int length, int nSeg);
void testTranspose(int w, int h);
void testTranspose_C2(int w, int h);
void testTranspose_batch(int width, int height, int n);

#endif
