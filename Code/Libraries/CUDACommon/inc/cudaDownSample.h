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

#ifndef __CUDA_DOWNSAMPLE_H
#define __CUDA_DOWNSAMPLE_H

#include <Vector3D.h>
#include <cuda_runtime.h>

class cplVector3DArray;
class cplDownsizeFilter3D;


void computeDSFilterParams(Vector3Df& sigma, Vector3Di& kRadius, int f);
void computeDSFilterParams(Vector3Df& sigma, Vector3Di& kRadius, const Vector3Di& factor);
    
template<class SmoothFilter>
void cplDownSample(float* d_o, float* d_i, const Vector3Di& size,
                   SmoothFilter* smFilter, cplDownsizeFilter3D* dsFilter,
                   float* d_temp0, float* d_temp1, cudaStream_t stream=NULL);

template<class SmoothFilter>
void cplDownSample(cplVector3DArray& d_o, cplVector3DArray& d_i, const Vector3Di& size,
                   SmoothFilter* smFilter, cplDownsizeFilter3D* dsFilter,
                   float* d_temp0, float* d_temp1, cudaStream_t stream=NULL);

template<class SmoothFilter>
void cplDownSample(float* d_o, float* d_i, const Vector3Di& osize, const Vector3Di& isize,
                   SmoothFilter* smFilter, float* d_temp0, float* d_temp1, cudaStream_t stream=NULL);

template<class SmoothFilter>
void cplDownSample(cplVector3DArray& d_o, cplVector3DArray& d_i, const Vector3Di& size,
                    SmoothFilter* smFilter, float* d_temp0, float* d_temp1, cudaStream_t stream=NULL);

#endif
