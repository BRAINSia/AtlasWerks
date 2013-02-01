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

#ifndef __CUDA_COMPLEX_H
#define __CUDA_COMPLEX_H

#include <cutil_math.h>

typedef float2 cplComplex;

inline __host__ __device__ cplComplex complexMul(cplComplex a, cplComplex b){
    return make_float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

inline __host__ __device__ cplComplex complexMulMulC(cplComplex a, cplComplex b, float c){
    return complexMul(a,b) * c;
}

void cplReal2Complex(cplComplex* d_o, float* d_i, int n, cudaStream_t stream=0);

void cplComplexMul(cplComplex* d_o, cplComplex* d_i, cplComplex* d_k, int n, cudaStream_t stream = 0);
void cplComplexMul_I(cplComplex* d_d, cplComplex* d_k, int n, cudaStream_t stream =0);

void cplComplexMulMulC_I(cplComplex *d_d, cplComplex *d_k, float c, int n, cudaStream_t stream=0);
void cplComplexMulNormalize_I(cplComplex *d_d, cplComplex *d_k, int n, cudaStream_t stream=0);


void cplReal2Complex(cplComplex *d_c, const unsigned int  c_w, const unsigned int  c_h,
                      float   *d_r, const unsigned int  r_w, const unsigned int  r_h,
                      cudaStream_t stream = 0);

void cplComplex2Real(float   *d_r, const unsigned int  r_w, const unsigned int  r_h,
                      cplComplex *d_c, const unsigned int  c_w, const unsigned int  c_h,
                      cudaStream_t stream = 0);


void cplComplex2Real(float*  d_r, cplComplex* d_c, float scale,
                      unsigned int  w, unsigned int  h,
                      cudaStream_t stream = 0);

void cplReal2Complex(cplComplex*d_c, float*   d_r,
                      unsigned int  w, unsigned int h,
                      cudaStream_t stream = 0);


#endif
