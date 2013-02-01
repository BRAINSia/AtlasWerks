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

#ifndef __CUDA_COMPLEX_CU
#define __CUDA_COMPLEX_CU

#include <cudaComplex.h>
#include <cpl.h>

__global__ void cplReal2Complex_kernel(cplComplex* d_o, float* d_i, int n){
    uint blockId = get_blockID();
    uint id      = get_threadID(blockId);
    if( id < n)
        d_o[id] = make_float2(d_i[id], 0.f);
}

void cplReal2Complex(cplComplex* d_o, float* d_i, int n, cudaStream_t stream){
    dim3 threads(256);
    dim3 grids(iDivUp(n,threads.x));
    checkConfig(grids);

    cplReal2Complex_kernel<<< grids, threads, 0, stream>>>(d_o, d_i, n);
}


/*-----------------------------------------------------------------------------*/
__global__ void cplComplexMul_kernel(cplComplex* d_o, cplComplex* d_i, cplComplex* d_k, int n){
    uint blockId = get_blockID();
    uint id      = get_threadID(blockId);
    
    if( id < n)
        d_o[id] = complexMul(d_i[id], d_k[id]);
}

void cplComplexMul(cplComplex* d_o, cplComplex* d_i, cplComplex* d_k, int n, cudaStream_t stream){
    dim3 threads(256);
    dim3 grids(iDivUp(n,threads.x));
    cplComplexMul_kernel<<<grids, threads, 0, stream>>>(d_o, d_i, d_k, n);
}

/*-----------------------------------------------------------------------------*/
__global__ void cplComplexMul_I_kernel(cplComplex* d_d, cplComplex* d_k, int n){
    uint blockId = get_blockID();
    uint id      = get_threadID(blockId);

    if (id < n)
        d_d[id] = complexMul(d_d[id], d_k[id]);
}

void cplComplexMul_I(cplComplex* d_d, cplComplex* d_k, int n, cudaStream_t stream){
    dim3 threads(256);
    dim3 grids(iDivUp(n,threads.x));
    checkConfig(grids);
    cplComplexMul_I_kernel<<<grids, threads, 0, stream>>>(d_d, d_k, n);
}

/*-----------------------------------------------------------------------------*/
__global__ void cplComplexMulMulC_I_kernel(cplComplex* d_d, cplComplex* d_k, float c, int n){
    uint blockId = get_blockID();
    uint id      = get_threadID(blockId);

    if (id < n)
        d_d[id] = complexMulMulC(d_d[id], d_k[id], c);
}

void cplComplexMulMulC_I(cplComplex* d_d, cplComplex* d_k, float c, int n, cudaStream_t stream){
    dim3 threads(256);
    dim3 grids(iDivUp(n,threads.x));
    checkConfig(grids);
    cplComplexMulMulC_I_kernel<<<grids, threads, 0, stream>>>(d_d, d_k, c, n);
}


void cplComplexMulNormalize_I(cplComplex *d_d, cplComplex *d_k, int n, cudaStream_t stream){
    cplComplexMulMulC_I(d_d, d_k, 1.f / n, n, stream);
}

/*-----------------------------------------------------------------------------*/
__global__ void real2Complex_kernel(
    cplComplex *d_c, uint  c_w, uint  c_h,
    float   *d_r, uint  r_w, uint  r_h
){
    uint      tidx   = blockDim.x * blockIdx.x + threadIdx.x;
    uint      tidy   = blockDim.y * blockIdx.y + threadIdx.y;
    uint     c_index = tidx + tidy * c_w;
    uint     r_index = tidx + tidy * r_w;

    if (tidx < r_w && tidy < r_h){
        d_c[c_index] = make_float2(d_r[r_index], 0);
    }
}

void cplReal2Complex(cplComplex *d_c, uint  c_w, uint  c_h,
                      float   *d_r, uint  r_w, uint  r_h,
                      cudaStream_t stream)
{
    dim3 threads(16, 16);
    dim3 grids(iDivUp(r_w, threads.x), iDivUp(r_h, threads.y));
    cplVectorOpers::SetMem(d_c, make_float2(0.f, 0.f), c_w * c_h);
    real2Complex_kernel<<<grids, threads, 0, stream>>>(d_c, c_w, c_h,
                                                       d_r, r_w, r_h);
}

/*-----------------------------------------------------------------------------*/
__global__ void complex2Real_kernel(float   *d_r, uint  r_w, uint  r_h,
                                    cplComplex *d_c, uint  c_w, uint  c_h,
                                    int off_x, int off_y)
{
    uint      tidx   = blockDim.x * blockIdx.x + threadIdx.x;
    uint      tidy   = blockDim.y * blockIdx.y + threadIdx.y;
    uint     c_index = (tidx + off_x) + (tidy + off_y) * c_w;
    uint     r_index = tidx + tidy * r_w;

    if (tidx < r_w && tidy < r_h){
        d_r[r_index] = d_c[c_index].x;
    }

}

void cplComplex2Real(float   *d_r, uint  r_w, uint  r_h,
                      cplComplex *d_c, uint  c_w, uint  c_h,
                      int off_x, int off_y,
                      cudaStream_t stream)
{
    dim3 threads(16, 16);
    dim3 grids(iDivUp(r_w, threads.x), iDivUp(r_h, threads.y));
    complex2Real_kernel<<<grids, threads, 0, stream>>>(d_r, r_w, r_h,
                                                       d_c, c_w, c_h,
                                                       off_x, off_y);
}


/*-------------------------------------------------------------------------------*/
__global__ void complex2real_kernel(float* d_r, cplComplex* d_c, float scale,
                                    uint  w, uint  h){
    uint  idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint  idy = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (idx < w && idy < h){
        uint  id   = idx + idy * w;
        volatile float2 cv = d_c[id];
        d_r[id] = cv.x * scale;
    }
}

void cplComplex2Real(float* d_r, cplComplex* d_c, float scale,
                      uint  w, uint  h,
                      cudaStream_t stream){
    dim3 threads(16, 16);
    dim3 grids(iDivUp(w, threads.x), iDivUp(h, threads.y));
    complex2real_kernel<<<grids, threads, 0, stream>>>(d_r, d_c, scale, w, h);
}
/*-------------------------------------------------------------------------------*/
__global__ void real2complex_kernel(cplComplex* d_c, float* d_r, uint  w, uint  h){
    uint  idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint  idy = blockIdx.y * blockDim.y + threadIdx.y;
    if (idx < w && idy < h){
        uint  id   = idx + idy * w;
        d_c[id]    = make_float2(d_r[id], 0.f);
    }
}

void cplReal2Complex(cplComplex* d_c, float* d_r,
                      uint  w, uint  h,
                      cudaStream_t stream){
    dim3 threads(16, 16);
    dim3 grids(iDivUp(w, threads.x), iDivUp(h, threads.y));
    real2complex_kernel<<<grids, threads, 0, stream>>>(d_c, d_r, w, h);
}

/*-----------------------------------------------------------------------------*/
__global__ void extentFromRealToComplex3D(
    float *d_r  , uint  r_w, uint  r_h, uint  r_l,
    cplComplex *d_c, uint  c_w, uint  c_h, uint  c_l
){
    uint     tidx   = blockDim.x * blockIdx.x + threadIdx.x;
    uint     tidy   = blockDim.y * blockIdx.y + threadIdx.y;
    uint     coffset = c_w * c_h;
    uint     roffset = r_w * r_h;
    
    uint     c_index = tidx + tidy * c_w;
    uint     r_index = tidx + tidy * r_w;

    if (tidx < r_w && tidy < r_h){
        for (uint  l = 0; l < r_l; l += 0){
            d_c[c_index] = make_float2(d_r[r_index], 0.f);
            c_index +=coffset;
            r_index +=roffset;
        }
    }
}

__global__ void collapseFromComplexToReal3D(
    cplComplex *d_c, uint  c_w, uint  c_h, uint  c_l,
    float   *d_r, uint  r_w, uint  r_h, uint  r_l,
    uint  off_x, uint  off_y, uint  off_z
){
    uint      tidx   = blockDim.x * blockIdx.x + threadIdx.x;
    uint      tidy   = blockDim.y * blockIdx.y + threadIdx.y;
    
    uint     coffset = c_w * c_h;
    uint     roffset = r_w * r_h;
    
    uint     c_index = (tidx + off_x) + (tidy + off_y)* c_w;
    uint     r_index = tidx + tidy * r_w;

    if (tidx < r_w && tidy < r_h){
        for (uint  l =  0; l < r_l; ++l){
            volatile float2 data = d_c[c_index];
            d_r[r_index] = data.x;
            c_index += coffset;
            r_index += roffset;
        }
    }
}

/*
 * Collapse the real input 2D array with (c_w, c_h) size to 
 * the real output 2D array (r_w, r_h) 
 */

__global__ void real2Real_kernel(
    float *d_ro, uint  w_o, uint  h_o,
    float *d_ri, uint  w_i, uint  h_i,
    uint  off_x, uint  off_y
){
    uint      tidx   = blockDim.x * blockIdx.x + threadIdx.x;
    uint      tidy   = blockDim.y * blockIdx.y + threadIdx.y;

    uint     i_index = (tidx + off_x) + (tidy + off_y) * w_i;
    uint     o_index =  tidx + tidy * w_o;
    
    if (tidx < w_o && tidy < h_o){
        d_ro[o_index] = d_ri[i_index];
    }
}

void real2Real(float *d_ro, uint  w_o, uint  h_o,
               float *d_ri, uint  w_i, uint  h_i,
               uint  off_x, uint  off_y,
               cudaStream_t stream)
{
    dim3 threads(16,16);
    dim3 grids(iDivUp(w_o, threads.x), iDivUp(h_o, threads.y));
    real2Real_kernel<<<grids, threads, 0, stream>>>(d_ro, w_o, h_o,
                                                    d_ri, w_i, h_i,
                                                    off_x, off_y);
}


#endif
