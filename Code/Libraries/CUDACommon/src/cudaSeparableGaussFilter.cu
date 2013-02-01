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

#include <cpl.h>
#include <cudaTranspose.h>
#include <cudaSeparableGaussFilter.h>
#include <cudaDownsizeFilter3D.h>

#define MAX_KERNEL_LENGTH 1025
#define MAX_KERNEL_RADIUS (MAX_KERNEL_LENGTH / 2)

// kernel coefficient function
__device__ __constant__ float c_x_Kernel[MAX_KERNEL_LENGTH];
__device__ __constant__ float c_y_Kernel[MAX_KERNEL_LENGTH];
__device__ __constant__ float c_z_Kernel[MAX_KERNEL_LENGTH];

// supplement kernel for the boundary
__device__ __constant__ float c_sx_Kernel[MAX_KERNEL_LENGTH+1];
__device__ __constant__ float c_sy_Kernel[MAX_KERNEL_LENGTH+1];
__device__ __constant__ float c_sz_Kernel[MAX_KERNEL_LENGTH+1];

////////////////////////////////////////////////////////////////////////////////
// Convolution kernel storage
////////////////////////////////////////////////////////////////////////////////
void setConvolutionKernelX(const float *h_Kernel, int kLength){
    cudaMemcpyToSymbol("c_x_Kernel", h_Kernel, kLength * sizeof(float));
}

void setConvolutionKernelY(const float *h_Kernel, int kLength){
    cudaMemcpyToSymbol(c_y_Kernel, h_Kernel, kLength * sizeof(float));
}

void setConvolutionKernelZ(const float *h_Kernel, int kLength){
    cudaMemcpyToSymbol(c_z_Kernel, h_Kernel, kLength * sizeof(float));
}

void setSupplementKernelX(const float *h_Kernel, int kRadius){
    cudaMemcpyToSymbol(c_sx_Kernel, h_Kernel, kRadius * sizeof(float));
}

void setSupplementKernelY(const float *h_Kernel, int kRadius){
    cudaMemcpyToSymbol(c_sy_Kernel, h_Kernel, kRadius * sizeof(float));
}

void setSupplementKernelZ(const float *h_Kernel, int kRadius){
    cudaMemcpyToSymbol(c_sz_Kernel, h_Kernel, kRadius * sizeof(float));
}


void generateGaussian(float* h_filter, float* h_s, float sigma, int kRadius){

    int kLength = kRadius * 2 + 1;
    float scale = 1.f / (2.f * sigma * sigma);

    // Compute the Gaussian coefficient 
    float sum  = h_filter[kRadius] = 1.f;
    for (int i=1; i <= kRadius; ++i){
        float fval = exp( -(float)(i * i) * scale);
        h_filter[kRadius - i] = h_filter[kRadius + i] = fval;
        sum += 2 * fval;
    }

    // Normalize the Gaussian coefficient
    sum = 1.f / sum;
    for (int i=0; i < kLength; ++i)
        h_filter[i] *= sum;

    sum    = 0.f;
    h_s[0] = 1.f;
    // Calculate the coefficient factor for the edge
    for (int i=1; i<= kRadius; ++i){
        sum   += h_filter[i - 1];
        h_s[i] = 1.f / (1.f - sum);
    }
}

__global__ void cplConvolutionX3D_kernel(float* d_o, const float* d_i,
                                          int kRadius,
                                          int sizeX, int sizeY, int sizeZ){

    const int ix = blockDim.x * blockIdx.x + threadIdx.x;
    const int iy = blockDim.y * blockIdx.y + threadIdx.y;
    
    if (ix >= sizeX || iy >= sizeY)
        return;

    int id = MAD(iy, sizeX, ix);
    for (int iz = 0; iz < sizeZ; ++iz){
        // compute the supplement elements
        float f = 1.f;
        if (ix - kRadius < 0)
            f = c_sx_Kernel[-(ix-kRadius)];
        else if (ix + kRadius > (sizeX-1))
            f = c_sx_Kernel[(ix + kRadius)-(sizeX - 1)];

        // 
        float sum = 0;
        for (int k=-kRadius; k <= kRadius; ++k){
            int x = ix + k;
            if (x >=0 && x < sizeX){
                sum += d_i[id + k] * c_x_Kernel[kRadius - k];
            }
        }
        d_o[id] = sum * f;
        id     += sizeX * sizeY;
    }
}

void cplConvolutionX3D(float* d_o, const float* d_i,
                        int kRadius,
                        int sizeX, int sizeY, int sizeZ, cudaStream_t stream){

    dim3 threads(16,16);
    dim3 grids(iDivUp(sizeX, threads.x),iDivUp(sizeY, threads.y));
    cplConvolutionX3D_kernel<<<grids, threads, 0, stream>>>(d_o, d_i, kRadius, sizeX, sizeY, sizeZ);
}


//////////////////////////////////////////////////////////////////////////////////
// Colution column
//////////////////////////////////////////////////////////////////////////////////
__global__ void cplConvolutionY3D_kernel(float* d_o, const float* d_i,
                                          int kRadius,
                                          int sizeX, int sizeY, int sizeZ){

    const int ix = blockDim.x * blockIdx.x + threadIdx.x;
    const int iy = blockDim.y * blockIdx.y + threadIdx.y;
    
    if (ix >= sizeX || iy >= sizeY)
        return;

    int id = MAD(iy, sizeX, ix);
    for (int iz = 0; iz < sizeZ; ++iz){
        // compute the supplement elements

        float f = 1.f;
        if (iy - kRadius < 0)
            f = c_sy_Kernel[-(iy-kRadius)];
        else if (iy + kRadius > (sizeY-1))
            f = c_sy_Kernel[(iy + kRadius) - (sizeY - 1)];

        // 
        float sum = 0;
        for (int k=-kRadius; k <= kRadius; ++k){
            int y = iy + k;
            if (y >=0 && y < sizeY){
                sum += d_i[id + k * sizeX ] * c_y_Kernel[kRadius - k];
            }
        }
        d_o[id] = sum * f;
        id     += sizeX * sizeY;
    }
}



void cplConvolutionY3D(float* d_o, const float* d_i,
                        int kRadius,
                        int sizeX, int sizeY, int sizeZ, cudaStream_t stream){

    dim3 threads(16,16);
    dim3 grids(iDivUp(sizeX, threads.x),iDivUp(sizeY, threads.y));
    cplConvolutionY3D_kernel<<<grids, threads, 0, stream>>>(d_o, d_i, kRadius, sizeX, sizeY, sizeZ);
}



__global__ void cplConvolutionZ3D_kernel(float* d_o, const float* d_i,
                                          int kRadius,
                                          int sizeX, int sizeY, int sizeZ){

    const int ix = blockDim.x * blockIdx.x + threadIdx.x;
    const int iy = blockDim.y * blockIdx.y + threadIdx.y;
    
    if (ix >= sizeX || iy >= sizeY)
        return;

    int id = MAD(iy, sizeX, ix);
    
    for (int iz = 0; iz < sizeZ; ++iz){
        // compute the supplement elements

        float f = 1.f;
        if (iz - kRadius < 0)
            f = c_sz_Kernel[-(iz-kRadius)];
        else if (iz + kRadius > (sizeZ - 1))
            f = c_sz_Kernel[(iz + kRadius) - (sizeZ - 1)];

        // 
        float sum = 0;
        for (int k=-kRadius; k <= kRadius; ++k){
            int z = iz + k;
            if (z >=0 && z < sizeZ){
                sum += d_i[id + k * sizeX * sizeY ] * c_z_Kernel[kRadius - k];
            }
        }
        d_o[id] = sum * f;
        id     += sizeX * sizeY;
    }
}

void cplConvolutionZ3D(float* d_o, const float* d_i,
                        int kRadius,
                        int sizeX, int sizeY, int sizeZ, cudaStream_t stream){
    dim3 threads(16,16);
    dim3 grids(iDivUp(sizeX, threads.x),iDivUp(sizeY, threads.y));
    cplConvolutionZ3D_kernel<<<grids, threads, 0, stream>>>(d_o, d_i, kRadius, sizeX, sizeY, sizeZ);
}


void cpuConvolutionX3D(float* h_o, const float* h_i,
                       const float* h_kernel, const float* h_sKernel,
                       int kRadius, int sizeX, int sizeY, int sizeZ){

    int id = 0;
    for (int iz=0; iz < sizeZ; ++iz)
        for (int iy=0; iy < sizeY; ++iy)
            for (int ix=0; ix < sizeX; ++ix, ++id){
                float f = 1.f;

                if (ix - kRadius < 0)
                    f = h_sKernel[kRadius -ix];
                else if (ix + kRadius > (sizeX -1))
                    f = h_sKernel[ix + kRadius - (sizeX-1)];

                float sum = 0;
                for(int k = -kRadius; k <= kRadius; k++){
                    int x = ix + k;
                    if (x >=0 && x < sizeX)
                        sum = h_i[id + k] * h_kernel[kRadius - k];
                }
                h_o[id] = sum * f;
            }
}


void cpuConvolutionY3D(float* h_o, const float* h_i,
                       const float* h_kernel, const float* h_sKernel,
                       int kRadius, int sizeX, int sizeY, int sizeZ){

    int id = 0;
    for (int iz=0; iz < sizeZ; ++iz)
        for (int iy=0; iy < sizeY; ++iy)
            for (int ix=0; ix < sizeX; ++ix, ++id){
                float f = 1.f;

                if (iy - kRadius < 0)
                    f = h_sKernel[kRadius -iy];
                else if (iy + kRadius > sizeY - 1)
                    f = h_sKernel[iy + kRadius - (sizeY - 1)];

                float sum = 0;
                for(int k = -kRadius; k <= kRadius; k++){
                    int y = iy + k;
                    if (y >=0 && y < sizeY)
                        sum = h_i[id + k * sizeX] * h_kernel[kRadius - k];
                }
                h_o[id] = sum * f;
            }
}

void cpuConvolutionZ3D(float* h_o, const float* h_i,
                       const float* h_kernel, const float* h_sKernel,
                       int kRadius, int sizeX, int sizeY, int sizeZ){

    int id = 0;
    for (int iz=0; iz < sizeZ; ++iz)
        for (int iy=0; iy < sizeY; ++iy)
            for (int ix=0; ix < sizeX; ++ix, ++id){
                float f = 1.f;

                if (iz - kRadius < 0)
                    f = h_sKernel[kRadius -iz];
                else if (iz + kRadius > (sizeZ - 1))
                    f = h_sKernel[iz + kRadius - (sizeZ - 1)];

                float sum = 0;
                for(int k = -kRadius; k <= kRadius; k++){
                    int z = iz + k;
                    if (z >=0 && z < sizeZ)
                        sum = h_i[id + k * sizeX * sizeY] * h_kernel[kRadius - k];
                }
                h_o[id] = sum * f;
            }
}


void cudaGaussianFilter(float* d_o, float* d_i,
                        const int3& size, const float3& sigma, int3& kRadius,
                        float* d_temp0, cudaStream_t stream)
{
    int nElems = size.x * size.y * size.z;
    
    bool need_temp = (d_temp0 == NULL);
    if (need_temp)
        dmemAlloc(d_temp0, nElems);

    // adjust the kernel size if needed
    if (kRadius.x > size.x/2 - 1)
        kRadius.x = size.x/2 - 1;

    if (kRadius.y > size.y/2 - 1)
        kRadius.y = size.y/2 - 1;

    if (kRadius.z > size.z/2 - 1)
        kRadius.z = size.z/2 - 1;

    int3 kLength = make_int3(kRadius.x * 2 + 1, kRadius.y * 2 + 1, kRadius.z * 2 + 1);

    // generate the kernel
    float* h_kX = new float [kLength.x];
    float* h_kY = new float [kLength.y];
    float* h_kZ = new float [kLength.z];

    float* h_sX = new float [kRadius.x + 1];
    float* h_sY = new float [kRadius.y + 1];
    float* h_sZ = new float [kRadius.z + 1];

    generateGaussian(h_kX, h_sX, sigma.x, kRadius.x);
    generateGaussian(h_kY, h_sY, sigma.y, kRadius.y);
    generateGaussian(h_kZ, h_sZ, sigma.z, kRadius.z);

    setConvolutionKernelX(h_kX, kLength.x);
    setConvolutionKernelY(h_kY, kLength.y);
    setConvolutionKernelZ(h_kZ, kLength.z);

    setSupplementKernelX(h_sX, kRadius.x + 1);
    setSupplementKernelY(h_sY, kRadius.y + 1);
    setSupplementKernelZ(h_sZ, kRadius.z + 1);
#if 1
    cplConvolutionX3D(d_o, d_i, kRadius.x, size.x, size.y, size.z);
    cplConvolutionY3D(d_temp0, d_o, kRadius.y, size.x, size.y, size.z);
    cplConvolutionZ3D(d_o, d_temp0, kRadius.z, size.x, size.y, size.z);
#else
    cplConvolutionX3D(d_temp0, d_i, kRadius.x, size.x, size.y, size.z);
    cplShiftCoordinate(d_o, d_temp0, size.x, size.y, size.z, 1);
    
    cplConvolutionX3D(d_temp0, d_o, kRadius.z, size.z, size.x, size.y);
    cplShiftCoordinate(d_o, d_temp0, size.z, size.x, size.y, 1);

    cplConvolutionX3D(d_temp0, d_o, kRadius.y, size.y, size.z, size.x);
    cplShiftCoordinate(d_o, d_temp0, size.y, size.z, size.x, 1);
#endif    
    
    delete []h_kX;
    delete []h_kY;
    delete []h_kZ;

    delete []h_sX;
    delete []h_sY;
    delete []h_sZ;

    if (need_temp)
        dmemFree(d_temp0);
}
