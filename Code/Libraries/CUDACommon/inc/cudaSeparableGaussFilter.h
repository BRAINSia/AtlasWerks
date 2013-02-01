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

#ifndef __CUDA_SEPARABLE_GAUSSIAN_FILTER_H
#define __CUDA_SEPARABLE_GAUSSIAN_FILTER_H

/**
 * Generate the Gaussian kernel and Gaussian supplement kernel (cut off) on the boundary 
 *  h_filter : Gaussian kernel 
 */
void generateGaussian(float* h_filter, float* h_s, float sigma, int kRadius);

/**
 * Load the kernel to the constant memory X
 */
void setConvolutionKernelX(const float *h_Kernel, int kLength);
void setConvolutionKernelY(const float *h_Kernel, int kLength);
void setConvolutionKernelZ(const float *h_Kernel, int kLength);

/**
 * Load the supplement kernel to the constant memory X
 */
void setSupplementKernelX(const float *h_Kernel, int kRadius);
void setSupplementKernelY(const float *h_Kernel, int kRadius);
void setSupplementKernelZ(const float *h_Kernel, int kRadius);


/**
 * Perform the GPU convolution on the X direction
 */
void cplConvolutionX3D(float* d_o, const float* d_i,
                        int kRadius, int sizeX, int sizeY, int sizeZ, cudaStream_t stream=NULL);
/**
 * Perform the GPU convolution on the Y direction
 */
void cplConvolutionY3D(float* d_o, const float* d_i,
                        int kRadius, int sizeX, int sizeY, int sizeZ, cudaStream_t stream=NULL);
/**
 * Perform the GPU convolution on the Z direction
 */
void cplConvolutionZ3D(float* d_o, const float* d_i,
                        int kRadius, int sizeX, int sizeY, int sizeZ, cudaStream_t stream=NULL);

/**
 * Perform the CPU convolution on the X direction
 */
void cpuConvolutionX3D(float* h_o, const float* h_i, const float* h_kernel, const float* h_sKernel,
                       int kRadius, int sizeX, int sizeY, int sizeZ);
/**
 * Perform the CPU convolution on the Y direction
 */
void cpuConvolutionY3D(float* h_o, const float* h_i, const float* h_kernel, const float* h_sKernel,
                       int kRadius, int sizeX, int sizeY, int sizeZ);
/**
 * Perform the CPU convolution on the Z direction
 */
void cpuConvolutionZ3D(float* h_o, const float* h_i, const float* h_kernel, const float* h_sKernel,
                       int kRadius, int sizeX, int sizeY, int sizeZ);


/**
 * This one still exist to maintain the compatible with previous version
 * however recommend not to use it but the cplGaussianFilter instead
 */
void cudaGaussianFilter(float* d_o, float* d_i,
                        const int3& size, const float3& sigma, int3& kRadius,
                        float* d_temp0, cudaStream_t stream);


#endif
