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

#ifndef __CUDA_RANDOM_H
#define __CUDA_RANDOM_H

#include <cuda_runtime.h>
#include <singleton.h>

typedef long long int INT64;
#define QRNG_DIMENSIONS 3
#define QRNG_RESOLUTION 31
#define INT_SCALE (1.0f / (float)0x80000001U)

class cplQRand {
public:
    cplQRand();
    ~cplQRand();

    void initTableGPU();

    void rand(float* d_i, unsigned int seed, unsigned int n, cudaStream_t s=NULL);
    void rand2(float* d_i, unsigned int seed, unsigned int n, unsigned int nAlign, cudaStream_t s=NULL);
    void rand3(float* d_i, unsigned int seed, unsigned int n, unsigned int nAlign, cudaStream_t s=NULL);
    
    void inverseCND(float *d_o, float *d_i, unsigned int N, cudaStream_t s=NULL);
    void inverseCND_d(float *d_o, float *d_i, unsigned int N, cudaStream_t s=NULL);
    void init();

    double getQuasirandomValue63(INT64 i, int dim);
    float getQuasirandomValue(int i, int dim);
private:
    void initQuasirandomGenerator();
    void GenerateCJ();

    INT64 cjn[63][QRNG_DIMENSIONS];
    unsigned int table[QRNG_DIMENSIONS][QRNG_RESOLUTION];
};

typedef Singleton<cplQRand> cplQRandSingleton;

#endif
