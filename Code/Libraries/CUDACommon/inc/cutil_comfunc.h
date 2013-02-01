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

#ifndef CUTIL_COMFUNC_H
#define CUTIL_COMFUNC_H

#include <iostream>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <host_defines.h>
#include <vector_types.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cutil.h>


#ifdef __DEVICE_EMULATION__
#define __SYNC __syncthreads();
#else
#define __SYNC
#endif

#define TIME_MEASURE( call) do {                                      \
    call;                                                             \
    cudaEventRecord(start, 0);                                        \
    for (int i=0; i< nIters; ++i)                                     \
        call;                                                         \
    cudaEventRecord(end, 0);                                          \
    cudaEventSynchronize(end);                                        \
    float runTime;                                                    \
    cudaEventElapsedTime(&runTime, start, end);                       \
	runTime /= float(nIters);                                         \
	fprintf(stderr, "Runtime %f ms\n", runTime);                      \
    } while (0)


#ifndef IMUL
#define IMUL __mul24
#endif

#ifndef UIMUL
#define UIMUL __umul24
#endif

#define UMUL(a, b)    __umul24( (a), (b) )
#define IMAD(a, b, c) ( __mul24((a), (b)) + (c) )
#define UMAD(a, b, c) ( UMUL((a), (b)) + (c) )
#define MAD(a,b,c)    ( (a) * (b) + (c))

inline int iDivUp(int a, int b){
    return (a + b - 1) / b;
}

template <typename Type>
void SafeDelete(Type *&ptr)
{
    if(ptr)
    {
        delete ptr;
        ptr = NULL;
    }
}

template <typename Type>
void SafeArrayDelete(Type *&ptr)
{
    if(ptr)
    {
        delete[] ptr;
        ptr = NULL;
    }
}


template <typename Type>
void cudaSafeDelete(Type *&ptr)
{
    if(ptr)
    {
        cudaFree(ptr);
        ptr = NULL;
    }
}

template<class T>
inline  __host__  __device__ void swap(T& x, T& y){
    T temp = x;
    x = y;
    y = temp;
}

inline int nextPowerOf2(int v){
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    return v + 1;
}

inline int log2_Pow2(int v){
    static const unsigned int b[] = {0xAAAAAAAA, 0xCCCCCCCC, 0xF0F0F0F0, 0xFF00FF00, 0xFFFF0000};
    register unsigned int r = (v & b[0]) != 0;
    for (int i = 4; i > 0; i--) // unroll for speed...
    {
        r |= ((v & b[i]) != 0) << i;
    }
    return r;
}

inline int isPow2(int v){
    return (!(v & (v - 1)) && v);
}

inline void checkConfig(dim3& grids){
    int nBlocks = grids.x;
    if (nBlocks >= (1 << 16)){
        int bly = nBlocks;
        int blx = 1;
        while (bly >= (1 << 16)){
            bly >>=1;
            blx <<=1;
        }
        grids.x = blx;
        grids.y = (blx * bly == nBlocks) ? bly : bly + 1;
    }
}

inline dim3 make_large_grid(uint num_blocks){
    if (num_blocks <= 65535){
        return dim3(num_blocks);
    } else {
        dim3 grids(num_blocks);
        checkConfig(grids);
        return grids;
    }
}

inline dim3 make_large_grid(uint num_threads, uint blocksize){
    uint num_blocks = iDivUp(num_threads, blocksize);
    if (num_blocks <= 65535){
        //fits in a 1D grid
        return dim3(num_blocks);
    } else {
        dim3 grids(num_blocks);
        checkConfig(grids);
        return grids;
    }
}

inline dim3 make_small_grid(uint num_blocks){
    if (num_blocks <= 65535){
        return dim3(num_blocks);
    } else {
        fprintf(stderr,"Requested size exceedes 1D grid dimensions\n");
        return dim3(0);
    }
}

inline dim3 make_small_grid(uint num_threads, uint blocksize){
    uint num_blocks = iDivUp(num_threads, blocksize);
    if (num_blocks <= 65535){
        return dim3(num_blocks);
    } else {
        fprintf(stderr,"Requested size exceedes 1D grid dimensions\n");
        return dim3(0);
    }
}

static const uint MultiplyDeBruijnBitPosition[32] = 
{
    0, 1, 28, 2, 29, 14, 24, 3, 30, 22, 20, 15, 25, 17, 4, 8, 
    31, 27, 13, 23, 21, 19, 16, 7, 26, 12, 18, 6, 11, 5, 10, 9
};

inline uint log2i(uint v) {
    v |= v >> 1; // first round down to power of 2 
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v = (v >> 1) + 1;
    return MultiplyDeBruijnBitPosition[(v * 0x077CB531UL) >> 27];
}

inline int isPowerOf2(int v) {
    return ((v & (v - 1)) == 0);
}


//Align a to nearest higher multiple of b
inline int iAlignUp(int a, int b){
    return (a % b != 0) ?  (a - a % b + b) : a;
}

inline std::ostream& operator<<(std::ostream& output, const float2& v){
    output << "(" << v.x << ", " << v.y << ")";
    return output;
}

inline std::ostream& operator<<(std::ostream& output, const int2& v){
    output << "(" << v.x << ", " << v.y << ")";
        return output;
}

inline std::ostream& operator<<(std::ostream& output, const uint2& v){
    output << "(" << v.x << ", " << v.y << ")";
    return output;
}

inline std::ostream& operator<<(std::ostream& output, const float4& v){
    output << "(" << v.x << ", " << v.y << ", " << v.z << ", " << v.w <<")";
    return output;
}

inline std::ostream& operator<<(std::ostream& output, const int4& v){
    output << "(" << v.x << ", " << v.y << ", " << v.z << ", " << v.w <<")";
    return output;
}

inline std::ostream& operator<<(std::ostream& output, const uint4& v){
    output << "(" << v.x << ", " << v.y << ", " << v.z << ", " << v.w <<")";
        return output;
}

/**
 * @brief Print out the 1D array (i/f: integer/float input)
 *
 * @param[in]  h_data   The reference array
 *             le       number of elelment
 *             name     name of the array 
 */

template<class T>
inline void printHostArray1D(T* h_i, int n, const char* name){
    std::cerr << name << std::endl;
    for (int i=0; i < n; ++i)
        std::cerr << h_i[i] << " ";
    std::cerr << std::endl;
}

template<class T>
inline void printDeviceArray1D(T* d_i, int n, const char* name){
    T* h_i = new T [n];
    cudaMemcpy(h_i, d_i, n * sizeof(T), cudaMemcpyDeviceToHost);
    printHostArray1D(h_i, n, name);
    delete []h_i;
}

/**
 * @brief Print out the 2D array
 *
 * @param[in]  h_data   The reference array
 *             w        number of column
 *             h        number of row
 *             name     name of the array
 */

template<class T>
inline void printHostArray2D(T* h_i, int w, int h, const char* name){
    std::cerr << name << std::endl;
    for (int j=0; j < h; ++j){
        for (int i=0; i <w; ++i)
            std::cerr << h_i[i + j * w] << " ";
        std::cerr << std::endl;
    }
}

template<class T>
inline void printDeviceArray2D(T* d_i, int w, int h, const char* name){
    T* h_i = new T [w * h];
    cudaMemcpy(h_i, d_i, w * h * sizeof(T), cudaMemcpyDeviceToHost);
    printHostArray2D(h_i, w, h, name);
    delete []h_i;
}

template<class T>
inline T getSumHostArray(T* h_i, int n)
{
    T sum = 0;
    for (int i=0; i < n; ++i)
        sum+=h_i[i];
    return sum;
}

template<class T>
inline T getSumDeviceArray(T* d_i, int n)
{
    T* h_i = new T[n];
    cudaMemcpy(h_i, d_i, n * sizeof(float), cudaMemcpyDeviceToHost);
    T sum = getSumHostArray(h_i, n);
    delete []h_i;
    return sum;
}


template<class T>
inline void printHostArray2D(T* h_i, int w, int h, int pitch, char* name){
    std::cerr << name << std::endl;
    for (int j=0; j < h; ++j){
        for (int i=0; i <w; ++i)
            std::cerr << h_i[i + j * pitch] << " ";
        std::cerr << std::endl;
    }
}

template<class T>
inline void printDeviceArray2D(T* d_i, int w, int h, int pitch, char* name){
    T* h_i = new T [h * pitch];
    cudaMemcpy(h_i, d_i, pitch * h  * sizeof(T), cudaMemcpyDeviceToHost);
    printHostArray2D(h_i, w, h, pitch, name);
    delete []h_i;
}

int testError(float* h_ref, float* d_a, float eps, int n, const char* msg);
template<class T>
int testError(T* h_ref, T* d_a, int n, const char* msg);
int testErrorUlps(float* h_ref, float* d_a, int maxUlps, int n, const char* msg);
    
/**
 * @brief Compute the mean square error .
 *
 * @param[in]  ref   The reference array
 *             a     The computed array
 *             n     The number of elements
 * @returns MSE 
 *
 */

bool isMatchD2D(float* d_i, float* d_i1, unsigned int n);
bool isMatchH2D(float* h_i, float* d_i1, unsigned int n);

// Use this function for CPU accumulation buffer
void cpuAdd_I(float* h_a, float* h_b, int n);
void cpuAdd(float** h_avgL, unsigned int nImgs, unsigned int nVox);

void resampling(float* in, float* out, int s_w, int s_h, int s_l, int d_w, int d_h, int d_l);
void halfSamplingInput(float* out, float* in, int w, int h, int l);
void upSamplingTransformation(float* out, float* in, int w, int h, int l);
void makeRandomUintVector(unsigned int *a, unsigned int numElements, unsigned int keybits);
void makeRandomUintVector(uint2 *a, unsigned int numElements, unsigned int keybits);

#endif
