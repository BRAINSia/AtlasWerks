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

#ifndef __CUDA_INTERFACE_CU
#define __CUDA_INTERFACE_CU

#include "cudaInterface.h"
#include <cutil_inline.h>
#include <stdio.h>
#include <libDefine.h>
#include <cplMacro.h>
#include <float.h>

int cudaDeviceFlags[MAX_NUMBER_DEVICES];

void checkCUDAError(const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err) 
    {
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, 
                                  cudaGetErrorString( err) );
        exit(EXIT_FAILURE);
    }
}

bool hasCUDAError(std::string &errMsg)
{
  errMsg = std::string("No Error");
  cudaError_t err = cudaGetLastError();
  if( cudaSuccess != err) 
    {
      errMsg = std::string("Cuda error: ");
      errMsg = errMsg + cudaGetErrorString( err);
      return true;
    }
  return false;
}

bool hasCUDAError()
{
  cudaError_t err = cudaGetLastError();
  return (cudaSuccess != err);
}

std::string getCUDAError()
{
  cudaError_t err = cudaGetLastError();
  if( cudaSuccess != err) 
    {
      std::string errStr("Cuda error: ");
      errStr = errStr + cudaGetErrorString( err);
    }
  return std::string("No Error");
}


void cudaInit(int argc, char **argv)
{   
    CUT_DEVICE_INIT(argc, argv);
}

void allocateDeviceArray(void **devPtr, unsigned int size)
{
    CUDA_SAFE_CALL(cudaMalloc(devPtr, size));
}

void freeDeviceArray(void *devPtr)
{
    if (devPtr) {
        CUDA_SAFE_CALL(cudaFree(devPtr));
        devPtr = NULL;
    }
}


bool isZeroCopyEnable(){
#if 0
    cudaDeviceProp devProp;
    int dev;
    cudaGetDevice(&dev);
    getDeviceInfo(devProp, dev, false);
    return devProp.canMapHostMemory;
#else
    return false;
#endif    
}

bool isZeroCopyEnable(int dev){
    cudaDeviceProp devProp;
    getDeviceInfo(devProp, dev);
    return devProp.canMapHostMemory;
}

void setDeviceFlag(int dev, int flag){
    cudaDeviceFlags[dev] = flag;
}

int getDeviceFlag(int dev){
    return cudaDeviceFlags[dev];
}

void setCurrentDeviceFlag(int flag){
    int curDev = getCurrentDeviceID();
    setDeviceFlag(curDev, flag);
    fprintf(stderr, "Get current flag %d \n ", getCurrentDeviceFlag());
}

int getCurrentDeviceFlag(){
    int curDev = getCurrentDeviceID();
    return getDeviceFlag(curDev);
}

int getCurrentDeviceID(){
    int dev = 0;
    cudaGetDevice(&dev);
    return dev;
}

int getNumberOfCapableCUDADevices(){
    int deviceCount =0;
    cudaGetDeviceCount(&deviceCount);
    return deviceCount;
}

void getDeviceInfo(cudaDeviceProp& deviceProp, int dev, bool print){
    cudaGetDeviceProperties(&deviceProp, dev);
    if (print){
        fprintf(stderr,"\nDevice %d: \"%s\"\n", dev, deviceProp.name);
        fprintf(stderr,"  Major revision number:                         %d\n",deviceProp.major);
        fprintf(stderr,"  Minor revision number:                         %d\n",deviceProp.minor);
        fprintf(stderr,"  Total amount of global memory:                 %ld bytes\n",deviceProp.totalGlobalMem);
#if CUDART_VERSION >= 2000
        fprintf(stderr,"  Number of multiprocessors:                     %d\n",deviceProp.multiProcessorCount);
        fprintf(stderr,"  Number of cores:                               %d\n",8 * deviceProp.multiProcessorCount);
#endif
        fprintf(stderr,"  Total amount of constant memory:               %lu bytes\n",deviceProp.totalConstMem); 
        fprintf(stderr,"  Total amount of shared memory per block:       %lu bytes\n",deviceProp.sharedMemPerBlock);
        fprintf(stderr,"  Total number of registers available per block: %d\n", deviceProp.regsPerBlock);
        fprintf(stderr,"  Warp size:                                     %d\n", deviceProp.warpSize);
        fprintf(stderr,"  Maximum number of threads per block:           %d\n", deviceProp.maxThreadsPerBlock);
        fprintf(stderr,"  Maximum sizes of each dimension of a block:    %d x %d x %d\n",
               deviceProp.maxThreadsDim[0],
               deviceProp.maxThreadsDim[1],
               deviceProp.maxThreadsDim[2]);
        fprintf(stderr,"  Maximum sizes of each dimension of a grid:     %d x %d x %d\n",
               deviceProp.maxGridSize[0],
               deviceProp.maxGridSize[1],
               deviceProp.maxGridSize[2]);
        fprintf(stderr,"  Maximum memory pitch:                          %lu bytes\n", deviceProp.memPitch);
        fprintf(stderr,"  Texture alignment:                             %lu bytes\n", deviceProp.textureAlignment);
        fprintf(stderr,"  Clock rate:                                    %.2f GHz\n", deviceProp.clockRate * 1e-6f);
#if CUDART_VERSION >= 2000
        fprintf(stderr,"  Concurrent copy and execution:                 %s\n",deviceProp.deviceOverlap ? "Yes" : "No");
#endif

#if CUDART_VERSION >= 2020
        fprintf(stderr,"  Integrated:                                    %s\n",deviceProp.integrated ? "Yes" : "No");
        fprintf(stderr,"  Support host page-loked memory mapping         %s\n",deviceProp.canMapHostMemory ? "Yes" : "No");
        if (deviceProp.computeMode == cudaComputeModeDefault){
            fprintf(stderr,"  This device can be used by multi-threads \n");
        } else if (deviceProp.computeMode == cudaComputeModeExclusive){
            fprintf(stderr,"  This device can be used by only one thread \n");
        } else if (deviceProp.computeMode == cudaComputeModeProhibited){
            fprintf(stderr,"  This device can not be used with thread\n");
        }
#endif
    }
}


template<typename T>
void allocatePinnedHostArray(T*& a, uint n){
    cudaHostAlloc((void**)&a, n * sizeof(T), cudaHostAllocPortable);
}

void freePinnedHostArray(void* a){
    if (a){
        cudaFreeHost(a);
    }
}

template void allocatePinnedHostArray(float*& a, uint n);
template void allocatePinnedHostArray(uint*& a, uint n);
template void allocatePinnedHostArray(int*& a, uint n);

template<typename T>
void dmemAlloc(T*& d_ptr, uint n)
{
    cudaMalloc((void**)&d_ptr, n * sizeof(T));
}

template void dmemAlloc(float*& d_ptr, uint n);
template void dmemAlloc(float2*& d_ptr, uint n);
template void dmemAlloc(float4*& d_ptr, uint n);

template void dmemAlloc(int*& d_ptr, uint n);
template void dmemAlloc(int2*& d_ptr, uint n);
template void dmemAlloc(int4*& d_ptr, uint n);

template void dmemAlloc(uint*& d_ptr, uint n);
template void dmemAlloc(uint2*& d_ptr, uint n);
template void dmemAlloc(uint4*& d_ptr, uint n);

void dmemFree(void *devPtr)
{
    if (devPtr) {
        CUDA_SAFE_CALL(cudaFree(devPtr));
        devPtr = NULL;
    }
}

void threadSync()
{
    CUDA_SAFE_CALL(cudaThreadSynchronize());
}

template<typename T>
void copyArrayFromDevice(T* host, const T* device, uint size)
{   
    CUDA_SAFE_CALL(cudaMemcpy(host, device, size * sizeof(T), cudaMemcpyDeviceToHost));
}

template void copyArrayFromDevice<float>(float* host, const float* device, uint size);
template void copyArrayFromDevice<int>(int* host, const int* device, uint size);
template void copyArrayFromDevice<uint>(uint* host, const uint* device, uint size);

template void copyArrayFromDevice<float2>(float2* host, const float2* device, uint size);
template void copyArrayFromDevice<int2>(int2* host, const int2* device, uint size);
template void copyArrayFromDevice<uint2>(uint2* host, const uint2* device, uint size);

template void copyArrayFromDevice<float4>(float4* host, const float4* device, uint size);
template void copyArrayFromDevice<int4>(int4* host, const int4* device, uint size);
template void copyArrayFromDevice<uint4>(uint4* host, const uint4* device, uint size);

template<typename T>
void copyArrayFromDeviceAsync(T* host, const T* device, uint size, cudaStream_t stream)
{   
    cudaMemcpyAsync(host, device, size * sizeof(T), cudaMemcpyDeviceToHost, stream);
    //cutilCheckMsg("Asyn copy - host should be pinned");
}

template void copyArrayFromDeviceAsync<float>(float* host, const float* device, uint size,cudaStream_t stream);
template void copyArrayFromDeviceAsync<int>(int* host, const int* device, uint size,cudaStream_t stream);
template void copyArrayFromDeviceAsync<uint>(uint* host, const uint* device, uint size,cudaStream_t stream);

template void copyArrayFromDeviceAsync<float2>(float2* host, const float2* device, uint size,cudaStream_t stream);
template void copyArrayFromDeviceAsync<int2>(int2* host, const int2* device, uint size,cudaStream_t stream);
template void copyArrayFromDeviceAsync<uint2>(uint2* host, const uint2* device, uint size,cudaStream_t stream);

template void copyArrayFromDeviceAsync<float4>(float4* host, const float4* device, uint size,cudaStream_t stream);
template void copyArrayFromDeviceAsync<int4>(int4* host, const int4* device, uint size,cudaStream_t stream);
template void copyArrayFromDeviceAsync<uint4>(uint4* host, const uint4* device, uint size,cudaStream_t stream);


template<typename T>
void copyArrayToDeviceAsync(T* device, const T* host, uint size, cudaStream_t stream){
    cudaMemcpyAsync(device, host, size * sizeof(T), cudaMemcpyHostToDevice, stream);
    //cutilCheckMsg("Asyn copy - host should be pinned");
}

template void copyArrayToDeviceAsync<float>(float* device, const float* host, uint size, cudaStream_t stream);
template void copyArrayToDeviceAsync<int>(int* device, const int* host, uint size, cudaStream_t stream);
template void copyArrayToDeviceAsync<uint>(uint* device, const uint* host, uint size, cudaStream_t stream);

template void copyArrayToDeviceAsync<float2>(float2* device, const float2* host, uint size, cudaStream_t stream);
template void copyArrayToDeviceAsync<int2>(int2* device, const int2* host, uint size, cudaStream_t stream);
template void copyArrayToDeviceAsync<uint2>(uint2* device, const uint2* host, uint size, cudaStream_t stream);

template void copyArrayToDeviceAsync<float4>(float4* device, const float4* host, uint size, cudaStream_t stream);
template void copyArrayToDeviceAsync<int4>(int4* device, const int4* host, uint size, cudaStream_t stream);
template void copyArrayToDeviceAsync<uint4>(uint4* device, const uint4* host, uint size, cudaStream_t stream);

template<typename T>
void copyArrayToDevice(T* device, const T* host, uint size)
{
    CUDA_SAFE_CALL(cudaMemcpy(device, host, size * sizeof(T), cudaMemcpyHostToDevice));
}

template void copyArrayToDevice<float>(float* device, const float* host, uint size);
template void copyArrayToDevice<int>(int* device, const int* host, uint size);
template void copyArrayToDevice<uint>(uint* device, const uint* host, uint size);

template void copyArrayToDevice<float2>(float2* device, const float2* host, uint size);
template void copyArrayToDevice<int2>(int2* device, const int2* host, uint size);
template void copyArrayToDevice<uint2>(uint2* device, const uint2* host, uint size);

template void copyArrayToDevice<float4>(float4* device, const float4* host, uint size);
template void copyArrayToDevice<int4>(int4* device, const int4* host, uint size);
template void copyArrayToDevice<uint4>(uint4* device, const uint4* host, uint size);


template<typename T>
void copyArrayDeviceToDevice(T* d_dest, const T* d_src, uint size)
{
    cudaMemcpy(d_dest, d_src, size * sizeof(T), cudaMemcpyDeviceToDevice);
}

template void copyArrayDeviceToDevice<float>(float* d_dest, const float* d_src, uint size);
template void copyArrayDeviceToDevice<int>(int* d_dest, const int* d_src, uint size);
template void copyArrayDeviceToDevice<uint>(uint* d_dest, const uint* d_src, uint size);

template void copyArrayDeviceToDevice<uint2>(uint2* d_dest, const uint2* d_src, uint size);
template void copyArrayDeviceToDevice<float2>(float2* d_dest, const float2* d_src, uint size);
template void copyArrayDeviceToDevice<int2>(int2* d_dest, const int2* d_src, uint size);

template void copyArrayDeviceToDevice<uint4>(uint4* d_dest, const uint4* d_src, uint size);
template void copyArrayDeviceToDevice<float4>(float4* d_dest, const float4* d_src, uint size);
template void copyArrayDeviceToDevice<int4>(int4* d_dest, const int4* d_src, uint size);

template<typename T>
void copyArrayDeviceToDeviceAsync(T* d_dest, const T* d_src, uint size, cudaStream_t stream)
{
    cudaMemcpyAsync(d_dest, d_src, size * sizeof(T), cudaMemcpyDeviceToDevice, stream);
}

template void copyArrayDeviceToDeviceAsync<float>(float* d_dest, const float* d_src, uint size, cudaStream_t stream);
template void copyArrayDeviceToDeviceAsync<int>(int* d_dest, const int* d_src, uint size, cudaStream_t stream);
template void copyArrayDeviceToDeviceAsync<uint>(uint* d_dest, const uint* d_src, uint size, cudaStream_t stream);

template void copyArrayDeviceToDeviceAsync<uint2>(uint2* d_dest, const uint2* d_src, uint size, cudaStream_t stream);
template void copyArrayDeviceToDeviceAsync<float2>(float2* d_dest, const float2* d_src, uint size, cudaStream_t stream);
template void copyArrayDeviceToDeviceAsync<int2>(int2* d_dest, const int2* d_src, uint size, cudaStream_t stream);

template void copyArrayDeviceToDeviceAsync<uint4>(uint4* d_dest, const uint4* d_src, uint size, cudaStream_t stream);
template void copyArrayDeviceToDeviceAsync<float4>(float4* d_dest, const float4* d_src, uint size, cudaStream_t stream);
template void copyArrayDeviceToDeviceAsync<int4>(int4* d_dest, const int4* d_src, uint size, cudaStream_t stream);


////////////////////////////////////////////////////////////////////////////////
// Constant <-> Host function - Host must be pinned memory
////////////////////////////////////////////////////////////////////////////////


template<typename T> 
void copyHostToConstant(const char* name, T* h_i, int cnt, cudaStream_t stream)
{
    cudaMemcpyToSymbolAsync(name, h_i, cnt * sizeof(T), 0, cudaMemcpyHostToDevice, stream);
    //cutilCheckMsg("Host to constant - host should be pinned");
}

template void copyHostToConstant<float>(const char* name, float* h_i, int cnt, cudaStream_t stream);
template void copyHostToConstant<int>(const char* name, int* h_i, int cnt, cudaStream_t stream);
template void copyHostToConstant<uint>(const char* name, uint* h_i, int cnt, cudaStream_t stream);

template<typename T> 
void copyConstantToHost(T* h_i, const char* name, int cnt, cudaStream_t stream)
{
    cudaMemcpyFromSymbolAsync(h_i, name, cnt * sizeof(T), 0, cudaMemcpyDeviceToHost, stream);
    //cutilCheckMsg("Constant to Host - host should be pinned");
}

template void copyConstantToHost<float>(float* h_i, const char* name, int cnt, cudaStream_t stream);
template void copyConstantToHost<int>(int* h_i, const char* name, int cnt, cudaStream_t stream);
template void copyConstantToHost<uint>(uint* h_i, const char* name, int cnt, cudaStream_t stream);

////////////////////////////////////////////////////////////////////////////////
// Constant <-> Device function
////////////////////////////////////////////////////////////////////////////////

template<typename T> 
void copyDeviceToConstant(const char* name, T* d_i, int cnt, cudaStream_t stream)
{
    cudaMemcpyToSymbolAsync(name, d_i, cnt * sizeof(T), 0, cudaMemcpyDeviceToDevice, stream);
    //cutilCheckMsg("Asyn Device to Host - host should be pinned");
}

template void copyDeviceToConstant<float>(const char* name, float* d_i, int cnt, cudaStream_t stream);
template void copyDeviceToConstant<int>(const char* name, int* d_i, int cnt, cudaStream_t stream);
template void copyDeviceToConstant<uint>(const char* name, uint* d_i, int cnt, cudaStream_t stream);

template<typename T> 
void copyConstantToDevice(T* d_i, const char* name, int cnt, cudaStream_t stream)
{
    cudaMemcpyFromSymbolAsync(d_i, name, cnt * sizeof(T), 0, cudaMemcpyDeviceToDevice, stream);
    //cutilCheckMsg("Asyn host - device should be pinned");
}

template void copyConstantToDevice<float>(float* d_i, const char* name, int cnt, cudaStream_t stream);
template void copyConstantToDevice<int>(int* d_i, const char* name, int cnt, cudaStream_t stream);
template void copyConstantToDevice<uint>(uint* d_i, const char* name, int cnt, cudaStream_t stream);

////////////////////////////////////////////////////////////////////////////////
// Function with complex structure
////////////////////////////////////////////////////////////////////////////////
/*
void copyArrayToDevice(Vector3D_XY_Z_Array& device, Vector3Df* host, uint cnt)
{   
    float2* xy = new float2[cnt];
    float * z  = new float [cnt];
    for (uint i=0; i<cnt; ++i){
        xy[i] = make_float2(host[i].x, host[i].y);
        z [i] = host[i].z;
    }

    copyArrayToDevice(device.xy, xy, cnt);
    copyArrayToDevice(device.z, z, cnt);

    delete []xy;
    delete []z;

}

void copyArrayFromDevice(Vector3Df* host, Vector3D_XY_Z_Array& device, uint cnt)
{   
    float2* xy = new float2[cnt];
    float * z  = new float [cnt];

    copyArrayFromDevice(xy, device.xy, cnt);
    copyArrayFromDevice(z, device.z, cnt);
    
    for (uint i=0; i<cnt; ++i){
        host[i].x = xy[i].x;
        host[i].y = xy[i].y;
        host[i].z = z[i];
    }

    delete []xy;
    delete []z;
}

void copyArrayToDevice(cplVector3DArray& d_o, Vector3Df* h_i, int n)
{
    int nAlign = iAlignUp(n, CUDA_DATA_BLOCK_ALIGN);
    float* x = new float[nAlign * 3];
    float* y = x + nAlign;
    float* z = x + 2 * nAlign;
    
    for (int i=0; i < n; ++i)
    {
        x[i] = h_i[i].x;
        y[i] = h_i[i].y;
        z[i] = h_i[i].z;
    }

    copyArrayToDevice(d_o.x, x, 3 * nAlign);
    
    delete []x;
}

void copyArrayFromDevice(Vector3Df* h_o, cplVector3DArray& d_i, int n)
{
    int nAlign = iAlignUp(n, CUDA_DATA_BLOCK_ALIGN);
    
    float* x = new float[nAlign * 3];
    float* y = x + nAlign;
    float* z = x + 2 * nAlign;

    copyArrayFromDevice(x, d_i.x, 3 * nAlign);
    
    for (int i=0; i < n; ++i)
        h_o[i] = Vector3Df(x[i], y[i], z[i]);

    delete []x;
}

*/

void cudaGetMemInfo(cuda_size& free, cuda_size& total){
    cuMemGetInfo(&free, &total);
}

void printCUDAMemoryUsage(){
    cuda_size free, total;
    cudaGetMemInfo(free, total);
    fprintf(stderr, "Amount of meory used %u from total %u MB \n", (total - free) >> 20, total >> 20);
}

template<typename T>
T hostArraySum(T* a, int n) {
    long i;
    T sum, correction, corrected_next_term, new_sum;
    
    sum = a[0];
    correction = (T) 0;
    for (i = 1; i < n; i++)
    {
        corrected_next_term = a[i] - correction;
        new_sum = sum + corrected_next_term;
        correction = (new_sum - sum) - corrected_next_term;
        sum = new_sum;
    }
    return sum;
}

template<typename T>
T deviceArraySum(T* d_a, int n){
    T * h_a = new T [n];
    cudaMemcpy(h_a, d_a, n * sizeof(float), cudaMemcpyDeviceToHost);
    T sum = hostArraySum(h_a, n);
    delete []h_a;
    return sum;
}

template float deviceArraySum(float* d_a, int n);
template int deviceArraySum(int* d_a, int n);
template unsigned int deviceArraySum(unsigned int* d_a, int n);

float getHostSumDouble(float* a, int n){
    double sum = 0;
    for (int i=0; i< n; ++i){
        sum += a[i];
    }
    return sum;
}

float getDeviceSumDouble(float* d_a, int n){
    float* h_a = new float [n];
    cudaMemcpy(h_a, d_a, n * sizeof(float), cudaMemcpyDeviceToHost);
    float sum = getHostSumDouble(h_a, n);
    delete []h_a;
    return sum;
}


float getHostSum2Double(float* a, int n){
    double sum = 0;
    for (int i=0; i< n; ++i){
        sum += a[i]*a[i];
    }
    return sum;
}

float getDeviceSum2Double(float* d_a, int n){
    float* h_a = new float [n];
    cudaMemcpy(h_a, d_a, n * sizeof(float), cudaMemcpyDeviceToHost);
    float sum = getHostSum2Double(h_a, n);
    delete []h_a;
    return sum;
}

float getHostDotDouble(float *a, float*b , int n)
{
    double sum;
    for (int i=0; i< n; ++i)
        sum +=a[i] * b[i];
    return sum;
}

float getDeviceDotDouble(float* d_a, float* d_b, int n)
{
    float* h_a = new float [n];
    float* h_b = new float [n];
    cudaMemcpy(h_a, d_a, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_b, d_b, n * sizeof(float), cudaMemcpyDeviceToHost);
    float sum = getHostDotDouble(h_a, d_b, n);
    delete []h_a;
    delete []h_b;
    return sum;
}

void getHostRange(float& minV, float& maxV, float* a, int n){
    minV = FLT_MAX;
    maxV = -FLT_MAX;
    for (int i=0; i< n; ++i)
    {
        minV = minV < a[i] ? minV : a[i];
        maxV = maxV > a[i] ? maxV : a[i];
    }
}

void getDeviceRange(float& minV, float& maxV, float* d_a, int n)
{
    float * h_a = new float [n];
    cudaMemcpy(h_a, d_a, n * sizeof(float), cudaMemcpyDeviceToHost);
    getHostRange(minV, maxV, h_a, n);
    delete []h_a;
}
#endif
