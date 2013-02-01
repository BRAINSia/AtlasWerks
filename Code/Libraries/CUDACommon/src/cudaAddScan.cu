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

#include <cutil_comfunc.h>
#include "cudaScan.h"
#include "typeConvert.h"
#include "baseScan.cu"
#include "cudpp/cudpp.h"
#include <cutil.h>


#define ADD_SCAN_CTA_SIZE     128
#define LOG_ADD_SCAN_CTA_SIZE 7

template <class T>
__device__ void addscanWarps(T x, T y, T *s_data)
{
    int idx = threadIdx.x;
    T val  = scanwarp<T, LOG_WARP_SIZE - 1>(x, s_data);
    __syncthreads(); 
    T val2 = scanwarp<T, LOG_WARP_SIZE - 1>(y, s_data);
    
    if ((idx & (WARP_SIZE -1))== (WARP_SIZE - 1))
    {
        s_data[idx >> LOG_WARP_SIZE]                = val + x;
        s_data[(idx + blockDim.x) >> LOG_WARP_SIZE] = val2+ y;
    }
    __syncthreads();

#ifndef __DEVICE_EMULATION__
    if (idx < WARP_SIZE)
#endif
    {
        s_data[idx] = scanwarp<T,(LOG_ADD_SCAN_CTA_SIZE-LOG_WARP_SIZE+1)>(s_data[idx], s_data);
    }
    __syncthreads();
    val  += s_data[idx >> LOG_WARP_SIZE];
    val2 += s_data[(idx + blockDim.x) >> LOG_WARP_SIZE];

    __syncthreads();
    
    s_data[idx] = val;
    s_data[idx+blockDim.x] = val2;
}

template <class T>
__device__ void addscanCTA(T            *s_data, 
                           T            *d_blockSums)
{
    T val  = s_data[threadIdx.x];
    T val2 = s_data[threadIdx.x + blockDim.x];
    
    __syncthreads();     
    addscanWarps<T>(val, val2, s_data);
    __syncthreads();  

    if (threadIdx.x == blockDim.x - 1)
        *d_blockSums = val2 + s_data[threadIdx.x + blockDim.x];
}

template <class T>
__device__ void addscanCTA(T            *s_data)
{
    T val  = s_data[threadIdx.x];
    T val2 = s_data[threadIdx.x + blockDim.x];
    __syncthreads();     
    addscanWarps<T>(val, val2, s_data);
    __syncthreads();  
}


template <class T>
__global__ void addScanBlock_kernel(T            *d_odata,
                                    T            *d_idata,
                                    T            *d_blockSum,
                                    int n) 
{
    const int blockSize = 2 * ADD_SCAN_CTA_SIZE;

#if (CUDART_VERSION  < 2020)
    __shared__ T s_data[blockSize];
#else
    volatile __shared__ T s_data[blockSize];
#endif
    uint blockId = blockIdx.x + blockIdx.y * gridDim.x;
    uint offset  = blockId * blockSize;
    
    uint ai = threadIdx.x + offset;
    uint bi = ai + blockDim.x;

    s_data[threadIdx.x]              = (ai < n)? d_idata[ai] : 0;
    s_data[threadIdx.x + blockDim.x] = (bi < n)? d_idata[bi] : 0;

    __syncthreads();
    
    addscanCTA<T>(s_data, d_blockSum + blockId);
    
    if (ai < n) d_odata[ai] = s_data[threadIdx.x];
    if (bi < n) d_odata[bi] = s_data[threadIdx.x + blockDim.x];
}

/*
 * This function is similar to the function offered by CUDACPP how
 * ever it is much faster with small number of input < 8M
 *
 */


template<class T>
__global__ void addScanBlock_kernel2(T         *d_odata,
                                     T         *d_idata,
                                     T         *d_blockSum,
                                     uint n) 
{
    #define T2 typename typeToVector<T,2>::Result
    const uint blockSize = 4 * ADD_SCAN_CTA_SIZE;
    __shared__ T s_data[2*ADD_SCAN_CTA_SIZE];

    uint blockId = blockIdx.x + blockIdx.y * gridDim.x;
    uint offset  = blockId * blockSize;

    T2* d_idata2 = (T2*)(d_idata + offset);
    T2 data0, data1;

    uint tid = threadIdx.x;
    uint ai  = tid;
    uint bi  = tid + blockDim.x;

    uint nElems = n - offset;
    uint i = (ai << 1);
    
    if ((ai<<1) < nElems)
        data0 = d_idata2[ai];
    if ((bi<<1) < nElems)
        data1 = d_idata2[bi];
    
    
    if (i  >=nElems) data0.x = 0;
    if (i+1>=nElems) data0.y = 0;
    data0.y += data0.x;
    s_data[ai] = data0.y;

    i = (bi << 1);
    if (i  >=nElems) data1.x = 0;
    if (i+1>=nElems) data1.y = 0;
    data1.y += data1.x;
    s_data[bi] = data1.y;
        
    __syncthreads();
    
    addscanCTA<T>(s_data, d_blockSum + blockId);

    data0.y = s_data[ai]+data0.x;
    data0.x = s_data[ai];

    data1.y = s_data[bi]+data1.x;
    data1.x = s_data[bi];
    
    T2* d_odata2 = (T2*)(d_odata + offset);

    i = (ai << 1);
    if (i + 1 < nElems)   d_odata2[ai]        = data0;
    else if (i  < nElems) d_odata[i  +offset] = data0.x;

    i = (bi << 1);
    if (i + 1 < nElems)   d_odata2[bi]        = data1;
    else if (i  < nElems) d_odata[i  +offset] = data1.x;
}

template<class T>
__global__ void addScanBlock_kernel4(T         *d_odata,
                                     T         *d_idata,
                                     T         *d_blockSum,
                                     uint n) 
{
    #define T4 typename typeToVector<T,4>::Result
    const uint blockSize = 8 * ADD_SCAN_CTA_SIZE;
    __shared__ T s_data[2 * ADD_SCAN_CTA_SIZE];

    uint blockId = blockIdx.x + blockIdx.y * gridDim.x;
    uint offset  = blockId * blockSize;

    T4* d_idata4 = (T4*)(d_idata + offset);
    T4 data0, data1;

    uint tid = threadIdx.x;
    uint ai  = tid;
    uint bi  = tid + blockDim.x;
    
    data0 = d_idata4[ai];
    data1 = d_idata4[bi];
    
    uint nElems = n - offset;
    
    uint i = 4 * ai;

    data0.x *= (i  < nElems);
    data0.y *= (i+1< nElems);
    data0.z *= (i+2< nElems);
    data0.w *= (i+3< nElems);

    data0.y += data0.x;
    data0.z += data0.y;
    data0.w += data0.z;
        
    s_data[ai] = data0.w;

    i = 4 * bi;
    data1.x *= (i  < nElems);
    data1.y *= (i+1< nElems);
    data1.z *= (i+2< nElems);
    data1.w *= (i+3< nElems);
    
    data1.y += data1.x;
    data1.z += data1.y;
    data1.w += data1.z;
    
    s_data[bi] = data1.w;
        
    __syncthreads();
    
    addscanCTA<T>(s_data, d_blockSum + blockId);

    data0.w = s_data[ai]+data0.z;
    data0.z = s_data[ai]+data0.y;
    data0.y = s_data[ai]+data0.x;
    data0.x = s_data[ai];

    data1.w = s_data[bi]+data1.z;
    data1.z = s_data[bi]+data1.y;
    data1.y = s_data[bi]+data1.x;
    data1.x = s_data[bi];
    
    T4* d_odata4 = (T4*)(d_odata + offset);

    i = 4 * ai;
    if (i + 3 < nElems) d_odata4[ai]        = data0;
    else {
        if (i  <nElems) d_odata[i + offset    ] = data0.x;
        if (i+1<nElems) d_odata[i + offset + 1] = data0.y;
        if (i+2<nElems) d_odata[i + offset + 2] = data0.z;
    }

    i = 4 * bi;
    if (i + 1 < nElems)   d_odata4[bi]        = data1;
    else {
        if (i   < nElems) d_odata[i+offset    ] = data1.x;
        if (i+1 < nElems) d_odata[i+offset + 1] = data1.y;
        if (i+2 < nElems) d_odata[i+offset + 2] = data1.z;
    }
}

template<class T>
void addScanBlock(T            *d_odata,
                  T            *d_idata,
                  T            *d_blockSum,
                  uint n) {
    dim3 threads(ADD_SCAN_CTA_SIZE);
    uint nBlocks = iDivUp(n, ADD_SCAN_CTA_SIZE * 2);
    dim3 grids(nBlocks);
    checkConfig(grids);
    addScanBlock_kernel<<<grids, threads>>>(d_odata, d_idata, d_blockSum, n);
}

template<class T>
void addScanBlock2(T         *d_odata,
                   T         *d_idata,
                   T         *d_blockSum,
                   uint n) {
    dim3 threads(ADD_SCAN_CTA_SIZE);
    uint nBlocks = iDivUp(n, ADD_SCAN_CTA_SIZE*4);
    dim3 grids(nBlocks);
    checkConfig(grids);
    addScanBlock_kernel2<<<grids, threads>>>(d_odata, d_idata, d_blockSum, n);
}

template<class T>
void addScanBlock4(T         *d_odata,
                   T         *d_idata,
                   T         *d_blockSum,
                   T n) {
    dim3 threads(ADD_SCAN_CTA_SIZE);
    uint nBlocks = iDivUp(n, ADD_SCAN_CTA_SIZE*8);
    dim3 grids(nBlocks);
    checkConfig(grids);
    addScanBlock_kernel4<<<grids, threads>>>(d_odata, d_idata, d_blockSum, n);
}

template<class T>
__global__ void addUniform_kernel(T*    d_data,
                                  T*    d_uni,
                                  uint  n)
{
    #define T2 typename typeToVector<T,2>::Result
    const uint blockSize = 4 * ADD_SCAN_CTA_SIZE;
    
    __shared__ T uni;
    uint blockId = blockIdx.x + blockIdx.y * gridDim.x;
    uint tid = threadIdx.x;

    if (tid == 0)
        uni = d_uni[blockId];

    uint blockOffset = blockId * blockSize;
    uint i           = 2 * tid;
    uint add         = blockOffset + i;

    T2  data;
    if (add < n){
        data = *((T2*)(d_data + add));
    }
    
    __syncthreads();

    data.x += uni;
    data.y += uni;
    
    if ( add + 1 < n) *((T2*)(d_data + add))= data;
    else if (add < n) *(d_data + add)       = data.x;
}

template<class T>
void addUniform(T*    d_data,
                T*    d_uni,
                uint  n){
    dim3 threads(ADD_SCAN_CTA_SIZE * 2);
    uint nBlocks = iDivUp(n, ADD_SCAN_CTA_SIZE * 4);
    dim3 grids(nBlocks);
    checkConfig(grids);
    addUniform_kernel<<<grids, threads>>>(d_data, d_uni, n);
}

template<class T>
__global__ void addUniform_kernel2(T*    d_data,
                                   T*    d_uni,
                                   uint  n)
{
    const uint numThreads = 2 * ADD_SCAN_CTA_SIZE;
    const uint blockSize  = 8 * ADD_SCAN_CTA_SIZE;
        
    #define T2 typename typeToVector<T,2>::Result
    __shared__ T2 uni;
    uint blockId = blockIdx.x + blockIdx.y * gridDim.x;
    uint tid     = threadIdx.x;
    
    if (tid == 0){
        uni = *((T2*)(d_uni + 2 * blockId));
    }

    uint blockOffset = blockId * blockSize;
    uint left = n - blockOffset;
    
    T2* d_data2 = (T2*) (d_data + blockOffset);
    uint ai = tid;
    uint bi = tid + numThreads;
    
    T2  data0 = d_data2[ai];
    T2  data1 = d_data2[bi];

    __syncthreads();


    data0.x += uni.x;
    data0.y += uni.x;

    data1.x += uni.y;
    data1.y += uni.y;

    uint i = 2 * ai;

    if ( i + 1< left) d_data2[ai] = data0;
    else if (i< left) d_data[i + blockOffset] = data0.x;

    i    = 2 * bi;
    if ( i + 1< left) d_data2[bi] = data1;
    else if (i< left) d_data[i + blockOffset] = data1.x;
}

template<class T>
void addUniform2(T*    d_data,
                 T*    d_uni,
                 uint  n){
    dim3 threads(2 * ADD_SCAN_CTA_SIZE);
    uint nBlocks = iDivUp(n, 8 * ADD_SCAN_CTA_SIZE);
    dim3 grids(nBlocks);
    checkConfig(grids);
    
    addUniform_kernel2<<<grids, threads>>>(d_data, d_uni, n);
}


void cudaAddScanPlan::clean(){
    cudaFree(d_buffer);
    d_buffer = NULL;
    L0 = 0; L1 = 0;
}

void cudaAddScanPlan::allocate(uint n){
    nMax = n;
    nMaxBlocks    = iDivUp(nMax,       ADD_SCAN_CTA_SIZE * 4);
    nMaxSubBlocks = iDivUp(nMaxBlocks, ADD_SCAN_CTA_SIZE * 4);
    L0 = iAlignUp(nMaxBlocks, 32);
    L1 = iAlignUp(nMaxSubBlocks,32);
    cudaMalloc((void**)&d_buffer, (L0 + L1 + 32) * sizeof(float));

    datatype =  CUDPP_INT;
    
    scanConfig.algorithm = CUDPP_SCAN;
    scanConfig.datatype  = datatype;
    scanConfig.op        = CUDPP_ADD;
    scanConfig.options   = CUDPP_OPTION_EXCLUSIVE | CUDPP_OPTION_FORWARD;
    cudppPlan(&mScanPlan, scanConfig, nMax, 1, 0);
}


////////////////////////////////////////////////////////////////////////////////
// To perform the scan biScan perform 2 scan path instead of one scan path
// This is faster than the one scan path with number of element < (1 << 23)
////////////////////////////////////////////////////////////////////////////////

template<class T>
void cudaAddScanPlan::biScan(T *d_odata,T  *d_idata, uint n)
{
    dim3 threads(ADD_SCAN_CTA_SIZE);
    uint nBlocks = iDivUp(n, ADD_SCAN_CTA_SIZE * 4);
    
    dim3 grids(nBlocks);
    checkConfig(grids);
    addScanBlock_kernel2<<<grids, threads>>>(d_odata, d_idata,L1_buffer<T>(), n);

    if (nBlocks > 1){
        uint nSubBlocks = iDivUp(nBlocks, ADD_SCAN_CTA_SIZE * 4);
        addScanBlock_kernel2<<<nSubBlocks, threads>>>(L1_buffer<T>(), L1_buffer<T>(), L2_buffer<T>(), nBlocks);
        
        if (nSubBlocks > 1){
            addScanBlock_kernel2<<<1, threads>>>(L2_buffer<T>(), L2_buffer<T>(), total_buffer<T>(), nSubBlocks);
            addUniform(L1_buffer<T>(), L2_buffer<T>(), nBlocks);
            //addUniform2(L1_buffer<T>(), L2_buffer<T>(), nBlocks);
        }
        addUniform(d_odata, L1_buffer<T>(),n);
        //addUniform2(d_odata, L1_buffer<T>(),n);
    }
}

template<class T>
void cudaAddScanPlan::scan(T *d_odata,T *d_idata, uint n){
    if (n > (1 << 23)){
        datatype= getCUDPPType(d_odata);
        if (datatype != scanConfig.datatype){
            scanConfig.datatype = datatype;
            cudppPlan(&mScanPlan, scanConfig, nMax, 1, 0);
        }
        cudppScan(mScanPlan, d_odata, d_idata, n);
    }
    else
        biScan(d_odata, d_idata, n);
}

template void cudaAddScanPlan::scan<float>(float *d_odata,float  *d_idata, uint n);
template void cudaAddScanPlan::scan<int>(int *d_odata,int  *d_idata, uint n);
template void cudaAddScanPlan::scan<uint>(uint *d_odata,uint  *d_idata, uint n);

template<class T>
void addScanCPU(T* h_odata, T* h_idata, int n){
    T sum = (T) 0;
    for (int i=0; i< n; ++i){
        h_odata[i] = sum;
        sum       += h_idata[i];
    }
}

template void addScanCPU<float>(float* h_odata, float* h_idata, int n);
template void addScanCPU<int>(int* h_odata, int* h_idata, int n);
template void addScanCPU<uint>(uint* h_odata, uint* h_idata, int n);


void cudaAddScanPlan::test(int n){
    float elapsedTime;
    int nIters = 500;
    
    float* h_i = new float [n];
    float* h_o = new float [n];

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    float* d_i;
    float* d_o;
    
    for (int i=0; i< n; ++i){
        h_i[i] = rand() & 0x03;
    }

    // Test the accuracy first
    cudaMalloc((void**)&d_i, n * sizeof(float));
    cudaMalloc((void**)&d_o, n * sizeof(float));

    cudaMemcpy(d_i, h_i, n * sizeof(float), cudaMemcpyHostToDevice);
    addScanCPU(h_o, h_i, n);

    this->scan(d_o, d_i, n);
    testError(h_o, d_o, 1e-5, n, "Scan ");
    
    cudaEventRecord(start,0);
    for (int i=0; i< nIters; ++i)
        this->scan(d_o, d_i, n);
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf( "\nCustomize scan processing time: %f (ms)\n\n", elapsedTime/nIters);

    // init cudaPlan
    
    CUDPPConfiguration scanConfig;
    scanConfig.algorithm = CUDPP_SCAN;
    scanConfig.datatype  = CUDPP_FLOAT;
    scanConfig.op        = CUDPP_ADD;
    scanConfig.options   = CUDPP_OPTION_EXCLUSIVE | CUDPP_OPTION_FORWARD;

    cudppPlan(&mScanPlan, scanConfig, n, 1, 0);
    cudppScan(mScanPlan, d_o, d_i, n);
    testError(h_o, d_o, 1e-5, n, "cudppScan");

    cudaEventRecord(start,0);
    for (int i=0; i< nIters; ++i)
        cudppScan(mScanPlan, d_o, d_i, n);
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf( "\nCUDPP scan processing time: %f (ms)\n\n", elapsedTime/nIters);

    delete []h_i;
    delete []h_o;
}
