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

#include <cutil.h>
#include <cutil_comfunc.h>
#include "typeConvert.h"
#include <VectorMath.h>
#include <cudaMap.h>
#include <cplMacro.h>
#include <cudaInterface.h>
#include <cudaTexFetch.h>
/*-------------------------------------------------------------------------------*/
// @brief Perform scatter operation 
//         Size of 2 array is the same
//         This function can be used for mapping value of permutation 
//  @param[in]  iPos   p1, p2, p3, ...pn : position of each ith element in the output
//              iData  input data
//  @param[out] output mapping forward from input to output  
 
template<class T>
__global__ void cplScatter_kernel(T *         oData,
                                  T *         iData,
                                  uint*       iPos,
                                  uint  n)
{
    const uint blockId = blockIdx.x  + blockIdx.y * gridDim.x;
    unsigned int   id = threadIdx.x + blockId * blockDim.x;
    if (id < n)
        oData[iPos[id]] = iData[id];
};


template<class T>
void cplScatter(T*oData, T* iData, uint* iPos, uint n, cudaStream_t stream){
    dim3 threads(256);
    dim3 grids(iDivUp(n,threads.x));
    checkConfig(grids);
    cplScatter_kernel<<<grids, threads, 0, stream>>>(oData, iData, iPos, n);
};


template void cplScatter(float*oData, float* iData, uint* iPos, uint n, cudaStream_t stream);
template void cplScatter(float2*oData, float2* iData, uint* iPos, uint n, cudaStream_t stream);
template void cplScatter(float4*oData, float4* iData, uint* iPos, uint n, cudaStream_t stream);

template void cplScatter(int*oData, int* iData, uint* iPos, uint n, cudaStream_t stream);
template void cplScatter(int2*oData, int2* iData, uint* iPos, uint n, cudaStream_t stream);
template void cplScatter(int4*oData, int4* iData, uint* iPos, uint n, cudaStream_t stream);

template void cplScatter(uint*oData, uint* iData, uint* iPos, uint n, cudaStream_t stream);
template void cplScatter(uint2*oData, uint2* iData, uint* iPos, uint n, cudaStream_t stream);
template void cplScatter(uint4*oData, uint4* iData, uint* iPos, uint n, cudaStream_t stream);


// @brief Perform scatter operation (multipass version)
//        Size of 2 array is the same
//        This function can be used for mapping value of permutation 
// @param[in]  iPos   p1, p2, p3, ...pn : position of each ith element in the output
//             iData  input data
// @param[out] output mapping forward from input to output  

template<class T>
__global__ void cplScatter_multipass_kernel(T *         oData,
                                            T *         iData,
                                            uint*       iPos,
                                            uint  n,
                                            int minIndex,
                                            int maxIndex)
{
    const uint blockId = blockIdx.x  + blockIdx.y * gridDim.x;
    unsigned int   id = threadIdx.x + blockId * blockDim.x;
    if (id < n){
        int writeId = iPos[id];
        if (writeId < maxIndex && writeId >= minIndex)
            oData[writeId] = iData[id];
    }
};


template<class T>
void cplScatter_multipass(T*oData, T* iData, uint* iPos, uint n, int minIndex, int maxIndex, cudaStream_t stream){
    dim3 threads(256);
    dim3 grids(iDivUp(n,threads.x));
    checkConfig(grids);
    cplScatter_multipass_kernel<<<grids, threads, 0, stream>>>(oData, iData, iPos, n, minIndex, maxIndex);
}

template<class T>
void cplScatter(T*oData, T* iData, uint* iPos, uint n, int nPar, cudaStream_t stream){
    unsigned int s     = 0;
    int pSize = iDivUp(n, nPar);
    while (s < n) {
        cplScatter_multipass(oData, iData, iPos, n, s, min(s + pSize, n), stream);
        s += pSize;
    }
}

template void cplScatter(float*oData, float* iData, uint* iPos, uint n, int nPar, cudaStream_t stream);
template void cplScatter(float2*oData, float2* iData, uint* iPos, uint n, int nPar, cudaStream_t stream);
template void cplScatter(float4*oData, float4* iData, uint* iPos, uint n, int nPar, cudaStream_t stream);

template void cplScatter(int*oData, int* iData, uint* iPos, uint n, int nPar, cudaStream_t stream);
template void cplScatter(int2*oData, int2* iData, uint* iPos, uint n, int nPar, cudaStream_t stream);
template void cplScatter(int4*oData, int4* iData, uint* iPos, uint n, int nPar, cudaStream_t stream);

template void cplScatter(uint*oData, uint* iData, uint* iPos, uint n, int nPar, cudaStream_t stream);
template void cplScatter(uint2*oData, uint2* iData, uint* iPos, uint n, int nPar, cudaStream_t stream);
template void cplScatter(uint4*oData, uint4* iData, uint* iPos, uint n, int nPar, cudaStream_t stream);
    
//
// @brief Perform mapping from the memory
//        Size of 2 array is the same
//        This function can be used for mapping value of permutation 
// @param[in]  iPos   p1, p2, p3, ...pn : position of each ith element in the input
//             iData  input data
// @param[out] output iData[p1], iData[p2] ....  
//

template <class T>
__global__ void cplMap_kernel(T *         oData,
                              T *         iData,
                              uint*       iPos,
                              uint  n)
{
    const uint blockId = blockIdx.x  + blockIdx.y * gridDim.x;
    unsigned int   id = threadIdx.x + blockId * blockDim.x;
    if (id < n)
        oData[id] = iData[iPos[id]];
}

template <class T>
void cplMap(T*oData, T* iData, uint* iPos, uint n, cudaStream_t stream){
    dim3 threads(256);
    dim3 grids(iDivUp(n,threads.x));
    checkConfig(grids);
    cplMap_kernel<<<grids, threads, 0, stream>>>(oData, iData, iPos, n);
}

template void cplMap(int*oData, int* iData, uint* iPos, uint n, cudaStream_t stream);
template void cplMap(int2*oData, int2* iData, uint* iPos, uint n, cudaStream_t stream);
template void cplMap(int4*oData, int4* iData, uint* iPos, uint n, cudaStream_t stream);

template void cplMap(uint*oData, uint* iData, uint* iPos, uint n, cudaStream_t stream);
template void cplMap(uint2*oData, uint2* iData, uint* iPos, uint n, cudaStream_t stream);
template void cplMap(uint4*oData, uint4* iData, uint* iPos, uint n, cudaStream_t stream);

template void cplMap(float*oData, float* iData, uint* iPos, uint n, cudaStream_t stream);
template void cplMap(float2*oData, float2* iData, uint* iPos, uint n, cudaStream_t stream);
template void cplMap(float4*oData, float4* iData, uint* iPos, uint n, cudaStream_t stream);


template<typename T>
__global__ void cplMap_kernel(T * oData, uint* iPos, uint  n)
{
    uint blockId = get_blockID();
    uint id      = get_threadID(blockId);
    if (id < n)
        oData[id] = fetch(iPos[id], (T*)NULL);
}

template<typename T>
void cplMap_tex(T* d_o, T* d_i, uint* d_iPos, uint n, cudaStream_t stream)
{
    dim3 threads(64);
    dim3 grids(iDivUp(n, threads.x));
    checkConfig(grids);

    cache_bind(d_i);
    cplMap_kernel<<<grids, threads, 0, stream>>>(d_o, d_iPos, n);
}

template void cplMap_tex(int*oData, int* iData, uint* iPos, uint n, cudaStream_t stream);
template void cplMap_tex(int2*oData, int2* iData, uint* iPos, uint n, cudaStream_t stream);
template void cplMap_tex(int4*oData, int4* iData, uint* iPos, uint n, cudaStream_t stream);

template void cplMap_tex(uint*oData, uint* iData, uint* iPos, uint n, cudaStream_t stream);
template void cplMap_tex(uint2*oData, uint2* iData, uint* iPos, uint n, cudaStream_t stream);
template void cplMap_tex(uint4*oData, uint4* iData, uint* iPos, uint n, cudaStream_t stream);

template void cplMap_tex(float*oData, float* iData, uint* iPos, uint n, cudaStream_t stream);
template void cplMap_tex(float2*oData, float2* iData, uint* iPos, uint n, cudaStream_t stream);
template void cplMap_tex(float4*oData, float4* iData, uint* iPos, uint n, cudaStream_t stream);

//
// @brief Perform mapping from the memory multipass version
//        Size of 2 array is the same
//        This function can be used for mapping value of permutation 
// @param[in]  iPos   p1, p2, p3, ...pn : position of each ith element in the input
//             iData  input data
// @param[out] output iData[p1], iData[p2] ....  
//

template <class T>
__global__ void cplMap_mpass_kernel(T* oData, T* iData, uint* iPos,
                                    uint n, int min_index, int max_index)
{
    uint blockId = get_blockID();
    uint id      = get_threadID(blockId);
    
    if (id < n){
        int read_id = iPos[id];
        if (read_id < max_index && read_id >= min_index)
            oData[id] = iData[read_id];
    }
}

template<class T>
void cplMap_mpass(T*oData, T* iData, uint* iPos, uint n, int min_index, int max_index, cudaStream_t stream)
{
    dim3 threads(256);
    dim3 grids(iDivUp(n,threads.x));
    checkConfig(grids);
    cplMap_mpass_kernel<<<grids, threads, 0, stream>>>(oData, iData, iPos, n, min_index, max_index);
}


template<class T>
void cplMap(T* oData, T* iData, uint* iPos, uint no, int ni, int nPar, cudaStream_t stream){
    int s     = 0;
    int pSize = iDivUp(ni, nPar);

    while (s < ni) {
        cplMap_mpass(oData, iData, iPos, no, s, min(s + pSize, ni), stream);
        s += pSize;
    }
}

template void cplMap(float* d_o, float* d_i, uint* d_iPos, uint no, int ni, int nPar, cudaStream_t stream);
template void cplMap(float2* d_o, float2* d_i, uint* d_iPos, uint no, int ni, int nPar, cudaStream_t stream);
template void cplMap(float4* d_o, float4* d_i, uint* d_iPos, uint no, int ni, int nPar, cudaStream_t stream);

template void cplMap(int* d_o, int* d_i, uint* d_iPos, uint no, int ni, int nPar, cudaStream_t stream);
template void cplMap(int2* d_o, int2* d_i, uint* d_iPos, uint no, int ni, int nPar, cudaStream_t stream);
template void cplMap(int4* d_o, int4* d_i, uint* d_iPos, uint no, int ni, int nPar, cudaStream_t stream);

template void cplMap(uint* d_o, uint* d_i, uint* d_iPos, uint no, int ni, int nPar, cudaStream_t stream);
template void cplMap(uint2* d_o, uint2* d_i, uint* d_iPos, uint no, int ni, int nPar, cudaStream_t stream);
template void cplMap(uint4* d_o, uint4* d_i, uint* d_iPos, uint no, int ni, int nPar, cudaStream_t stream);

//
// @brief Perform mapping from the memory multipass version with texture
//        Size of 2 array is the same
//        This function can be used for mapping value of permutation 
// @param[in]  iPos   p1, p2, p3, ...pn : position of each ith element in the input
//             iData  input data
// @param[out] output iData[p1], iData[p2] ....  
//
//---------------------------------------------------------------------------=
template<typename T>
__global__ void cplMap_multipass_tex_kernel(T* oData, uint* iPos,
                                            uint n,int min_index,int max_index)
{
    const uint blockId = blockIdx.x  + blockIdx.y * gridDim.x;
    unsigned int   id  = threadIdx.x + blockId * blockDim.x;

    if (id < n){
        int read_id = iPos[id];
        if (read_id < max_index && read_id >= min_index)
            oData[id] = fetch(read_id, (T*)NULL);
    }
}

template<typename T>
void cplMap_multipass_tex(T*oData, T* iData, uint* iPos,
                          uint n, int min_index, int max_index, cudaStream_t stream)
{
    dim3 threads(256);
    dim3 grids(iDivUp(n,threads.x));
    checkConfig(grids);

    cache_bind(iData);
    cplMap_multipass_tex_kernel<<<grids, threads, 0, stream>>>(oData, iPos, n, min_index, max_index);
}


template<class T>
void cplMap_tex(T* oData, T* iData, uint* iPos, uint no, int ni, int nPar, cudaStream_t stream){
    int s     = 0;
    int pSize = iDivUp(ni, nPar);

    while (s < ni) {
        cplMap_multipass_tex(oData, iData, iPos, no, s, min(s + pSize, ni), stream);
        s += pSize;
    }
}


template void cplMap_tex(float* d_o, float* d_i, uint* d_iPos, uint no, int ni, int nPar, cudaStream_t stream);
template void cplMap_tex(float2* d_o, float2* d_i, uint* d_iPos, uint no, int ni, int nPar, cudaStream_t stream);
template void cplMap_tex(float4* d_o, float4* d_i, uint* d_iPos, uint no, int ni, int nPar, cudaStream_t stream);

template void cplMap_tex(int* d_o, int* d_i, uint* d_iPos, uint no, int ni, int nPar, cudaStream_t stream);
template void cplMap_tex(int2* d_o, int2* d_i, uint* d_iPos, uint no, int ni, int nPar, cudaStream_t stream);
template void cplMap_tex(int4* d_o, int4* d_i, uint* d_iPos, uint no, int ni, int nPar, cudaStream_t stream);

template void cplMap_tex(uint* d_o, uint* d_i, uint* d_iPos, uint no, int ni, int nPar, cudaStream_t stream);
template void cplMap_tex(uint2* d_o, uint2* d_i, uint* d_iPos, uint no, int ni, int nPar, cudaStream_t stream);
template void cplMap_tex(uint4* d_o, uint4* d_i, uint* d_iPos, uint no, int ni, int nPar, cudaStream_t stream);

void testGather(int n){
    std::cout << "Problem size " << n << std::endl;
    
    uint timer;
    CUT_SAFE_CALL( cutCreateTimer( &timer));
    CUT_SAFE_CALL( cutStartTimer( timer));

    int* h_idata = new int[n];
    uint*  h_pos   = new uint[n];
    int* h_odata = new int[n];

    for (int i=0; i< n; ++i)
        h_idata[i] = rand() % n;

    for (int i=0; i< n; ++i)
        h_pos[i] = i;

    for (int i=0; i< n; ++i){
        int t = h_pos[i];
        int r = rand() % n;
        h_pos[i] = h_pos[r];
        h_pos[r] = t;
    }

    for (int i=0; i< n; ++i)
        h_odata[i] = h_idata[h_pos[i]];
                 
    int* d_idata, *d_odata;
    uint* d_pos;
    
    int size = n * sizeof(uint);

    dmemAlloc(d_idata, n);
    dmemAlloc(d_pos  , n);
    dmemAlloc(d_odata, n);

    cudaMemcpy(d_idata, h_idata, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_pos  , h_pos  , size, cudaMemcpyHostToDevice);

    int nIter = 200;

    cplMap(d_odata, d_idata, d_pos, n);
    testError(h_odata, d_odata, n, "Normal gather");

    CUT_SAFE_CALL( cutResetTimer( timer));
    CUT_SAFE_CALL( cutStartTimer( timer));
    for (int i=0; i < nIter; ++i) 
        cplMap(d_odata, d_idata, d_pos, n);
    cudaThreadSynchronize();
    CUT_SAFE_CALL( cutStopTimer( timer));
    printf( "Processing time: %f (ms)\n", cutGetTimerValue(timer)/nIter);

    int nPar;
    
    for (nPar = 2; nPar <= 64; nPar<<=1){
        cplVectorOpers::SetMem(d_odata, 0, n);

        CUT_SAFE_CALL( cutResetTimer( timer));
        CUT_SAFE_CALL( cutStartTimer( timer));
        for (int i=0; i < nIter; ++i)
            cplMap(d_odata, d_idata, d_pos, n, n, nPar);
        cudaThreadSynchronize();
        CUT_SAFE_CALL( cutStopTimer( timer));

        testError(h_odata, d_odata, n, "Multipass gather ");
        printf( "Processing time with %d part: %f (ms)\n", nPar, cutGetTimerValue(timer)/nIter);
    }

    for (nPar = 2; nPar <= 64; nPar<<=1){
        cplVectorOpers::SetMem(d_odata, 0, n);
        
        CUT_SAFE_CALL( cutResetTimer( timer));
        CUT_SAFE_CALL( cutStartTimer( timer));
        for (int i=0; i < nIter; ++i)
            cplMap_tex(d_odata, d_idata, d_pos, n, n, nPar, 0);

        cudaThreadSynchronize();
        CUT_SAFE_CALL( cutStopTimer( timer));

        testError(h_odata, d_odata, n, "Multipass gather with texture");
        printf( "Processing time with %d part: %f (ms)\n", nPar, cutGetTimerValue(timer)/nIter);
    }

    dmemFree(d_idata);
    dmemFree(d_odata);
    dmemFree(d_pos);

    delete []h_odata;
    delete []h_idata;
    delete []h_pos;

    CUT_SAFE_CALL( cutDeleteTimer( timer));

}

void testScatter(int n){
    std::cout << "Problem size " << n << std::endl;
    
    uint timer;

    CUT_SAFE_CALL( cutCreateTimer( &timer));
    CUT_SAFE_CALL( cutStartTimer( timer));

    float* h_idata = new float[n];
    uint* h_pos   = new uint[n];
    float* h_odata = new float[n];

    for (int i=0; i< n; ++i)
        h_idata[i] = rand() % n;

    for (int i=0; i< n; ++i)
        h_pos[i] = i;

    for (int i=0; i< n; ++i){
        int t = h_pos[i];
        int r = rand() % n;
        h_pos[i] = h_pos[r];
        h_pos[r] = t;
    }

    for (int i=0; i< n; ++i)
        h_odata[h_pos[i]] = h_idata[i];
                 
    float* d_idata, *d_odata;
    uint* d_pos;
    
    int size = n * sizeof(uint);

    dmemAlloc(d_idata, n);
    dmemAlloc(d_pos  , n);
    dmemAlloc(d_odata, n);

    cudaMemcpy(d_idata, h_idata, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_pos  , h_pos  , size, cudaMemcpyHostToDevice);

    int nIter = 200;

    cplScatter(d_odata, d_idata, d_pos, n);
    testError(h_odata, d_odata, 1e-6, n, "Normal scatter");

    CUT_SAFE_CALL( cutResetTimer( timer));
    CUT_SAFE_CALL( cutStartTimer( timer));
    for (int i=0; i < nIter; ++i) 
        cplScatter(d_odata, d_idata, d_pos, n);
    cudaThreadSynchronize();
    CUT_SAFE_CALL( cutStopTimer( timer));
    printf( "Processing time: %f (ms)\n", cutGetTimerValue(timer)/nIter);


    cplVectorOpers::SetMem(d_odata, 0.f, n);
    cplScatter(d_odata, d_idata, d_pos, n, 2);
    testError(h_odata, d_odata, 1e-6, n, "Multipass scatter");

    for (int nPar = 2; nPar <= 64; nPar<<=1){
        CUT_SAFE_CALL( cutResetTimer( timer));
        CUT_SAFE_CALL( cutStartTimer( timer));
        for (int i=0; i < nIter; ++i)
            cplScatter(d_odata, d_idata, d_pos, n, nPar);

        cudaThreadSynchronize();
        CUT_SAFE_CALL( cutStopTimer( timer));
        printf( "Processing time with %d part: %f (ms)\n", nPar, cutGetTimerValue(timer)/nIter);
    }
        
    dmemFree(d_idata);
    dmemFree(d_odata);
    dmemFree(d_pos);

    delete []h_odata;
    delete []h_idata;
    delete []h_pos;

    CUT_SAFE_CALL( cutDeleteTimer( timer));

}

void testSparseRead(int n, int s){

    uint timer;
    CUT_SAFE_CALL( cutCreateTimer( &timer));
    CUT_SAFE_CALL( cutStartTimer( timer));

    float* h_idata = new float[n];
    uint*  h_pos   = new uint[n];
    float* h_odata = new float[s];

    for (int i=0; i< n; ++i)
        h_idata[i] = rand() % n;


    for (int i=0; i< n; ++i)
        h_pos[i] = i;

    for (int i=0; i< s; ++i){
        int t = h_pos[i];
        int r = rand() % n;
        h_pos[i] = h_pos[r];
        h_pos[r] = t;
    }

    for (int i=0; i< s; ++i)
        h_odata[i] = h_idata[h_pos[i]];
                 
    float* d_idata, *d_odata;
    uint* d_pos;
    
    dmemAlloc(d_idata, n);
    dmemAlloc(d_pos  , s);
    dmemAlloc(d_odata, s);

    cudaMemcpy(d_idata, h_idata, n* sizeof(uint), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pos, h_pos, s* sizeof(uint), cudaMemcpyHostToDevice);

    int nIter = 100;
    
    cplMap(d_odata, d_idata, d_pos, s);
    testError(h_odata, d_odata, 1e-6, s, "Global mem sparse read");
        
    CUT_SAFE_CALL( cutResetTimer( timer));
    CUT_SAFE_CALL( cutStartTimer( timer));
    for (int i=0; i < nIter; ++i) 
        cplMap(d_odata, d_idata, d_pos, s);
    cudaThreadSynchronize();
    CUT_SAFE_CALL( cutStopTimer( timer));
    printf( "Processing time: %f (ms)\n", cutGetTimerValue(timer)/nIter);

    cplMap(d_odata, d_idata, d_pos, s, n, 8);
    testError(h_odata, d_odata, 1e-6, s, "Multipass sparse read");
    CUT_SAFE_CALL( cutResetTimer( timer));
    CUT_SAFE_CALL( cutStartTimer( timer));
    for (int i=0; i < nIter; ++i) 
        cplMap(d_odata, d_idata, d_pos, s, n, 8);
    cudaThreadSynchronize();
    CUT_SAFE_CALL( cutStopTimer( timer));
    printf( "Processing time: %f (ms)\n", cutGetTimerValue(timer)/nIter);
    
    dmemFree(d_idata);
    dmemFree(d_odata);
    dmemFree(d_pos);
            
    delete []h_idata;
    delete []h_pos;

    CUT_SAFE_CALL( cutDeleteTimer( timer));
}

/*----------------------------------------------------------------------
  Ectract function base on texture that look up value from input
  d_o[i] = d_i[d_iPos[i]] if    d_iPos[i] < ni
  keep           other wise 

  Inputs : d_i    : input array
  d_pos  : position array that have same size with output (no)
           
  no     : size of the output
  ni     : size of the input

  Output :
  d_o    : Extract output   
  ---------------------------------------------------------------------*/
template<typename T>
__global__ void extract_kernel_tex(T* d_o, uint* d_iPos, uint no, uint ni){
    uint blockId= blockIdx.x  + blockIdx.y * gridDim.x;
    uint id     = threadIdx.x + blockId * blockDim.x;
    
    if (id < no){
        int l = d_iPos [id];
        if (l < ni)
            d_o[id] = fetch(l, (T*)NULL);
    }
}

template<typename T>
void extract_tex(T* d_o, T* d_i, uint* d_iPos, uint no, uint ni, cudaStream_t stream){
    dim3 threads(256);
    dim3 grids(iDivUp(no, 256));
    checkConfig(grids);

    cache_bind(d_i);
    extract_kernel_tex<<<grids, threads, 0, stream>>>(d_o, d_iPos, no, ni);
}

template void extract_tex(float* d_o, float* d_i, uint* d_iPos, uint no, uint ni, cudaStream_t stream);
template void extract_tex(float2* d_o, float2* d_i, uint* d_iPos, uint no, uint ni, cudaStream_t stream);
template void extract_tex(float4* d_o, float4* d_i, uint* d_iPos, uint no, uint ni, cudaStream_t stream);

template void extract_tex(int* d_o, int* d_i, uint* d_iPos, uint no, uint ni, cudaStream_t stream);
template void extract_tex(int2* d_o, int2* d_i, uint* d_iPos, uint no, uint ni, cudaStream_t stream);
template void extract_tex(int4* d_o, int4* d_i, uint* d_iPos, uint no, uint ni, cudaStream_t stream);

template void extract_tex(uint* d_o, uint* d_i, uint* d_iPos, uint no, uint ni, cudaStream_t stream);
template void extract_tex(uint2* d_o, uint2* d_i, uint* d_iPos, uint no, uint ni, cudaStream_t stream);
template void extract_tex(uint4* d_o, uint4* d_i, uint* d_iPos, uint no, uint ni, cudaStream_t stream);



__global__ void extract4fTof4_kernel(float4* d_o,
                                     float*  d_ix, float*  d_iy, float*  d_iz, float*  d_iw,  
                                     uint* d_iPos, uint no, uint ni){
    uint blockId= blockIdx.x  + blockIdx.y * gridDim.x;
    uint id     = threadIdx.x + blockId * blockDim.x;
    
    if (id < no){
        int l = d_iPos [id];
        if (l < ni){
            float4 r;

            r.x = d_ix[l];
            r.y = d_iy[l];
            r.z = d_iz[l];
            r.w = d_iw[l];
            
            d_o[id] = r;
        }
    }
}

void extract4fTof4(float4* d_o,
                   float*  d_ix, float*  d_iy, float*  d_iz, float*  d_iw,
                   uint* d_iPos, uint no, uint ni, cudaStream_t stream){
    dim3 threads(256);
    dim3 grids(iDivUp(no, 256));
    checkConfig(grids);

    extract4fTof4_kernel<<<grids, threads, 0, stream>>>(d_o,
                                                        d_ix, d_iy, d_iz, d_iw, 
                                                        d_iPos, no, ni);
}


/*----------------------------------------------------------------------
  Extract function base on texture that look up value from input
  d_o[i] += d_i[d_iPos[i]] if    d_iPos[i] < ni
  keep           other wise 

  Inputs : d_i    : input array
  d_pos  : position array that have same size with output (no)
           
  no     : size of the output
  ni     : size of the input

  Output :
  d_o    : Add extract result to the input 
  ---------------------------------------------------------------------*/
template<typename T>
__global__ void addExtract_kernel_tex(T* d_o, uint* d_iPos, uint no, uint ni){
    uint blockId= blockIdx.x  + blockIdx.y * gridDim.x;
    uint id     = threadIdx.x + blockId * blockDim.x;
    if (id < no){
        int l = d_iPos[id];
        if (l < ni)
            d_o[id] += fetch(l, (T*)NULL);
    }
}

template<typename T>
void addExtract_tex(T* d_o, T* d_i, uint* last, uint no, uint ni, cudaStream_t stream){
    dim3 threads(256);
    dim3 grids(iDivUp(no, 256));
    checkConfig(grids);

    cache_bind(d_i);
    addExtract_kernel_tex<<<grids, threads, 0, stream>>>(d_o, last, no, ni);
}


template void addExtract_tex(float* d_o, float* d_i, uint* d_iPos, uint no, uint ni, cudaStream_t stream);
template void addExtract_tex(float2* d_o, float2* d_i, uint* d_iPos, uint no, uint ni, cudaStream_t stream);
template void addExtract_tex(float4* d_o, float4* d_i, uint* d_iPos, uint no, uint ni, cudaStream_t stream);

template void addExtract_tex(int* d_o, int* d_i, uint* d_iPos, uint no, uint ni, cudaStream_t stream);
template void addExtract_tex(int2* d_o, int2* d_i, uint* d_iPos, uint no, uint ni, cudaStream_t stream);
template void addExtract_tex(int4* d_o, int4* d_i, uint* d_iPos, uint no, uint ni, cudaStream_t stream);

template void addExtract_tex(uint* d_o, uint* d_i, uint* d_iPos, uint no, uint ni, cudaStream_t stream);
template void addExtract_tex(uint2* d_o, uint2* d_i, uint* d_iPos, uint no, uint ni, cudaStream_t stream);
template void addExtract_tex(uint4* d_o, uint4* d_i, uint* d_iPos, uint no, uint ni, cudaStream_t stream);
