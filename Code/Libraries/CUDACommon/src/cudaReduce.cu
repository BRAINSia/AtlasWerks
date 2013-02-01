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
#include <iostream>

#define MAX_NUMBER_REDUCE_STREAMS     4
#define MAX_NUMBER_REDUCE_THREADS     128
#define MAX_NUMBER_REDUCE_BLOCKS      128

template<typename T, typename opers>
T accumulate(T* h_temp, int N){
    T sum = opers::identity();
    for (int i=0; i< N; ++i)
        opers::iop(sum, h_temp[i]);
    return sum;
}

void getNumBlocksAndThreads(int n, int maxBlocks, int maxThreads, int &blocks, int &threads)
{
    threads = nextPowerOf2(iDivUp(n, 2));
    threads = (threads < maxThreads) ? threads : maxThreads;
    
    blocks = iDivUp(n, threads * 2);
    blocks = blocks < maxBlocks ? blocks : maxBlocks;
}

/**
 * Allocate the memory for reduce temporary buffer
 */
void cplReduce::init(){
    uint size = MAX_NUMBER_REDUCE_BLOCKS * MAX_NUMBER_REDUCE_STREAMS * sizeof(int);
    // Check if zero copy is enable
    m_zeroCopy = isZeroCopyEnable();

    if(!m_zeroCopy) {
        //fprintf(stderr, "Zero copy is not available");
        h_temp   =(void*) malloc(size);
        cudaMalloc((void**)&d_temp, size);
    } else {
        //fprintf(stderr, "Zero copy is enable");
        cudaHostAlloc((void **)&h_temp, size, cudaHostAllocMapped);
        cudaHostGetDevicePointer((void **)&d_temp, (void *)h_temp, 0);
    }
    cutilCheckMsg("cplReduce::init");

    int dev = getCurrentDeviceID();
    //std::cerr << "Device " <<  dev << " Create reduce object " << std::endl;
}

/**
 * Release the temporary buffer
 */
void cplReduce::clean(){
    if (m_zeroCopy) {
        cudaFreeHost(h_temp);
    }else {
        free(h_temp);
        cudaFree(d_temp);
    }
}


template<class T, class traits, uint blockSize>
__inline__  __device__ void reduce_shared(T& mySum, volatile T* sdata, uint tid)
{
    // do reduction in shared mem
    if (blockSize >= 512) {
        if (tid < 256) { sdata[tid] = mySum = traits::op(mySum, sdata[tid + 256]); } __syncthreads();
    }
    if (blockSize >= 256) {
        if (tid < 128) { sdata[tid] = mySum = traits::op(mySum, sdata[tid + 128]); } __syncthreads();
    }
    if (blockSize >= 128) {
        if (tid <  64) { sdata[tid] = mySum = traits::op(mySum, sdata[tid +  64]); } __syncthreads();
    }
                            
    if (tid < 32)
    {
        if (blockSize >=  64) sdata[tid] = mySum = traits::op(mySum, sdata[tid + 32]);
        if (blockSize >=  32) sdata[tid] = mySum = traits::op(mySum, sdata[tid + 16]);
        if (blockSize >=  16) sdata[tid] = mySum = traits::op(mySum, sdata[tid +  8]);
        if (blockSize >=   8) sdata[tid] = mySum = traits::op(mySum, sdata[tid +  4]);
        if (blockSize >=   4) sdata[tid] = mySum = traits::op(mySum, sdata[tid +  2]);
        if (blockSize >=   2) sdata[tid] = mySum = traits::op(mySum, sdata[tid +  1]);
    }
}

/**
 * @brief Perform cuda kernel parallel reductin 
 *        s = a1 + a2 + a3 + .... + an
 * @param[in]  T         Input data type (currently int, float)
 *             traits    Binary operation (+, max, min)
 *             blockSize Size of block (related to optimize problem)
 *             g_idata   Input data
 *             n         Size of the input
 * @param[out] array of output redution perform for each block
 *
*/
template <class T, class traits, uint blockSize>
__global__ void reduce_kernel(const T *g_idata, T *g_odata, uint n)
{
    volatile __shared__ T sdata[MAX_NUMBER_REDUCE_THREADS];
    // reading from global memory, writing to shared memory
    uint tid      = threadIdx.x;
    uint i        = blockIdx.x*(blockSize*2) + tid;
    uint gridSize = blockSize * 2 * gridDim.x;
    
    T mySum = traits::identity();;
    while (i + blockSize < n )
    {
        traits::iop(mySum,traits::op(g_idata[i],g_idata[i+blockSize]));
        i += gridSize;
    }
    if ( i < n) traits::iop(mySum, g_idata[i]);
    sdata[tid] = mySum;

    __syncthreads();
    reduce_shared<T, traits, blockSize>(mySum, sdata, tid);

    // write result for this block to global mem 
    if (tid == 0) {
        g_odata[blockIdx.x] = mySum;
    }
}

template <class T, class traits>
void reduce(uint n, int threads, int blocks, const T *d_i, T *d_o){
    dim3 dimBlock(threads);
    dim3 dimGrid(blocks);
    switch (threads)  {
        case 512: reduce_kernel<T, traits, 512><<< dimGrid, dimBlock>>>(d_i, d_o, n);
            break;
        case 256: reduce_kernel<T, traits, 256><<< dimGrid, dimBlock>>>(d_i, d_o, n);
            break;
        case 128: reduce_kernel<T, traits, 128><<< dimGrid, dimBlock>>>(d_i, d_o, n);
            break;
        case 64:  reduce_kernel<T, traits, 64><<< dimGrid, dimBlock>>>(d_i, d_o, n);
            break;
        case 32:  reduce_kernel<T, traits, 32><<< dimGrid, dimBlock>>>(d_i, d_o, n);
            break;
        case 16:  reduce_kernel<T, traits, 16><<< dimGrid, dimBlock>>>(d_i, d_o, n);
            break;
        case  8:  reduce_kernel<T, traits, 8><<< dimGrid, dimBlock>>>(d_i, d_o, n);
            break;
        case  4:  reduce_kernel<T, traits, 4><<< dimGrid, dimBlock>>>(d_i, d_o, n);
            break;
        case  2:  reduce_kernel<T, traits, 2><<< dimGrid, dimBlock>>>(d_i, d_o, n);
            break;
        case  1:  reduce_kernel<T, traits, 1><<< dimGrid, dimBlock>>>(d_i, d_o, n);
            break;
    }
}

template void reduce<float, MOperator<float, MATH_MAX> >(uint n, int threads, int blocks, const float *d_i, float *d_o);
template void reduce<float, MOperator<float, MATH_MIN> >(uint n, int threads, int blocks, const float *d_i, float *d_o);
template void reduce<float, MOperator<float, MATH_ADD> >(uint n, int threads, int blocks, const float *d_i, float *d_o);

template void reduce<int, MOperator<int, MATH_MAX> >(uint n, int threads, int blocks, const int *d_i, int *d_o);
template void reduce<int, MOperator<int, MATH_MIN> >(uint n, int threads, int blocks, const int *d_i, int *d_o);
template void reduce<int, MOperator<int, MATH_ADD> >(uint n, int threads, int blocks, const int *d_i, int *d_o);

template void reduce<uint, MOperator<uint, MATH_MAX> >(uint n, int threads, int blocks, const uint *d_i, uint *d_o);
template void reduce<uint, MOperator<uint, MATH_MIN> >(uint n, int threads, int blocks, const uint *d_i, uint *d_o);
template void reduce<uint, MOperator<uint, MATH_ADD> >(uint n, int threads, int blocks, const uint *d_i, uint *d_o);


template <class T, class trait>
T cplReduce::cplSOPReduce(const T* d_i, int n){
    int blocks, threads;

    getNumBlocksAndThreads(n, MAX_NUMBER_REDUCE_BLOCKS, MAX_NUMBER_REDUCE_THREADS, blocks, threads);
    reduce<T, trait> (n, threads, blocks, d_i, (T*)d_temp);
    
    if    (m_zeroCopy)  cudaThreadSynchronize();
    else  cudaMemcpy(h_temp, d_temp, sizeof(T) * blocks, cudaMemcpyDeviceToHost);
    
    T s = trait::identity();
    for (int i=0; i< blocks; ++i){
        trait::iop(s, ((T*) h_temp)[i]);
    }
    return s;
}

// Instantiate
template<typename T>
T cplReduce::Max(const T* d_i, uint n) {
    return cplSOPReduce<T, MOperator<T, MATH_MAX> > (d_i, n);
}

template<typename T>
T cplReduce::Min(const T* d_i, uint n) {
    return cplSOPReduce<T, MOperator<T, MATH_MIN> > (d_i, n);
}

template<typename T>
T cplReduce::Sum(const T* d_i, uint n) {
    return cplSOPReduce<T, MOperator<T, MATH_ADD> > (d_i, n);
}

template float cplReduce::Max(const float* d_i, uint n);
template float cplReduce::Min(const float* d_i, uint n);
template float cplReduce::Sum(const float* d_i, uint n);
template uint cplReduce::Max(const uint* d_i, uint n);
template uint cplReduce::Min(const uint* d_i, uint n);
template uint cplReduce::Sum(const uint* d_i, uint n);
template int cplReduce::Max(const int* d_i, uint n);
template int cplReduce::Min(const int* d_i, uint n);
template int cplReduce::Sum(const int* d_i, uint n);


/**
 * @brief Perform cuda kernel parallel reductin 
 *        s = a1 + a2 + a3 + .... + an
 * @param[in]  T         Input data type (currently int, float)
 *             traits    Binary operation (+, max, min)
 *             traits1    Self data function (square, cude, sqrt, abs) 
 *             blockSize Size of block (related to optimize problem)
 *             g_idata   Input data
 *             n         Size of the input
 * @param[out] array of output redution perform for each block
 *
*/
// Correct version of reduce function
template <class T, class traits, class traits1,  uint blockSize>
__global__ void reduce_kernel(const T *g_idata, T *g_odata, uint n)
{
    volatile __shared__ T sdata[MAX_NUMBER_REDUCE_THREADS];
    
    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    uint tid = threadIdx.x;
    uint i   = blockIdx.x*(blockSize*2) + tid;
    uint gridSize = blockSize*2*gridDim.x;

    T mySum = traits::identity();
    // we reduce multiple elements per thread.  The number is determined by the 
    // number of active thread blocks (via gridSize).  More blocks will result
    // in a larger gridSize and therefore fewer elements per thread
    while (i + blockSize < n ) {
        traits::iop(mySum,traits::op(traits1::op(g_idata[i]),traits1::op(g_idata[i+blockSize])));
        i += gridSize;
    }

    if ( i < n)
        traits::iop(mySum, traits1::op(g_idata[i]));
    sdata[tid] = mySum;
    
    __syncthreads();

    reduce_shared<T, traits, blockSize>(mySum, sdata, tid);
    // write result for this block to global mem 
    if (tid == 0)
        g_odata[blockIdx.x] = mySum;
}

template <class T, class traits, class traits1>
void reduce(uint n, int threads, int blocks, const T *d_i, T *d_o){
    dim3 dimBlock(threads, 1, 1);
    dim3 dimGrid(blocks, 1, 1);
    int smem = threads * sizeof(T);
    switch (threads)
    {
        case 512:  reduce_kernel<T, traits, traits1, 512><<< dimGrid, dimBlock, smem>>>(d_i, d_o, n);
            break;
        case 256:  reduce_kernel<T, traits, traits1, 256><<< dimGrid, dimBlock, smem>>>(d_i, d_o, n);
            break;
        case 128:  reduce_kernel<T, traits, traits1, 128><<< dimGrid, dimBlock, smem>>>(d_i, d_o, n);
            break;
        case 64:   reduce_kernel<T, traits, traits1, 64><<< dimGrid, dimBlock, smem>>>(d_i, d_o, n);
            break;
        case 32:   reduce_kernel<T, traits, traits1, 32><<< dimGrid, dimBlock, smem>>>(d_i, d_o, n);
            break;
        case 16:   reduce_kernel<T, traits, traits1, 16><<< dimGrid, dimBlock, smem>>>(d_i, d_o, n);
            break;
        case  8:   reduce_kernel<T, traits, traits1, 8><<< dimGrid, dimBlock, smem>>>(d_i, d_o, n);
            break;
        case  4:   reduce_kernel<T, traits, traits1, 4><<< dimGrid, dimBlock, smem>>>(d_i, d_o, n);
            break;
        case  2:   reduce_kernel<T, traits, traits1, 2><<< dimGrid, dimBlock, smem>>>(d_i, d_o, n);
            break;
        case  1:   reduce_kernel<T, traits, traits1, 1><<< dimGrid, dimBlock, smem>>>(d_i, d_o, n);
            break;
    }
}

template void reduce<float, MOperator<float, MATH_MAX>, MOperator<float, MATH_ABS> >(uint n, int threads, int blocks, const float *d_i, float *d_o);
template void reduce<int, MOperator<int, MATH_MAX>, MOperator<int, MATH_ABS> >(uint n, int threads, int blocks, const int *d_i, int *d_o);
template void reduce<float, MOperator<float, MATH_ADD>, MOperator<float, MATH_SQR> >(uint n, int threads, int blocks, const float *d_i,float *d_o);
template void reduce<int, MOperator<int, MATH_ADD>, MOperator<int, MATH_SQR> >(uint n, int threads, int blocks, const int *d_i, int *d_o);

template <class T, class trait, class trait1>
T cplReduce::cplBOPReduce(const T* d_i, int n){
    int blocks, threads;

    getNumBlocksAndThreads(n, MAX_NUMBER_REDUCE_BLOCKS, MAX_NUMBER_REDUCE_THREADS, blocks, threads);
    reduce<T, trait, trait1 > (n, threads, blocks, d_i, (T*) d_temp);

    if (m_zeroCopy)  cudaThreadSynchronize();
    else             cudaMemcpy(h_temp, d_temp, sizeof(T) * blocks, cudaMemcpyDeviceToHost);

#if 1
    return accumulate<T, trait>((T*)h_temp, blocks);
#else    
    T s = trait::identity();
    for (int i=0; i< blocks; ++i){
        trait::iop(s, ((T*) h_temp)[i]);
    }
    return s;
#endif
}
template<typename T>
T cplReduce::MaxAbs(const T* d_i, uint n) {
    return cplBOPReduce<T, MOperator<T, MATH_MAX>, MOperator<T, MATH_ABS> > (d_i, n);
}

template<typename T>
T cplReduce::Sum2(const T* d_i, uint n) {
    return cplBOPReduce<T, MOperator<T, MATH_ADD>, MOperator<T, MATH_SQR> > (d_i, n);
}

template<typename T>
T cplReduce::SumAbs(const T* d_i, uint n) {
    return cplBOPReduce<T, MOperator<T, MATH_ADD>, MOperator<T, MATH_ABS> > (d_i, n);
}

template float cplReduce::MaxAbs(const float* d_i, uint n) ;
template float cplReduce::Sum2(const float* d_i, uint n);
template float cplReduce::SumAbs(const float* d_i, uint n);

template int cplReduce::MaxAbs(const int* d_i, uint n) ;
template int cplReduce::Sum2(const int* d_i, uint n);
template int cplReduce::SumAbs(const int* d_i, uint n);

template uint cplReduce::Sum2(const uint* d_i, uint n);

////////////////////////////////////////////////////////////////////////////////
//
////////////////////////////////////////////////////////////////////////////////
template <class T, class oper, class oper1, uint blockSize>
__global__ void biReduce_kernel(T *g_o, T *g_o1, const T *g_idata, uint n)
{
    volatile __shared__ T s0[MAX_NUMBER_REDUCE_THREADS];
    volatile __shared__ T s1[MAX_NUMBER_REDUCE_THREADS];
        
    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    uint tid      = threadIdx.x;
    uint i        = blockIdx.x*(blockSize*2) + tid;
    uint gridSize = blockSize * 2 * gridDim.x;

    T  mySum  = oper::identity();
    T  mySum1 = oper1::identity();

    while (i + blockSize < n )
    {
        oper::iop(mySum, oper::op(g_idata[i],g_idata[i+blockSize]));
        oper1::iop(mySum1 ,oper1::op(g_idata[i],g_idata[i+blockSize]));
        i += gridSize;
    }

    if ( i < n){
        oper::iop(mySum, g_idata[i]);
        oper1::iop(mySum1, g_idata[i]);
    }

    s0[tid] = mySum;
    s1[tid] = mySum1;
    
    __syncthreads();

    // do reduction in shared mem
    if (blockSize >= 512) {
        if (tid < 256) {
            s0[tid] = mySum = oper::op(mySum, s0[tid + 256]);
            s1[tid] = mySum1 = oper1::op(mySum1, s1[tid + 256]);
        }
        __syncthreads();
    }

    if (blockSize >= 256) {
        if (tid < 128) {
            s0[tid] = mySum = oper::op(mySum, s0[tid + 128]);
            s1[tid] = mySum1 = oper1::op(mySum1, s1[tid + 128]);
        }
        __syncthreads();
    }

    if (blockSize >= 128) {
        if (tid < 64) {
            s0[tid] = mySum = oper::op(mySum, s0[tid + 64]);
            s1[tid] = mySum1 = oper1::op(mySum1, s1[tid + 64]);
        }
        __syncthreads();
    }

    if (tid < 32)
    {
        if (blockSize >=  64) {
            s0[tid] = mySum = oper::op(mySum, s0[tid + 32]);
            s1[tid] = mySum1 = oper1::op(mySum1, s1[tid + 32]);
        }

        if (blockSize >=  32) {
            s0[tid] = mySum = oper::op(mySum, s0[tid + 16]);
            s1[tid] = mySum1 = oper1::op(mySum1, s1[tid + 16]);
        }

        if (blockSize >=  16) {
            s0[tid] = mySum = oper::op(mySum, s0[tid + 8]);
            s1[tid] = mySum1 = oper1::op(mySum1, s1[tid + 8]);
        }

        if (blockSize >=  8) {
            s0[tid] = mySum = oper::op(mySum, s0[tid + 4]);
            s1[tid] = mySum1 = oper1::op(mySum1, s1[tid + 4]);
        }

        if (blockSize >=  4) {
            s0[tid] = mySum = oper::op(mySum, s0[tid + 2]);
            s1[tid] = mySum1 = oper1::op(mySum1, s1[tid + 2]);
        }

        if (blockSize >=  2) {
            s0[tid] = mySum = oper::op(mySum, s0[tid + 1]);
            s1[tid] = mySum1 = oper1::op(mySum1, s1[tid + 1]);
        }
    }
    // write result for this block to global mem 
    if (tid == 0){
        g_o[blockIdx.x] = mySum;
        g_o1[blockIdx.x] = mySum1;
    }
}

template <class T, class oper, class oper1>
void cplReduce::biReduce(T& rd0, T& rd1, const T*g_idata, uint n)
{
    const uint blockSize = MAX_NUMBER_REDUCE_THREADS;

    dim3 threads(blockSize);
    uint nBlocks = min(iDivUp(n, 2 * blockSize),  MAX_NUMBER_REDUCE_BLOCKS);
    dim3 grids(nBlocks);

    T* d_rd0 = (T*) d_temp;
    T* d_rd1 = d_rd0 + MAX_NUMBER_REDUCE_BLOCKS;

    T* h_rd0 = (T*) h_temp;
    T* h_rd1 = h_rd0 + MAX_NUMBER_REDUCE_BLOCKS;
    biReduce_kernel<T, oper, oper1, blockSize><<<grids, threads>>>(d_rd0, d_rd1, g_idata, n);
    
    if (m_zeroCopy) cudaThreadSynchronize();
    else
        cudaMemcpy(h_rd0, d_rd0, sizeof(T) * (nBlocks + MAX_NUMBER_REDUCE_BLOCKS), cudaMemcpyDeviceToHost);
    
    rd0 = oper::identity();
    rd1 = oper1::identity();

    for (uint i=0; i< nBlocks; ++i){
        oper::iop(rd0, h_rd0[i]);
        oper1::iop(rd1, h_rd1[i]);
    }
    cutilCheckMsg("cplReduce::biReduce");
}

template<class T>
void  cplReduce::MaxMin(T &maxV, T &minV, const T* d_data, uint n){
    biReduce<T, MOperator<T, MATH_MAX>, MOperator<T, MATH_MIN> >(maxV, minV, d_data, n);
}

template void  cplReduce::MaxMin(float &maxV, float &minV, const float* d_data, uint n);
template void  cplReduce::MaxMin(int &maxV, int &minV, const int* d_data, uint n);
template void  cplReduce::MaxMin(uint &maxV, uint &minV, const uint* d_data, uint n);

template<class T>
void  cplReduce::MaxSum(T &maxV, T &minV,const T* d_data, uint n){
    biReduce<T, MOperator<T, MATH_MAX>, MOperator<T, MATH_ADD> >(maxV, minV, d_data, n);
}

template void  cplReduce::MaxSum(float &maxV, float &sumV, const float* d_data, uint n);
template void  cplReduce::MaxSum(int &maxV, int &sumV, const int* d_data, uint n);
template void  cplReduce::MaxSum(uint &maxV, uint &sumV, const uint* d_data, uint n);

////////////////////////////////////////////////////////////////////////////////
//
////////////////////////////////////////////////////////////////////////////////
template <class T, class traits, class traits1, uint blockSize>
__global__ void reduceProduct_kernel(T *g_odata, const T*g_idata, const T*g_idata1, uint n)
{
    volatile __shared__ T sdata[MAX_NUMBER_REDUCE_THREADS];
        
    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    uint blockId  = blockIdx.x;
    uint tid      = threadIdx.x;
    uint i        = blockId * (blockSize * 2) + tid;
    uint gridSize = (blockSize * 2) * gridDim.x;

    T mySum = traits::identity();
    
    // we reduce multiple elements per thread.  The number is determined by the 
    // number of active thread blocks (via gridSize).  More blocks will result
    // in a larger gridSize and therefore fewer elements per thread
    while (i + blockSize < n ) {
        T t1 = traits1::op(g_idata[i], g_idata1[i]);
        T t2 = traits1::op(g_idata[i + blockSize], g_idata1[i + blockSize]);
        traits::iop(mySum,traits::op(t1, t2));
        i += gridSize;
    }

    if ( i < n) {
        T t1 = traits1::op(g_idata[i], g_idata1[i]);
        traits::iop(mySum,t1);
    }

    sdata[tid] = mySum;
    
    __syncthreads();

    // do reduction in shared mem
    reduce_shared<T, traits, blockSize>(mySum, sdata, tid);
    
    // write result for this block to global mem 
    if (tid == 0)
        g_odata[blockIdx.x] = sdata[0];
}

template <class T, class traits, class traits1>
T cplReduce::reduceProduct(const T*g_idata, const T*g_idata1, uint n)
{
    const uint blockSize = MAX_NUMBER_REDUCE_THREADS;
    dim3 threads(blockSize);
    uint nBlocks = min(iDivUp(n, 2 * blockSize),  MAX_NUMBER_REDUCE_BLOCKS);

    dim3 grids(nBlocks);
    reduceProduct_kernel<T, traits, traits1, blockSize><<<grids, threads>>>((T*)d_temp, g_idata, g_idata1, n);

    if (m_zeroCopy)
        cudaThreadSynchronize();
    else 
        cudaMemcpy(h_temp, d_temp, sizeof(T) * nBlocks, cudaMemcpyDeviceToHost);
    
    
    T sum = traits::identity();
    for (uint i=0; i< nBlocks; ++i){
        traits::iop(sum, ((T*) h_temp)[i]);
    }
    cutilCheckMsg("cplReduce::reduceProduct");
    return sum;
}

template<typename T>
T cplReduce::Dot(const T* d_i, const T* d_i1, uint n){
    return reduceProduct<T, MOperator<T, MATH_ADD>, MOperator<T, MATH_MUL> > (d_i, d_i1, n);
}

template<typename T>
T cplReduce::SumAdd(const T* d_i, const T* d_i1, uint n){
    return reduceProduct<T, MOperator<T, MATH_ADD>, MOperator<T, MATH_ADD> > (d_i, d_i1, n);
}

template<typename T>
T cplReduce::MaxAdd(const T* d_i, const T* d_i1, uint n){
    return reduceProduct<T, MOperator<T, MATH_MAX>, MOperator<T, MATH_ADD> > (d_i, d_i1, n);
}

template<typename T>
T cplReduce::MinAdd(const T* d_i, const T* d_i1, uint n){
    return reduceProduct<T, MOperator<T, MATH_MIN>, MOperator<T, MATH_ADD> > (d_i, d_i1, n);
}

template float cplReduce::Dot(const float* d_i, const float* d_i1, uint n);
template float cplReduce::SumAdd(const float* d_i, const float* d_i1, uint n);
template float cplReduce::MaxAdd(const float* d_i, const float* d_i1, uint n);
template float cplReduce::MinAdd(const float* d_i, const float* d_i1, uint n);

template uint cplReduce::Dot(const uint* d_i, const uint* d_i1, uint n);
template uint cplReduce::SumAdd(const uint* d_i, const uint* d_i1, uint n);
template uint cplReduce::MaxAdd(const uint* d_i, const uint* d_i1, uint n);
template uint cplReduce::MinAdd(const uint* d_i, const uint* d_i1, uint n);

template int cplReduce::Dot(const int* d_i, const int* d_i1, uint n);
template int cplReduce::SumAdd(const int* d_i, const int* d_i1, uint n);
template int cplReduce::MaxAdd(const int* d_i, const int* d_i1, uint n);
template int cplReduce::MinAdd(const int* d_i, const int* d_i1, uint n);

bool cplReduce::selfTest(int n){
    int test = true;
    int* h_i = new int [n];
    int* h_i1 = new int [n];

    for (int j=0; j< n; ++j) h_i[j] = (rand() & 0x03);
    for (int j=0; j< n; ++j) h_i1[j] = (rand() & 0x03);
    
    int *d_i;
    dmemAlloc(d_i, n);
    copyArrayToDevice(d_i, h_i, n);
    
    int *d_i1;
    dmemAlloc(d_i1,n);
    copyArrayToDevice(d_i1,h_i1,n);

    int h_max = -INT_MAX, h_min = INT_MAX;
    int h_maxAbs = 0;
    int h_sumSQR = 0;
    int h_sum = 0;
    int h_dot = 0;
    
    for (int i=0; i< n; ++i)
    {
        h_max = max(h_max, h_i[i]);
        h_maxAbs = max(h_maxAbs, h_i[i]);
        h_min = min(h_min, h_i[i]);
        h_sum += h_i[i];
        h_sumSQR += h_i[i]*h_i[i];
        h_dot += h_i1[i] * h_i[i];
    }

    int d_sum = Sum(d_i, n);
    int d_max = Max(d_i, n);
    int d_min = Min(d_i, n);
    
    int d_maxAbs = MaxAbs(d_i, n);
    int d_sumSQR = Sum2(d_i, n);
    int d_dot = Dot(d_i,d_i1, n);

    fprintf(stderr, "Maximum value from CPU %d from GPU %d\n",h_max, d_max);
    fprintf(stderr, "Minumum value from CPU %d from GPU %d\n",h_min, d_min);
    fprintf(stderr, "Total value from CPU %d from GPU %d\n",h_sum, d_sum);
    
    fprintf(stderr, "Maximum abosulte value from CPU %d from GPU %d\n",h_maxAbs, d_maxAbs);
    fprintf(stderr, "Total square value from CPU %d from GPU %d\n",h_sumSQR, d_sumSQR);
    fprintf(stderr, "Dot product value from CPU %d from GPU %d\n",h_dot, d_dot);

    MaxMin(d_max, d_min, d_i, n);
    fprintf(stderr, "Max min value from GPU %d %d\n",d_max, d_min);

    //Extensive test
    h_max = -INT_MAX, h_min = INT_MAX;
    h_maxAbs = 0;
    h_sumSQR = 0;
    h_sum = 0;
    h_dot = 0;
    
    for (int l=1; l < min(n, 10001);++l){
        h_max = max(h_max, h_i[l-1]);
        h_maxAbs = max(h_maxAbs, h_i[l-1]);
        h_min = min(h_min, h_i[l-1]);
        h_sum += h_i[l-1];
        h_sumSQR += h_i[l-1]*h_i[l-1];
        h_dot += h_i1[l-1] * h_i[l-1];

        int d_max = Max(d_i, l);
        int d_min = Min(d_i, l);
        int d_sum = Sum(d_i, l);
        int d_maxAbs = MaxAbs(d_i, l);
        int d_sumSQR = Sum2(d_i, l);
        int d_dot    = Dot(d_i, d_i1, l);
        
        if (d_max != h_max){
            fprintf(stderr, "Max Test FAILED at %d GPU %d CPU %d\n", l, d_max, h_max );
            test = false;
        }

        if (d_min != h_min){
            fprintf(stderr, "Min Test FAILED at %d GPU %d CPU %d\n", l, d_min, h_min );
            test = false;
        }

        if (d_maxAbs!= h_maxAbs){
            fprintf(stderr, "MaxAbs Test FAILED at %d GPU %d CPU %d\n", l, d_maxAbs, h_maxAbs );
            test = false;
        }

        if (d_sum!= h_sum){
            fprintf(stderr, "Sum Test FAILED at %d GPU %d CPU %d\n", l, d_sum, h_sum );
            test = false;
        }

        if (d_sumSQR!= h_sumSQR){
            fprintf(stderr, "Sum SQR Test FAILED at %d GPU %d CPU %d\n", l, d_sumSQR, h_sumSQR );
            test = false;
        }

        if (d_dot!= h_dot){
            fprintf(stderr, "Dot Test FAILED at %d GPU %d CPU %d\n", l, d_dot, h_dot );
            test = false;
        }

        MaxMin(d_max, d_min, d_i, l);
        if (d_min != h_min || d_max != h_max){
            fprintf(stderr, "MaxMin Test FAILED at %d GPU %d %d CPU %d %d\n", l, d_max, d_min, h_max, h_min );
            test = false;
        }

        if (test == false)
            break;
    }

    if (test)
        fprintf(stderr, "Test PASSED  \n");
    
    delete []h_i1;
    delete []h_i;
    cudaFree(d_i);
    cudaFree(d_i1);

    return test;
}

template<class T>
T reduceSumCPU(const T *data, int size)
{
    T sum = data[0];
    T c = (T)0.0;              
    for (int i = 1; i < size; i++)
    {
        T y = data[i] - c;  
        T t = sum + y;      
        c = (t - sum) - y;  
        sum = t;            
    }
    return sum;
}

void cplReduce::timeTest(int n){
    int* h_i = new int [n];
    for (int j=0; j< n; ++j)
        h_i[j] = rand() & 0xFF;
    
    int *d_i;
    dmemAlloc(d_i, n);
    copyArrayToDevice(d_i, h_i, n);

    int h_sum = reduceSumCPU(h_i, n);
    
    uint timer = 0;
    
    cutilCheckError(cutCreateTimer( &timer));
    int nIters = 100;
    int d_sum = 0;
    cudaThreadSynchronize();
    cutStartTimer(timer);
    for (int i=0; i < nIters; ++i)
        d_sum = Sum(d_i, n);
    cudaThreadSynchronize();
    cutStopTimer(timer);
    float reduceTime = cutGetTimerValue(timer) / nIters;
    fprintf(stderr, "Result CPU %d GPU %d \n ",h_sum, d_sum);
    fprintf(stderr, "Average time: %f ms\n", reduceTime);
    fprintf(stderr, "Bandwidth:    %f GB/s\n\n", (n * sizeof(int)) / (reduceTime * 1.0e6));
    cutDeleteTimer(timer);
    delete []h_i;
    dmemFree(d_i);

    cutilCheckMsg("Free memory");
}

#include <cudaVector3DArray.h>

// Operation on the Vector3DArray
Vector3Df MaxAbs(cplReduce& rdPlan, const cplVector3DArray& d_i, uint n){
    Vector3Df maxV;
    maxV.x = rdPlan.MaxAbs(d_i.x, n);
    maxV.y = rdPlan.MaxAbs(d_i.y, n);
    maxV.z = rdPlan.MaxAbs(d_i.z, n);
    return maxV;
}


float MaxAbsAll(cplReduce& rdPlan, const cplVector3DArray& d_i, uint n){
    if ((n == d_i.n) && (d_i.isContinous())){
        return rdPlan.MaxAbs(d_i.x, 3 * n);
    }
    else {
        float max_x = rdPlan.MaxAbs(d_i.x, n);
        float max_y = rdPlan.MaxAbs(d_i.y, n);
        float max_z = rdPlan.MaxAbs(d_i.z, n);
        return max(max(max_x, max_y),max_z);
    }
}

void  MaxMin(cplReduce& rd, Vector3Df& maxV, Vector3Df& minV,
                 const cplVector3DArray& d_i, uint n)
{
    rd.MaxMin(maxV.x, minV.x, d_i.x, n);
    rd.MaxMin(maxV.y, minV.y, d_i.y, n);
    rd.MaxMin(maxV.z, minV.z, d_i.z, n);
}

Vector3Df Sum(cplReduce& rd, const cplVector3DArray& d_i, uint n)
{
    Vector3Df sum;
    sum.x =  rd.Sum(d_i.x, n);
    sum.y =  rd.Sum(d_i.y, n);
    sum.z =  rd.Sum(d_i.z, n);
    return sum;
}

Vector3Df Sum2(cplReduce& rd, const cplVector3DArray& d_i, uint n)
{
    Vector3Df sum;
    sum.x =  rd.Sum2(d_i.x, n);
    sum.y =  rd.Sum2(d_i.y, n);
    sum.z =  rd.Sum2(d_i.z, n);
    return sum;
}

