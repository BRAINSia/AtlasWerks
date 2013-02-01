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

#include <cudaReduceStream.h>
#include "cutil_inline.h"
#include <cpl.h>

#if __CUDA_ARCH__ != 200
#define NUMBER_OF_CORES               (240 * 16)
#else
#define NUMBER_OF_CORES               (240)
#endif

#define OPTIMAL_THREAD_SIZE           64
#define MAX_NUMBER_OF_REDUCE_STREAM   8
#define M    NUMBER_OF_CORES          // Number of core 
#define N    OPTIMAL_THREAD_SIZE      // Best number of thread per block 

template<typename T, class opers>
__global__ void reduce_level2_kernel(T *data) {
    __shared__ T shm[N];
    uint idx=threadIdx.x;
    T s=opers::identity();
#if M%(2*N)==0
    for(uint j=idx; j<M; j+=N*2)
        opers::iop(s, opers::op(data[j], data[j + N]));
#else
    for(uint j=idx; j<M; j+=N)
        opers::iop(s, data[j]);
#endif
    shm[threadIdx.x]=s;
    __syncthreads();
    if(idx==0) {
        T s=opers::identity();
        for(uint i=0; i<N; i++)
            opers::iop(s, shm[i]);
        data[0]=s;
    }
}

template<typename T, class opers, bool accumulate>
__global__ void reduce_level2_kernel(T* d_o, T *data) {
    __shared__ T shm[N];
    uint idx=threadIdx.x;
    T s=opers::identity();
#if M%(2*N)==0
    for(uint j=idx; j<M; j+=N*2)
        opers::iop(s, opers::op(data[j], data[j + N]));
#else
    for(uint j=idx; j<M; j+=N)
        opers::iop(s, data[j]);
#endif
    shm[threadIdx.x]=s;
    __syncthreads();
    if(idx==0) {
        T s=opers::identity();
        for(uint i=0; i<N; i++)
            opers::iop(s, shm[i]);

        d_o[0]=(accumulate) ? opers::op(d_o[0], s) : s;
    }
}

template<typename T, class opers>
__global__ void reduce_level1_kernel(T *data, T *res, int size) {
    __shared__ T shm[N];
    int idx=blockIdx.x*blockDim.x+threadIdx.x;

    T s=opers::identity();
    int j=idx;
    for(; j + M * N <size; j+=M*N*2)
        opers::iop(s, opers::op(data[j], data[j + M*N]));
    shm[threadIdx.x]= (j < size) ? opers::op(s, data[j]) : s;
    __syncthreads();
    if(threadIdx.x==0) {
        T s=opers::identity();
        for(int i=0; i<N; i++)
            opers::iop(s, shm[i]);
        res[blockIdx.x]=s;
    }
}

template<typename T, class opers, class opers1>
__global__ void compReduce_level1_kernel(T *data, T *res, int size) {
    __shared__ T shm[N];
    int idx=blockIdx.x*blockDim.x+threadIdx.x;

    T s=opers::identity();
    int j=idx;
    for(; j + M * N <size; j+=M*N*2)
        opers::iop(s, opers::op(opers1::op(data[j]), opers1::op(data[j + M*N])));
    shm[threadIdx.x]= (j < size) ? opers::op(s, opers1::op(data[j])) : s;
    __syncthreads();
    if(threadIdx.x==0) {
        T s=opers::identity();
        for(int i=0; i<N; i++)
            opers::iop(s, shm[i]);
        res[blockIdx.x]=s;
    }
}

template<typename T, class opers, class opers1>
__global__ void product_level1_kernel(T *d_i0, T *d_i1, T *res, int size) {
    __shared__ T shm[N];
    int idx=blockIdx.x*blockDim.x+threadIdx.x;
    T s=opers::identity();
    int j=idx;
    for(; j + M * N <size; j+=M*N*2)
        opers::iop(s, opers::op(opers1::op(d_i0[j],d_i1[j]), opers1::op(d_i0[j + M*N], d_i1[j + M*N])));
    shm[threadIdx.x]= (j < size) ? opers::op(s, opers1::op(d_i0[j],d_i1[j])) : s;
    __syncthreads();
    if(threadIdx.x==0) {
        T s=opers::identity();
        for(int i=0; i<N; i++)
            opers::iop(s, shm[i]);
        res[blockIdx.x]=s;
    }
}

template<typename T, class opers, class opers1>
__global__ void bireduce_level1_kernel(T *data, T *res, int size) {
    __shared__ T shm[N];
    __shared__ T shm1[N];
    
    int idx=blockIdx.x*blockDim.x+threadIdx.x;

    T s =opers::identity();
    T s1=opers1::identity();
    
    int j=idx;
    for(; j + M * N <size; j+=M*N*2) {
        opers::iop(s, opers::op(data[j], data[j + M*N]));
        opers1::iop(s1, opers1::op(data[j], data[j + M*N]));
    }

    shm[threadIdx.x]  = (j < size) ? opers::op(s, data[j]) : s;
    shm1[threadIdx.x] = (j < size) ? opers1::op(s1, data[j]) : s1;
    
    __syncthreads();
#if 1
    if(threadIdx.x==0) {
        T s =opers::identity();
        T s1=opers1::identity();
        
        for(int i=0; i<N; i++){
            opers::iop(s, shm[i]);
            opers1::iop(s1, shm1[i]);
        }
        res[blockIdx.x]=s;
        res[blockIdx.x + M * N]=s1;
    }
#else
    if(threadIdx.x==0) {
        T s =opers::identity();
        for(int i=0; i<N; i++)
            opers::iop(s, shm[i]);
        res[blockIdx.x]=s;
    }
    if(threadIdx.x==1) {
        T s1 =opers1::identity();
        for(int i=0; i<N; i++)
            opers1::iop(s1, shm1[i]);
        res[blockIdx.x + M * N]=s1;
    }
#endif
}

template<typename T, class opers, class opers1>
__global__ void bireduce_level2_kernel(T *data) {
    T* data1 = data + M * N;
    __shared__ T shm[N];
    __shared__ T shm1[N];
    uint idx=threadIdx.x;
    T s =opers::identity();
    T s1=opers1::identity();
#if M%(2*N)==0
    for(uint j=idx; j<M; j+=N*2){
        opers::iop(s, opers::op(data[j], data[j + N]));
        opers1::iop(s1, opers1::op(data1[j], data1[j + N]));
    }
#else
    for(uint j=idx; j<M; j+=N){
        opers::iop(s, data[j]);
        opers1::iop(s1, data1[j]);
    }
#endif
    shm[threadIdx.x]=s;
    shm1[threadIdx.x]=s1;
    __syncthreads();
    if(idx==0) {
        T s=opers::identity();
        T s1=opers1::identity();
        for(uint i=0; i<N; i++){
            opers::iop(s, shm[i]);
            opers1::iop(s1, shm1[i]);
        }
        data[0] =s;
        data[1] =s1;
    }
}

template<typename T, class opers, class opers1, bool accumulate>
__global__ void bireduce_level2_o_kernel(T* d_o, T *data) {
    T* data1 = data + M * N;
    __shared__ T shm[N];
    __shared__ T shm1[N];
    uint idx=threadIdx.x;
    T s =opers::identity();
    T s1=opers1::identity();
#if M%(2*N)==0
    for(uint j=idx; j<M; j+=N*2){
        opers::iop(s, opers::op(data[j], data[j + N]));
        opers1::iop(s1, opers1::op(data1[j], data1[j + N]));
    }
#else
    for(uint j=idx; j<M; j+=N){
        opers::iop(s, data[j]);
        opers1::iop(s1, data1[j]);
    }
#endif
    shm[threadIdx.x]=s;
    shm1[threadIdx.x]=s1;
    __syncthreads();
    if(idx==0) {
        T s=opers::identity();
        T s1=opers1::identity();
        for(uint i=0; i<N; i++){
            opers::iop(s, shm[i]);
            opers1::iop(s1, shm1[i]);
        }
        if (accumulate) {
            opers::iop(d_o[0], s);
            opers1::iop(d_o[1], s1);
        } else {
            d_o[0] =s;
            d_o[1] =s1;
        }
    }
}

template<typename T, class opers, class opers1>
__global__ void bireduce_level1_kernel(T *data, T *res, T* res1, int size) {
    __shared__ T shm[N];
    __shared__ T shm1[N];
    
    int idx=blockIdx.x*blockDim.x+threadIdx.x;

    T s =opers::identity();
    T s1=opers1::identity();
    
    int j=idx;
    for(; j + M * N <size; j+=M*N*2) {
        opers::iop(s, opers::op(data[j], data[j + M*N]));
        opers1::iop(s1, opers1::op(data[j], data[j + M*N]));
    }

    shm[threadIdx.x]  = (j < size) ? opers::op(s, data[j]) : s;
    shm1[threadIdx.x] = (j < size) ? opers1::op(s1, data[j]) : s1;
    
    __syncthreads();
    if(threadIdx.x==0) {
        T s =opers::identity();
        T s1=opers1::identity();
        
        for(int i=0; i<N; i++){
            opers::iop(s, shm[i]);
            opers1::iop(s1, shm1[i]);
        }
        res[blockIdx.x]=s;
        res1[blockIdx.x]=s1;
    }
}

template<typename T, class opers, class opers1>
__global__ void bireduce_level2_kernel(T *data, T* data1) {
    __shared__ T shm[N];
    __shared__ T shm1[N];
    uint idx=threadIdx.x;
    T s =opers::identity();
    T s1=opers1::identity();
#if M%(2*N)==0
    for(uint j=idx; j<M; j+=N*2){
        opers::iop(s, opers::op(data[j], data[j + N]));
        opers1::iop(s1, opers1::op(data1[j], data1[j + N]));
    }
#else
    for(uint j=idx; j<M; j+=N){
        opers::iop(s, data[j]);
        opers1::iop(s1, data1[j]);
    }
#endif
    shm[threadIdx.x]=s;
    __syncthreads();
    if(idx==0) {
        T s=opers::identity();
        T s1=opers1::identity();
        for(uint i=0; i<N; i++){
            opers::iop(s, shm[i]);
            opers1::iop(s1, shm1[i]);
        }
        data[0] =s;
        data[1]=s1;
    }
}


template <typename T, class oper, class oper1>
void cplReduceS::Bireduce(T* d_i, unsigned int n, cudaStream_t stream)
{
    T* d_buf  = (T*) d_rdBuf;
#if 0
    T* d_buf1 = d_buf + M * N;
    bireduce_level1_kernel<T, oper, oper1><<<M, N, 0, stream>>>(d_i, d_buf, d_buf1, n);
    bireduce_level2_kernel<T, oper, oper1><<<1, N, 0, stream>>>(d_buf, d_buf1);
#else
    bireduce_level1_kernel<T, oper, oper1><<<M, N, 0, stream>>>(d_i, d_buf, n);
    bireduce_level2_kernel<T, oper, oper1><<<1, N, 0, stream>>>(d_buf);
#endif
}

template<typename T>
void cplReduceS::MaxMin(T* data, unsigned int n, cudaStream_t stream)
{
    Bireduce<T, MOperator<T, MATH_MAX>, MOperator<T, MATH_MIN> >(data, n, stream);
}

template void cplReduceS::MaxMin(float* data, unsigned int n, cudaStream_t stream);
template void cplReduceS::MaxMin(int* data, unsigned int n, cudaStream_t stream);
template void cplReduceS::MaxMin(uint* data, unsigned int n, cudaStream_t stream);

template<typename T>
void cplReduceS::MaxSum(T* data, unsigned int n, cudaStream_t stream)
{
    Bireduce<T, MOperator<T, MATH_MAX>, MOperator<T, MATH_ADD> >(data, n, stream);
}
template void cplReduceS::MaxSum(float* data, unsigned int n, cudaStream_t stream);
template void cplReduceS::MaxSum(int* data, unsigned int n, cudaStream_t stream);
template void cplReduceS::MaxSum(uint* data, unsigned int n, cudaStream_t stream);

////////////////////////////////////////////////////////////////////////////////

template <typename T, class oper, class oper1, bool accumulate>
void cplReduceS::Bireduce(T* d_o, T* d_i, unsigned int n, cudaStream_t stream)
{
    T* d_buf  = (T*) d_rdBuf;
    bireduce_level1_kernel<T, oper, oper1><<<M, N, 0, stream>>>(d_i, d_buf, n);
    bireduce_level2_o_kernel<T, oper, oper1, accumulate><<<1, N, 0, stream>>>(d_o, d_buf);
}

template<typename T>
void cplReduceS::MaxMin(T* d_o, T* data, unsigned int n, cudaStream_t stream)
{
    Bireduce<T, MOperator<T, MATH_MAX>, MOperator<T, MATH_MIN>, false >(d_o, data, n, stream);
}

template void cplReduceS::MaxMin(float* d_o, float* data, unsigned int n, cudaStream_t stream);
template void cplReduceS::MaxMin(int* d_o, int* data, unsigned int n, cudaStream_t stream);
template void cplReduceS::MaxMin(uint* d_o, uint* data, unsigned int n, cudaStream_t stream);

template<typename T>
void cplReduceS::MaxSum(T* d_o, T* data, unsigned int n, cudaStream_t stream)
{
    Bireduce<T, MOperator<T, MATH_MAX>, MOperator<T, MATH_ADD>, false >(d_o, data, n, stream);
}
template void cplReduceS::MaxSum(float* d_o, float* data, unsigned int n, cudaStream_t stream);
template void cplReduceS::MaxSum(int* d_o, int* data, unsigned int n, cudaStream_t stream);
template void cplReduceS::MaxSum(uint* d_o, uint* data, unsigned int n, cudaStream_t stream);

template<typename T>
void cplReduceS::MaxMinA(T* d_o, T* data, unsigned int n, cudaStream_t stream)
{
    Bireduce<T, MOperator<T, MATH_MAX>, MOperator<T, MATH_MIN>, true >(d_o, data, n, stream);
}

template void cplReduceS::MaxMinA(float* d_o, float* data, unsigned int n, cudaStream_t stream);
template void cplReduceS::MaxMinA(int* d_o, int* data, unsigned int n, cudaStream_t stream);
template void cplReduceS::MaxMinA(uint* d_o, uint* data, unsigned int n, cudaStream_t stream);

template<typename T>
void cplReduceS::MaxSumA(T* d_o, T* data, unsigned int n, cudaStream_t stream)
{
    Bireduce<T, MOperator<T, MATH_MAX>, MOperator<T, MATH_ADD>, true >(d_o, data, n, stream);
}
template void cplReduceS::MaxSumA(float* d_o, float* data, unsigned int n, cudaStream_t stream);
template void cplReduceS::MaxSumA(int* d_o, int* data, unsigned int n, cudaStream_t stream);
template void cplReduceS::MaxSumA(uint* d_o, uint* data, unsigned int n, cudaStream_t stream);

////////////////////////////////////////////////////////////////////////////////

void cplReduceS::init(){
    int *d_buf;
    dmemAlloc(d_buf, M * N * MAX_NUMBER_OF_REDUCE_STREAM);
    d_rdBuf = d_buf;
}

void cplReduceS::clean()
{
    if (d_rdBuf != NULL)
        dmemFree(d_rdBuf);
    d_rdBuf = NULL;
}

template <typename T, class opers>
void cplReduceS::Reduce(T* pSrc, int n, cudaStream_t stream)
{
    reduce_level1_kernel<T, opers><<<M, N, 0, stream>>>(pSrc, (T*)d_rdBuf, n);
    reduce_level2_kernel<T, opers><<<1, N, 0, stream>>>((T*)d_rdBuf);
}


template<typename T>
void cplReduceS::Max(T* pSrc, unsigned int n, cudaStream_t stream){
    Reduce<T, MOperator<T, MATH_MAX> >(pSrc, n, stream);
}

template void cplReduceS::Max(float* pSrc, unsigned int n, cudaStream_t stream);
template void cplReduceS::Max(int* pSrc, unsigned int n, cudaStream_t stream);
template void cplReduceS::Max(uint* pSrc, unsigned int n, cudaStream_t stream);

template<typename T>
void cplReduceS::Min(T* pSrc, unsigned int n, cudaStream_t stream){
    Reduce<T, MOperator<T, MATH_MIN> >(pSrc, n, stream);
}

template void cplReduceS::Min(float* pSrc, unsigned int n, cudaStream_t stream);
template void cplReduceS::Min(int* pSrc, unsigned int n, cudaStream_t stream);
template void cplReduceS::Min(uint* pSrc, unsigned int n, cudaStream_t stream);

template<typename T>
void cplReduceS::Sum(T* pSrc, unsigned int n, cudaStream_t stream){
    Reduce<T, MOperator<T, MATH_ADD> >(pSrc, n, stream);
}

template void cplReduceS::Sum(float* pSrc, unsigned int n, cudaStream_t stream);
template void cplReduceS::Sum(int* pSrc, unsigned int n, cudaStream_t stream);
template void cplReduceS::Sum(uint* pSrc, unsigned int n, cudaStream_t stream);

////////////////////////////////////////////////////////////////////////////////
template <typename T, class opers, bool accumulate>
void cplReduceS::Reduce(T* d_o, T* pSrc, int n, cudaStream_t stream)
{
    reduce_level1_kernel<T, opers><<<M, N, 0, stream>>>(pSrc, (T*)d_rdBuf, n);
    reduce_level2_kernel<T, opers, accumulate><<<1, N, 0, stream>>>(d_o, (T*) d_rdBuf);
}

////////////////////////////////////////////////////////////////////////////////
template<typename T>
void cplReduceS::Max(T* d_o, T* pSrc, unsigned int n, cudaStream_t stream){
    Reduce<T, MOperator<T, MATH_MAX>, false>(d_o, pSrc, n, stream);
}
template void cplReduceS::Max(float* d_o, float* pSrc, unsigned int n, cudaStream_t stream);
template void cplReduceS::Max(int* d_o, int* pSrc, unsigned int n, cudaStream_t stream);
template void cplReduceS::Max(uint* d_o, uint* pSrc, unsigned int n, cudaStream_t stream);

template<typename T>
void cplReduceS::MaxA(T* d_o, T* pSrc, unsigned int n, cudaStream_t stream){
    Reduce<T, MOperator<T, MATH_MAX>, true>(d_o, pSrc, n, stream);
}
template void cplReduceS::MaxA(float* d_o, float* pSrc, unsigned int n, cudaStream_t stream);
template void cplReduceS::MaxA(int* d_o, int* pSrc, unsigned int n, cudaStream_t stream);
template void cplReduceS::MaxA(uint* d_o, uint* pSrc, unsigned int n, cudaStream_t stream);


////////////////////////////////////////////////////////////////////////////////
template<typename T>
void cplReduceS::Min(T* d_o, T* pSrc, unsigned int n, cudaStream_t stream){
    Reduce<T, MOperator<T, MATH_MIN>, false >(d_o, pSrc, n, stream);
}
template void cplReduceS::Min(float* d_o, float* pSrc, unsigned int n, cudaStream_t stream);
template void cplReduceS::Min(int* d_o, int* pSrc, unsigned int n, cudaStream_t stream);
template void cplReduceS::Min(uint* d_o, uint* pSrc, unsigned int n, cudaStream_t stream);

template<typename T>
void cplReduceS::MinA(T* d_o, T* pSrc, unsigned int n, cudaStream_t stream){
    Reduce<T, MOperator<T, MATH_MIN>, true >(d_o, pSrc, n, stream);
}
template void cplReduceS::MinA(float* d_o, float* pSrc, unsigned int n, cudaStream_t stream);
template void cplReduceS::MinA(int* d_o, int* pSrc, unsigned int n, cudaStream_t stream);
template void cplReduceS::MinA(uint* d_o, uint* pSrc, unsigned int n, cudaStream_t stream);

////////////////////////////////////////////////////////////////////////////////
template<typename T>
void cplReduceS::Sum(T* d_o, T* pSrc, unsigned int n, cudaStream_t stream){
    Reduce<T, MOperator<T, MATH_ADD>, false>(d_o, pSrc, n, stream);
}

template void cplReduceS::Sum(float* d_o, float* pSrc, unsigned int n, cudaStream_t stream);
template void cplReduceS::Sum(int* d_o, int* pSrc, unsigned int n, cudaStream_t stream);
template void cplReduceS::Sum(uint* d_o, uint* pSrc, unsigned int n, cudaStream_t stream);

template<typename T>
void cplReduceS::SumA(T* d_o, T* pSrc, unsigned int n, cudaStream_t stream){
    Reduce<T, MOperator<T, MATH_ADD>, true>(d_o, pSrc, n, stream);
}

template void cplReduceS::SumA(float* d_o, float* pSrc, unsigned int n, cudaStream_t stream);
template void cplReduceS::SumA(int* d_o, int* pSrc, unsigned int n, cudaStream_t stream);
template void cplReduceS::SumA(uint* d_o, uint* pSrc, unsigned int n, cudaStream_t stream);

////////////////////////////////////////////////////////////////////////////////
template <typename T, class opers, class opers1>
void cplReduceS::CompReduce(T* pSrc, int n, cudaStream_t stream)
{
    compReduce_level1_kernel<T, opers, opers1><<<M, N, 0, stream>>>(pSrc, (T*)d_rdBuf, n);
    reduce_level2_kernel<T, opers><<<1, N, 0, stream>>>((T*)d_rdBuf);
}

template<typename T>
void cplReduceS::MaxAbs(T* pSrc, unsigned int n, cudaStream_t stream){
    CompReduce<T, MOperator<T, MATH_MAX>, MOperator<T, MATH_ABS> >(pSrc, n, stream);
};

template void cplReduceS::MaxAbs(float* pSrc, unsigned int n, cudaStream_t stream);
template void cplReduceS::MaxAbs(int* pSrc, unsigned int n, cudaStream_t stream);
template void cplReduceS::MaxAbs(uint* pSrc, unsigned int n, cudaStream_t stream);

template<typename T>
void cplReduceS::Sum2(T* pSrc, unsigned int n, cudaStream_t stream){
    CompReduce<T, MOperator<T, MATH_ADD>, MOperator<T, MATH_SQR> >(pSrc, n, stream);
};
template void cplReduceS::Sum2(float* pSrc, unsigned int n, cudaStream_t stream);
template void cplReduceS::Sum2(int* pSrc, unsigned int n, cudaStream_t stream);
template void cplReduceS::Sum2(uint* pSrc, unsigned int n, cudaStream_t stream);

template<typename T>
void cplReduceS::SumAbs(T* pSrc, unsigned int n, cudaStream_t stream){
    CompReduce<T, MOperator<T, MATH_ADD>, MOperator<T, MATH_ABS> >(pSrc, n, stream);
};

template void cplReduceS::SumAbs(float* pSrc, unsigned int n, cudaStream_t stream);
template void cplReduceS::SumAbs(int* pSrc, unsigned int n, cudaStream_t stream);
template void cplReduceS::SumAbs(uint* pSrc, unsigned int n, cudaStream_t stream);

////////////////////////////////////////////////////////////////////////////////
template <typename T, class opers, class opers1, bool accumulate>
void cplReduceS::CompReduce(T* d_o, T* pSrc, int n, cudaStream_t stream)
{
    compReduce_level1_kernel<T, opers, opers1><<<M, N, 0, stream>>>(pSrc, (T*)d_rdBuf, n);
    reduce_level2_kernel<T, opers, accumulate><<<1, N, 0, stream>>>(d_o, (T*)d_rdBuf);
}

template<typename T>
void cplReduceS::MaxAbs(T* d_o, T* pSrc, unsigned int n, cudaStream_t stream){
    CompReduce<T, MOperator<T, MATH_MAX>, MOperator<T, MATH_ABS>, false >(d_o, pSrc, n, stream);
};

template void cplReduceS::MaxAbs(float* d_o, float* pSrc, unsigned int n, cudaStream_t stream);
template void cplReduceS::MaxAbs(int* d_o, int* pSrc, unsigned int n, cudaStream_t stream);
template void cplReduceS::MaxAbs(uint* d_o, uint* pSrc, unsigned int n, cudaStream_t stream);

template<typename T>
void cplReduceS::MaxAbsA(T* d_o, T* pSrc, unsigned int n, cudaStream_t stream){
    CompReduce<T, MOperator<T, MATH_MAX>, MOperator<T, MATH_ABS>, true >(d_o, pSrc, n, stream);
};

template void cplReduceS::MaxAbsA(float* d_o, float* pSrc, unsigned int n, cudaStream_t stream);
template void cplReduceS::MaxAbsA(int* d_o, int* pSrc, unsigned int n, cudaStream_t stream);
template void cplReduceS::MaxAbsA(uint* d_o, uint* pSrc, unsigned int n, cudaStream_t stream);

////////////////////////////////////////////////////////////////////////////////
template<typename T>
void cplReduceS::Sum2(T* d_o, T* pSrc, unsigned int n, cudaStream_t stream){
    CompReduce<T, MOperator<T, MATH_ADD>, MOperator<T, MATH_SQR>, false >(d_o, pSrc, n, stream);
};
template void cplReduceS::Sum2(float* d_o, float* pSrc, unsigned int n, cudaStream_t stream);
template void cplReduceS::Sum2(int* d_o, int* pSrc, unsigned int n, cudaStream_t stream);
template void cplReduceS::Sum2(uint* d_o,uint* pSrc, unsigned int n, cudaStream_t stream);

template<typename T>
void cplReduceS::Sum2A(T* d_o, T* pSrc, unsigned int n, cudaStream_t stream){
    CompReduce<T, MOperator<T, MATH_ADD>, MOperator<T, MATH_SQR>, true >(d_o, pSrc, n, stream);
};
template void cplReduceS::Sum2A(float* d_o, float* pSrc, unsigned int n, cudaStream_t stream);
template void cplReduceS::Sum2A(int* d_o, int* pSrc, unsigned int n, cudaStream_t stream);
template void cplReduceS::Sum2A(uint* d_o,uint* pSrc, unsigned int n, cudaStream_t stream);

////////////////////////////////////////////////////////////////////////////////
template<typename T>
void cplReduceS::SumAbs(T* d_o, T* pSrc, unsigned int n, cudaStream_t stream){
    CompReduce<T, MOperator<T, MATH_ADD>, MOperator<T, MATH_ABS>, false >(d_o, pSrc, n, stream);
};

template void cplReduceS::SumAbs(float* d_o, float* pSrc, unsigned int n, cudaStream_t stream);
template void cplReduceS::SumAbs(int* d_o, int* pSrc, unsigned int n, cudaStream_t stream);
template void cplReduceS::SumAbs(uint* d_o, uint* pSrc, unsigned int n, cudaStream_t stream);

template<typename T>
void cplReduceS::SumAbsA(T* d_o, T* pSrc, unsigned int n, cudaStream_t stream){
    CompReduce<T, MOperator<T, MATH_ADD>, MOperator<T, MATH_ABS>, true >(d_o, pSrc, n, stream);
};

template void cplReduceS::SumAbsA(float* d_o, float* pSrc, unsigned int n, cudaStream_t stream);
template void cplReduceS::SumAbsA(int* d_o, int* pSrc, unsigned int n, cudaStream_t stream);
template void cplReduceS::SumAbsA(uint* d_o, uint* pSrc, unsigned int n, cudaStream_t stream);

////////////////////////////////////////////////////////////////////////////////
template <typename T, class opers, class opers1>
void cplReduceS::Product(T* d_i0, T* d_i1, unsigned int n, cudaStream_t stream)
{
    product_level1_kernel<T, opers, opers1><<<M, N, 0, stream>>>(d_i0, d_i1, (T*)d_rdBuf, n);
    reduce_level2_kernel<T, opers><<<1, N, 0, stream>>>((T*)d_rdBuf);
}

template<typename T>
void cplReduceS::Dot(T* d_i0, T* d_i1, unsigned int n, cudaStream_t stream)
{
    Product<T, MOperator<T, MATH_ADD>, MOperator<T, MATH_MUL> >(d_i0, d_i1, n, stream);
}

template void cplReduceS::Dot(float* d_i0, float* d_i1, unsigned int n, cudaStream_t stream);
template void cplReduceS::Dot(int* d_i0, int* d_i1, unsigned int n, cudaStream_t stream);
template void cplReduceS::Dot(uint* d_i0, uint* d_i1, unsigned int n, cudaStream_t stream);

template<typename T>
void cplReduceS::SumAdd(T* d_i0, T* d_i1, unsigned int n, cudaStream_t stream)
{
    Product<T, MOperator<T, MATH_ADD>, MOperator<T, MATH_ADD> >(d_i0, d_i1, n, stream);
}

template void cplReduceS::SumAdd(float* d_i0, float* d_i1, unsigned int n, cudaStream_t stream);
template void cplReduceS::SumAdd(int* d_i0, int* d_i1, unsigned int n, cudaStream_t stream);
template void cplReduceS::SumAdd(uint* d_i0, uint* d_i1, unsigned int n, cudaStream_t stream);


template<typename T>
void cplReduceS::MinAdd(T* d_i0, T* d_i1, unsigned int n, cudaStream_t stream)
{
    Product<T, MOperator<T, MATH_MIN>, MOperator<T, MATH_ADD> >(d_i0, d_i1, n, stream);
}

template void cplReduceS::MinAdd(float* d_i0, float* d_i1, unsigned int n, cudaStream_t stream);
template void cplReduceS::MinAdd(int* d_i0, int* d_i1, unsigned int n, cudaStream_t stream);
template void cplReduceS::MinAdd(uint* d_i0, uint* d_i1, unsigned int n, cudaStream_t stream);


template<typename T>
void cplReduceS::MaxAdd(T* d_i0, T* d_i1, unsigned int n, cudaStream_t stream)
{
    Product<T, MOperator<T, MATH_MAX>, MOperator<T, MATH_ADD> >(d_i0, d_i1, n, stream);
}

template void cplReduceS::MaxAdd(float* d_i0, float* d_i1, unsigned int n, cudaStream_t stream);
template void cplReduceS::MaxAdd(int* d_i0, int* d_i1, unsigned int n, cudaStream_t stream);
template void cplReduceS::MaxAdd(uint* d_i0, uint* d_i1, unsigned int n, cudaStream_t stream);

////////////////////////////////////////////////////////////////////////////////
template <typename T, class opers, class opers1, bool accumulate>
void cplReduceS::Product(T* d_o, T* d_i0, T* d_i1, unsigned int n, cudaStream_t stream)
{
    product_level1_kernel<T, opers, opers1><<<M, N, 0, stream>>>(d_i0, d_i1, (T*)d_rdBuf, n);
    reduce_level2_kernel<T, opers, accumulate><<<1, N, 0, stream>>>(d_o, (T*)d_rdBuf);
}

template<typename T>
void cplReduceS::Dot(T* d_o, T* d_i0, T* d_i1, unsigned int n, cudaStream_t stream)
{
    Product<T, MOperator<T, MATH_ADD>, MOperator<T, MATH_MUL>, false >(d_o, d_i0, d_i1, n, stream);
}

template void cplReduceS::Dot(float* d_o, float* d_i0, float* d_i1, unsigned int n, cudaStream_t stream);
template void cplReduceS::Dot(int* d_o, int* d_i0, int* d_i1, unsigned int n, cudaStream_t stream);
template void cplReduceS::Dot(uint* d_o, uint* d_i0, uint* d_i1, unsigned int n, cudaStream_t stream);

template<typename T>
void cplReduceS::DotA(T* d_o, T* d_i0, T* d_i1, unsigned int n, cudaStream_t stream)
{
    Product<T, MOperator<T, MATH_ADD>, MOperator<T, MATH_MUL>, true >(d_o, d_i0, d_i1, n, stream);
}

template void cplReduceS::DotA(float* d_o, float* d_i0, float* d_i1, unsigned int n, cudaStream_t stream);
template void cplReduceS::DotA(int* d_o, int* d_i0, int* d_i1, unsigned int n, cudaStream_t stream);
template void cplReduceS::DotA(uint* d_o, uint* d_i0, uint* d_i1, unsigned int n, cudaStream_t stream);

////////////////////////////////////////////////////////////////////////////////
template<typename T>
void cplReduceS::SumAdd(T* d_o, T* d_i0, T* d_i1, unsigned int n, cudaStream_t stream)
{
    Product<T, MOperator<T, MATH_ADD>, MOperator<T, MATH_ADD>, false >(d_o, d_i0, d_i1, n, stream);
}

template void cplReduceS::SumAdd(float* d_o, float* d_i0, float* d_i1, unsigned int n, cudaStream_t stream);
template void cplReduceS::SumAdd(int* d_o, int* d_i0, int* d_i1, unsigned int n, cudaStream_t stream);
template void cplReduceS::SumAdd(uint* d_o, uint* d_i0, uint* d_i1, unsigned int n, cudaStream_t stream);

template<typename T>
void cplReduceS::SumAddA(T* d_o, T* d_i0, T* d_i1, unsigned int n, cudaStream_t stream)
{
    Product<T, MOperator<T, MATH_ADD>, MOperator<T, MATH_ADD>, true >(d_o, d_i0, d_i1, n, stream);
}

template void cplReduceS::SumAddA(float* d_o, float* d_i0, float* d_i1, unsigned int n, cudaStream_t stream);
template void cplReduceS::SumAddA(int* d_o, int* d_i0, int* d_i1, unsigned int n, cudaStream_t stream);
template void cplReduceS::SumAddA(uint* d_o, uint* d_i0, uint* d_i1, unsigned int n, cudaStream_t stream);

////////////////////////////////////////////////////////////////////////////////
template<typename T>
void cplReduceS::MinAdd(T* d_o, T* d_i0, T* d_i1, unsigned int n, cudaStream_t stream)
{
    Product<T, MOperator<T, MATH_MIN>, MOperator<T, MATH_ADD>, false >(d_o, d_i0, d_i1, n, stream);
}

template void cplReduceS::MinAdd(float* d_o, float* d_i0, float* d_i1, unsigned int n, cudaStream_t stream);
template void cplReduceS::MinAdd(int* d_o, int* d_i0, int* d_i1, unsigned int n, cudaStream_t stream);
template void cplReduceS::MinAdd(uint* d_o, uint* d_i0, uint* d_i1, unsigned int n, cudaStream_t stream);

template<typename T>
void cplReduceS::MinAddA(T* d_o, T* d_i0, T* d_i1, unsigned int n, cudaStream_t stream)
{
    Product<T, MOperator<T, MATH_MIN>, MOperator<T, MATH_ADD>, true >(d_o, d_i0, d_i1, n, stream);
}

template void cplReduceS::MinAddA(float* d_o, float* d_i0, float* d_i1, unsigned int n, cudaStream_t stream);
template void cplReduceS::MinAddA(int* d_o, int* d_i0, int* d_i1, unsigned int n, cudaStream_t stream);
template void cplReduceS::MinAddA(uint* d_o, uint* d_i0, uint* d_i1, unsigned int n, cudaStream_t stream);

////////////////////////////////////////////////////////////////////////////////
template<typename T>
void cplReduceS::MaxAdd(T* d_o, T* d_i0, T* d_i1, unsigned int n, cudaStream_t stream)
{
    Product<T, MOperator<T, MATH_MAX>, MOperator<T, MATH_ADD>, false >(d_o, d_i0, d_i1, n, stream);
}

template void cplReduceS::MaxAdd(float* d_o, float* d_i0, float* d_i1, unsigned int n, cudaStream_t stream);
template void cplReduceS::MaxAdd(int* d_o, int* d_i0, int* d_i1, unsigned int n, cudaStream_t stream);
template void cplReduceS::MaxAdd(uint* d_o, uint* d_i0, uint* d_i1, unsigned int n, cudaStream_t stream);

template<typename T>
void cplReduceS::MaxAddA(T* d_o, T* d_i0, T* d_i1, unsigned int n, cudaStream_t stream)
{
    Product<T, MOperator<T, MATH_MAX>, MOperator<T, MATH_ADD>, true >(d_o, d_i0, d_i1, n, stream);
}

template void cplReduceS::MaxAddA(float* d_o, float* d_i0, float* d_i1, unsigned int n, cudaStream_t stream);
template void cplReduceS::MaxAddA(int* d_o, int* d_i0, int* d_i1, unsigned int n, cudaStream_t stream);
template void cplReduceS::MaxAddA(uint* d_o, uint* d_i0, uint* d_i1, unsigned int n, cudaStream_t stream);


template<typename T>
void cplReduceS::GetResultBuffer(T* h_r, int n)
{
    copyArrayFromDevice(h_r, (T*)d_rdBuf, n);
}

template void cplReduceS::GetResultBuffer(float* h_r, int n);
template void cplReduceS::GetResultBuffer(int* h_r, int n);
template void cplReduceS::GetResultBuffer(uint* h_r, int n);

template<typename T>
T cplReduceS::GetResult(int off) {
    T h_r;
    copyArrayFromDevice(&h_r, (T*)d_rdBuf, 1);
    return h_r;
}


template float cplReduceS::GetResult<float>(int off);
template int cplReduceS::GetResult<int>(int off);
template uint cplReduceS::GetResult<uint>(int off);
template float2 cplReduceS::GetResult<float2>(int off);
template int2 cplReduceS::GetResult<int2>(int off);
template uint2 cplReduceS::GetResult<uint2>(int off);

bool cplReduceS::selfTest(int n){
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

    int d_max = -INT_MAX, d_min = INT_MAX;
    int d_maxAbs = 0;
    int d_sumSQR = 0;
    int d_sum = 0;
    int d_dot = 0;
    int2 d_pair;
    
    this->Sum(d_i, n); d_sum = this->GetResult<int>();
    this->Max(d_i, n); d_max = this->GetResult<int>();
    this->Min(d_i, n); d_min = this->GetResult<int>();
    this->MaxAbs(d_i, n); d_maxAbs = this->GetResult<int>();
    this->Sum2(d_i, n); d_sumSQR   = this->GetResult<int>();
    this->Dot(d_i,d_i1, n); d_dot  = this->GetResult<int>();
    this->MaxMin(d_i, n);   d_pair = this->GetResult<int2>();
    
    fprintf(stderr, "Maximum value from CPU %d from GPU %d\n",h_max, d_max);
    fprintf(stderr, "Minumum value from CPU %d from GPU %d\n",h_min, d_min);
    fprintf(stderr, "Total value from CPU %d from GPU %d\n",h_sum, d_sum);
    
    fprintf(stderr, "Maximum abosulte value from CPU %d from GPU %d\n",h_maxAbs, d_maxAbs);
    fprintf(stderr, "Total square value from CPU %d from GPU %d\n",h_sumSQR, d_sumSQR);
    fprintf(stderr, "Dot product value from CPU %d from GPU %d\n",h_dot, d_dot);
    fprintf(stderr, "Max Min value from CPU %d %d from GPU %d %d\n",h_max, h_min, d_pair.x, d_pair.y);
    
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

        this->Sum(d_i, l); d_sum = this->GetResult<int>();
        this->Max(d_i, l); d_max = this->GetResult<int>();
        this->Min(d_i, l); d_min = this->GetResult<int>();
        this->MaxAbs(d_i, l); d_maxAbs = this->GetResult<int>();
        this->Sum2(d_i, l); d_sumSQR   = this->GetResult<int>();
        this->Dot(d_i,d_i1, l); d_dot  = this->GetResult<int>();
        this->MaxMin(d_i, l);   d_pair = this->GetResult<int2>();

        
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
        
        if ( d_pair.x != h_max || d_pair.y != h_min){
            fprintf(stderr, "Max Min Test FAILED at %d GPU %d %d CPU %d %d\n",
                    l, d_pair.x, d_pair.y, h_max, h_min);
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


void cplReduceS::benchmark(){
    const int step    = 2500000;
    const int maxSize = 40 * step;
    unsigned int timer = 0;
    int nIters = 500;
    int *d_i, *d_i1;
    dmemAlloc(d_i, maxSize);
    dmemAlloc(d_i1, maxSize);
        
    cplVectorOpers::SetMem(d_i, 1, maxSize);
    cplVectorOpers::SetMem(d_i1, 1, maxSize);
    
    this->Sum(d_i, maxSize);
    cutCreateTimer(&timer);
    cudaThreadSynchronize();
    for (int is = step; is < maxSize; is += step) {
        cutResetTimer(timer);
        cutStartTimer(timer);
        for (int i=0; i < nIters; ++i)
            this->Sum(d_i, is);
        cudaThreadSynchronize();
        cutStopTimer(timer);
        float reduceTime = cutGetTimerValue(timer) / nIters;
        fprintf(stderr, "Size %d Sum Bandwidth:    %f GB/s\n", is, (is * sizeof(int)) / (reduceTime * 1.0e6));
    }

    this->Sum2(d_i, maxSize);
    for (int is = step; is < maxSize; is += step) {
        cutResetTimer(timer);
        cutStartTimer(timer);
        for (int i=0; i < nIters; ++i)
            this->Sum2(d_i, is);
        cudaThreadSynchronize();
        cutStopTimer(timer);
        float reduceTime = cutGetTimerValue(timer) / nIters;
        fprintf(stderr, "Size %d Sum2 Bandwidth:    %f GB/s\n", is, (is * sizeof(int)) / (reduceTime * 1.0e6));
    }

    this->Dot(d_i, d_i1, maxSize);
    for (int is = step; is < maxSize; is += step) {
        cutResetTimer(timer);
        cutStartTimer(timer);
        for (int i=0; i < nIters; ++i)
            this->Dot(d_i, d_i1,  is);
        cudaThreadSynchronize();
        cutStopTimer(timer);
        float reduceTime = cutGetTimerValue(timer) / nIters;
        fprintf(stderr, "Size %d Dot Bandwidth:    %f GB/s\n", is, (is * sizeof(int2)) / (reduceTime * 1.0e6));
    }

    this->MaxMin(d_i, maxSize);
    for (int is = step; is < maxSize; is += step) {
        cutResetTimer(timer);
        cutStartTimer(timer);
        for (int i=0; i < nIters; ++i)
            this->MaxMin(d_i, is);
        cudaThreadSynchronize();
        cutStopTimer(timer);
        float reduceTime = cutGetTimerValue(timer) / nIters;
        fprintf(stderr, "Size %d Bireduce Bandwidth:    %f GB/s\n", is, (is * sizeof(int)) / (reduceTime * 1.0e6));
    }
    cutDeleteTimer(timer);
    dmemFree(d_i);dmemFree(d_i1);
    cutilCheckMsg("Free memory");
}
