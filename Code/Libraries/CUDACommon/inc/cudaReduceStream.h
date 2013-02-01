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

#ifndef __CUDA_REDUCE_STREAM_H
#define __CUDA_REDUCE_STREAM_H

#include <cuda_runtime.h>
class cplReduceS{
public:
    cplReduceS():d_rdBuf(NULL){};
    ~cplReduceS(){clean();};
    
    void init();
    void clean();

    template<typename T>
    void Max(T* d_i, unsigned int n, cudaStream_t stream=NULL);
    template<typename T>
    void Min(T* d_i, unsigned int n, cudaStream_t stream=NULL);
    template<typename T>
    void Sum(T* d_i, unsigned int n, cudaStream_t stream=NULL);

    template<typename T>
    void MaxAbs(T* d_i, unsigned int n, cudaStream_t stream=NULL);
    template<typename T>
    void Sum2(T* d_i, unsigned int n, cudaStream_t stream=NULL);
    template<typename T>
    void SumAbs(T* d_i, unsigned int n, cudaStream_t stream=NULL);

    template<class T>
    void MaxMin(T* d_i, unsigned int n, cudaStream_t stream=NULL);
    template<class T>
    void MaxSum(T* d_i, unsigned int n, cudaStream_t stream=NULL);

    template<typename T>
    void Dot(T* d_i, T* d_i1,  unsigned int n, cudaStream_t stream=NULL);
    template<typename T>
    void SumAdd(T* d_i, T* d_i1,  unsigned int n, cudaStream_t stream=NULL);
    template<typename T>
    void MaxAdd(T* d_i, T* d_i1,  unsigned int n, cudaStream_t stream=NULL);
    template<typename T>
    void MinAdd(T* d_i, T* d_i1,  unsigned int n, cudaStream_t stream=NULL);

    template<typename T>
    void Max(T* d_o, T* d_i, unsigned int n, cudaStream_t stream=NULL);
    template<typename T>
    void Min(T* d_o, T* d_i, unsigned int n, cudaStream_t stream=NULL);
    template<typename T>
    void Sum(T* d_o, T* d_i, unsigned int n, cudaStream_t stream=NULL);

    template<typename T>
    void MaxAbs(T* d_o, T* d_i, unsigned int n, cudaStream_t stream=NULL);
    template<typename T>
    void Sum2(T* d_o, T* d_i, unsigned int n, cudaStream_t stream=NULL);

    template<typename T>
    void SumAbs(T* d_o, T* d_i, unsigned int n, cudaStream_t stream=NULL);
    template<class T>
    void MaxMin(T* d_o, T* d_i, unsigned int n, cudaStream_t stream=NULL);
    template<class T>
    void MaxSum(T* d_o, T* d_i, unsigned int n, cudaStream_t stream=NULL);

    template<typename T>
    void Dot(T* d_o, T* d_i, T* d_i1,  unsigned int n, cudaStream_t stream=NULL);
    template<typename T>
    void SumAdd(T* d_o, T* d_i, T* d_i1,  unsigned int n, cudaStream_t stream=NULL);
    template<typename T>
    void MaxAdd(T* d_o, T* d_i, T* d_i1,  unsigned int n, cudaStream_t stream=NULL);
    template<typename T>
    void MinAdd(T* d_o, T* d_i, T* d_i1,  unsigned int n, cudaStream_t stream=NULL);

    template<typename T>
    void MaxA(T* d_o, T* d_i, unsigned int n, cudaStream_t stream=NULL);
    template<typename T>
    void MinA(T* d_o, T* d_i, unsigned int n, cudaStream_t stream=NULL);
    template<typename T>
    void SumA(T* d_o, T* d_i, unsigned int n, cudaStream_t stream=NULL);

    template<typename T>
    void MaxAbsA(T* d_o, T* d_i, unsigned int n, cudaStream_t stream=NULL);
    template<typename T>
    void Sum2A(T* d_o, T* d_i, unsigned int n, cudaStream_t stream=NULL);
    template<typename T>
    void SumAbsA(T* d_o, T* d_i, unsigned int n, cudaStream_t stream=NULL);
    
    template<class T>
    void MaxMinA(T* d_o, T* d_i, unsigned int n, cudaStream_t stream=NULL);
    template<class T>
    void MaxSumA(T* d_o, T* d_i, unsigned int n, cudaStream_t stream=NULL);

    template<typename T>
    void DotA(T* d_o, T* d_i, T* d_i1,  unsigned int n, cudaStream_t stream=NULL);
    template<typename T>
    void SumAddA(T* d_o, T* d_i, T* d_i1,  unsigned int n, cudaStream_t stream=NULL);
    template<typename T>
    void MaxAddA(T* d_o, T* d_i, T* d_i1,  unsigned int n, cudaStream_t stream=NULL);
    template<typename T>
    void MinAddA(T* d_o, T* d_i, T* d_i1,  unsigned int n, cudaStream_t stream=NULL);

    bool selfTest(int n);
    void benchmark();

    template<typename T>
    T* GetDeviceResultPtr() { return (T*)d_rdBuf; }

    template<typename T>
    void GetResultBuffer(T* h_r, int length);

    template<typename T>
    T GetResult(int off=0);
protected:
    template <typename T, class trait>
    void Reduce(T* d_i, int n, cudaStream_t stream=NULL);
    template <typename T, class traits, class traits1>
    void CompReduce(T* d_i, int n, cudaStream_t stream=NULL);
    template <typename T, class traits, class traits1>
    void Product(T* d_i0, T* d_i1, unsigned int n, cudaStream_t stream=NULL);
    template <typename T, class oper, class oper1>
    void Bireduce(T* d_i, unsigned int n, cudaStream_t stream=NULL);
    
    template <typename T, class trait, bool accumulate>
    void Reduce(T* d_o, T* d_i, int n, cudaStream_t stream=NULL);
    template <typename T, class traits, class traits1, bool accumulate>
    void CompReduce(T* d_o, T* d_i, int n, cudaStream_t stream=NULL);
    template <typename T, class traits, class traits1, bool accumulate>
    void Product(T* d_o, T* d_i0, T* d_i1, unsigned int n, cudaStream_t stream=NULL);
    template <typename T, class oper, class oper1, bool accumulate>
    void Bireduce(T* d_o, T* d_i, unsigned int n, cudaStream_t stream=NULL);
    void* d_rdBuf;
};

    


#endif
