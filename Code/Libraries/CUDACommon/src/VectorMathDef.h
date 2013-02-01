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

#ifndef __VECTOR_MATH_DEF_H
#define __VECTOR_MATH_DEF_H

#include <oper_util.h>
#include <VectorMath.h>
#include <cudaInterface.h>
#include <cutil_math.h>
#include <cutil_comfunc.h>
#include <cplMacro.h>
#include <cudaTexFetch.h>

namespace cplVectorOpers {
    template<class T>
    __global__ void Copy_kernel(T* d_o, const T* d_i, unsigned int n){
        uint blockId = get_blockID();
        uint      id = get_threadID(blockId);

        if (id < n){
            d_o[id] = d_i[id];
        }
    }

    template<class T>
    void Copy(T* d_o, const T* d_i, unsigned int n, cudaStream_t stream){
        dim3 threads(256);
        dim3 grids(iDivUp(n, threads.x));
        checkConfig(grids);
        Copy_kernel<<<grids, threads, 0, stream>>>(d_o, d_i, n);
    }

/////////////////////////////////////////////////////////////////////////////////
// cplVectorOpers::SetMem
//  the function provided by CUDA is slow and is not flexible enough to set
//  value with different type
//////////////////////////////////////////////////////////////////////////////////
    template<class T>
    __global__ void SetMem_kernel(T* g_data, T c, int n){
        uint blockId = get_blockID();
        uint      id = get_threadID(blockId);
        if (id < n) 
            g_data[id] = c;
    }

    template<class T>
    void SetMem(T* d_data, T c, int n, cudaStream_t stream) {
        dim3 threads(256);
        dim3 grids(iDivUp(n, threads.x));
        checkConfig(grids);
        SetMem_kernel<<<grids, threads,0,stream>>>(d_data, c, n);
    }

/////////////////////////////////////////////////////////////////////////////////
//  SetLinear
//  Initiate the value of an unsigned array with a linear ram
//////////////////////////////////////////////////////////////////////////////////
    template<typename T>
    __global__ void SetLinear_kernel(T* g_data, uint n){
        uint blockId = get_blockID();
        uint id      = get_threadID(blockId);
        if (id < n) 
            g_data[id] = (T)id;
    }
    
    template<typename T>
    void SetLinear(T* d_data,  int n, cudaStream_t stream) {
        dim3 threads(256);
        dim3 grids(iDivUp(n,threads.x));
        checkConfig(grids);
    
        SetLinear_kernel<<<grids, threads,0,stream>>>(d_data, n);
    }


/////////////////////////////////////////////////////////////////////////////////
// SetLinearDown
//  Initiate the value of an unsigned array with a linear ram down
//////////////////////////////////////////////////////////////////////////////////
    template<typename T>
    __global__ void SetLinearDown_kernel(T* g_data, uint n){
        uint blockId = get_blockID();
        uint id      = get_threadID(blockId);
        if (id < n) 
            g_data[id] = (T)(n - id - 1);
    }

    template<typename T>
    void SetLinearDown(T* d_data,  int n, cudaStream_t stream) {
        dim3 threads(256);
        dim3 grids(iDivUp(n,threads.x));
        checkConfig(grids);
    
        SetLinearDown_kernel<<<grids, threads,0,stream>>>(d_data, n);
    }

/////////////////////////////////////////////////////////////////////////////////
// Comp_unary 
//  Return the result on a single operation on the input : abs, negative, sqrt ...
//////////////////////////////////////////////////////////////////////////////////
    template<class T, class trait>
    __global__ void Comp_unary_kernel(T* d_o, const T* d_i, int n){
        uint blockId = get_blockID();
        uint      id = get_threadID(blockId);
        if (id < n){
            d_o[id] = trait::op(d_i[id]);
        }
    }

    template<class T, class trait>
    void Comp_unary(T* d_o, const T* d_i, int n, cudaStream_t stream){
        dim3 threads(256);
        dim3 grids(iDivUp(n,threads.x));
        checkConfig(grids);
        Comp_unary_kernel<T, trait><<<grids, threads,0,stream>>>(d_o, d_i, n);
    }

    template<class T>
    void Abs(T* odata, const T* idata, unsigned int size, cudaStream_t stream){
        Comp_unary<T, MOperator<T, MATH_ABS> >(odata, idata, size, stream);
    }

    template<class T>
    void Cube(T* odata, const T* idata, unsigned int size, cudaStream_t stream){
        Comp_unary<T, MOperator<T, MATH_CUBE> >(odata, idata, size, stream);
    }

    template<class T>
    void Sqr(T* odata, const T* idata, unsigned int size, cudaStream_t stream){
        Comp_unary<T, MOperator<T, MATH_SQR> >(odata, idata, size, stream);
    }

    template<class T>
    void Neg(T* odata, const T* idata, unsigned int size, cudaStream_t stream){
        Comp_unary<T, MOperator<T, MATH_NEG> >(odata, idata, size, stream);
    }

/////////////////////////////////////////////////////////////////////////////////
// Comp_unary Inplace version
//////////////////////////////////////////////////////////////////////////////////
    template<class T, class trait>
    __global__ void Comp_unary_kernel_I(T* d_i, int n){
        uint blockId = get_blockID();
        uint      id = get_threadID(blockId);
        if (id < n){
            d_i[id] = trait::op(d_i[id]);
        }
    }

    template<class T, class trait>
    void Comp_unary_I(T* d_i, int n, cudaStream_t stream){
        dim3 threads(256);
        dim3 grids(iDivUp(n,threads.x));
        checkConfig(grids);
        Comp_unary_kernel_I<T, trait><<<grids, threads,0,stream>>>(d_i, n);
    }

    template<class T>
    void Abs_I(T* idata, unsigned int size, cudaStream_t stream){
        Comp_unary_I<T, MOperator<T, MATH_ABS> >(idata, size, stream);
    }

    template<class T>
    void Cube_I(T* idata, unsigned int size, cudaStream_t stream){
        Comp_unary_I<T, MOperator<T, MATH_CUBE> >(idata, size, stream);
    }

    template<class T>
    void Sqr_I(T* idata, unsigned int size, cudaStream_t stream){
        Comp_unary_I<T, MOperator<T, MATH_SQR> >(idata, size, stream);
    }

    template<class T>
    void Neg_I(T* idata, unsigned int size, cudaStream_t stream){
        Comp_unary_I<T, MOperator<T, MATH_NEG> >(idata, size, stream);
    }

    void Sqrt(float* odata,const  float* idata, unsigned int size, cudaStream_t stream){
        Comp_unary<float, MOperator<float, MATH_SQRT> >(odata, idata, size, stream);
    }

    void Inv(float* odata, const float* idata, unsigned int size, cudaStream_t stream){
        Comp_unary<float, MOperator<float, MATH_INV> >(odata, idata, size, stream);
    }

    void Sqrt_I(float* idata, unsigned int size, cudaStream_t stream){
        Comp_unary_I<float, MOperator<float, MATH_SQRT> >(idata, size, stream);
    }

    void Inv_I(float* idata, unsigned int size, cudaStream_t stream){
        Comp_unary_I<float, MOperator<float, MATH_INV> >(idata, size, stream);
    }

/////////////////////////////////////////////////////////////////////////////////
// Binary functions
//////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////
// Add a constant
//////////////////////////////////////////////////////////////////////////////////
    template<class T> __global__ void AddC_kernel(T* d_o, T* d_i, T c, int n){
        uint blockId = get_blockID();
        uint      id = get_threadID(blockId);
        if (id < n){
            d_o[id] = d_i[id] + c;
        }
    }
    template<class T>
    void AddC(T* d_o, T* d_i, T c, int n, cudaStream_t stream){
        dim3 threads(256);
        dim3 grids(iDivUp(n,threads.x));
        checkConfig(grids);
        AddC_kernel<<<grids, threads,0,stream>>>(d_o, d_i, c, n);
    }

/////////////////////////////////////////////////////////////////////////////////
// Add constant inplace
//////////////////////////////////////////////////////////////////////////////////
    template<class T>
    __global__ void AddC_kernel_I(T* d_o, T c, int n){
        uint blockId = get_blockID();
        uint      id = get_threadID(blockId);
        if (id < n){
            d_o[id] += c;
        }
    }
    template<class T>
    void AddC_I(T* d_o, T c, int n, cudaStream_t stream){
        dim3 threads(256);
        dim3 grids(iDivUp(n,threads.x));
        checkConfig(grids);
        AddC_kernel_I<<<grids, threads,0,stream>>>(d_o, c, n);
    }

/////////////////////////////////////////////////////////////////////////////////
// Add two array
//////////////////////////////////////////////////////////////////////////////////
    template<class T>
    __global__ void Add_kernel(T* d_o , T* d_i, T* d_i1, int n){
        uint blockId = get_blockID();
        uint      id = get_threadID(blockId);
        if (id < n){
            d_o[id] = d_i[id] + d_i1[id];
        }
    }
    
    template<class T>
    void Add(T* d_o, T* d_i, T* d_i1, int n, cudaStream_t stream){
        dim3 threads(256);
        dim3 grids(iDivUp(n,threads.x));
        checkConfig(grids);
        Add_kernel<<<grids, threads,0,stream>>>(d_o, d_i, d_i1, n);
    }

/////////////////////////////////////////////////////////////////////////////////
// Add two array inplace version
//////////////////////////////////////////////////////////////////////////////////
    template<class T>
    __global__ void Add_kernel_I(T* d_o, T* d_i, int n){
        uint blockId = get_blockID();
        uint      id = get_threadID(blockId);
        if (id < n){
            d_o[id] += d_i[id];
        }
    }
    
    template<class T>
    void Add_I(T* d_o, T* d_i, int n, cudaStream_t stream){
        dim3 threads(256);
        dim3 grids(iDivUp(n,threads.x));
        checkConfig(grids);
        Add_kernel_I<<<grids, threads,0,stream>>>(d_o, d_i, n);
    }

/////////////////////////////////////////////////////////////////////////////////
// Sub a constant
//////////////////////////////////////////////////////////////////////////////////
    template<class T> __global__ void SubC_kernel(T* d_o, T* d_i, T c, int n){
        uint blockId = get_blockID();
        uint      id = get_threadID(blockId);
        if (id < n){
            d_o[id] = d_i[id] - c;
        }
    }
    
    template<class T>
    void SubC(T* d_o, T* d_i, T c, int n, cudaStream_t stream){
        dim3 threads(256);
        dim3 grids(iDivUp(n,threads.x));
        checkConfig(grids);
        SubC_kernel<<<grids, threads,0,stream>>>(d_o, d_i, c, n);
    }

/////////////////////////////////////////////////////////////////////////////////
// Sub a constant inplace version
//////////////////////////////////////////////////////////////////////////////////
    template<class T>
    __global__ void SubC_kernel_I(T* d_o, T c, int n){
        uint blockId = get_blockID();
        uint      id = get_threadID(blockId);
        if (id < n){
            d_o[id] -= c;
        }
    }
    
    template<class T>
    void SubC_I(T* d_o, T c, int n, cudaStream_t stream){
        dim3 threads(256);
        dim3 grids(iDivUp(n,threads.x));
        checkConfig(grids);
        SubC_kernel_I<<<grids, threads,0,stream>>>(d_o, c, n);
    }


/////////////////////////////////////////////////////////////////////////////////
// Sub two arrays
//////////////////////////////////////////////////////////////////////////////////

    template<class T>
    __global__ void Sub_kernel(T* d_o , T* d_i, T* d_i1, int n){
        uint blockId = get_blockID();
        uint      id = get_threadID(blockId);
        if (id < n){
            d_o[id] = d_i[id] - d_i1[id];
        }
    }
    
    template<class T>
    void Sub(T* d_o, T* d_i, T* d_i1, int n, cudaStream_t stream){
        dim3 threads(256);
        dim3 grids(iDivUp(n,threads.x));
        checkConfig(grids);
        Sub_kernel<<<grids, threads,0,stream>>>(d_o, d_i, d_i1, n);
    }

/////////////////////////////////////////////////////////////////////////////////
// Sub two arrays inplace version
//////////////////////////////////////////////////////////////////////////////////
    template<class T>
    __global__ void Sub_kernel_I(T* d_o, T* d_i, int n){
        uint blockId = get_blockID();
        uint      id = get_threadID(blockId);
        if (id < n){
            d_o[id] -= d_i[id];
        }
    }
    
    template<class T>
    void Sub_I(T* d_o, T* d_i, int n, cudaStream_t stream){
        dim3 threads(256);
        dim3 grids(iDivUp(n,threads.x));
        checkConfig(grids);
        Sub_kernel_I<<<grids, threads,0,stream>>>(d_o, d_i, n);
    }

/////////////////////////////////////////////////////////////////////////////////
// Multiply a constant
// It is possible that we have the array type is a complex type
// that multiply operator is defined 
//////////////////////////////////////////////////////////////////////////////////
    template<class T, class T2>
    __global__ void MulC_kernel(T2* d_o, const T2* d_i, T c, int n){
        uint blockId = get_blockID();
        uint      id = get_threadID(blockId);
        if (id < n){
            d_o[id] = d_i[id] * c;
        }
    }
    template<class T, class T2>
    void MulC(T2* d_o, const T2* d_i, T c, int n, cudaStream_t stream){
        dim3 threads(256);
        dim3 grids(iDivUp(n,threads.x));
        checkConfig(grids);
        MulC_kernel<<<grids, threads,0,stream>>>(d_o, d_i, c, n);
    }

/////////////////////////////////////////////////////////////////////////////////
// Multiply a constant inplace version
//////////////////////////////////////////////////////////////////////////////////
    template<class T, class T2> 
    __global__ void MulC_kernel_I(T2* d_o, T c, int n){
        uint blockId = get_blockID();
        uint      id = get_threadID(blockId);
        if (id < n){
            d_o[id] *= c;
        }
    }
    template<class T, class T2> 
    void MulC_I(T2* d_o, T c, int n, cudaStream_t stream){
        dim3 threads(256);
        dim3 grids(iDivUp(n,threads.x));
        checkConfig(grids);
        MulC_kernel_I<<<grids, threads,0,stream>>>(d_o, c, n);
    }
    
/////////////////////////////////////////////////////////////////////////////////
// Multiply two array element by element 
//////////////////////////////////////////////////////////////////////////////////
    template<class T>
    __global__ void Mul_kernel(T* d_o , T* d_i, T* d_i1, int n){
        uint blockId = get_blockID();
        uint      id = get_threadID(blockId);
        if (id < n){
            d_o[id] = d_i[id] * d_i1[id];
        }
    }
    
    template<class T>
    void Mul(T* d_o, T* d_i, T* d_i1, int n, cudaStream_t stream){
        dim3 threads(256);
        dim3 grids(iDivUp(n,threads.x));
        checkConfig(grids);
        Mul_kernel<<<grids, threads,0,stream>>>(d_o, d_i, d_i1, n);
    }

/////////////////////////////////////////////////////////////////////////////////
// Multiply two arrays inplace version
//////////////////////////////////////////////////////////////////////////////////
    template<class T>
    __global__ void Mul_kernel_I(T* d_o, T* d_i, int n){
        uint blockId = get_blockID();
        uint      id = get_threadID(blockId);
        if (id < n){
            d_o[id] *= d_i[id];
        }
    }
    
    template<class T>
    void Mul_I(T* d_o, T* d_i, int n, cudaStream_t stream){
        dim3 threads(256);
        dim3 grids(iDivUp(n,threads.x));
        checkConfig(grids);
        Mul_kernel_I<<<grids, threads,0,stream>>>(d_o, d_i, n);
    }
/////////////////////////////////////////////////////////////////////////////////
// Divide a constant
// It is possible that we have the array type is a complex type
// that multiply operator is defined 
//////////////////////////////////////////////////////////////////////////////////
    template<class T2, class T>
    __global__ void DivC_kernel(T2* d_o, const T2* d_i, T c, int n){
        uint blockId = get_blockID();
        uint      id = get_threadID(blockId);
        if (id < n){
            d_o[id] = d_i[id] / c;
        }
    }
    template<class T2, class T>
    void DivC(T2* d_o, const T2* d_i, T c, int n, cudaStream_t stream){
        dim3 threads(256);
        dim3 grids(iDivUp(n,threads.x));
        checkConfig(grids);
        DivC_kernel<<<grids, threads, 0, stream>>>(d_o, d_i, c, n);
    }

/////////////////////////////////////////////////////////////////////////////////
// Divide a constant inplace version
//////////////////////////////////////////////////////////////////////////////////
    template<class T2, class T> 
    __global__ void DivC_kernel_I(T2* d_o, T c, int n){
        uint blockId = get_blockID();
        uint      id = get_threadID(blockId);
        if (id < n){
            d_o[id] /= c;
        }
    }
    template<class T2, class T> 
    void DivC_I(T2* d_o, T c, int n, cudaStream_t stream){
        dim3 threads(256);
        dim3 grids(iDivUp(n,threads.x));
        checkConfig(grids);
        DivC_kernel_I<<<grids, threads, 0, stream>>>(d_o, c, n);
    }
    
/////////////////////////////////////////////////////////////////////////////////
// Divide two array element by element 
//////////////////////////////////////////////////////////////////////////////////
    template<class T>
    __global__ void Div_kernel(T* d_o , T* d_i, T* d_i1, int n){
        uint blockId = get_blockID();
        uint      id = get_threadID(blockId);
        if (id < n){
            d_o[id] = d_i[id] / d_i1[id];
        }
    }
    
    template<class T>
    void Div(T* d_o, T* d_i, T* d_i1, int n, cudaStream_t stream){
        dim3 threads(256);
        dim3 grids(iDivUp(n,threads.x));
        checkConfig(grids);
        Div_kernel<<<grids, threads, 0, stream>>>(d_o, d_i, d_i1, n);
    }

/////////////////////////////////////////////////////////////////////////////////
// Divide two arrays inplace version
//////////////////////////////////////////////////////////////////////////////////
    template<class T>
    __global__ void Div_kernel_I(T* d_o, T* d_i, int n){
        uint blockId = get_blockID();
        uint      id = get_threadID(blockId);
        if (id < n){
            d_o[id] /= d_i[id];
        }
    }
    
    template<class T>
    void Div_I(T* d_o, T* d_i, int n, cudaStream_t stream){
        dim3 threads(256);
        dim3 grids(iDivUp(n,threads.x));
        checkConfig(grids);
        Div_kernel_I<<<grids, threads, 0, stream>>>(d_o, d_i, n);
    }
/////////////////////////////////////////////////////////////////////////////////
// Trinary function
//////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////
// Absolute value of the difference
//////////////////////////////////////////////////////////////////////////////////
    template<class T>
    __global__ void AbsDiff_kernel(T* d_o , const T* d_i, const T* d_i1, int n){
        uint blockId = get_blockID();
        uint id      = get_threadID(blockId);
        if (id < n){
            d_o[id] = (d_i[id] >= d_i1[id]) ? (d_i[id] - d_i1[id]) : (d_i1[id] - d_i[id]);
        }
    }
    template<class T>
    void AbsDiff(T* d_o, const T* d_i, const T* d_i1, int n, cudaStream_t stream){
        dim3 threads(256);
        dim3 grids(iDivUp(n,threads.x));
        checkConfig(grids);
    
        AbsDiff_kernel<<<grids, threads,0,stream>>>(d_o, d_i, d_i1, n);
    }

/////////////////////////////////////////////////////////////////////////////////
// Absolute value of the difference
//////////////////////////////////////////////////////////////////////////////////

    template<class T>
    __global__ void AbsDiff_kernel_I(T* d_o, const T* d_i, int n){
        uint blockId = get_blockID();
        uint id      = get_threadID(blockId);
        if (id < n){
            d_o[id] = (d_o[id] >= d_i[id]) ? d_o[id] - d_i[id] : d_o[id] = d_i[id] - d_o[id];
        }
    }
    
    template<class T>
    void AbsDiff_I(T* d_o,const  T* d_i, int n, cudaStream_t stream){
        dim3 threads(256);
        dim3 grids(iDivUp(n,threads.x));
        checkConfig(grids);
        AbsDiff_kernel_I<<<grids, threads,0,stream>>>(d_o, d_i, n);
    }

    /////////////////////////////////////////////////////////////////////////////////
// Square value of the difference
//////////////////////////////////////////////////////////////////////////////////
    template<class T>
    __global__ void SqrDiff_kernel(T* d_o , const T* d_i, const T* d_i1, int n){
        uint blockId = get_blockID();
        uint id      = get_threadID(blockId);
        if (id < n){
            d_o[id] = (d_i[id] - d_i1[id]) * (d_i[id] - d_i1[id]);
        }
    }
    template<class T>
    void SqrDiff(T* d_o, const T* d_i, const T* d_i1, int n, cudaStream_t stream){
        dim3 threads(256);
        dim3 grids(iDivUp(n,threads.x));
        checkConfig(grids);
    
        SqrDiff_kernel<<<grids, threads,0,stream>>>(d_o, d_i, d_i1, n);
    }

/////////////////////////////////////////////////////////////////////////////////
// Square value of the difference
//////////////////////////////////////////////////////////////////////////////////

    template<class T>
    __global__ void SqrDiff_kernel_I(T* d_o, const T* d_i, int n){
        uint blockId = get_blockID();
        uint id      = get_threadID(blockId);
        if (id < n){
            d_o[id] = (d_o[id] - d_i[id]) * (d_o[id] - d_i[id]);
        }
    }
    
    template<class T>
    void SqrDiff_I(T* d_o,const  T* d_i, int n, cudaStream_t stream){
        dim3 threads(256);
        dim3 grids(iDivUp(n,threads.x));
        checkConfig(grids);
        SqrDiff_kernel_I<<<grids, threads,0,stream>>>(d_o, d_i, n);
    }


/////////////////////////////////////////////////////////////////////////////////
// d_o = a * b * c
//////////////////////////////////////////////////////////////////////////////////
   template<class T1, class T> 
    __global__ void MulMulC_kernel(T1* d_o,T1* a, T* b, T c, int n){
        uint blockId = get_blockID();
        uint      id = get_threadID(blockId);
        if (id <n){
            d_o[id] = a[id] * (b[id] * c);
        }
    }
    template<class T1, class T>
    void MulMulC(T1* d_o, T1* a, T* b, T c, int n, cudaStream_t stream){
        dim3 threads(256);
        dim3 grids(iDivUp(n, threads.x));
        checkConfig(grids);
        MulMulC_kernel<<<grids, threads,0,stream>>>(d_o, a, b, c, n);
    }

    template<class T1, class T> 
    __global__ void MulMulC_I_kernel(T1* a, T* b, T c, int n){
        uint blockId = get_blockID();
        uint      id = get_threadID(blockId);
        if (id <n){
            a[id] *= (b[id] * c);
        }
    }

    template<class T1, class T>
    void MulMulC_I(T1* a, T* b, T c, int n, cudaStream_t stream){
        dim3 threads(256);
        dim3 grids(iDivUp(n, threads.x));
        checkConfig(grids);
        MulMulC_I_kernel<<<grids, threads,0,stream>>>(a, b, c, n);
    }

/////////////////////////////////////////////////////////////////////////////////
// Multiply three arrays together
//////////////////////////////////////////////////////////////////////////////////
    template<class T1, class T>
    __global__ void MulMul_kernel(T1* d_o,T1* a, T* b, T* c, int n){
        uint blockId = get_blockID();
        uint      id = get_threadID(blockId);
        if (id <n){
            d_o[id] = a[id] * (b[id] * c[id]);
        }
    }

    template<class T1, class T>
    void MulMul(T1* d_o, T1* a, T* b, T* c, int n, cudaStream_t stream){
        dim3 threads(256);
        dim3 grids(iDivUp(n,threads.x));
        checkConfig(grids);
        MulMul_kernel<<<grids, threads,0,stream>>>(d_o, a, b, c, n);
    }

/////////////////////////////////////////////////////////////////////////////////
// Multiply three arrays together inplace version
//////////////////////////////////////////////////////////////////////////////////
    template<class T1, class T>
    __global__ void MulMul_I_kernel(T1* a, T* b, T* c, int n){
        uint blockId = get_blockID();
        uint      id = get_threadID(blockId);
        if (id <n){
            T1 va =  a[id];
            va   *=  (b[id] * c[id]);
            a[id] = va;
        }
    }

    template<class T1, class T>
    void MulMul_I(T1* a, T* b, T* c, int n, cudaStream_t stream){
        dim3 threads(256);
        dim3 grids(iDivUp(n,threads.x));
        checkConfig(grids);
        MulMul_I_kernel<<<grids, threads,0,stream>>>(a, b, c, n);
    }


/////////////////////////////////////////////////////////////////////////////////
// Add two array together and divide by the third array
//////////////////////////////////////////////////////////////////////////////////

    template<class T1, class T>
    __global__ void AddDiv_kernel(T1* d_o, T1* a, T1* b, T* c, int n){
        uint blockId = get_blockID();
        uint      id = get_threadID(blockId);
        if (id < n)
            d_o[id] = (a[id] + b[id]) / c[id];
    }

    template<class T1, class T>
    void AddDiv(T1* d_o, T1* a, T1* b, T* c, int n, cudaStream_t stream){
        dim3 threads(256);
        dim3 grids(iDivUp(n,threads.x));
        checkConfig(grids);
        AddDiv_kernel<<<grids, threads,0,stream>>>(d_o, a, b, c, n);
    }

/////////////////////////////////////////////////////////////////////////////////
// Add two array together and divide by the third array inplace 
//////////////////////////////////////////////////////////////////////////////////
    template<class T1, class T>
    __global__ void AddDiv_I_kernel(T1* d_o, T1* b, T* c, int n){
        uint blockId = get_blockID();
        uint      id = get_threadID(blockId);
        if (id < n)
            d_o[id] = (d_o[id] + b[id]) / c[id];
    }

    template<class T1, class T>
    void AddDiv_I(T1* d_o, T1* b, T* c, int n, cudaStream_t stream){
        dim3 threads(256);
        dim3 grids(iDivUp(n,threads.x));
        checkConfig(grids);
        AddDiv_I_kernel<<<grids, threads,0,stream>>>(d_o, b, c, n);
    }

/////////////////////////////////////////////////////////////////////////////////
// Sub two arrays and divide by the third array
//////////////////////////////////////////////////////////////////////////////////
    template<class T1, class T>
    __global__ void SubDiv_kernel(T1* d_o, T1* a, T1* b, T* c, int n){
        uint blockId = get_blockID();
        uint      id = get_threadID(blockId);
        if (id < n)
            d_o[id] = (a[id] - b[id]) / c[id];
    }

    template<class T1, class T>
    void SubDiv(T1* d_o, T1* a, T1* b, T* c, int n, cudaStream_t stream){
        dim3 threads(256);
        dim3 grids(iDivUp(n,threads.x));
        checkConfig(grids);
        SubDiv_kernel<<<grids, threads,0,stream>>>(d_o, a, b, c, n);
    }

/////////////////////////////////////////////////////////////////////////////////
// Sub two arrays and divide by the third array inplace
//////////////////////////////////////////////////////////////////////////////////
    template<class T1, class T>
    __global__ void SubDiv_I_kernel(T1* d_o, T1* b, T* c, int n){
        uint blockId = get_blockID();
        uint      id = get_threadID(blockId);
        if (id < n)
            d_o[id] = (d_o[id] - b[id]) / c[id];
    }

    template<class T1, class T>
    void SubDiv_I(T1* d_o, T1* b, T* c, int n, cudaStream_t stream){
        dim3 threads(256);
        dim3 grids(iDivUp(n,threads.x));
        checkConfig(grids);
        SubDiv_I_kernel<<<grids, threads,0,stream>>>(d_o, b, c, n);
    }

/////////////////////////////////////////////////////////////////////////////////
// Multiply two array together and add the third one
//////////////////////////////////////////////////////////////////////////////////
    template<class T1, class T>
    __global__ void MulAdd_kernel(T1* d_o, T1* a, T* b, T1* c, int n){
        uint blockId = get_blockID();
        uint      id = get_threadID(blockId);
        if (id < n)
            d_o[id] = a[id] * b[id] + c[id];
    }

    template<class T1, class T>
    void MulAdd(T1* d_o, T1* a, T* b, T1* c, int n, cudaStream_t stream){
        dim3 threads(256);
        dim3 grids(iDivUp(n,threads.x));
        checkConfig(grids);
        MulAdd_kernel<<<grids, threads,0,stream>>>(d_o, a, b, c, n);
    }


/////////////////////////////////////////////////////////////////////////////////
// Multiply two array together and add the third one inplace version
//////////////////////////////////////////////////////////////////////////////////
    template<class T1, class T>
    __global__ void MulAdd_I_kernel(T1* d_o, T* b, T1* c, int n){
        uint blockId = get_blockID();
        uint      id = get_threadID(blockId);
        if (id < n)
            d_o[id] = d_o[id] * b[id] + c[id];
    }

    template<class T1, class T>
    void MulAdd_I(T1* d_o, T* b, T1* c, int n, cudaStream_t stream){
        dim3 threads(256);
        dim3 grids(iDivUp(n,threads.x));
        checkConfig(grids);
        MulAdd_I_kernel<<<grids, threads,0,stream>>>(d_o, b, c, n);
    }


/////////////////////////////////////////////////////////////////////////////////
// Multiply two array together and sub the third one 
//////////////////////////////////////////////////////////////////////////////////
    template<class T1, class T>
    __global__ void MulSub_kernel(T1* d_o, T1* a, T* b, T1* c, int n){
        uint blockId = get_blockID();
        uint      id = get_threadID(blockId);
        if (id < n)
            d_o[id] = a[id] * b[id] - c[id];
    }

    template<class T1, class T>
    void MulSub(T1* d_o, T1* a, T* b, T1* c, int n, cudaStream_t stream){
        dim3 threads(256);
        dim3 grids(iDivUp(n,threads.x));
        checkConfig(grids);
        MulSub_kernel<<<grids, threads,0,stream>>>(d_o, a, b, c, n);
    }

/////////////////////////////////////////////////////////////////////////////////
// Multiply two array together and sub the third one inplace version
//////////////////////////////////////////////////////////////////////////////////

    template<class T1, class T>
    __global__ void MulSub_I_kernel(T1* d_o, T* b, T1* c, int n){
        uint blockId = get_blockID();
        uint      id = get_threadID(blockId);
        if (id < n)
            d_o[id] = d_o[id] * b[id] - c[id];
    }

    template<class T1, class T>
    void MulSub_I(T1* d_o, T* b, T1* c, int n, cudaStream_t stream){
        dim3 threads(256);
        dim3 grids(iDivUp(n,threads.x));
        checkConfig(grids);
        MulSub_I_kernel<<<grids, threads,0,stream>>>(d_o, b, c, n);
    }


/////////////////////////////////////////////////////////////////////////////////
// Add two arrays and multiply by the third one
//////////////////////////////////////////////////////////////////////////////////
    template<class T1, class T>
    __global__ void AddMul_kernel(T1* d_o, T1* a, T1* b, T* c, int n){
        uint blockId = get_blockID();
        uint      id = get_threadID(blockId);
        if (id < n)
            d_o[id] = (a[id] + b[id]) * c[id];
    }

    template<class T1, class T>
    void AddMul(T1* d_o, T1* a, T1* b, T* c, int n, cudaStream_t stream){
        dim3 threads(256);
        dim3 grids(iDivUp(n,threads.x));
        checkConfig(grids);
        AddMul_kernel<<<grids, threads,0,stream>>>(d_o, a, b, c, n);
    }

/////////////////////////////////////////////////////////////////////////////////
// Add two array together and multiply by the third one inplace version
//////////////////////////////////////////////////////////////////////////////////
    template<class T1, class T>
    __global__ void AddMul_I_kernel(T1* d_o, T1* b, T* c, int n){
        uint blockId = get_blockID();
        uint      id = get_threadID(blockId);
        if (id < n)
            d_o[id] = (d_o[id] + b[id]) * c[id];
    }

    template<class T1, class T>
    void AddMul_I(T1* d_o, T1* b, T* c, int n, cudaStream_t stream){
        dim3 threads(256);
        dim3 grids(iDivUp(n,threads.x));
        checkConfig(grids);
        AddMul_I_kernel<<<grids, threads,0,stream>>>(d_o, b, c, n);
    }

/////////////////////////////////////////////////////////////////////////////////
// Sub two arrays and multiply by the third one
//////////////////////////////////////////////////////////////////////////////////

    template<class T1, class T>
    __global__ void SubMul_kernel(T1* d_o, T1* a, T1* b, T* c, int n){
        uint blockId = get_blockID();
        uint      id = get_threadID(blockId);
        if (id < n)
            d_o[id] = (a[id] - b[id]) * c[id];
    }

    template<class T1, class T>
    void SubMul(T1* d_o, T1* a, T1* b, T* c, int n, cudaStream_t stream){
        dim3 threads(256);
        dim3 grids(iDivUp(n,threads.x));
        checkConfig(grids);
        SubMul_kernel<<<grids, threads,0,stream>>>(d_o, a, b, c, n);
    }

/////////////////////////////////////////////////////////////////////////////////
// Sub two arrays and multiply by the third one inplace version
//////////////////////////////////////////////////////////////////////////////////

    template<class T1, class T>
    __global__ void SubMul_I_kernel(T1* d_o, T1* b, T* c, int n){
        uint blockId = get_blockID();
        uint      id = get_threadID(blockId);
        if (id < n)
            d_o[id] = (d_o[id] - b[id]) * c[id];
    }

    template<class T1, class T>
    void SubMul_I(T1* d_o, T1* b, T* c, int n, cudaStream_t stream){
        dim3 threads(256);
        dim3 grids(iDivUp(n,threads.x));
        checkConfig(grids);
        SubMul_I_kernel<<<grids, threads,0,stream>>>(d_o, b, c, n);
    }

/////////////////////////////////////////////////////////////////////////////////
// Add an array by a constant then multiply by other constant
// Used with normalized function
//////////////////////////////////////////////////////////////////////////////////

    template<class T>
    __global__ void AddCMulC_kernel(T* d_odata, T*d_idata, T a, T b, int n){
        uint blockId = get_blockID();
        uint      id = get_threadID(blockId);
        if (id < n)
            d_odata[id] = (d_idata[id] + a) * b;
    }

    template<class T>
    void AddCMulC(T* d_odata, T* d_idata, T a, T b, int n, cudaStream_t stream){
        dim3 threads(256);
        dim3 grids(iDivUp(n,threads.x));
        checkConfig(grids);
        AddCMulC_kernel<<<grids, threads,0,stream>>>(d_odata, d_idata, a, b, n);
    }


    template<class T>
    __global__ void AddCMulC_I_kernel(T* d_data, T a, T b, int n){
        uint blockId = get_blockID();
        uint      id = get_threadID(blockId);
        if (id < n)
            d_data[id] = (d_data[id] + a) * b;
    }

    template<class T>
    void AddCMulC_I(T* d_data, T a, T b, int n, cudaStream_t stream){
        dim3 threads(256);
        dim3 grids(iDivUp(n,threads.x));
        checkConfig(grids);
        AddCMulC_I_kernel<<<grids, threads,0,stream>>>(d_data, a, b, n);
    }

/////////////////////////////////////////////////////////////////////////////////
// Multiply an array by a constant and add the second one
//////////////////////////////////////////////////////////////////////////////////

    template<class T1, class T>
    __global__ void MulCAdd_kernel(T1* d_o, T1* a, T b, T1* c, int n){
        uint blockId = get_blockID();
        uint      id = get_threadID(blockId);
        if (id < n)
            d_o[id] = a[id] * b + c[id];
    }

    template<class T1, class T>
    void MulCAdd(T1* d_o, T1* a, T b, T1* c, int n, cudaStream_t stream){
        dim3 threads(256);
        dim3 grids(iDivUp(n,threads.x));
        checkConfig(grids);
        MulCAdd_kernel<<<grids, threads,0,stream>>>(d_o, a, b, c, n);
    }

/////////////////////////////////////////////////////////////////////////////////
// Multiply an array by a constant and add the second one inplace version
//////////////////////////////////////////////////////////////////////////////////

    template<class T1, class T>
    __global__ void MulCAdd_I_kernel(T1* d_o, T b, T1* c, int n){
        uint blockId = get_blockID();
        uint      id = get_threadID(blockId);
        if (id < n)
            d_o[id] = d_o[id] * b + c[id];
    }

    template<class T1, class T>
    void MulCAdd_I(T1* d_o, T b, T1* c, int n, cudaStream_t stream){
        dim3 threads(256);
        dim3 grids(iDivUp(n,threads.x));
        checkConfig(grids);
        MulCAdd_I_kernel<<<grids, threads,0,stream>>>(d_o, b, c, n);
    }

/////////////////////////////////////////////////////////////////////////////////
// Multiply an array by a constant and add the second constant 
//////////////////////////////////////////////////////////////////////////////////

    template<class T1, class T>
    __global__ void MulCAddC_kernel(T1* d_o, T1* a, T b, T1 c, int n){
        uint blockId = get_blockID();
        uint      id = get_threadID(blockId);
        if (id < n)
            d_o[id] = a[id] * b + c;
    }

    template<class T1, class T>
    void MulCAddC(T1* d_o, T1* a, T b, T1 c, int n, cudaStream_t stream){
        dim3 threads(256);
        dim3 grids(iDivUp(n,threads.x));
        checkConfig(grids);
        MulCAddC_kernel<<<grids, threads,0,stream>>>(d_o, a, b, c, n);
    }

/////////////////////////////////////////////////////////////////////////////////
// Multiply an array by a constant and add the second constant inplace version
//////////////////////////////////////////////////////////////////////////////////

    template<class T1, class T>
    __global__ void MulCAddC_I_kernel(T1* d_o, T b, T1 c, int n){
        uint blockId = get_blockID();
        uint      id = get_threadID(blockId);
        if (id < n)
            d_o[id] = d_o[id] * b + c;
    }

    template<class T1, class T>
    void MulCAddC_I(T1* d_o, T b, T1 c, int n, cudaStream_t stream){
        dim3 threads(256);
        dim3 grids(iDivUp(n,threads.x));
    
        checkConfig(grids);
        MulCAddC_I_kernel<<<grids, threads,0,stream>>>(d_o, b, c, n);
    }


/////////////////////////////////////////////////////////////////////////////////
// Multiply an array by a constant and sub the second one
//////////////////////////////////////////////////////////////////////////////////

    template<class T1, class T>
    __global__ void MulCSub_kernel(T1* d_o, T1* a, T b, T1* c, int n){
        uint blockId = get_blockID();
        uint      id = get_threadID(blockId);
        if (id < n)
            d_o[id] = a[id] * b - c[id];
    }

    template<class T1, class T>
    void MulCSub(T1* d_o, T1* a, T b, T1* c, int n, cudaStream_t stream){
        dim3 threads(256);
        dim3 grids(iDivUp(n,threads.x));
        checkConfig(grids);
        MulCSub_kernel<<<grids, threads,0,stream>>>(d_o, a, b, c, n);
    }

/////////////////////////////////////////////////////////////////////////////////
// Multiply an array by a constant and sub the second one inplace version
//////////////////////////////////////////////////////////////////////////////////

    template<class T1, class T>
    __global__ void MulCSub_I_kernel(T1* d_o, T b, T1* c, int n){
        uint blockId = get_blockID();
        uint      id = get_threadID(blockId);
        if (id < n)
            d_o[id] = d_o[id] * b - c[id];
    }

    template<class T1, class T>
    void MulCSub_I(T1* d_o, T b, T1* c, int n, cudaStream_t stream){
        dim3 threads(256);
        dim3 grids(iDivUp(n,threads.x));
        checkConfig(grids);
        MulCSub_I_kernel<<<grids, threads,0,stream>>>(d_o, b, c, n);
    }


/////////////////////////////////////////////////////////////////////////////////
// Add two array then multiply by a constant 
//////////////////////////////////////////////////////////////////////////////////
    template<class T1, class T>
    __global__ void AddMulC_kernel(T1* d_o, T1* a, T1* b, T c, int n){
        uint blockId = get_blockID();
        uint      id = get_threadID(blockId);
        if (id < n)
            d_o[id] = (a[id] + b[id]) * c;
    }

    template<class T1, class T>
    void AddMulC(T1* d_o, T1* a, T1* b, T c, int n, cudaStream_t stream){
        dim3 threads(256);
        dim3 grids(iDivUp(n,threads.x));
        checkConfig(grids);
        AddMulC_kernel<<<grids, threads,0,stream>>>(d_o, a, b, c, n);
    }

/////////////////////////////////////////////////////////////////////////////////
// Add two array then multiply by a constant inplace version
//////////////////////////////////////////////////////////////////////////////////
    template<class T1, class T>
    __global__ void AddMulC_I_kernel(T1* d_o, T1* b, T c, int n){
        uint blockId = get_blockID();
        uint      id = get_threadID(blockId);
        if (id < n)
            d_o[id] = (d_o[id] + b[id]) * c;
    }

    template<class T1, class T>
    void AddMulC_I(T1* d_o, T1* b, T c, int n, cudaStream_t stream){
        dim3 threads(256);
        dim3 grids(iDivUp(n,threads.x));
        checkConfig(grids);
        AddMulC_I_kernel<<<grids, threads,0,stream>>>(d_o, b, c, n);
    }

/////////////////////////////////////////////////////////////////////////////////
// Sub two arrays then multiply by a constant inplace version
//////////////////////////////////////////////////////////////////////////////////


    template<class T1, class T>
    __global__ void SubMulC_kernel(T1* d_o, T1* a, T1* b, T c, int n){
        uint blockId = get_blockID();
        uint      id = get_threadID(blockId);
        if (id < n)
            d_o[id] = (a[id] - b[id]) * c;
    }

    template<class T1, class T>
    void SubMulC(T1* d_o, T1* a, T1* b, T c, int n, cudaStream_t stream){
        dim3 threads(256);
        dim3 grids(iDivUp(n,threads.x));
        checkConfig(grids);
        SubMulC_kernel<<<grids, threads,0,stream>>>(d_o, a, b, c, n);
    }


/////////////////////////////////////////////////////////////////////////////////
// Sub two arrays then multiply by a constant inplace version
//////////////////////////////////////////////////////////////////////////////////

    template<class T1, class T>
    __global__ void SubMulC_I_kernel(T1* d_o, T1* b, T c, int n){
        uint blockId = get_blockID();
        uint      id = get_threadID(blockId);
        if (id < n)
            d_o[id] = (d_o[id] - b[id]) * c;
    }

    template<class T1, class T>
    void SubMulC_I(T1* d_o, T1* b, T c, int n, cudaStream_t stream){
        dim3 threads(256);
        dim3 grids(iDivUp(n,threads.x));
        checkConfig(grids);
        SubMulC_I_kernel<<<grids, threads,0,stream>>>(d_o, b, c, n);
    }




/////////////////////////////////////////////////////////////////////////////////
// Add an arrays with another array that multiply by a constant 
//  d_data = d_data + d_a * c
//////////////////////////////////////////////////////////////////////////////////

    template<class T>
    __global__ void Add_MulC_kernel_I(T* d_data, T* d_a, T c, int n) {
        uint blockId = get_blockID();
        uint      id = get_threadID(blockId);
        if (id < n)
            d_data[id] += d_a[id] * c;
    }

    template<class T>
    void Add_MulC_I(T* d_data, T* d_a, T c, int n, cudaStream_t stream) {
        dim3 threads(256);
        dim3 grids(iDivUp(n, threads.x));
        checkConfig(grids);
        Add_MulC_kernel_I<<<grids, threads,0,stream>>>(d_data, d_a, c, n);
    }

/////////////////////////////////////////////////////////////////////////////////
// Add an arrays with another array that multiply by a constant 
//  d_o = d_i + d_a * c
//////////////////////////////////////////////////////////////////////////////////

    template<class T>
    __global__ void Add_MulC_kernel(T* d_o, T* d_i, T* d_a, T c, int n) {
        uint blockId = get_blockID();
        uint      id = get_threadID(blockId);
        if (id < n)
            d_o[id] = d_i[id] + d_a[id] * c;
    }

    template<class T>
    void Add_MulC(T* d_o, T* d_i, T* d_a, T c, int n, cudaStream_t stream) {
        dim3 threads(256);
        dim3 grids(iDivUp(n, threads.x));
        checkConfig(grids);
        Add_MulC_kernel<<<grids, threads,0,stream>>>(d_o, d_i, d_a, c, n);
    }

/////////////////////////////////////////////////////////////////////////////////
// Add an arrays with another array that multiply by an array
//  d_data = d_data + d_a * d_c
//////////////////////////////////////////////////////////////////////////////////

    template<class T>
    __global__ void Add_Mul_kernel_I(T* d_data, T* d_a, T* d_c, int n) {
        uint blockId = get_blockID();
        uint      id = get_threadID(blockId);
        if (id < n)
            d_data[id] += d_a[id] * d_c[id];
    }

    template<class T>
    void Add_Mul_I(T* d_data, T* d_a, T* d_c, int n, cudaStream_t stream) {
        dim3 threads(256);
        dim3 grids(iDivUp(n, threads.x));
        checkConfig(grids);
        Add_Mul_kernel_I<<<grids, threads,0,stream>>>(d_data, d_a, d_c, n);
    }

/////////////////////////////////////////////////////////////////////////////////
// Add an arrays with another array that multiply by an array 
//  d_o = d_i + d_a * d_c
//////////////////////////////////////////////////////////////////////////////////

    template<class T>
    __global__ void Add_Mul_kernel(T* d_o,T* d_i, T* d_a, T* d_c, int n) {
        uint blockId = get_blockID();
        uint      id = get_threadID(blockId);
        if (id < n)
            d_o[id] = d_i[id] + d_a[id] * d_c[id];
    }

    template<class T>
    void Add_Mul(T* d_o, T* d_i, T* d_a, T* d_c, int n, cudaStream_t stream) {
        dim3 threads(256);
        dim3 grids(iDivUp(n, threads.x));
        checkConfig(grids);
        Add_Mul_kernel<<<grids, threads,0,stream>>>(d_o, d_i, d_a, d_c, n);
    }


/////////////////////////////////////////////////////////////////////////////////
// Quadary function
//////////////////////////////////////////////////////////////////////////////////
    template<class T2, class T>
    __global__ void MulC_Add_MulC_kernel(T2* d_o, T2* d_i, T a, T2* d_i1, T b,  int n){
        uint blockId = get_blockID();
        uint      id = get_threadID(blockId);
        if (id < n){
            d_o[id] = d_i[id] * a + d_i1[id] * b;
        }
    }
    
    template<class T2, class T>
    void MulC_Add_MulC(T2* d_o, T2* d_i, T a, T2* d_i1, T b, int n, cudaStream_t stream){
        dim3 threads(256);
        dim3 grids(iDivUp(n,threads.x));
        checkConfig(grids);
        MulC_Add_MulC_kernel<T2, T><<<grids, threads,0,stream>>>(d_o, d_i, a, d_i1, b, n);
    }

    template<class T2, class T>
    __global__ void MulC_Add_MulC_kernel_I(T2* d_o, T a, T2* d_i, T b,  int n){
        uint blockId = get_blockID();
        uint      id = get_threadID(blockId);
        if (id < n){
            d_o[id] = d_o[id] * a + d_i[id] * b;
        }
    }

    template<class T2, class T>
    void MulC_Add_MulC_I(T2* d_o, T a, T2* d_i, T b, int n, cudaStream_t stream){
        dim3 threads(256);
        dim3 grids(iDivUp(n, threads.x));
        checkConfig(grids);
        MulC_Add_MulC_kernel_I<T2, T><<<grids, threads,0,stream>>>(d_o, a, d_i, b, n);
    }

////////////////////////////////////////////////////////////////////////////////
//
////////////////////////////////////////////////////////////////////////////////
    template<class T>
    __global__ void Add_AddMulC_kernel(T* d_o, T* d_a, T* d_b, T* d_c, T d, int n){
        uint blockId = get_blockID();
        uint      id = get_threadID(blockId);
        if (id < n){
            d_o[id] = d_a[id] + (d_b[id] + d_c[id]) * d;
        }
    }

    template<class T>
    void Add_AddMulC(T* d_o, T* d_a, T* d_b, T* d_c, T d, int n, cudaStream_t stream){
        dim3 threads(256);
        dim3 grids(iDivUp(n, threads.x));
        checkConfig(grids);
        Add_AddMulC_kernel<T><<<grids, threads,0,stream>>>(d_o, d_a, d_b, d_c, d, n);
    }

////////////////////////////////////////////////////////////////////////////////
//
////////////////////////////////////////////////////////////////////////////////
    template<class T>
    __global__ void Add_AddMulC_I_kernel(T* d_a, T* d_b, T* d_c, T d, int n){
        uint blockId = get_blockID();
        uint      id = get_threadID(blockId);
        if (id < n){
            d_a[id] += (d_b[id] + d_c[id]) * d;
        }
    }

    template<class T>
    void Add_AddMulC_I(T* d_a, T* d_b, T* d_c, T d, int n, cudaStream_t stream){
        dim3 threads(256);
        dim3 grids(iDivUp(n, threads.x));
        checkConfig(grids);
        Add_AddMulC_I_kernel<T><<<grids, threads,0,stream>>>(d_a, d_b, d_c, d, n);
    }

////////////////////////////////////////////////////////////////////////////////
//
////////////////////////////////////////////////////////////////////////////////

    template<class T>
    __global__ void Add_SubMulC_kernel(T* d_o, const T* d_a, const T* d_b, const T* d_c, T d, int n){
        uint blockId = get_blockID();
        uint      id = get_threadID(blockId);
        if (id < n){
            d_o[id] = d_a[id] + (d_b[id] - d_c[id]) * d;
        }
    }

    template<class T>
    void Add_SubMulC(T* d_o, const T* d_a, const T* d_b, const T* d_c, T d, int n, cudaStream_t stream){
        dim3 threads(256);
        dim3 grids(iDivUp(n, threads.x));
        checkConfig(grids);
        Add_SubMulC_kernel<T><<<grids, threads,0,stream>>>(d_o, d_a, d_b, d_c, d, n);
    }

    template<class T>
    __global__ void Add_SubMulC_I_kernel(T* d_o, const T* d_b, const T* d_c, T d, int n){
        uint blockId = get_blockID();
        uint      id = get_threadID(blockId);
        if (id < n){
            d_o[id] += (d_b[id] - d_c[id]) * d;
        }
    }

    template<class T>
    void Add_SubMulC_I(T* d_o, const T* d_b, const T* d_c, T d, int n, cudaStream_t stream){
        dim3 threads(256);
        dim3 grids(iDivUp(n, threads.x));
        checkConfig(grids);
        Add_SubMulC_I_kernel<T><<<grids, threads,0,stream>>>(d_o, d_b, d_c, d, n);
    }

////////////////////////////////////////////////////////////////////////////////
//
////////////////////////////////////////////////////////////////////////////////
    template<class T>
    __global__ void Add_MulMulC_kernel(T* d_o, const T* d_a, const T* d_b, const T* d_c, T d, int n){
        uint blockId = get_blockID();
        uint      id = get_threadID(blockId);
        if (id < n){
            d_o[id] = d_a[id] + (d_b[id] * d_c[id]) * d;
        }
    }

    template<class T>
    void Add_MulMulC(T* d_o, const T* d_a, const T* d_b, const T* d_c, T d, int n, cudaStream_t stream){
        dim3 threads(256);
        dim3 grids(iDivUp(n, threads.x));
        checkConfig(grids);
        Add_MulMulC_kernel<T><<<grids, threads,0,stream>>>(d_o, d_a, d_b, d_c, d, n);
    }

////////////////////////////////////////////////////////////////////////////////
// d_o = d_o + (d_b * d_c * d)
////////////////////////////////////////////////////////////////////////////////
    template<class T>
    __global__ void Add_MulMulC_I_kernel(T* d_o, const T* d_b, const T* d_c, T d, int n){
        uint blockId = get_blockID();
        uint      id = get_threadID(blockId);
        if (id < n){
            d_o[id] += (d_b[id] * d_c[id]) * d;
        }
    }

    template<class T>
    void Add_MulMulC_I(T* d_o, const T* d_b, const T* d_c, T d, int n, cudaStream_t stream){
        dim3 threads(256);
        dim3 grids(iDivUp(n, threads.x));
        checkConfig(grids);
        Add_MulMulC_I_kernel<T><<<grids, threads,0,stream>>>(d_o, d_b, d_c, d, n);
    }

////////////////////////////////////////////////////////////////////////////////
//
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// d_o = (d_i + a) * b + c
////////////////////////////////////////////////////////////////////////////////

    template<class T>
    __global__ void AddCMulCAddC_kernel(T* d_o, const T* d_i, T a, T b, T c, int n){
        uint blockId = get_blockID();
        uint id      = get_threadID(blockId);
        if (id < n)
            d_o[id] = (d_i[id] + a)*b + c;
    }

    template<class T>
    void AddCMulCAddC(T* d_o, const T* d_i, T a, T b, T c, int n, cudaStream_t stream){
        dim3 threads(256);
        dim3 grids(iDivUp(n, threads.x));
        checkConfig(grids);
        AddCMulCAddC_kernel<<<grids, threads, 0, stream>>>(d_o, d_i, a, b, c, n);
    }

////////////////////////////////////////////////////////////////////////////////
// d_o = (d_o + a) * b + c
////////////////////////////////////////////////////////////////////////////////
    template<class T>
    __global__ void AddCMulCAddC_I_kernel(T* d_o, T a, T b, T c, int n){
        uint blockId = get_blockID();
        uint id      = get_threadID(blockId);
        if (id < n)
            d_o[id] = (d_o[id] + a) * b + c;
    }

    template<class T> void AddCMulCAddC_I(T* d_o, T a, T b, T c, int n, cudaStream_t stream){
        dim3 threads(256);
        dim3 grids(iDivUp(n, threads.x));
        checkConfig(grids);
        AddCMulCAddC_I_kernel<<<grids, threads, 0, stream>>>(d_o, a, b, c, n);
    }

////////////////////////////////////////////////////////////////////////////////
// Reverse the order of an array 
////////////////////////////////////////////////////////////////////////////////
    template<typename T>
    __global__ void ReverseOrder_kernel(T* d_o, int n){
        uint blockId = get_blockID();
        uint id      = get_threadID(blockId);
        if (id < n){
            d_o[id] = fetch( n - 1 - id, (T*)NULL);
        }
    }

    template<typename T>
    void ReverseOrder(T* d_o, const T* d_i, int n, cudaStream_t stream){
        dim3 threads(256);
        dim3 grids(iDivUp(n, threads.x));
        checkConfig(grids);

        cache_bind(d_i);
        ReverseOrder_kernel<<<grids, threads, 0, stream>>>(d_o, n);
    }



////////////////////////////////////////////////////////////////////////////////
// Convert one array from fixed point presentation to floating point
////////////////////////////////////////////////////////////////////////////////
    __global__ void fixedToFloating_kernel(float* d_dst, int* d_src, int n)
    {
        uint blockId = get_blockID();
        uint id      = get_threadID(blockId);

        if (id < n){
            d_dst[id] = S2n20(d_src[id]);
        }
    }

    void FixedToFloating(float* d_dst, int* d_src, unsigned int n, cudaStream_t stream)
    {
        dim3 threads(256);
        dim3 grids(iDivUp(n, threads.x));
        checkConfig(grids);
        fixedToFloating_kernel<<<grids, threads, 0, stream>>>(d_dst, d_src, n);
    }

    __global__ void fixedToFloatingUnnomalized_kernel(float* d_dst, int* d_src, float c, int n)
    {
        uint blockId = get_blockID();
        uint id      = get_threadID(blockId);

        if (id < n){
            d_dst[id] = S2n20(d_src[id]) * c;
        }
    }

    void FixedToFloatingUnnomalized(float* d_dst, int* d_src, float c, unsigned int n, cudaStream_t stream)
    {
        dim3 threads(256);
        dim3 grids(iDivUp(n, threads.x));
        checkConfig(grids);
        fixedToFloatingUnnomalized_kernel<<<grids, threads, 0, stream>>>(d_dst, d_src, c, n);
    }

/////////////////////////////////////////////////////////////////////////////////
// d_data = d_data - eps * d_var
//////////////////////////////////////////////////////////////////////////////////
    template<class T>
    void EpsUpdate(T* d_data, T* d_var, T eps, int n, cudaStream_t stream)
    {
        Add_MulC_I(d_data, d_var, -eps, n, stream);
    }

////////////////////////////////////////////////////////////////////////////////
// Interpolate the image using the linear intepolation 
////////////////////////////////////////////////////////////////////////////////
    __global__ void Interpolate_kernel(float* d_o, float* d_i0, float* d_i1, float t, int n){
        uint blockId = get_blockID();
        uint      id = get_threadID(blockId);

        if (id < n){
            d_o[id] = d_i0[id] * (1 - t) + t * d_i1[id];
        }
    }

    void Interpolate(float* d_o, float* d_i0, float* d_i1, float t, int n, cudaStream_t stream){
        dim3 threads(256);
        dim3 grids(iDivUp(n, threads.x));
        checkConfig(grids);
        Interpolate_kernel<<<grids, threads,0,stream>>>(d_o, d_i0, d_i1, t, n);
    }

/////////////////////////////////////////////////////////////////////////////////
// v = v + eps * d_U
//////////////////////////////////////////////////////////////////////////////////
    void AddEpsilonStep(float* v, float* U, float epsilon, int n, cudaStream_t stream){
        Add_MulC_I(v, U, epsilon, n, stream);
    }
};
#endif
