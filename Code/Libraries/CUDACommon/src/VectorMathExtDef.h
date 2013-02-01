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

#include "VectorMathExt.h"
#include <cpl.h>
#include <cudaTexFetch.h>

namespace cplVectorCPDOpers {
/////////////////////////////////////////////////////////////////////////////////
// cplVectorOpers::SetMem
//  the function provided by CUDA is slow and is not flexible enough to set
//  value with different type
//////////////////////////////////////////////////////////////////////////////////
    template<class T>
    __global__ void SetMem_kernel(T* d_o, const T* d_c, int n){
        uint blockId = get_blockID();
        uint id      = get_threadID(blockId);

        if (id < n) 
            d_o[id] = fetch(0, d_c);
    }

    template<class T>
    void SetMem(T* d_o, const T* d_c, int n, cudaStream_t stream) {
        dim3 threads(256);
        dim3 grids = make_large_grid(n, threads.x);
        cache_bind(d_c);
        SetMem_kernel<<<grids, threads, 0, stream>>>(d_o, d_c, n);
    }

////////////////////////////////////////////////////////////////////////////////
//
////////////////////////////////////////////////////////////////////////////////
    template<class T>
    __global__ void AddC_kernel(T* d_o, T* d_i, const T* d_c, int n)
    {
        uint blockId = get_blockID();
        uint id      = get_threadID(blockId);

        if (id < n)
            d_o[id] = d_i[id] + fetch(0, d_c);
    }

    template<class T>
    __global__ void AddC_I_kernel(T* d_o, const T* d_c, int n)
    {
        uint blockId = get_blockID();
        uint id      = get_threadID(blockId);

        if (id < n)
            d_o[id] += fetch(0, d_c);
    }

/// d_o[i] = d_i[i] + d_c[0]
    template<class T>
    void AddC(T* d_o, T* d_i, const T* d_c, int n, cudaStream_t stream)
    {
        dim3 threads(256);
        dim3 grids = make_large_grid(n, threads.x);
        cache_bind(d_c);
        AddC_kernel<<<grids, threads, 0, stream>>>(d_o, d_i, d_c, n);
    }

/// d_o[i] += d_c[0]
    template<class T>
    void AddC_I(T* d_o, const T* d_c, int n, cudaStream_t stream)
    {
        dim3 threads(256);
        dim3 grids = make_large_grid(n, threads.x);
        cache_bind(d_c);

        AddC_I_kernel<<<grids, threads, 0, stream>>>(d_o, d_c, n);
    }

////////////////////////////////////////////////////////////////////////////////
//
////////////////////////////////////////////////////////////////////////////////
    template<class T>
    __global__ void SubC_kernel(T* d_o, T* d_i, const T* d_c, int n)
    {
        uint blockId = get_blockID();
        uint id      = get_threadID(blockId);

        if (id < n)
            d_o[id] = d_i[id] - fetch(0, d_c);
    }

    template<class T>
    __global__ void SubC_I_kernel(T* d_o, const T* d_c, int n)
    {
        uint blockId = get_blockID();
        uint id      = get_threadID(blockId);

        if (id < n)
            d_o[id] -= fetch(0, d_c);
    }

/// d_o[i] = d_i[i] - d_c[0]
    template<class T>
    void SubC(T* d_o, T* d_i, const T* d_c, int n, cudaStream_t stream)
    {
        dim3 threads(256);
        dim3 grids = make_large_grid(n, threads.x);
        cache_bind(d_c);

        SubC_kernel<<<grids, threads, 0, stream>>>(d_o, d_i, d_c, n);
    }

/// d_o[i] -= d_c[0]
    template<class T>
    void SubC_I(T* d_o, const T* d_c, int n, cudaStream_t stream)
    {
        dim3 threads(256);
        dim3 grids = make_large_grid(n, threads.x);
        cache_bind(d_c);

        SubC_I_kernel<<<grids, threads, 0, stream>>>(d_o, d_c, n);
    }

////////////////////////////////////////////////////////////////////////////////
//
////////////////////////////////////////////////////////////////////////////////
    template<class T>
    __global__ void MulC_kernel(T* d_o, T* d_i, const T* d_c, int n)
    {
        uint blockId = get_blockID();
        uint id      = get_threadID(blockId);

        if (id < n)
            d_o[id] = d_i[id] * fetch(0, d_c);
    }

    template<class T>
    __global__ void MulC_I_kernel(T* d_o, const T* d_c, int n)
    {
        uint blockId = get_blockID();
        uint id      = get_threadID(blockId);

        if (id < n)
            d_o[id] *= fetch(0, d_c);
    }

/// d_o[i] = d_i[i] * d_c[0]
    template<class T>
    void MulC(T* d_o, T* d_i, const T* d_c, int n, cudaStream_t stream)
    {
        dim3 threads(256);
        dim3 grids = make_large_grid(n, threads.x);
        cache_bind(d_c);

        MulC_kernel<<<grids, threads, 0, stream>>>(d_o, d_i, d_c, n);
    }

/// d_o[i] *= d_c[0]
    template<class T>
    void MulC_I(T* d_o, const T* d_c, int n, cudaStream_t stream)
    {
        dim3 threads(256);
        dim3 grids = make_large_grid(n, threads.x);
        cache_bind(d_c);

        MulC_I_kernel<<<grids, threads, 0, stream>>>(d_o, d_c, n);
    }

////////////////////////////////////////////////////////////////////////////////
//
////////////////////////////////////////////////////////////////////////////////
    template<class T>
    __global__ void DivC_kernel(T* d_o, T* d_i, const T* d_c, int n)
    {
        uint blockId = get_blockID();
        uint id      = get_threadID(blockId);

        if (id < n)
            d_o[id] = d_i[id] / fetch(0, d_c);
    }

    template<class T>
    __global__ void DivC_I_kernel(T* d_o, const T* d_c, int n)
    {
        uint blockId = get_blockID();
        uint id      = get_threadID(blockId);

        if (id < n)
            d_o[id] /= fetch(0, d_c);
    }

/// d_o[i] = d_i[i] * d_c[0]
    template<class T>
    void DivC(T* d_o, T* d_i, const T* d_c, int n, cudaStream_t stream)
    {
        dim3 threads(256);
        dim3 grids = make_large_grid(n, threads.x);
        cache_bind(d_c);

        DivC_kernel<<<grids, threads, 0, stream>>>(d_o, d_i, d_c, n);
    }

/// d_o[i] *= d_c[0]
    template<class T>
    void DivC_I(T* d_o, const T* d_c, int n, cudaStream_t stream)
    {
        dim3 threads(256);
        dim3 grids = make_large_grid(n, threads.x);
        cache_bind(d_c);

        DivC_I_kernel<<<grids, threads, 0, stream>>>(d_o, d_c, n);
    }

////////////////////////////////////////////////////////////////////////////////
//
////////////////////////////////////////////////////////////////////////////////
    template<class T1, class T>
    __global__ void AddMulC_kernel(T1* d_o, T1* d_a, T1* d_b, const T* d_c, int n)
    {
        uint blockId = get_blockID();
        uint id      = get_threadID(blockId);

        if (id < n)
            d_o[id] = (d_a[id] + d_b[id]) * fetch(0, d_c);
    }

    template<class T1, class T>
    void AddMulC(T1* d_o, T1* d_a, T1* d_b, const T* d_c, int n, cudaStream_t stream)
    {
        dim3 threads(256);
        dim3 grids = make_large_grid(n, threads.x);
        cache_bind(d_c);

        AddMulC_kernel<<<grids, threads, 0, stream>>>(d_o, d_a, d_b, d_c, n);
    };

    template<class T1, class T>
    __global__ void AddMulC_I_kernel(T1* d_o, T1* d_b, const T* d_c, int n)
    {
        uint blockId = get_blockID();
        uint id      = get_threadID(blockId);

        if (id < n)
            d_o[id] = (d_o[id] + d_b[id]) * fetch(0, d_c);
    }

    template<class T1, class T>
    void AddMulC_I(T1* d_o, T1* d_b, const T* d_c, int n, cudaStream_t stream)
    {
        dim3 threads(256);
        dim3 grids = make_large_grid(n, threads.x);
        cache_bind(d_c);

        AddMulC_I_kernel<<<grids, threads, 0, stream>>>(d_o, d_b, d_c, n);
    }

////////////////////////////////////////////////////////////////////////////////
//
////////////////////////////////////////////////////////////////////////////////
    template<class T1, class T>
    __global__ void SubMulC_kernel(T1* d_o, T1* d_a, T1* d_b, const T* d_c, int n)
    {
        uint blockId = get_blockID();
        uint id      = get_threadID(blockId);

        if (id < n)
            d_o[id] = (d_a[id] - d_b[id]) * fetch(0, d_c);
    }

    template<class T1, class T>
    void SubMulC(T1* d_o, T1* d_a, T1* d_b, const T* d_c, int n, cudaStream_t stream)
    {
        dim3 threads(256);
        dim3 grids = make_large_grid(n, threads.x);
        cache_bind(d_c);

        SubMulC_kernel<<<grids, threads, 0, stream>>>(d_o, d_a, d_b, d_c, n);
    };

    template<class T1, class T>
    __global__ void SubMulC_I_kernel(T1* d_o, T1* d_b, const T* d_c, int n)
    {
        uint blockId = get_blockID();
        uint id      = get_threadID(blockId);

        if (id < n)
            d_o[id] = (d_o[id] - d_b[id]) * fetch(0, d_c);
    }

    template<class T1, class T>
    void SubMulC_I(T1* d_o, T1* d_b, const T* d_c, int n, cudaStream_t stream)
    {
        dim3 threads(256);
        dim3 grids = make_large_grid(n, threads.x);
        cache_bind(d_c);

        SubMulC_I_kernel<<<grids, threads, 0, stream>>>(d_o, d_b, d_c, n);
    }

////////////////////////////////////////////////////////////////////////////////
//
////////////////////////////////////////////////////////////////////////////////
    template<class T1, class T>
    __global__ void MulMulC_kernel(T1* d_o, T1* d_a, T1* d_b, const T* d_c, int n)
    {
        uint blockId = get_blockID();
        uint id      = get_threadID(blockId);

        if (id < n)
            d_o[id] = (d_a[id] * d_b[id]) * fetch(0, d_c);
    }

    template<class T1, class T>
    void MulMulC(T1* d_o, T1* d_a, T1* d_b, const T* d_c, int n, cudaStream_t stream)
    {
        dim3 threads(256);
        dim3 grids = make_large_grid(n, threads.x);
        cache_bind(d_c);

        MulMulC_kernel<<<grids, threads, 0, stream>>>(d_o, d_a, d_b, d_c, n);
    };

    template<class T1, class T>
    __global__ void MulMulC_I_kernel(T1* d_o, T1* d_b, const T* d_c, int n)
    {
        uint blockId = get_blockID();
        uint id      = get_threadID(blockId);

        if (id < n)
            d_o[id] = (d_o[id] * d_b[id]) * fetch(0, d_c);
    }

    template<class T1, class T>
    void MulMulC_I(T1* d_o, T1* d_b, const T* d_c, int n, cudaStream_t stream)
    {
        dim3 threads(256);
        dim3 grids = make_large_grid(n, threads.x);
        cache_bind(d_c);

        MulMulC_I_kernel<<<grids, threads, 0, stream>>>(d_o, d_b, d_c, n);
    }

////////////////////////////////////////////////////////////////////////////////
//
////////////////////////////////////////////////////////////////////////////////
    template<class T1, class T>
    __global__ void DivMulC_kernel(T1* d_o, T1* d_a, T1* d_b, const T* d_c, int n)
    {
        uint blockId = get_blockID();
        uint id      = get_threadID(blockId);

        if (id < n)
            d_o[id] = (d_a[id] / d_b[id]) * fetch(0, d_c);
    }

    template<class T1, class T>
    void DivMulC(T1* d_o, T1* d_a, T1* d_b, const T* d_c, int n, cudaStream_t stream)
    {
        dim3 threads(256);
        dim3 grids = make_large_grid(n, threads.x);
        cache_bind(d_c);

        DivMulC_kernel<<<grids, threads, 0, stream>>>(d_o, d_a, d_b, d_c, n);
    };

    template<class T1, class T>
    __global__ void DivMulC_I_kernel(T1* d_o, T1* d_b, const T* d_c, int n)
    {
        uint blockId = get_blockID();
        uint id      = get_threadID(blockId);

        if (id < n)
            d_o[id] = (d_o[id] / d_b[id]) * fetch(0, d_c);
    }

    template<class T1, class T>
    void DivMulC_I(T1* d_o, T1* d_b, const T* d_c, int n, cudaStream_t stream)
    {
        dim3 threads(256);
        dim3 grids = make_large_grid(n, threads.x);
        cache_bind(d_c);

        DivMulC_I_kernel<<<grids, threads, 0, stream>>>(d_o, d_b, d_c, n);
    }

////////////////////////////////////////////////////////////////////////////////
//
////////////////////////////////////////////////////////////////////////////////
    template<class T1, class T>
    __global__ void MulCAdd_kernel(T1* d_o, T1* d_a, const T* d_c, T1* d_b, int n)
    {
        uint blockId = get_blockID();
        uint id      = get_threadID(blockId);

        if (id < n)
            d_o[id] = d_a[id] * fetch(0, d_c) + d_b[id];
    };

    template<class T1, class T>
    __global__ void MulCAdd_I_kernel(T1* d_o, const T* d_c, T1* d_b, int n)
    {
        uint blockId = get_blockID();
        uint id      = get_threadID(blockId);
    
        if (id < n)
            d_o[id] = d_o[id] * fetch(0, d_c) + d_b[id];
    };


    template<class T1, class T>
    void MulCAdd(T1* d_o, T1* d_a, const T* d_c, T1* d_b, int n, cudaStream_t stream)
    {
        dim3 threads(256);
        dim3 grids = make_large_grid(n, threads.x);
        cache_bind(d_c);

        MulCAdd_kernel<<<grids, threads, 0, stream>>>(d_o, d_a, d_c, d_b);
    }

    template<class T1, class T>
    void MulCAdd_I(T1* d_o, const T* d_c, T1* d_b, int n, cudaStream_t stream)
    {
        dim3 threads(256);
        dim3 grids = make_large_grid(n, threads.x);
        cache_bind(d_c);

        MulCAdd_I_kernel<<<grids, threads, 0, stream>>>(d_o, d_c, d_b);
    }

////////////////////////////////////////////////////////////////////////////////
//
////////////////////////////////////////////////////////////////////////////////
    template<class T1, class T>
    __global__ void MulCSub_kernel(T1* d_o, T1* d_a, const T* d_c, T1* d_b, int n)
    {
        uint blockId = get_blockID();
        uint id      = get_threadID(blockId);

        if (id < n)
            d_o[id] = d_a[id] * fetch(0, d_c) - d_b[id];
    };

    template<class T1, class T>
    __global__ void MulCSub_I_kernel(T1* d_o, const T* d_c, T1* d_b, int n)
    {
        uint blockId = get_blockID();
        uint id      = get_threadID(blockId);
    
        if (id < n)
            d_o[id] = d_o[id] * fetch(0, d_c) - d_b[id];
    };


    template<class T1, class T>
    void MulCSub(T1* d_o, T1* d_a, const T* d_c, T1* d_b, int n, cudaStream_t stream)
    {
        dim3 threads(256);
        dim3 grids = make_large_grid(n, threads.x);
        cache_bind(d_c);

        MulCSub_kernel<<<grids, threads, 0, stream>>>(d_o, d_a, d_c, d_b);
    }

    template<class T1, class T>
    void MulCSub_I(T1* d_o, const T* d_c, T1* d_b, int n, cudaStream_t stream)
    {
        dim3 threads(256);
        dim3 grids = make_large_grid(n, threads.x);
        cache_bind(d_c);

        MulCSub_I_kernel<<<grids, threads, 0, stream>>>(d_o, d_c, d_b);
    }

////////////////////////////////////////////////////////////////////////////////
//
////////////////////////////////////////////////////////////////////////////////
    template<class T1, class T>
    __global__ void MulCAddC_kernel(T1* d_o, T1* d_a, const T* d_bc, int n)
    {
        uint blockId = get_blockID();
        uint id      = get_threadID(blockId);
        if (id < n){
            d_o[id] = d_a[id] * fetch(0, d_bc) + fetch(1, d_bc);
        }
    };

    template<class T1, class T>
    __global__ void MulCAddC_I_kernel(T1* d_o, const T* d_bc, int n)
    {
        uint blockId = get_blockID();
        uint id      = get_threadID(blockId);
        if (id < n){
            d_o[id] = d_o[id] * fetch(0, d_bc) + fetch(1, d_bc);
        }
    };

    template<class T1, class T>
    void MulCAddC(T1* d_o, T1* d_a, const T* d_bc, int n, cudaStream_t stream)
    {
        dim3 threads(256);
        dim3 grids = make_large_grid(n, threads.x);
        cache_bind(d_bc);

        MulCAddC_kernel<<<grids, threads, 0, stream>>>(d_o, d_a, d_bc, n);
    };

    template<class T1, class T>
    void MulCAddC_I(T1* d_o, const T* d_bc, int n, cudaStream_t stream)
    {
        dim3 threads(256);
        dim3 grids = make_large_grid(n, threads.x);
        cache_bind(d_bc);

        MulCAddC_I_kernel<<<grids, threads, 0, stream>>>(d_o, d_bc, n);
    };

////////////////////////////////////////////////////////////////////////////////
//
////////////////////////////////////////////////////////////////////////////////
    template<class T1, class T>
    __global__ void MulCSubC_kernel(T1* d_o, T1* d_a, const T* d_bc, int n)
    {
        uint blockId = get_blockID();
        uint id      = get_threadID(blockId);
        if (id < n){
            d_o[id] = d_a[id] * fetch(0, d_bc) - fetch(1, d_bc);
        }
    };

    template<class T1, class T>
    __global__ void MulCSubC_I_kernel(T1* d_o, const T* d_bc, int n)
    {
        uint blockId = get_blockID();
        uint id      = get_threadID(blockId);
        if (id < n){
            d_o[id] = d_o[id] * fetch(0, d_bc) - fetch(1, d_bc);
        }
    };

    template<class T1, class T>
    void MulCSubC(T1* d_o, T1* d_a, const T* d_bc, int n, cudaStream_t stream)
    {
        dim3 threads(256);
        dim3 grids = make_large_grid(n, threads.x);
        cache_bind(d_bc);

        MulCSubC_kernel<<<grids, threads, 0, stream>>>(d_o, d_a, d_bc, n);
    };

    template<class T1, class T>
    void MulCSubC_I(T1* d_o, const T* d_bc, int n, cudaStream_t stream)
    {
        dim3 threads(256);
        dim3 grids = make_large_grid(n, threads.x);
        cache_bind(d_bc);

        MulCSubC_I_kernel<<<grids, threads, 0, stream>>>(d_o, d_bc, n);
    };


////////////////////////////////////////////////////////////////////////////////
//
////////////////////////////////////////////////////////////////////////////////
    template<class T>
    __global__ void AddCMulC_kernel(T* d_o, T* d_i, const T* d_ab, int n)
    {
        uint blockId = get_blockID();
        uint id      = get_threadID(blockId);
        if (id < n){
            d_o[id] = (d_i[id] + fetch(0, d_ab)) * fetch(1, d_ab);
        }
    };

    template<class T>
    __global__ void AddCMulC_I_kernel(T* d_o, const  T* d_ab, int n)
    {
        uint blockId = get_blockID();
        uint id      = get_threadID(blockId);
        if (id < n){
            d_o[id] = (d_o[id] + fetch(0, d_ab)) * fetch(1, d_ab);
        }
    };


    template<class T>
    void AddCMulC(T* d_o, T* d_i, const T* d_ab, int n, cudaStream_t stream)
    {
        dim3 threads(256);
        dim3 grids = make_large_grid(n, threads.x);
        cache_bind(d_ab);
        AddCMulC_kernel<<<grids, threads, 0, stream>>>(d_o, d_i, d_ab, n);
    }

    template<class T>
    void AddCMulC_I(T* d_o, const  T* d_ab, int n, cudaStream_t stream)
    {
        dim3 threads(256);
        dim3 grids = make_large_grid(n, threads.x);
        cache_bind(d_ab);
        AddCMulC_I_kernel<<<grids, threads, 0, stream>>>(d_o, d_ab, n);
    }

////////////////////////////////////////////////////////////////////////////////
//
////////////////////////////////////////////////////////////////////////////////
    template<class T>
    __global__ void SubCMulC_kernel(T* d_o, T* d_i, const T* d_ab, int n)
    {
        uint blockId = get_blockID();
        uint id      = get_threadID(blockId);
        if (id < n){
            d_o[id] = (d_i[id] - fetch(0, d_ab)) * fetch(1, d_ab);
        }
    };

    template<class T>
    __global__ void SubCMulC_I_kernel(T* d_o, const  T* d_ab, int n)
    {
        uint blockId = get_blockID();
        uint id      = get_threadID(blockId);
        if (id < n){
            d_o[id] = (d_o[id] - fetch(0, d_ab)) * fetch(1, d_ab);
        }
    };


    template<class T>
    void SubCMulC(T* d_o, T* d_i, const T* d_ab, int n, cudaStream_t stream)
    {
        dim3 threads(256);
        dim3 grids = make_large_grid(n, threads.x);
        cache_bind(d_ab);
        SubCMulC_kernel<<<grids, threads, 0, stream>>>(d_o, d_i, d_ab, n);
    }

    template<class T>
    void SubCMulC_I(T* d_o, const  T* d_ab, int n, cudaStream_t stream)
    {
        dim3 threads(256);
        dim3 grids = make_large_grid(n, threads.x);
        cache_bind(d_ab);
        SubCMulC_I_kernel<<<grids, threads, 0, stream>>>(d_o, d_ab, n);
    }

////////////////////////////////////////////////////////////////////////////////
//
////////////////////////////////////////////////////////////////////////////////
    template<class T>
    __global__ void AddCDivC_kernel(T* d_o, T* d_i, const T* d_ab, int n)
    {
        uint blockId = get_blockID();
        uint id      = get_threadID(blockId);
        if (id < n){
            d_o[id] = (d_i[id] + fetch(0, d_ab)) / fetch(1, d_ab);
        }
    };

    template<class T>
    __global__ void AddCDivC_I_kernel(T* d_o, const  T* d_ab, int n)
    {
        uint blockId = get_blockID();
        uint id      = get_threadID(blockId);
        if (id < n){
            d_o[id] = (d_o[id] + fetch(0, d_ab)) / fetch(1, d_ab);
        }
    };


    template<class T>
    void AddCDivC(T* d_o, T* d_i, const T* d_ab, int n, cudaStream_t stream)
    {
        dim3 threads(256);
        dim3 grids = make_large_grid(n, threads.x);
        cache_bind(d_ab);
        AddCDivC_kernel<<<grids, threads, 0, stream>>>(d_o, d_i, d_ab, n);
    }

    template<class T>
    void AddCDivC_I(T* d_o, const  T* d_ab, int n, cudaStream_t stream)
    {
        dim3 threads(256);
        dim3 grids = make_large_grid(n, threads.x);
        cache_bind(d_ab);
        AddCDivC_I_kernel<<<grids, threads, 0, stream>>>(d_o, d_ab, n);
    }

////////////////////////////////////////////////////////////////////////////////
//
////////////////////////////////////////////////////////////////////////////////
    template<class T>
    __global__ void SubCDivC_kernel(T* d_o, T* d_i, const T* d_ab, int n)
    {
        uint blockId = get_blockID();
        uint id      = get_threadID(blockId);
        if (id < n){
            d_o[id] = (d_i[id] - fetch(0, d_ab)) / fetch(1, d_ab);
        }
    };

    template<class T>
    __global__ void SubCDivC_I_kernel(T* d_o, const  T* d_ab, int n)
    {
        uint blockId = get_blockID();
        uint id      = get_threadID(blockId);
        if (id < n){
            d_o[id] = (d_o[id] - fetch(0, d_ab)) / fetch(1, d_ab);
        }
    };


    template<class T>
    void SubCDivC(T* d_o, T* d_i, const T* d_ab, int n, cudaStream_t stream)
    {
        dim3 threads(256);
        dim3 grids = make_large_grid(n, threads.x);
        cache_bind(d_ab);
        SubCDivC_kernel<<<grids, threads, 0, stream>>>(d_o, d_i, d_ab, n);
    }

    template<class T>
    void SubCDivC_I(T* d_o, const  T* d_ab, int n, cudaStream_t stream)
    {
        dim3 threads(256);
        dim3 grids = make_large_grid(n, threads.x);
        cache_bind(d_ab);
        SubCDivC_I_kernel<<<grids, threads, 0, stream>>>(d_o, d_ab, n);
    }

////////////////////////////////////////////////////////////////////////////////
//
////////////////////////////////////////////////////////////////////////////////
    template<class T>
    __global__ void Add_MulC_kernel(T* d_o, T* d_i, T* d_a, const T* d_c,  int n)
    {
        uint blockId = get_blockID();
        uint id      = get_threadID(blockId);
        if (id < n){
            d_o[id] = d_i[id] + d_a[id] * fetch(0, d_c);
        }    
    }

    template<class T>
    void Add_MulC(T* d_o, T* d_i, T* d_a, const T* d_c,  int n, cudaStream_t stream)
    {
        dim3 threads(256);
        dim3 grids = make_large_grid(n, threads.x);
        cache_bind(d_c);
        Add_MulC_kernel<<<grids, threads, 0, stream>>>(d_o, d_i, d_a, d_c, n);
    };

    template<class T>
    __global__ void Add_MulC_I_kernel(T* d_o, T* d_i, T* d_a, const T* d_c,  int n)
    {
        uint blockId = get_blockID();
        uint id      = get_threadID(blockId);
        if (id < n){
            d_o[id] += d_a[id] * fetch(0, d_c);
        }    
    }

    template<class T>
    void Add_MulC_I(T* d_o, T* d_a, const T* d_c,  int n, cudaStream_t stream)
    {
        dim3 threads(256);
        dim3 grids = make_large_grid(n, threads.x);
        cache_bind(d_c);
        Add_MulC_I_kernel<<<grids, threads, 0, stream>>>(d_o, d_a, d_c, n);
    };

////////////////////////////////////////////////////////////////////////////////
//
////////////////////////////////////////////////////////////////////////////////
    template<class T>
    __global__ void Sub_MulC_kernel(T* d_o, T* d_i, T* d_a, const T* d_c,  int n)
    {
        uint blockId = get_blockID();
        uint id      = get_threadID(blockId);
        if (id < n){
            d_o[id] = d_i[id] - d_a[id] * fetch(0, d_c);
        }    
    }

    template<class T>
    void Sub_MulC(T* d_o, T* d_i, T* d_a, const T* d_c,  int n, cudaStream_t stream)
    {
        dim3 threads(256);
        dim3 grids = make_large_grid(n, threads.x);
        cache_bind(d_c);
        Sub_MulC_kernel<<<grids, threads, 0, stream>>>(d_o, d_i, d_a, d_c, n);
    };

    template<class T>
    __global__ void Sub_MulC_I_kernel(T* d_o, T* d_i, T* d_a, const T* d_c,  int n)
    {
        uint blockId = get_blockID();
        uint id      = get_threadID(blockId);
        if (id < n){
            d_o[id] -= d_a[id] * fetch(0, d_c);
        }    
    }

    template<class T>
    void Sub_MulC_I(T* d_o, T* d_a, const T* d_c,  int n, cudaStream_t stream)
    {
        dim3 threads(256);
        dim3 grids = make_large_grid(n, threads.x);
        cache_bind(d_c);
        Sub_MulC_I_kernel<<<grids, threads, 0, stream>>>(d_o, d_a, d_c, n);
    };


////////////////////////////////////////////////////////////////////////////////
//
////////////////////////////////////////////////////////////////////////////////
    template<class T>
    __global__ void Add_AddMulC_kernel(T* d_o, T* d_a, T* d_b, T* d_d, const T* d_c, int n, cudaStream_t stream)
    {
        uint blockId = get_blockID();
        uint id      = get_threadID(blockId);
        if (id < n){
            d_o[id] = d_a[id] + (d_b[id] + d_d[id]) * fetch(0, d_c);
        }
    }

    template<class T>
    void Add_AddMulC(T* d_o, T* d_a, T* d_b, T* d_d, const T* d_c, int n, cudaStream_t stream)
    {
        dim3 threads(256);
        dim3 grids = make_large_grid(n, threads.x);
        cache_bind(d_c);
        Add_AddMulC_kernel<<<grids, threads, 0, stream>>>(d_o, d_a, d_b, d_d, d_c, n);
    };

    template<class T>
    __global__ void Add_AddMulC_I_kernel(T* d_o, T* d_a, T* d_b, T* d_d, const T* d_c, int n, cudaStream_t stream)
    {
        uint blockId = get_blockID();
        uint id      = get_threadID(blockId);
        if (id < n){
            d_o[id] = d_o[id] + (d_b[id] + d_d[id]) * fetch(0, d_c);
        }
    }

    template<class T>
    void Add_AddMulC_I(T* d_o, T* d_b, T* d_d, const T* d_c, int n, cudaStream_t stream)
    {
        dim3 threads(256);
        dim3 grids = make_large_grid(n, threads.x);
        cache_bind(d_c);
        Add_AddMulC_I_kernel<<<grids, threads, 0, stream>>>(d_o, d_b, d_d, d_c, n);
    };


////////////////////////////////////////////////////////////////////////////////
//
////////////////////////////////////////////////////////////////////////////////
    template<class T>
    __global__ void Add_MulMulC_kernel(T* d_o, T* d_a, T* d_b, T* d_d, const T* d_c, int n, cudaStream_t stream)
    {
        uint blockId = get_blockID();
        uint id      = get_threadID(blockId);
        if (id < n){
            d_o[id] = d_a[id] + (d_b[id] * d_d[id]) * fetch(0, d_c);
        }
    }

    template<class T>
    void Add_MulMulC(T* d_o, T* d_a, T* d_b, T* d_d, const T* d_c, int n, cudaStream_t stream)
    {
        dim3 threads(256);
        dim3 grids = make_large_grid(n, threads.x);
        cache_bind(d_c);
        Add_MulMulC_kernel<<<grids, threads, 0, stream>>>(d_o, d_a, d_b, d_d, d_c, n);
    };

    template<class T>
    __global__ void Add_MulMulC_I_kernel(T* d_o, T* d_a, T* d_b, T* d_d, const T* d_c, int n, cudaStream_t stream)
    {
        uint blockId = get_blockID();
        uint id      = get_threadID(blockId);
        if (id < n){
            d_o[id] = d_o[id] + (d_b[id] * d_d[id]) * fetch(0, d_c);
        }
    }

    template<class T>
    void Add_MulMulC_I(T* d_o, T* d_b, T* d_d, const T* d_c, int n, cudaStream_t stream)
    {
        dim3 threads(256);
        dim3 grids = make_large_grid(n, threads.x);
        cache_bind(d_c);
        Add_MulMulC_I_kernel<<<grids, threads, 0, stream>>>(d_o, d_b, d_d, d_c, n);
    };

////////////////////////////////////////////////////////////////////////////////
//
////////////////////////////////////////////////////////////////////////////////
    template<class T>
    __global__ void Add_SubMulC_kernel(T* d_o, T* d_a, T* d_b, T* d_d, const T* d_c, int n, cudaStream_t stream)
    {
        uint blockId = get_blockID();
        uint id      = get_threadID(blockId);
        if (id < n){
            d_o[id] = d_a[id] + (d_b[id] - d_d[id]) * fetch(0, d_c);
        }
    }

    template<class T>
    void Add_SubMulC(T* d_o, T* d_a, T* d_b, T* d_d, const T* d_c, int n, cudaStream_t stream)
    {
        dim3 threads(256);
        dim3 grids = make_large_grid(n, threads.x);
        cache_bind(d_c);
        Add_SubMulC_kernel<<<grids, threads, 0, stream>>>(d_o, d_a, d_b, d_d, d_c, n);
    };

    template<class T>
    __global__ void Add_SubMulC_I_kernel(T* d_o, T* d_a, T* d_b, T* d_d, const T* d_c, int n, cudaStream_t stream)
    {
        uint blockId = get_blockID();
        uint id      = get_threadID(blockId);
        if (id < n){
            d_o[id] = d_o[id] + (d_b[id] - d_d[id]) * fetch(0, d_c);
        }
    }

    template<class T>
    void Add_SubMulC_I(T* d_o, T* d_b, T* d_d, const T* d_c, int n, cudaStream_t stream)
    {
        dim3 threads(256);
        dim3 grids = make_large_grid(n, threads.x);
        cache_bind(d_c);
        Add_SubMulC_I_kernel<<<grids, threads, 0, stream>>>(d_o, d_b, d_d, d_c, n);
    };


////////////////////////////////////////////////////////////////////////////////
//
////////////////////////////////////////////////////////////////////////////////
    template<class T>
    __global__ void Sub_AddMulC_kernel(T* d_o, T* d_a, T* d_b, T* d_d, const T* d_c, int n, cudaStream_t stream)
    {
        uint blockId = get_blockID();
        uint id      = get_threadID(blockId);
        if (id < n){
            d_o[id] = d_a[id] - (d_b[id] + d_d[id]) * fetch(0, d_c);
        }
    }

    template<class T>
    void Sub_AddMulC(T* d_o, T* d_a, T* d_b, T* d_d, const T* d_c, int n, cudaStream_t stream)
    {
        dim3 threads(256);
        dim3 grids = make_large_grid(n, threads.x);
        cache_bind(d_c);
        Sub_AddMulC_kernel<<<grids, threads, 0, stream>>>(d_o, d_a, d_b, d_d, d_c, n);
    };

    template<class T>
    __global__ void Sub_AddMulC_I_kernel(T* d_o, T* d_a, T* d_b, T* d_d, const T* d_c, int n, cudaStream_t stream)
    {
        uint blockId = get_blockID();
        uint id      = get_threadID(blockId);
        if (id < n){
            d_o[id] = d_o[id] - (d_b[id] + d_d[id]) * fetch(0, d_c);
        }
    }

    template<class T>
    void Sub_AddMulC_I(T* d_o, T* d_b, T* d_d, const T* d_c, int n, cudaStream_t stream)
    {
        dim3 threads(256);
        dim3 grids = make_large_grid(n, threads.x);
        cache_bind(d_c);
        Sub_AddMulC_I_kernel<<<grids, threads, 0, stream>>>(d_o, d_b, d_d, d_c, n);
    };


////////////////////////////////////////////////////////////////////////////////
//
////////////////////////////////////////////////////////////////////////////////
    template<class T>
    __global__ void Sub_MulMulC_kernel(T* d_o, T* d_a, T* d_b, T* d_d, const T* d_c, int n, cudaStream_t stream)
    {
        uint blockId = get_blockID();
        uint id      = get_threadID(blockId);
        if (id < n){
            d_o[id] = d_a[id] - (d_b[id] * d_d[id]) * fetch(0, d_c);
        }
    }

    template<class T>
    void Sub_MulMulC(T* d_o, T* d_a, T* d_b, T* d_d, const T* d_c, int n, cudaStream_t stream)
    {
        dim3 threads(256);
        dim3 grids = make_large_grid(n, threads.x);
        cache_bind(d_c);
        Sub_MulMulC_kernel<<<grids, threads, 0, stream>>>(d_o, d_a, d_b, d_d, d_c, n);
    };

    template<class T>
    __global__ void Sub_MulMulC_I_kernel(T* d_o, T* d_a, T* d_b, T* d_d, const T* d_c, int n, cudaStream_t stream)
    {
        uint blockId = get_blockID();
        uint id      = get_threadID(blockId);
        if (id < n){
            d_o[id] = d_o[id] - (d_b[id] * d_d[id]) * fetch(0, d_c);
        }
    }

    template<class T>
    void Sub_MulMulC_I(T* d_o, T* d_b, T* d_d, const T* d_c, int n, cudaStream_t stream)
    {
        dim3 threads(256);
        dim3 grids = make_large_grid(n, threads.x);
        cache_bind(d_c);
        Sub_MulMulC_I_kernel<<<grids, threads, 0, stream>>>(d_o, d_b, d_d, d_c, n);
    };

////////////////////////////////////////////////////////////////////////////////
//
////////////////////////////////////////////////////////////////////////////////
    template<class T>
    __global__ void Sub_SubMulC_kernel(T* d_o, T* d_a, T* d_b, T* d_d, const T* d_c, int n, cudaStream_t stream)
    {
        uint blockId = get_blockID();
        uint id      = get_threadID(blockId);
        if (id < n){
            d_o[id] = d_a[id] - (d_b[id] - d_d[id]) * fetch(0, d_c);
        }
    }

    template<class T>
    void Sub_SubMulC(T* d_o, T* d_a, T* d_b, T* d_d, const T* d_c, int n, cudaStream_t stream)
    {
        dim3 threads(256);
        dim3 grids = make_large_grid(n, threads.x);
        cache_bind(d_c);
        Sub_SubMulC_kernel<<<grids, threads, 0, stream>>>(d_o, d_a, d_b, d_d, d_c, n);
    };

    template<class T>
    __global__ void Sub_SubMulC_I_kernel(T* d_o, T* d_a, T* d_b, T* d_d, const T* d_c, int n, cudaStream_t stream)
    {
        uint blockId = get_blockID();
        uint id      = get_threadID(blockId);
        if (id < n){
            d_o[id] = d_o[id] - (d_b[id] - d_d[id]) * fetch(0, d_c);
        }
    }

    template<class T>
    void Sub_SubMulC_I(T* d_o, T* d_b, T* d_d, const T* d_c, int n, cudaStream_t stream)
    {
        dim3 threads(256);
        dim3 grids = make_large_grid(n, threads.x);
        cache_bind(d_c);
        Sub_SubMulC_I_kernel<<<grids, threads, 0, stream>>>(d_o, d_b, d_d, d_c, n);
    };


    template<class T>
    void EpsUpdate(T* d_o, T* d_v, const T* eps, int n, cudaStream_t stream)
    {
        Sub_MulC_I(d_o, d_v, eps, n, stream);
    };

    template<class T>
    void AddEpsilonStep(T* d_o, T* d_v, const T* eps, int n, cudaStream_t stream)
    {
        Add_MulC_I(d_o, d_v, eps, n, stream);
    }




////////////////////////////////////////////////////////////////////////////////
//
////////////////////////////////////////////////////////////////////////////////
// d_o = (d_i + a) * b + c
    template<class T>
    __global__  void AddCMulCAddC_kernel(T* d_o, T* d_i, T* d_abc, int n)
    {
        uint blockId = get_blockID();
        uint id      = get_threadID(blockId);
    
        if (id < n){
            d_o[id] = (d_i[id] + fetch(0, d_abc)) * fetch(1, d_abc) + fetch(2, d_abc);
        }
    }

    template<class T>
    void AddCMulCAddC(T* d_o, T* d_i, const T* d_abc, int n, cudaStream_t stream)
    {
        dim3 threads(256);
        dim3 grids = make_large_grid(n, threads.x);
        cache_bind(d_abc);
        AddCMulCAddC_kernel<<<grids, threads, 0, stream>>>(d_o, d_i, d_abc, n);
    }

// d_o = (d_o + a) * b + c
    template<class T>
    __global__  void AddCMulCAddC_I_kernel(T* d_o, T* d_abc, int n)
    {
        uint blockId = get_blockID();
        uint id      = get_threadID(blockId);
    
        if (id < n){
            d_o[id] = (d_o[id] + fetch(0, d_abc)) * fetch(1, d_abc) + fetch(2, d_abc);
        }
    }

    template<class T>
    void AddCMulCAddC_I(T* d_o, const T* d_abc, int n, cudaStream_t stream)
    {
        dim3 threads(256);
        dim3 grids = make_large_grid(n, threads.x);
        cache_bind(d_abc);
        AddCMulCAddC_I_kernel<<<grids, threads, 0, stream>>>(d_o, d_abc, n);
    }

////////////////////////////////////////////////////////////////////////////////
//
////////////////////////////////////////////////////////////////////////////////
    template<class T2, class T>
    __global__ void MulC_Add_MulC_kernel(T2* d_o, T2* d_i, T2* d_i1, const T* d_ab, int n)
    {
        uint blockId = get_blockID();
        uint id      = get_threadID(blockId);
    
        if (id < n){
            d_o[id] = d_i[id] * fetch(0, d_ab) + d_i1[id] * fetch(1, d_ab);
        }
    };

    template<class T2, class T>
    void MulC_Add_MulC(T2* d_o, T2* d_i, T2* d_i1, const T* d_ab, int n, cudaStream_t stream)
    {
        dim3 threads(256);
        dim3 grids = make_large_grid(n, threads.x);
        cache_bind(d_ab);
        MulC_Add_MulC_kernel<<<grids, threads, 0, stream>>>(d_o, d_i, d_i1, d_ab, n);
    };


    template<class T2, class T>
    __global__ void MulC_Add_MulC_I_kernel(T2* d_o, T2* d_i,  const T* d_ab, int n, cudaStream_t stream)
    {
        uint blockId = get_blockID();
        uint id      = get_threadID(blockId);
    
        if (id < n){
            d_o[id] = d_o[id] * fetch(0, d_ab) + d_i[id] * fetch(1, d_ab);
        }

    };

    template<class T2, class T>
    void MulC_Add_MulC_I(T2* d_o, T2* d_i,  const T* d_ab, int n, cudaStream_t stream)
    {
        dim3 threads(256);
        dim3 grids = make_large_grid(n, threads.x);
        cache_bind(d_ab);
        MulC_Add_MulC_I_kernel<<<grids, threads, 0, stream>>>(d_o, d_i, d_ab, n);
    };


////////////////////////////////////////////////////////////////////////////////
// Interpolate the image using the linear intepolation
//    d_o = d_i0 * (1 - t) + d_i1 * t
////////////////////////////////////////////////////////////////////////////////
    __global__ void Interpolate_kernel(float* d_o, float* d_i0, float* d_i1, const float* d_t, int n)
    {
        uint blockId = get_blockID();
        uint id      = get_threadID(blockId);
    
        if (id < n){
            d_o[id] = d_i0[id] * (1.f - fetch(0, d_t)) + d_i1[id] * fetch(0, d_t);
        }
    }
    
    void Interpolate(float* d_o, float* d_i0, float* d_i1, const float* d_t, int n, cudaStream_t stream)
    {
        dim3 threads(256);
        dim3 grids = make_large_grid(n, threads.x);
        cache_bind(d_t);
        Interpolate_kernel<<<grids, threads, 0, stream>>>(d_o, d_i0, d_i1, d_t, n);
    };
};
#endif
