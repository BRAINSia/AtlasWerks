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

#ifndef  __CUDA__VECTOR_MATH__H
#define  __CUDA__VECTOR_MATH__H

#include <cutil_math.h>
namespace cplVectorCPDOpers{
/// d_o[i] = d_c
    template<class T> void SetMem(T* d_o, T* d_c, int len, cudaStream_t stream=NULL);

/// d_o[i] = d_i[i] + d_c[0]
    template<class T>
    void AddC(T* d_o, T* d_i, const T* d_c, int n, cudaStream_t stream=NULL);

/// d_o[i] += d_c[0]
    template<class T>
    void AddC_I(T* d_o, const T* d_c, int n, cudaStream_t stream=NULL);

/// d_o[i] = d_i[i] - d_c[0]
    template<class T>
    void SubC(T* d_o, T* d_i, const T* d_c, int n, cudaStream_t stream=NULL);

/// d_o[i] -= d_c[0]
    template<class T>
    void SubC_I(T* d_o, const T* d_c, int n, cudaStream_t stream=NULL);

/// d_o[i] = d_i[i] * d_c[0]
    template<class T, class T2>
    void MulC(T2* d_o, T2* d_i, const T* d_c, int n, cudaStream_t stream=NULL);

/// d_o[i] *= d_c[0]
    template<class T, class T2>
    void MulC_I(T2* d_o, const T* d_c, int n, cudaStream_t stream=NULL);

/// d_o[i] = d_i[i] / d_c[0]
    template<class T, class T2>
    void DivC(T2* d_o, T2* d_i, const T* d_c, int n, cudaStream_t stream=NULL);

/// d_o[i] /= d_c[0]
    template<class T, class T2>
    void DivC_I(T2* d_o, const T* d_c, int n, cudaStream_t stream=NULL);

// Function with 3 inputs
    template<class T1, class T>
    void MulMulC(T1* d_o, T1* d_a, T1* d_b, const T* d_c, int n, cudaStream_t stream=NULL);

    template<class T1, class T>
    void MulMulC_I(T1* d_o, T1* d_b, const T* d_c, int n, cudaStream_t stream=NULL);

    template<class T1, class T>
    void AddMulC(T1* d_o, T1* d_a, T1* d_b, const T* d_c, int n, cudaStream_t stream=NULL);

    template<class T1, class T>
    void AddMulC_I(T1* d_o, T1* d_b, const T* d_c, int n, cudaStream_t stream=NULL);

    template<class T1, class T>
    void SubMulC(T1* d_o, T1* d_a, T1* d_b, const T* d_c, int n, cudaStream_t stream=NULL);

    template<class T1, class T>
    void SubMulC_I(T1* d_o, T1* d_b, const T* d_c, int n, cudaStream_t stream=NULL);

    template<class T1, class T>
    void DivMulC(T1* d_o, T1* d_a, T* d_b, const T* d_c, int n, cudaStream_t stream=NULL);

    template<class T1, class T>
    void DivMulC_I(T1* d_a, T* d_b, const T* d_c, int n, cudaStream_t stream=NULL);


    template<class T1, class T>
    void MulCAdd(T1* d_o, T1* d_a, const T* d_c, T1* d_b, int n, cudaStream_t stream=NULL);
    template<class T1, class T>
    void MulCAdd_I(T1* d_o, const T* d_c, T1* d_b, int n, cudaStream_t stream=NULL);
    template<class T1, class T>
    void MulCSub(T1* d_o, T1* d_a, const T* d_c, T1* d_b, int n, cudaStream_t stream=NULL);
    template<class T1, class T>
    void MulCSub_I(T1* d_o, const T* d_c, T1* d_b, int n, cudaStream_t stream=NULL);


    template<class T1, class T>
    void MulCAddC(T1* d_o, T1* d_a, const T* d_bc, int n, cudaStream_t stream=NULL);
    template<class T1, class T>
    void MulCAddC_I(T1* d_o, const T* d_bc, int n, cudaStream_t stream=NULL);
    template<class T1, class T>
    void MulCSubC(T1* d_o, T1* d_a, const T* d_bc, int n, cudaStream_t stream=NULL);
    template<class T1, class T>
    void MulCSubC_I(T1* d_o, const T* d_bc, int n, cudaStream_t stream=NULL);


    template<class T>
    void AddCMulC(T* d_o, T* d_i, const T* d_ab, int n, cudaStream_t stream=NULL);
    template<class T>
    void AddCMulC_I(T* d_data, const  T* d_ab, int n, cudaStream_t stream=NULL);
    template<class T>
    void SubCMulC_I(T* d_data, const  T* d_ab, int n, cudaStream_t stream=NULL);
    template<class T>
    void SubCMulC(T* d_o, T* d_i, const T* d_ab, int n, cudaStream_t stream=NULL);

    template<class T>
    void AddCDivC(T* d_o, T* d_i, const T* d_ab, int n, cudaStream_t stream=NULL);
    template<class T>
    void AddCDivC_I(T* d_data, const  T* d_ab, int n, cudaStream_t stream=NULL);
    template<class T>
    void SubCDivC_I(T* d_data, const  T* d_ab, int n, cudaStream_t stream=NULL);
    template<class T>
    void SubCDivC(T* d_o, T* d_i, const T* d_ab, int n, cudaStream_t stream=NULL);


    template<class T>
    void Add_MulC(T* d_o, T* d_i, T* d_a, const T* d_c,  int n, cudaStream_t stream=NULL);
    template<class T>
    void Add_MulC_I(T* d_data, T* d_a, const T* d_c, int n, cudaStream_t stream=NULL);

    template<class T>
    void Sub_MulC(T* d_o, T* d_i, T* d_a, const T* d_c,  int n, cudaStream_t stream);
    template<class T>
    void Sub_MulC_I(T* d_data, T* d_a, const T* d_c, int n, cudaStream_t stream=NULL);


    template<class T>
    void Add_AddMulC(T* d_o, T* d_a, T* d_b, T* d_d, const T* d_c, int n, cudaStream_t stream=NULL);
    template<class T>
    void Add_AddMulC_I(T* d_a, T* d_b, T* d_d, const T* d_c, int n, cudaStream_t stream=NULL);
    template<class T>
    void Add_SubMulC(T* d_o, T* d_a, T* d_b, T* d_d, const T* d_c, int n, cudaStream_t stream=NULL);
    template<class T>
    void Add_SubMulC_I(T* d_a, T* d_b, T* d_d, const T* d_c, int n, cudaStream_t stream=NULL);
    template<class T>
    void Add_MulMulC(T* d_o, T* d_a, T* d_b, T* d_d, const T* d_c, int n, cudaStream_t stream=NULL);
    template<class T>
    void Add_MulMulC_I(T* d_a, T* d_b, T* d_d, const T* d_c, int n, cudaStream_t stream=NULL);

    template<class T>
    void Sub_AddMulC(T* d_o, T* d_a, T* d_b, T* d_d, const T* d_c, int n, cudaStream_t stream=NULL);
    template<class T>
    void Sub_AddMulC_I(T* d_a, T* d_b, T* d_d, const T* d_c, int n, cudaStream_t stream=NULL);
    template<class T>
    void Sub_SubMulC(T* d_o, T* d_a, T* d_b, T* d_d, const T* d_c, int n, cudaStream_t stream=NULL);
    template<class T>
    void Sub_SubMulC_I(T* d_a, T* d_b, T* d_d, const T* d_c, int n, cudaStream_t stream=NULL);
    template<class T>
    void Sub_MulMulC(T* d_o, T* d_a, T* d_b, T* d_d, const T* d_c, int n, cudaStream_t stream=NULL);
    template<class T>
    void Sub_MulMulC_I(T* d_a, T* d_b, T* d_d, const T* d_c, int n, cudaStream_t stream=NULL);

//
// d_o -= d_v * eps 
//
    template<class T>
    void EpsUpdate(T* d_o, T* d_v, const T* eps, int n, cudaStream_t stream=NULL);

//
// d_o += d_v * eps 
//
    template<class T>
    void AddEpsilonStep(T* d_o, T* d_v, const T* eps, int n, cudaStream_t stream=NULL);

// Functions with 4 inputs
    template<class T2, class T>
    void MulC_Add_MulC(T2* d_o, T2* d_i, T2* d_i1, const T* d_ab, int n, cudaStream_t stream=NULL);
    template<class T2, class T>
    void MulC_Add_MulC_I(T2* d_o, T2* d_i,  const T* d_ab, int n, cudaStream_t stream=NULL);

// d_o = (d_i + a) * b + c
    template<class T>
    void AddCMulCAddC(T* d_o, T* d_i, const T* d_abc, int n, cudaStream_t stream=NULL);

// d_o = (d_o + a) * b + c
    template<class T>
    void AddCMulCAddC_I(T* d_o, const T* d_abc, int n, cudaStream_t stream=NULL);


////////////////////////////////////////////////////////////////////////////////
// Interpolate the image using the linear intepolation
//    d_o = d_i0 * (1 - t) + d_i1 * t
////////////////////////////////////////////////////////////////////////////////
    void Interpolate(float* d_o, float* d_i0, float* d_i1, const float* d_t, int n, cudaStream_t stream=NULL);
}
#endif
