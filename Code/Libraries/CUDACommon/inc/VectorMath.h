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

#include <cuda_runtime.h>

namespace cplVectorOpers {
/**
 * Copy array (d_o = d_i)
 */
    template<class T> void Copy(T* d_o, const  T* d_i, unsigned int n, cudaStream_t stream=NULL);
/**
 * Fill array with constant value
 */
    template<class T>void SetMem(T* d_data, T c, int len, cudaStream_t stream=NULL);
/**
 * Set unsined array with the linear ramp up
 * d[i] = i
 */
    template<typename T>
    void SetLinear(T* d_data,  int len, cudaStream_t stream=NULL);

/**
 * Set unsined array with the linear ramp down
 * d[i] = n - 1 - i;
 */
    template<typename T>
    void SetLinearDown(T* d_data,  int len, cudaStream_t stream=NULL);

/////////////////////////////////////////////////////////////////////////////////
// Unary functions
//////////////////////////////////////////////////////////////////////////////////
 /**
 * Absolute value of array (d_o = abs(d_i))
 */
    template<class T> void Abs(T* d_o, const T* d_i, unsigned int size, cudaStream_t stream=NULL);
/**
 * Absolute value of array, in-place (d_o = abs(d_o))
 */
    template<class T> void Abs_I(T* d_i, unsigned int size, cudaStream_t stream=NULL);
/**
 * Cube each value in array (d_o = d_i^3)
 */
    template<class T> void Cube(T* d_o, const T* d_i, unsigned int size, cudaStream_t stream=NULL);
/**
 * Cube each value in array, in-place (d_o = d_o^3)
 */
    template<class T> void Cube_I(T* d_o, unsigned int size, cudaStream_t stream=NULL);
/**
 * Negate each value in array (d_o = -d_i)
 */
    template<class T> void Neg(T* d_o, const T* d_i, unsigned int size, cudaStream_t stream=NULL);
/**
 * Negate each value in array, in-place (d_o = -d_o)
 */
    template<class T> void Neg_I(T* d_o, unsigned int size, cudaStream_t stream=NULL);
/**
 * Square of each value in array (d_o = sqr(d_i))
 */
    template<class T> void Sqr(T* d_o, const T* d_i, unsigned int size, cudaStream_t stream=NULL);

/**
 * Square of each value in array, in-place (d_o = sqr(d_o))
 */
    template<class T> void Sqr_I(T* d_i, unsigned int size, cudaStream_t stream=NULL);

/**
 * Square root of each value in array (d_o = sqr(d_i))
 */
    void Sqrt(float* d_o, const  float* d_i, unsigned int size, cudaStream_t stream);

/**
 * Square of root each value in array (d_o = sqr(d_o)) inplace version
 */
    void Sqrt_I(float* d_o, unsigned int size, cudaStream_t stream);

/**
 * Inverse of each value in array (d_o = 1.0/d_i)
 */
    void Inv(float* d_o, const float* d_i, unsigned int size, cudaStream_t stream);

/**
 * Inverse of each value in array (d_o = 1.0/d_o)
 */
    void Inv_I(float* d_o, unsigned int size, cudaStream_t stream=NULL);

/////////////////////////////////////////////////////////////////////////////////
// Binary functions
//////////////////////////////////////////////////////////////////////////////////
/**
 * Add constant to an array (d_o = d_i + c)
 */
    template<class T> void AddC(T* d_o, T* d_i, T c, int n, cudaStream_t stream=NULL);

/**
 * Add constant to an array, in-place (d_o = d_o + c)
 */
    template<class T> void AddC_I(T* d_o, T c, int n, cudaStream_t stream=NULL);

/**
 * Add two arrays, (d_o = d_i + d_i1)
 */
    template<class T> void Add(T* d_o, T* d_i, T* d_i1, int n, cudaStream_t stream=NULL);

/**
 * Add two arrays, in-place (d_o = d_o + d_i)
 */
    template<class T> void Add_I(T* d_o, T* d_i, int n, cudaStream_t stream=NULL);

/**
 * Subtract constant from an array (d_o = d_i - c)
 */
    template<class T> void SubC(T* d_o, T* d_i, T c, int n, cudaStream_t stream=NULL);

/**
 * Subtract constant from an array, in-place (d_o = d_o - c)
 */
    template<class T> void SubC_I(T* d_o, T c, int n, cudaStream_t stream=NULL);

/**
 * Subtract two arrays, (d_o = d_i - d_i1)
 */
    template<class T> void Sub(T* d_o, T* d_i, T* d_i1, int n, cudaStream_t stream=NULL);

/**
 * Subtract two arrays, in-place (d_o = d_o - d_i)
 */
    template<class T> void Sub_I(T* d_o, T* d_i, int n, cudaStream_t stream=NULL);

/**
 * Multiply an array by a constant (scale array) (d_o = d_i * c)
 */
    template<class T, class T2> void MulC(T2* d_o, const T2* d_i, T c, int n, cudaStream_t stream=NULL);

/**
 * Multiply an array by a constant (scale array) in-place (d_o = d_o * c)
 */
    template<class T, class T2> void MulC_I(T2* d_o, T c, int n, cudaStream_t stream=NULL);

/**
 * Multiply two arrays (d_o = d_i * d_i1)
 */
    template<class T> void Mul(T* d_o, T* d_i, T* d_i1, int n, cudaStream_t stream=NULL);

/**
 * Multiply two arrays, in-place (d_o = d_o * d_i)
 */
    template<class T> void Mul_I(T* d_o, T* d_i, int n, cudaStream_t stream=NULL);

/**
 * Divide an array by a constant (scale array) (d_o = d_i / c)
 */
    template<class T2, class T> void DivC(T2* d_o, const T2* d_i, T c, int n, cudaStream_t stream=NULL);

/**
 * Divide an array by a constant (scale array) in-place (d_o = d_o / c)
 */
    template<class T2, class T> void DivC_I(T2* d_o, T c, int n, cudaStream_t stream=NULL);

/**
 * Divide two arrays (d_o = d_i / d_i1)
 */
    template<class T> void Div(T* d_o, T* d_i, T* d_i1, int n, cudaStream_t stream=NULL);

/**
 * Divide two arrays, in-place (d_o = d_o / d_i)
 */
    template<class T> void Div_I(T* d_o, T* d_i, int n, cudaStream_t stream=NULL);

/////////////////////////////////////////////////////////////////////////////////
// Triary functions
//////////////////////////////////////////////////////////////////////////////////

/**
 * Compute the absolution different between 2 vector
 * d_o = abs(d_i - d_i1)
 */
    template<class T> void AbsDiff(T* d_o, const T* d_i, const T* d_i1, int n, cudaStream_t stream=NULL);

/**
 * Compute the absolution different between 2 vector in-place version
 * d_o = abs(d_o - d_i1)
 */
    template<class T> void AbsDiff_I(T* d_o, const T* d_i, int n, cudaStream_t stream=NULL);

/**
 * Compute the absolution different between 2 vector
 * d_o = sqr(d_i - d_i1)
 */
    template<class T> void SqrDiff(T* d_o, const T* d_i, const T* d_i1, int n, cudaStream_t stream=NULL);

/**
 * Compute the absolution different between 2 vector in-place version
 * d_o = sqr(d_o - d_i1)
 */
    template<class T> void SqrDiff_I(T* d_o, const T* d_i, int n, cudaStream_t stream=NULL);

/**
 * d_o = A * (B * c)
 */
    template<class T1, class T> void MulMulC(T1* d_o, T1* A,T* B, T c, int n, cudaStream_t stream=NULL);

/**
 * A = A * (B * c)
 */
    template<class T1, class T> void MulMulC_I(T1* A, T* B, T c, int n, cudaStream_t stream=NULL);

/**
 * d_o = A * (B * C)
 */
    template<class T1, class T> void MulMul(T1* d_o, T1* A, T* B, T* C, int n, cudaStream_t stream=NULL);

/**
 * A = A * (B * C)
 */
    template<class T1, class T> void MulMul_I(T1* d_o, T* B, T* C, int n, cudaStream_t stream=NULL);

/**
 * d_o = A * B + C
 */
    template<class T1, class T> void MulAdd(T1* d_o, T1* A, T* B, T1* C, int n, cudaStream_t stream=NULL);

/**
 * d_o = d_o * B + C
 */
    template<class T1, class T> void MulAdd_I(T1* d_o, T* B, T1* C, int n, cudaStream_t stream=NULL);

/**
 * d_o = A * B - C
 */
    template<class T1, class T> void MulSub(T1* d_o, T1* A, T* B, T1* C, int n, cudaStream_t stream=NULL);

/**
 * d_o = d_o * B - C
 */
    template<class T1, class T> void MulSub_I(T1* d_o, T* B, T1* C, int n, cudaStream_t stream=NULL);

/**
 * d_o = (A + B) * C
 */
    template<class T1, class T> void AddMul(T1* d_o, T1* A, T1* B, T* C, int n, cudaStream_t stream=NULL);

/**
 * d_o = (d_o + B) * C
 */
    template<class T1, class T> void AddMul_I(T1* d_o, T1* B, T* C, int n, cudaStream_t stream=NULL);

/**
 * d_o = (A + B) / C
 */
    template<class T1, class T> void AddDiv(T1* d_o, T1* A, T1* B, T* C, int n, cudaStream_t stream=NULL);
/**
 * d_o = (d_o + B) / C
 */
    template<class T1, class T> void AddDiv_I(T1* d_o, T1* B, T* C, int n, cudaStream_t stream=NULL);

/**
 * d_o = (A - B) * C
 */
    template<class T1, class T> void SubMul(T1* d_o, T1* A, T1* B, T* C, int n, cudaStream_t stream=NULL);
/**
 * d_o = (d_o - B) * C
 */
    template<class T1, class T> void SubMul_I(T1* d_o, T1* B, T* C, int n, cudaStream_t stream=NULL);
/**
 * d_o = (A - B) / C
 */
    template<class T1, class T> void SubDiv(T1* d_o, T1* A, T1* B, T* C, int n, cudaStream_t stream=NULL);
/**
 * d_o = (d_o + B) / C
 */
    template<class T1, class T> void SubDiv_I(T1* d_o, T1* B, T* C, int n, cudaStream_t stream=NULL);

/**
 * d_o = (A * b) + C
 */
    template<class T1, class T> void MulCAdd(T1* d_o, T1* A, T b, T1* C, int n, cudaStream_t stream=NULL);
/**
 * d_o = (d_o * b) + C
 */
    template<class T1, class T> void MulCAdd_I(T1* d_o, T b, T1* C, int n, cudaStream_t stream=NULL);
/**
 * d_o = (A * b) - C
 */
    template<class T1, class T> void MulCSub(T1* d_o, T1* A, T b, T1* C, int n, cudaStream_t stream=NULL);
/**
 * d_o = (d_o * b) - C
 */
    template<class T1, class T> void MulCSub_I(T1* d_o, T b, T1* C, int n, cudaStream_t stream=NULL);

/**
 * d_o = (A * b) + c
 */
    template<class T1, class T> void MulCAddC(T1* d_o, T1* A, T b, T1 c, int n, cudaStream_t stream=NULL);
/**
 * d_o = (d_o * b) + c
 */
    template<class T1, class T> void MulCAddC_I(T1* d_o, T b, T1 c, int n, cudaStream_t stream=NULL);

/**
 * d_o = (A + B) * c
 */
    template<class T1, class T> void AddMulC(T1* d_o, T1* A, T1* B, T c, int n, cudaStream_t stream=NULL);
/**
 * d_o = (d_o + B) * c
 */
    template<class T1, class T> void AddMulC_I(T1* d_o, T1* B, T c, int n, cudaStream_t stream=NULL);
/**
 * d_o = (A - B) * c
 */
    template<class T1, class T> void SubMulC(T1* d_o, T1* A, T1* B, T c, int n, cudaStream_t stream=NULL);
/**
 * d_o = (d_o - B) * c
 */
    template<class T1, class T> void SubMulC_I(T1* d_o, T1* B, T c, int n, cudaStream_t stream=NULL);

/**
 * d_o = (d_i + a) * b
 */
    template<class T> void AddCMulC(T* d_o, T* d_i, T a, T b, int n, cudaStream_t stream=NULL);

/**
 * d_o = (d_o + a) * b
 */
    template<class T> void AddCMulC_I(T* d_o, T a, T b, int n, cudaStream_t stream=NULL);

/**
 * d_o = d_i + d_A * c
 */
    template<class T> void Add_MulC(T* d_o, T* d_i, T* d_A, T c, int n, cudaStream_t stream=NULL);
/**
 * d_o = d_o + d_A * c
 */
    template<class T> void Add_MulC_I(T* d_o, T* d_A, T c, int n, cudaStream_t stream=NULL);

/**
 * d_o = d_i + d_A * d_C
 */
    template<class T> void Add_Mul(T* d_o, T* d_i, T* d_A, T* d_C, int n, cudaStream_t stream=NULL);
/**
 * d_o = d_o + d_A * d_C
 */
    template<class T> void Add_Mul_I(T* d_data, T* d_A, T* d_C, int n, cudaStream_t stream=NULL);

/////////////////////////////////////////////////////////////////////////////////
// n-ary functions
//////////////////////////////////////////////////////////////////////////////////

/**
 * d_o = d_A + (d_b + d_C) * d
 */
    template<class T> void Add_AddMulC(T* d_o, const T* d_A, const T* d_B, const T* d_C, T d, int n, cudaStream_t stream=NULL);

/**
 * d_o = d_A + (d_b - d_C) * d
 */
    template<class T> void Add_SubMulC(T* d_o, const T* d_A, const T* d_B, const T* d_C, T d, int n, cudaStream_t stream=NULL);

/**
 * d_o = d_A + d_b * d_C * d
 */
    template<class T> void Add_MulMulC(T* d_o, const T* d_A, const T* d_B, const T* d_C, T d, int n, cudaStream_t stream=NULL);

/**
 * d_o = d_o + (d_b + d_C) * d
 */
    template<class T> void Add_AddMulC_I(T* d_o, const T* d_B, const T* d_C, T d, int n, cudaStream_t stream=NULL);
/**
 * d_o = d_o + (d_b - d_C) * d
 */
    template<class T> void Add_SubMulC_I(T* d_o, const T* d_B, const T* d_C, T d, int n, cudaStream_t stream=NULL);
/**
 * d_o = d_o + d_b * d_C * d
 */
    template<class T>
    void Add_MulMulC_I(T* d_o, const T* d_B, const T* d_C, T d, int n, cudaStream_t stream=NULL);

// Functions with 4 inputs

/**
 * d_o = d_i * a + d_i1 * b
 */
    template<class T2, class T>
    void MulC_Add_MulC(T2* d_o, T2* d_i, T a, T2* d_i1, T b, int n, cudaStream_t stream=NULL);

/**
 * d_o = d_o * a + d_i1 * b
 */
    template<class T2, class T>
    void MulC_Add_MulC_I(T2* d_o, T a, T2* d_i, T b, int n, cudaStream_t stream=NULL);

/**
 * d_o = (d_i + a) * b + c
 */
    template<class T>
    void AddCMulCAddC(T* d_o, const T* d_i, T a, T b, T c, int n, cudaStream_t stream=NULL);

/**
 * d_o = (d_o + a) * b + c
 */
    template<class T>
    void AddCMulCAddC_I(T* d_o, T a, T b, T c, int n, cudaStream_t stream=NULL);

/**
 * d_o[i] = d_i[n-1 - i]
 */
    template<typename T>
    void ReverseOrder(T* d_o, const T* d_i, int n, cudaStream_t stream=NULL);

    void FixedToFloating(float* d_dst, int* d_src, unsigned int n, cudaStream_t stream=NULL);
    void FixedToFloatingUnnomalized(float* d_dst, int* d_src, float c, unsigned int n, cudaStream_t stream=NULL);


/**
 * d_data -= d_var * eps 
 */
    template<class T> void EpsUpdate(T* d_data, T* d_var, T eps, int n, cudaStream_t stream=NULL);

/**
 * v[id] += epsilon * U[id]; 
 */
    void AddEpsilonStep(float* v, float* U, float epsilon, int nElems, cudaStream_t stream=NULL);

/**
 * Interpolate the image using the linear intepolation
 *    d_o = d_i0 * (1 - t) + d_i1 * t
 */
    void Interpolate(float* d_o, float* d_i0, float* d_i1, float t, int nElems, cudaStream_t stream=NULL);

}
#endif
