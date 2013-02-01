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

#include <cutil_math.h>
#include "VectorMathExt.h"
#include "VectorMathExtDef.h"

// Constant Paramatter from Device 
namespace cplVectorCPDOpers {
    template void SetMem(float* d_o, const float* d_c, int n, cudaStream_t stream);
    template void SetMem(float2* d_o, const float2* d_c, int n, cudaStream_t stream);
    template void SetMem(float4* d_o, const float4* d_c, int n, cudaStream_t stream);

    template void SetMem(int* d_o, const int* d_c, int n, cudaStream_t stream);
    template void SetMem(int2* d_o, const int2* d_c, int n, cudaStream_t stream);
    template void SetMem(int4* d_o, const int4* d_c, int n, cudaStream_t stream);

    template void SetMem(uint* d_o, const uint* d_c, int n, cudaStream_t stream);
    template void SetMem(uint2* d_o, const uint2* d_c, int n, cudaStream_t stream);
    template void SetMem(uint4* d_o, const uint4* d_c, int n, cudaStream_t stream);
    
////////////////////////////////////////////////////////////////////////////////
//
////////////////////////////////////////////////////////////////////////////////
    template void AddC(float* d_o, float* d_i, const float* d_c, int n, cudaStream_t stream);
    template void AddC(float2* d_o, float2* d_i, const float2* d_c, int n, cudaStream_t stream);
    template void AddC(float4* d_o, float4* d_i, const float4* d_c, int n, cudaStream_t stream);

    template void AddC(int* d_o, int* d_i, const int* d_c, int n, cudaStream_t stream);
    template void AddC(int2* d_o, int2* d_i, const int2* d_c, int n, cudaStream_t stream);
    template void AddC(int4* d_o, int4* d_i, const int4* d_c, int n, cudaStream_t stream);

    template void AddC(uint* d_o, uint* d_i, const uint* d_c, int n, cudaStream_t stream);
    template void AddC(uint2* d_o, uint2* d_i, const uint2* d_c, int n, cudaStream_t stream);
    template void AddC(uint4* d_o, uint4* d_i, const uint4* d_c, int n, cudaStream_t stream);

    template void AddC_I(float* d_o, const float* d_c, int n, cudaStream_t stream);
    template void AddC_I(float2* d_o, const float2* d_c, int n, cudaStream_t stream);
    template void AddC_I(float4* d_o, const float4* d_c, int n, cudaStream_t stream);

    template void AddC_I(int* d_o, const int* d_c, int n, cudaStream_t stream);
    template void AddC_I(int2* d_o, const int2* d_c, int n, cudaStream_t stream);
    template void AddC_I(int4* d_o, const int4* d_c, int n, cudaStream_t stream);

    template void AddC_I(uint* d_o, const uint* d_c, int n, cudaStream_t stream);
    template void AddC_I(uint2* d_o, const uint2* d_c, int n, cudaStream_t stream);
    template void AddC_I(uint4* d_o, const uint4* d_c, int n, cudaStream_t stream);
////////////////////////////////////////////////////////////////////////////////
//
////////////////////////////////////////////////////////////////////////////////
    template void SubC(float* d_o, float* d_i, const float* d_c, int n, cudaStream_t stream);
    template void SubC(float2* d_o, float2* d_i, const float2* d_c, int n, cudaStream_t stream);
    template void SubC(float4* d_o, float4* d_i, const float4* d_c, int n, cudaStream_t stream);

    template void SubC(int* d_o, int* d_i, const int* d_c, int n, cudaStream_t stream);
    template void SubC(int2* d_o, int2* d_i, const int2* d_c, int n, cudaStream_t stream);
    template void SubC(int4* d_o, int4* d_i, const int4* d_c, int n, cudaStream_t stream);

    template void SubC(uint* d_o, uint* d_i, const uint* d_c, int n, cudaStream_t stream);
    template void SubC(uint2* d_o, uint2* d_i, const uint2* d_c, int n, cudaStream_t stream);
    template void SubC(uint4* d_o, uint4* d_i, const uint4* d_c, int n, cudaStream_t stream);

    template void SubC_I(float* d_o, const float* d_c, int n, cudaStream_t stream);
    template void SubC_I(float2* d_o, const float2* d_c, int n, cudaStream_t stream);
    template void SubC_I(float4* d_o, const float4* d_c, int n, cudaStream_t stream);

    template void SubC_I(int* d_o, const int* d_c, int n, cudaStream_t stream);
    template void SubC_I(int2* d_o, const int2* d_c, int n, cudaStream_t stream);
    template void SubC_I(int4* d_o, const int4* d_c, int n, cudaStream_t stream);

    template void SubC_I(uint* d_o, const uint* d_c, int n, cudaStream_t stream);
    template void SubC_I(uint2* d_o, const uint2* d_c, int n, cudaStream_t stream);
    template void SubC_I(uint4* d_o, const uint4* d_c, int n, cudaStream_t stream);

////////////////////////////////////////////////////////////////////////////////
//
////////////////////////////////////////////////////////////////////////////////
    template void MulC(float* d_o, float* d_i, const float* d_c, int n, cudaStream_t stream);
    template void MulC(float2* d_o, float2* d_i, const float2* d_c, int n, cudaStream_t stream);
    template void MulC(float4* d_o, float4* d_i, const float4* d_c, int n, cudaStream_t stream);

    template void MulC(int* d_o, int* d_i, const int* d_c, int n, cudaStream_t stream);
    template void MulC(int2* d_o, int2* d_i, const int2* d_c, int n, cudaStream_t stream);
    template void MulC(int4* d_o, int4* d_i, const int4* d_c, int n, cudaStream_t stream);

    template void MulC(uint* d_o, uint* d_i, const uint* d_c, int n, cudaStream_t stream);
    template void MulC(uint2* d_o, uint2* d_i, const uint2* d_c, int n, cudaStream_t stream);
    template void MulC(uint4* d_o, uint4* d_i, const uint4* d_c, int n, cudaStream_t stream);

    template void MulC_I(float* d_o, const float* d_c, int n, cudaStream_t stream);
    template void MulC_I(float2* d_o, const float2* d_c, int n, cudaStream_t stream);
    template void MulC_I(float4* d_o, const float4* d_c, int n, cudaStream_t stream);

    template void MulC_I(int* d_o, const int* d_c, int n, cudaStream_t stream);
    template void MulC_I(int2* d_o, const int2* d_c, int n, cudaStream_t stream);
    template void MulC_I(int4* d_o, const int4* d_c, int n, cudaStream_t stream);

    template void MulC_I(uint* d_o, const uint* d_c, int n, cudaStream_t stream);
    template void MulC_I(uint2* d_o, const uint2* d_c, int n, cudaStream_t stream);
    template void MulC_I(uint4* d_o, const uint4* d_c, int n, cudaStream_t stream);

////////////////////////////////////////////////////////////////////////////////
//
////////////////////////////////////////////////////////////////////////////////
    template void DivC(float* d_o, float* d_i, const float* d_c, int n, cudaStream_t stream);
    template void DivC(float2* d_o, float2* d_i, const float2* d_c, int n, cudaStream_t stream);
    template void DivC(float4* d_o, float4* d_i, const float4* d_c, int n, cudaStream_t stream);

    template void DivC_I(float* d_o, const float* d_c, int n, cudaStream_t stream);
    template void DivC_I(float2* d_o, const float2* d_c, int n, cudaStream_t stream);
    template void DivC_I(float4* d_o, const float4* d_c, int n, cudaStream_t stream);

////////////////////////////////////////////////////////////////////////////////
//
////////////////////////////////////////////////////////////////////////////////
    template void AddMulC(float * d_o, float * d_a, float * d_b, const float * d_c, int n, cudaStream_t stream);
    template void AddMulC(float2* d_o, float2* d_a, float2* d_b, const float2* d_c, int n, cudaStream_t stream);
    template void AddMulC(float4* d_o, float4* d_a, float4* d_b, const float4* d_c, int n, cudaStream_t stream);

    template void AddMulC(uint * d_o, uint * d_a, uint * d_b, const uint * d_c, int n, cudaStream_t stream);
    template void AddMulC(uint2* d_o, uint2* d_a, uint2* d_b, const uint2* d_c, int n, cudaStream_t stream);
    template void AddMulC(uint4* d_o, uint4* d_a, uint4* d_b, const uint4* d_c, int n, cudaStream_t stream);

    template void AddMulC(int * d_o, int * d_a, int * d_b, const int * d_c, int n, cudaStream_t stream);
    template void AddMulC(int2* d_o, int2* d_a, int2* d_b, const int2* d_c, int n, cudaStream_t stream);
    template void AddMulC(int4* d_o, int4* d_a, int4* d_b, const int4* d_c, int n, cudaStream_t stream);

////////////////////////////////////////////////////////////////////////////////
//
////////////////////////////////////////////////////////////////////////////////
    template void SubMulC(float * d_o, float * d_a, float * d_b, const float * d_c, int n, cudaStream_t stream);
    template void SubMulC(float2* d_o, float2* d_a, float2* d_b, const float2* d_c, int n, cudaStream_t stream);
    template void SubMulC(float4* d_o, float4* d_a, float4* d_b, const float4* d_c, int n, cudaStream_t stream);

    template void SubMulC(uint * d_o, uint * d_a, uint * d_b, const uint * d_c, int n, cudaStream_t stream);
    template void SubMulC(uint2* d_o, uint2* d_a, uint2* d_b, const uint2* d_c, int n, cudaStream_t stream);
    template void SubMulC(uint4* d_o, uint4* d_a, uint4* d_b, const uint4* d_c, int n, cudaStream_t stream);

    template void SubMulC(int * d_o, int * d_a, int * d_b, const int * d_c, int n, cudaStream_t stream);
    template void SubMulC(int2* d_o, int2* d_a, int2* d_b, const int2* d_c, int n, cudaStream_t stream);
    template void SubMulC(int4* d_o, int4* d_a, int4* d_b, const int4* d_c, int n, cudaStream_t stream);


////////////////////////////////////////////////////////////////////////////////
//
////////////////////////////////////////////////////////////////////////////////
    template void MulMulC(float * d_o, float * d_a, float * d_b, const float * d_c, int n, cudaStream_t stream);
    template void MulMulC(float2* d_o, float2* d_a, float2* d_b, const float2* d_c, int n, cudaStream_t stream);
    template void MulMulC(float4* d_o, float4* d_a, float4* d_b, const float4* d_c, int n, cudaStream_t stream);

    template void MulMulC(uint * d_o, uint * d_a, uint * d_b, const uint * d_c, int n, cudaStream_t stream);
    template void MulMulC(uint2* d_o, uint2* d_a, uint2* d_b, const uint2* d_c, int n, cudaStream_t stream);
    template void MulMulC(uint4* d_o, uint4* d_a, uint4* d_b, const uint4* d_c, int n, cudaStream_t stream);

    template void MulMulC(int * d_o, int * d_a, int * d_b, const int * d_c, int n, cudaStream_t stream);
    template void MulMulC(int2* d_o, int2* d_a, int2* d_b, const int2* d_c, int n, cudaStream_t stream);
    template void MulMulC(int4* d_o, int4* d_a, int4* d_b, const int4* d_c, int n, cudaStream_t stream);

////////////////////////////////////////////////////////////////////////////////
//
////////////////////////////////////////////////////////////////////////////////
    template void AddMulC_I(float * d_o, float * d_b, const float * d_c, int n, cudaStream_t stream);
    template void AddMulC_I(float2* d_o, float2* d_b, const float2* d_c, int n, cudaStream_t stream);
    template void AddMulC_I(float4* d_o, float4* d_b, const float4* d_c, int n, cudaStream_t stream);

    template void AddMulC_I(uint * d_o, uint * d_b, const uint * d_c, int n, cudaStream_t stream);
    template void AddMulC_I(uint2* d_o, uint2* d_b, const uint2* d_c, int n, cudaStream_t stream);
    template void AddMulC_I(uint4* d_o, uint4* d_b, const uint4* d_c, int n, cudaStream_t stream);

    template void AddMulC_I(int * d_o, int * d_b, const int * d_c, int n, cudaStream_t stream);
    template void AddMulC_I(int2* d_o, int2* d_b, const int2* d_c, int n, cudaStream_t stream);
    template void AddMulC_I(int4* d_o, int4* d_b, const int4* d_c, int n, cudaStream_t stream);

////////////////////////////////////////////////////////////////////////////////
//
////////////////////////////////////////////////////////////////////////////////
    template void SubMulC_I(float * d_o, float * d_b, const float * d_c, int n, cudaStream_t stream);
    template void SubMulC_I(float2* d_o, float2* d_b, const float2* d_c, int n, cudaStream_t stream);
    template void SubMulC_I(float4* d_o, float4* d_b, const float4* d_c, int n, cudaStream_t stream);

    template void SubMulC_I(uint * d_o, uint * d_b, const uint * d_c, int n, cudaStream_t stream);
    template void SubMulC_I(uint2* d_o, uint2* d_b, const uint2* d_c, int n, cudaStream_t stream);
    template void SubMulC_I(uint4* d_o, uint4* d_b, const uint4* d_c, int n, cudaStream_t stream);

    template void SubMulC_I(int * d_o, int * d_b, const int * d_c, int n, cudaStream_t stream);
    template void SubMulC_I(int2* d_o, int2* d_b, const int2* d_c, int n, cudaStream_t stream);
    template void SubMulC_I(int4* d_o, int4* d_b, const int4* d_c, int n, cudaStream_t stream);


////////////////////////////////////////////////////////////////////////////////
//
////////////////////////////////////////////////////////////////////////////////
    template void MulMulC_I(float * d_o, float * d_b, const float * d_c, int n, cudaStream_t stream);
    template void MulMulC_I(float2* d_o, float2* d_b, const float2* d_c, int n, cudaStream_t stream);
    template void MulMulC_I(float4* d_o, float4* d_b, const float4* d_c, int n, cudaStream_t stream);

    template void MulMulC_I(uint * d_o, uint * d_b, const uint * d_c, int n, cudaStream_t stream);
    template void MulMulC_I(uint2* d_o, uint2* d_b, const uint2* d_c, int n, cudaStream_t stream);
    template void MulMulC_I(uint4* d_o, uint4* d_b, const uint4* d_c, int n, cudaStream_t stream);

    template void MulMulC_I(int * d_o, int * d_b, const int * d_c, int n, cudaStream_t stream);
    template void MulMulC_I(int2* d_o, int2* d_b, const int2* d_c, int n, cudaStream_t stream);
    template void MulMulC_I(int4* d_o, int4* d_b, const int4* d_c, int n, cudaStream_t stream);
}; // cplVectorCPDOpers
