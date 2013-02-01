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
#include "VectorMathDef.h"

namespace cplVectorOpers {
    template void Copy(float* d_o, const float* d_i, unsigned int n, cudaStream_t stream);
    template void Copy(float2* d_o, const float2* d_i, unsigned int n, cudaStream_t stream);
    template void Copy(float4* d_o, const float4* d_i, unsigned int n, cudaStream_t stream);

    template void Copy(uint* d_o, const uint* d_i, unsigned int n, cudaStream_t stream);
    template void Copy(uint2* d_o, const uint2* d_i, unsigned int n, cudaStream_t stream);
    template void Copy(uint4* d_o, const uint4* d_i, unsigned int n, cudaStream_t stream);

    template void Copy(int* d_o, const int* d_i, unsigned int n, cudaStream_t stream);
    template void Copy(int2* d_o, const int2* d_i, unsigned int n, cudaStream_t stream);
    template void Copy(int4* d_o, const int4* d_i, unsigned int n, cudaStream_t stream);

//////////////////////////////////////////////////////////////////////////////////
// instatiate function
/////////////////////////////////////////////////////////////////////////////////
    template void SetMem(float*, float , int, cudaStream_t stream);
    template void SetMem(float2*, float2 , int, cudaStream_t stream);
    template void SetMem(float4*, float4 , int, cudaStream_t stream);

    template void SetMem(int*, int , int, cudaStream_t stream);
    template void SetMem(int2*, int2 , int, cudaStream_t stream);
    template void SetMem(int4*, int4 , int, cudaStream_t stream);

    template void SetMem(uint*, uint , int, cudaStream_t stream);
    template void SetMem(uint2*, uint2 , int, cudaStream_t stream);
    template void SetMem(uint4*, uint4 , int, cudaStream_t stream);

    template void SetLinear(float* d_data,  int len, cudaStream_t stream);
    template void SetLinear(uint* d_data,  int len, cudaStream_t stream);
    template void SetLinear(int* d_data,  int len, cudaStream_t stream);

    template void SetLinearDown(float* d_data,  int len, cudaStream_t stream);
    template void SetLinearDown(uint* d_data,  int len, cudaStream_t stream);
    template void SetLinearDown(int* d_data,  int len, cudaStream_t stream);
/////////////////////////////////////////////////////////////////////////////////
// Unary functions
//////////////////////////////////////////////////////////////////////////////////
    template void Abs(float* d_o, const float* d_i, unsigned int size, cudaStream_t stream);
    template void Sqr(float* d_o, const float* d_i, unsigned int size, cudaStream_t stream);
    template void Cube(float* d_o,const  float* d_i, unsigned int size, cudaStream_t stream);
    template void Neg(float* d_o,const  float* d_i, unsigned int size, cudaStream_t stream);

    template void Abs(int* d_o,const  int* d_i, unsigned int size, cudaStream_t stream);
    template void Sqr(int* d_o,const  int* d_i, unsigned int size, cudaStream_t stream);
    template void Cube(int* d_o,const  int* d_i, unsigned int size, cudaStream_t stream);
    template void Neg(int* d_o,const  int* d_i, unsigned int size, cudaStream_t stream);

    template void Sqr(uint* d_o,const  uint* d_i, unsigned int size, cudaStream_t stream);
    template void Cube(uint* d_o,const  uint* d_i, unsigned int size, cudaStream_t stream);

    template void Abs_I(float* d_o, unsigned int size, cudaStream_t stream);
    template void Sqr_I(float* d_o, unsigned int size, cudaStream_t stream);
    template void Cube_I(float* d_o, unsigned int size, cudaStream_t stream);
    template void Neg_I(float* d_o, unsigned int size, cudaStream_t stream);

    template void Abs_I(int* d_o, unsigned int size, cudaStream_t stream);
    template void Sqr_I(int* d_o, unsigned int size, cudaStream_t stream);
    template void Cube_I(int* d_o, unsigned int size, cudaStream_t stream);
    template void Neg_I(int* d_o, unsigned int size, cudaStream_t stream);

    template void Sqr_I(uint* d_o, unsigned int size, cudaStream_t stream);
    template void Cube_I(uint* d_o, unsigned int size, cudaStream_t stream);

/////////////////////////////////////////////////////////////////////////////////
// Binary function
//////////////////////////////////////////////////////////////////////////////////
    template void AddC(float* d_o, float* d_i, float c, int n, cudaStream_t stream);
    template void AddC_I(float* d_o, float c, int n, cudaStream_t stream);
    template void Add(float* d_o, float* d_i, float* d_i1, int n, cudaStream_t stream);
    template void Add_I(float* d_o, float* d_i, int n, cudaStream_t stream);

    template void SubC(float* d_o, float* d_i, float c, int n, cudaStream_t stream);
    template void SubC_I(float* d_o, float c, int n, cudaStream_t stream);
    template void Sub(float* d_o, float* d_i, float* d_i1, int n, cudaStream_t stream);
    template void Sub_I(float* d_o, float* d_i, int n, cudaStream_t stream);

    template void MulC(float* d_o, const float* d_i, float c, int n, cudaStream_t stream);
    template void MulC_I(float* d_o, float c, int n, cudaStream_t stream);
    template void Mul(float* d_o, float* d_i, float* d_i1, int n, cudaStream_t stream);
    template void Mul_I(float* d_o, float* d_i, int n, cudaStream_t stream);

    template void MulC_I(float2* d_o, float c, int n, cudaStream_t stream);
    template void MulC(float2* d_o, const float2* d_i, float c, int n, cudaStream_t stream);

    template void AddC(int* d_o, int* d_i, int c, int n, cudaStream_t stream);
    template void AddC_I(int* d_o, int c, int n, cudaStream_t stream);
    template void Add(int* d_o, int* d_i, int* d_i1, int n, cudaStream_t stream);
    template void Add_I(int* d_o, int* d_i, int n, cudaStream_t stream);

    template void SubC(int* d_o, int* d_i, int c, int n, cudaStream_t stream);
    template void SubC_I(int* d_o, int c, int n, cudaStream_t stream);
    template void Sub(int* d_o, int* d_i, int* d_i1, int n, cudaStream_t stream);
    template void Sub_I(int* d_o, int* d_i, int n, cudaStream_t stream);

    template void MulC(int* d_o, const int* d_i, int c, int n, cudaStream_t stream);
    template void MulC_I(int* d_o, int c, int n, cudaStream_t stream);
    template void Mul(int* d_o, int* d_i, int* d_i1, int n, cudaStream_t stream);
    template void Mul_I(int* d_o, int* d_i, int n, cudaStream_t stream);

    template void MulC_I(int2* d_o, int c, int n, cudaStream_t stream);
    template void MulC(int2* d_o, const int2* d_i, int c, int n, cudaStream_t stream);

    template void AddC(uint* d_o, uint* d_i, uint c, int n, cudaStream_t stream);
    template void AddC_I(uint* d_o, uint c, int n, cudaStream_t stream);
    template void Add(uint* d_o, uint* d_i, uint* d_i1, int n, cudaStream_t stream);
    template void Add_I(uint* d_o, uint* d_i, int n, cudaStream_t stream);

    template void SubC(uint* d_o, uint* d_i, uint c, int n, cudaStream_t stream);
    template void SubC_I(uint* d_o, uint c, int n, cudaStream_t stream);
    template void Sub(uint* d_o, uint* d_i, uint* d_i1, int n, cudaStream_t stream);
    template void Sub_I(uint* d_o, uint* d_i, int n, cudaStream_t stream);

    template void MulC(uint* d_o, const uint* d_i, uint c, int n, cudaStream_t stream);
    template void MulC_I(uint* d_o, uint c, int n, cudaStream_t stream);
    template void Mul(uint* d_o, uint* d_i, uint* d_i1, int n, cudaStream_t stream);
    template void Mul_I(uint* d_o, uint* d_i, int n, cudaStream_t stream);

    template void MulC_I(uint2* d_o, uint c, int n, cudaStream_t stream);
    template void MulC(uint2* d_o, const uint2* d_i, uint c, int n, cudaStream_t stream);

    // Division only make sense with float
    template void DivC(float* d_o, const float* d_i, float c, int n, cudaStream_t stream);
    template void DivC_I(float* d_o, float c, int n, cudaStream_t stream);
    template void Div(float* d_o, float* d_i, float* d_i1, int n, cudaStream_t stream);
    template void Div_I(float* d_o, float* d_i, int n, cudaStream_t stream);

    template void DivC_I(float2* d_o, float c, int n, cudaStream_t stream);
    template void DivC(float2* d_o, const float2* d_i, float c, int n, cudaStream_t stream);

/////////////////////////////////////////////////////////////////////////////////
// Trinary functions
//////////////////////////////////////////////////////////////////////////////////
    template void AbsDiff(float* d_o,const float* d_i,const  float* d_i1, int n, cudaStream_t stream);
    template void AbsDiff(int* d_o,const int* d_i,const int* d_i1, int n, cudaStream_t stream);
    template void AbsDiff(uint* d_o,const uint* d_i,const uint* d_i1, int n, cudaStream_t stream);

    template void AbsDiff_I(float* d_o,const  float* d_i, int n, cudaStream_t stream);
    template void AbsDiff_I(int* d_o,const  int* d_i, int n, cudaStream_t stream);
    template void AbsDiff_I(uint* d_o,const  uint* d_i, int n, cudaStream_t stream);

    template void SqrDiff(float* d_o,const float* d_i,const  float* d_i1, int n, cudaStream_t stream);
    template void SqrDiff(int* d_o,const int* d_i,const int* d_i1, int n, cudaStream_t stream);
    template void SqrDiff(uint* d_o,const uint* d_i,const uint* d_i1, int n, cudaStream_t stream);

    template void SqrDiff_I(float* d_o,const  float* d_i, int n, cudaStream_t stream);
    template void SqrDiff_I(int* d_o,const  int* d_i, int n, cudaStream_t stream);
    template void SqrDiff_I(uint* d_o,const  uint* d_i, int n, cudaStream_t stream);

    template void MulMulC<float, float>(float* d_o,float* a, float* b, float c, int n, cudaStream_t stream);
    template void MulMulC<float2, float>(float2* d_o,float2* a, float* b, float c, int n, cudaStream_t stream);
    template void MulMulC_I<float, float>(float* a, float* b, float c, int n, cudaStream_t stream);
    template void MulMulC_I<float2, float>(float2* a, float* b, float c, int n, cudaStream_t stream);

    template void MulMul<float, float>(float* d_o, float* a, float* b, float* c, int n, cudaStream_t stream);
    template void MulMul<float2, float>(float2* d_o,float2* a, float* b, float* c, int n, cudaStream_t stream);
    template void MulMul<float4, float>(float4* d_o,float4* a, float* b, float* c, int n, cudaStream_t stream);
    template void MulMul_I<float, float>(float* a, float* b, float* c, int n, cudaStream_t stream);
    template void MulMul_I<float2, float>(float2* a, float* b, float* c, int n, cudaStream_t stream);
    template void MulMul_I<float4, float>(float4* a, float* b, float* c, int n, cudaStream_t stream);

    template void MulAdd<float, float>(float* d_o, float* a, float* b, float* c, int n, cudaStream_t stream);
    template void MulAdd<float2, float>(float2* d_o, float2* a, float* b, float2* c, int n, cudaStream_t stream);
    template void MulAdd<float4, float>(float4* d_o, float4* a, float* b, float4* c, int n, cudaStream_t stream);
    template void MulAdd_I<float, float>(float* d_o,   float* b, float* c, int n, cudaStream_t stream);
    template void MulAdd_I<float2, float>(float2* d_o, float* b, float2* c, int n, cudaStream_t stream);
    template void MulAdd_I<float4, float>(float4* d_o, float* b, float4* c, int n, cudaStream_t stream);

    template void MulSub<float, float>(float* d_o, float* a, float* b, float* c, int n, cudaStream_t stream);
    template void MulSub<float2, float>(float2* d_o, float2* a, float* b, float2* c, int n, cudaStream_t stream);
    template void MulSub<float4, float>(float4* d_o, float4* a, float* b, float4* c, int n, cudaStream_t stream);
    template void MulSub_I<float, float>(float* d_o,   float* b, float* c, int n, cudaStream_t stream);
    template void MulSub_I<float2, float>(float2* d_o, float* b, float2* c, int n, cudaStream_t stream);
    template void MulSub_I<float4, float>(float4* d_o, float* b, float4* c, int n, cudaStream_t stream);

    template void AddDiv<float, float>(float* d_o, float* a, float* b, float* c, int n, cudaStream_t stream);
    template void AddDiv<float2, float>(float2* d_o, float2* a, float2* b, float* c, int n, cudaStream_t stream);
    template void AddDiv<float4, float>(float4* d_o, float4* a, float4* b, float* c, int n, cudaStream_t stream);
    template void AddDiv_I<float, float>(float* d_o, float* b, float* c, int n, cudaStream_t stream);
    template void AddDiv_I<float2, float>(float2* d_o, float2* b, float* c, int n, cudaStream_t stream);
    template void AddDiv_I<float4, float>(float4* d_o, float4* b, float* c, int n, cudaStream_t stream);

    template void AddMul<float, float>(float* d_o, float* a, float* b, float* c, int n, cudaStream_t stream);
    template void AddMul<float2, float>(float2* d_o, float2* a, float2* b, float* c, int n, cudaStream_t stream);
    template void AddMul<float4, float>(float4* d_o, float4* a, float4* b, float* c, int n, cudaStream_t stream);
    template void AddMul_I<float, float>(float* d_o, float* b, float* c, int n, cudaStream_t stream);
    template void AddMul_I<float2, float>(float2* d_o, float2* b, float* c, int n, cudaStream_t stream);
    template void AddMul_I<float4, float>(float4* d_o, float4* b, float* c, int n, cudaStream_t stream);

    template void SubMul<float, float>(float* d_o, float* a, float* b, float* c, int n, cudaStream_t stream);
    template void SubMul<float2, float>(float2* d_o, float2* a, float2* b, float* c, int n, cudaStream_t stream);
    template void SubMul<float4, float>(float4* d_o, float4* a, float4* b, float* c, int n, cudaStream_t stream);
    template void SubMul_I<float, float>(float* d_o, float* b, float* c, int n, cudaStream_t stream);
    template void SubMul_I<float2, float>(float2* d_o, float2* b, float* c, int n, cudaStream_t stream);
    template void SubMul_I<float4, float>(float4* d_o, float4* b, float* c, int n, cudaStream_t stream);

    template void MulCAdd<float, float>(float* d_o, float* a, float b, float* c, int n, cudaStream_t stream);
    template void MulCAdd<float2, float>(float2* d_o, float2* a, float b, float2* c, int n, cudaStream_t stream);
    template void MulCAdd<float4, float>(float4* d_o, float4* a, float b, float4* c, int n, cudaStream_t stream);

    template void MulCAdd_I<float, float>(float* d_o, float b, float* c, int n, cudaStream_t stream);
    template void MulCAdd_I<float2, float>(float2* d_o, float b, float2* c, int n, cudaStream_t stream);
    template void MulCAdd_I<float4, float>(float4* d_o, float b, float4* c, int n, cudaStream_t stream);

    template void MulCAddC<float, float>(float* d_o, float* a, float b, float c, int n, cudaStream_t stream);
    template void MulCAddC<float2, float>(float2* d_o, float2* a, float b, float2 c, int n, cudaStream_t stream);
    template void MulCAddC<float4, float>(float4* d_o, float4* a, float b, float4 c, int n, cudaStream_t stream);

    template void MulCAddC_I<float, float>(float* d_o, float b, float c, int n, cudaStream_t stream);
    template void MulCAddC_I<float2, float>(float2* d_o, float b, float2 c, int n, cudaStream_t stream);
    template void MulCAddC_I<float4, float>(float4* d_o, float b, float4 c, int n, cudaStream_t stream);

    template void MulCSub<float, float>(float* d_o, float* a, float b, float* c, int n, cudaStream_t stream);
    template void MulCSub<float2, float>(float2* d_o, float2* a, float b, float2* c, int n, cudaStream_t stream);
    template void MulCSub<float4, float>(float4* d_o, float4* a, float b, float4* c, int n, cudaStream_t stream);

    template void MulCSub_I<float, float>(float* d_o, float b, float* c, int n, cudaStream_t stream);
    template void MulCSub_I<float2, float>(float2* d_o, float b, float2* c, int n, cudaStream_t stream);
    template void MulCSub_I<float4, float>(float4* d_o, float b, float4* c, int n, cudaStream_t stream);

    template void AddMulC<float, float>(float* d_o, float* a, float* b, float c, int n, cudaStream_t stream);
    template void AddMulC<float2, float>(float2* d_o, float2* a, float2* b, float c, int n, cudaStream_t stream);
    template void AddMulC<float4, float>(float4* d_o, float4* a, float4* b, float c, int n, cudaStream_t stream);

    template void AddMulC_I<float, float>(float* d_o,  float* b, float c, int n, cudaStream_t stream);
    template void AddMulC_I<float2, float>(float2* d_o,float2* b, float c, int n, cudaStream_t stream);
    template void AddMulC_I<float4, float>(float4* d_o,float4* b, float c, int n, cudaStream_t stream);

    template void SubMulC<float, float>(float* d_o, float* a, float* b, float c, int n, cudaStream_t stream);
    template void SubMulC<float2, float>(float2* d_o, float2* a, float2* b, float c, int n, cudaStream_t stream);
    template void SubMulC<float4, float>(float4* d_o, float4* a, float4* b, float c, int n, cudaStream_t stream);

    template void SubMulC_I<float, float>(float* d_o,  float* b, float c, int n, cudaStream_t stream);
    template void SubMulC_I<float2, float>(float2* d_o,float2* b, float c, int n, cudaStream_t stream);
    template void SubMulC_I<float4, float>(float4* d_o,float4* b, float c, int n, cudaStream_t stream);

    template void EpsUpdate<float>(float* d_data, float* d_var, float eps, int n, cudaStream_t stream);

    template void AddCMulC_I(float* d_o, float a, float b, int n, cudaStream_t stream);
    template void AddCMulC(float* d_o, float* d_i, float a, float b, int n, cudaStream_t stream);

    template void Add_MulC_I(float* d_data, float* d_a, float c, int n, cudaStream_t stream);
    template void Add_MulC_I(uint* d_data, uint* d_a, uint c, int n, cudaStream_t stream);
    template void Add_MulC_I(int* d_data, int* d_a, int c, int n, cudaStream_t stream);

    template void Add_MulC(float* d_o, float* d_i, float* d_a, float c, int n, cudaStream_t stream);
    template void Add_MulC(uint* d_o, uint* d_i, uint* d_a, uint c, int n, cudaStream_t stream);
    template void Add_MulC(int* d_o, int* d_i, int* d_a, int c, int n, cudaStream_t stream);

    template void Add_Mul_I(float* d_data, float* d_a, float* c, int n, cudaStream_t stream);
    template void Add_Mul_I(uint* d_data, uint* d_a, uint* c, int n, cudaStream_t stream);
    template void Add_Mul_I(int* d_data, int* d_a, int* c, int n, cudaStream_t stream);

    template void Add_Mul(float* d_o, float* d_i, float* d_a, float* c, int n, cudaStream_t stream);
    template void Add_Mul(uint* d_o, uint* d_i, uint* d_a, uint* c, int n, cudaStream_t stream);
    template void Add_Mul(int* d_o, int* d_i, int* d_a, int* c, int n, cudaStream_t stream);

/////////////////////////////////////////////////////////////////////////////////
// n-ary functions
//////////////////////////////////////////////////////////////////////////////////
    template void MulC_Add_MulC<float, float>(float*, float*, float, float*, float, int, cudaStream_t stream);
    template void MulC_Add_MulC<float2, float>(float2*, float2*, float, float2*, float, int, cudaStream_t stream);
    template void MulC_Add_MulC<float4, float>(float4*, float4*, float, float4*, float, int, cudaStream_t stream);

    template void MulC_Add_MulC_I<float, float>(float*, float, float*, float, int, cudaStream_t stream);
    template void MulC_Add_MulC_I<float2, float>(float2*, float, float2*, float, int, cudaStream_t stream);
    template void MulC_Add_MulC_I<float4, float>(float4*, float, float4*, float, int, cudaStream_t stream);

    template void Add_AddMulC(float* d_o, float* d_a, float* d_b, float* d_c, float d, int n, cudaStream_t stream);
    template void Add_AddMulC(uint* d_o, uint* d_a, uint* d_b, uint* d_c, uint d, int n, cudaStream_t stream);
    template void Add_AddMulC(int* d_o, int* d_a, int* d_b, int* d_c, int d, int n, cudaStream_t stream);

    template void Add_AddMulC_I(float* d_o, float* d_b, float* d_c, float d, int n, cudaStream_t stream);
    template void Add_AddMulC_I(uint* d_o, uint* d_b, uint* d_c, uint d, int n, cudaStream_t stream);
    template void Add_AddMulC_I(int* d_o, int* d_b, int* d_c, int d, int n, cudaStream_t stream);

    template void Add_SubMulC(float* d_o, const float* d_a, const float* d_b, const float* d_c, float d, int n, cudaStream_t stream);
    template void Add_SubMulC(uint* d_o, const uint* d_a, const uint* d_b, const uint* d_c, uint d, int n, cudaStream_t stream);
    template void Add_SubMulC(int* d_o, const int* d_a, const int* d_b, const int* d_c, int d, int n, cudaStream_t stream);

    template void Add_SubMulC_I(float* d_o, const float* d_b, const float* d_c, float d, int n, cudaStream_t stream);
    template void Add_SubMulC_I(uint* d_o, const uint* d_b, const uint* d_c, uint d, int n, cudaStream_t stream);
    template void Add_SubMulC_I(int* d_o, const int* d_b, const int* d_c, int d, int n, cudaStream_t stream);

    template void Add_MulMulC(float* d_o, const float* d_a, const float* d_b, const float* d_c, float d, int n, cudaStream_t stream);
    template void Add_MulMulC(uint* d_o, const uint* d_a, const uint* d_b, const uint* d_c, uint d, int n, cudaStream_t stream);
    template void Add_MulMulC(int* d_o, const int* d_a, const int* d_b, const int* d_c, int d, int n, cudaStream_t stream);

    template void Add_MulMulC_I(float* d_o, const float* d_b, const float* d_c, float d, int n, cudaStream_t stream);
    template void Add_MulMulC_I(uint* d_o, const uint* d_b, const uint* d_c, uint d, int n, cudaStream_t stream);
    template void Add_MulMulC_I(int* d_o, const int* d_b, const int* d_c, int d, int n, cudaStream_t stream);

    template void AddCMulCAddC(float* d_o, const float* d_i, float a, float b, float c, int n, cudaStream_t stream);
    template void AddCMulCAddC(int* d_o, const int* d_i, int a, int b, int c, int n, cudaStream_t stream);
    template void AddCMulCAddC(uint* d_o, const uint* d_i, uint a, uint b, uint c, int n, cudaStream_t stream);

    template void AddCMulCAddC_I(float* d_o, float a, float b, float c, int n, cudaStream_t stream);
    template void AddCMulCAddC_I(int* d_o, int a, int b, int c, int n, cudaStream_t stream);
    template void AddCMulCAddC_I(uint* d_o, uint a, uint b, uint c, int n, cudaStream_t stream);


    template void ReverseOrder(float* d_o, const float* d_i, int n, cudaStream_t stream);
    template void ReverseOrder(int* d_o, const int* d_i, int n, cudaStream_t stream);
    template void ReverseOrder(uint* d_o, const uint* d_i, int n, cudaStream_t stream);
};
