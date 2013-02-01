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

#ifndef __OPER_UTIL_H__
#define __OPER_UTIL_H__

#include <limits.h>
#include <float.h>
#include <cutil_math.h>

enum MathOperator {
    MATH_ADD,
    MATH_AND,
    MATH_ABS,
    MATH_CUBE,
    MATH_DIV,
    MATH_INV,
    MATH_MAX,
    MATH_MIN,
    MATH_MUL,
    MATH_NEG,
    MATH_OR,
    MATH_SQRT,
    MATH_SQR,
    MATH_SUB,
    MATH_LERP,
    MATH_XOR,
    MATH_CLAMP,    
}; 

enum CompareOperator {
    CMP_GREATER,
    CMP_LESS,
    CMP_EQUAL,
    CMP_LESS_OR_EQUAL,
    CMP_GREATER_OR_EQUAL
};

template <typename T, CompareOperator oper>
class CmpOperator
{
public:
    //unary function 
    static __host__ __device__ uint op(T a, T b){
        switch (oper){
            case CMP_GREATER:
                return (a > b);
            case CMP_LESS :
                return (a < b);
            case CMP_EQUAL:
                return (a == b);
            case CMP_LESS_OR_EQUAL:
                return (a <= b);
            case CMP_GREATER_OR_EQUAL:
                return (a >= b);
            default:
                return 0;
        }         
    }

    static __host__ __device__ T different(T c) {
        switch (oper){
            case CMP_GREATER:
                return c;
            case CMP_LESS :
                return c;
            case CMP_EQUAL:
                return c + 1;
            case CMP_LESS_OR_EQUAL:
                return c + 1;
            case CMP_GREATER_OR_EQUAL:
                return c - 1;
            default:
                return c;
        }
    }
};

    
template <typename T, MathOperator oper>
class MOperator
{
public:
    //unary function 
    static __host__ __device__ T op(T a){
        switch (oper){
            default:
            case MATH_CUBE:
                return a * a * a;
            case MATH_NEG:
                return -a;
            case MATH_SQR: 
                return a * a;
        }         
    }

    // binary function 
    static __host__ __device__ T op(T a, T b){
        switch (oper){
            default:
            case MATH_ADD: 
                return a + b;

            case MATH_DIV:
                return a / b;
                
            case MATH_MUL:
                return a * b;
                
            case MATH_MIN: 
                return min(a, b);
                
            case MATH_MAX:
                return max(a, b);

            case MATH_SUB: 
                return a - b;
                
        }
    }

    static __host__ __device__ T op(T a , T b, T t){
        switch (oper){
            default:
            case MATH_LERP:
                return a + t * (b-a);
            case MATH_CLAMP:
                return max(a, min(b, t));
        }
    }
    
    // inplace operation 
    static __host__ __device__ void iop(T &a, T b){
        switch (oper){
            default:
            case MATH_ADD: 
                a += b;
                return;
                
            case MATH_DIV: 
                a /= b;
                return;

            case MATH_SUB: 
                a -= b;
                return;
                
            case MATH_MUL:
                a *= b;
                return;
                
            case MATH_MIN: 
                a = min(a, b);
                return;
                
            case MATH_MAX:
                a = max(a, b);
                return;
        }         
    }

    static __host__ __device__ T identity() {
        return 0;
    }
};

template <MathOperator oper>
class MOperator<float, oper>
{
public:
    //unary function 
    static __host__ __device__ float op(float a){
        switch (oper){
            default:
            case MATH_ABS:
                return fabsf(a);

            case MATH_CUBE:
                return a * a * a;
                
            case MATH_SQRT: 
                return sqrtf(a);

            case MATH_SQR: 
                return a * a;

            case MATH_INV:
                return 1.f / a;
                
            case MATH_NEG:
                return -a;
        }         
    }

    // binary function 
    static __host__ __device__ float op(float a, float b){
        switch (oper){
            default:
            case MATH_ADD: 
                return a + b;

            case MATH_DIV:
                return a / b;
                
            case MATH_MUL:
                return a * b;
                
            case MATH_MIN: 
                return fminf(a, b);
                
            case MATH_MAX:
                return fmaxf(a, b);

            case MATH_SUB: 
                return a - b;
        }
    }

    static __host__ __device__ float op(float a , float b, float t){
        switch (oper){
            default:
            case MATH_LERP:
                return a + t * (b-a);
            case MATH_CLAMP:
                return fmaxf(a, fminf(b, t));
        }
    }
    
    // inplace operation 
    static __host__ __device__ void iop(float &a, float b){
        switch (oper){
            default:

            case MATH_ADD: 
                a += b;
                return;
                
            case MATH_DIV: 
                a /= b;
                return;

            case MATH_SUB: 
                a -= b;
                return;
                
            case MATH_MUL:
                a *= b;
                return;
                
            case MATH_MIN: 
                a = fminf(a, b);
                return;
                
            case MATH_MAX:
                a = fmaxf(a, b);
                return;
        }
    }
    
    static __host__ __device__ float identity() {
        switch (oper){
            default:
            case MATH_ADD: 
                return 0;
                
            case MATH_SUB: 
                return 0;

            case MATH_MUL:
                return 1;

            case MATH_DIV:
                return 1;
                
            case MATH_MIN: 
                return FLT_MAX;
                
            case MATH_MAX:
                return -FLT_MAX;
        }
    }
};

template <MathOperator oper>
class MOperator<int, oper>
{
public:

    //unary function 
    static __host__ __device__ int op(int a){
        switch (oper){
            default:
            case MATH_ABS:
                return abs(a);

            case MATH_CUBE:
                return a * a * a;
                
            case MATH_SQR: 
                return a * a;

            case MATH_NEG:
                return -a;
        }         
    }

    // binary function 
    static __host__ __device__ int op(int a, int b){
        switch (oper){
            default:
            case MATH_ADD: 
                return a + b;
                
            case MATH_AND: 
                return a & b;

            case MATH_OR: 
                return a | b;

            case MATH_XOR: 
                return a ^ b;

            case MATH_DIV:
                return a / b;
                
            case MATH_MUL:
                return a * b;
                
            case MATH_MIN: 
                return min(a, b);
                
            case MATH_MAX:
                return max(a, b);

            case MATH_SUB: 
                return a - b;
                
        }
    }

    // inplace operation 
    static __host__ __device__ void iop(int &a, int b){
        switch (oper){
            default:
            case MATH_ADD: 
                a += b;
                return;

            case MATH_AND:
                a &= b;
                return;

            case MATH_OR:
                a |= b;
                return;

            case MATH_XOR:
                a ^= b;
                return;

            case MATH_DIV: 
                a /= b;
                return;

            case MATH_SUB: 
                a -= b;
                return;
                
            case MATH_MUL:
                a *= b;
                return;
                
            case MATH_MIN: 
                a = min(a, b);
                return;
                
            case MATH_MAX:
                a = max(a, b);
                return;
        }         
    }

    static __host__ __device__ int identity() {
        switch (oper){
            default:
            case MATH_AND:
                return 0xFFFFFFFF;

            case MATH_OR:
                return 0;

            case MATH_ADD: 
                return 0;
                
            case MATH_MUL:
                return 1;
                
            case MATH_MIN: 
                return INT_MAX;
                
            case MATH_MAX:
                return INT_MIN;
        }
    }

};

template <MathOperator oper>
class MOperator<unsigned int, oper>
{
public:
    //unary function 
    static __host__ __device__ uint op(uint a){
        switch (oper){
            default:
            case MATH_CUBE:
                return a * a * a;
                
            case MATH_SQR: 
                return a * a;
        }         
    }

    // binary function 
    static __host__ __device__ uint op(uint a, uint b){
        switch (oper){
            default:
            case MATH_ADD: 
                return a + b;
                
            case MATH_AND: 
                return a & b;

            case MATH_OR: 
                return a | b;

            case MATH_XOR: 
                return a ^ b;

            case MATH_DIV:
                return a / b;
                
            case MATH_MUL:
                return a * b;
                
            case MATH_MIN: 
                return min(a, b);
                
            case MATH_MAX:
                return max(a, b);
        }
    }
    
    // inplace operation 
    static __host__ __device__ void iop(uint &a, uint b){
        switch (oper){
            default:
            case MATH_ADD: 
                a += b;
                return;

            case MATH_AND:
                a &= b;
                return;

            case MATH_OR:
                a |= b;
                return;

            case MATH_XOR:
                a ^= b;
                return;

            case MATH_DIV: 
                a /= b;
                return;

            case MATH_SUB: 
                a -= b;
                return;
                
            case MATH_MUL:
                a *= b;
                return;
                
            case MATH_MIN: 
                a = min(a, b);
                return;
                
            case MATH_MAX:
                a = max(a, b);
                return;
        }         
    }

    static __host__ __device__ uint identity() {
        switch (oper){
            default:
            case MATH_AND:
                return 0xFFFFFFFF;

            case MATH_OR:
                return 0;

            case MATH_ADD: 
                return 0;
                
            case MATH_MUL:
                return 1;
                
            case MATH_MIN: 
                return 0xffffffff;
                
            case MATH_MAX:
                return 0;
        }
    }

};


#endif
