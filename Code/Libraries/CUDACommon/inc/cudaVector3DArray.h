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

#ifndef __CUDA_VECTOR3DARRAY__H
#define __CUDA_VECTOR3DARRAY__H

#include <cutil_math.h>
#include <libDefine.h>
#include <Vector3D.h>

struct Vector3D_XY_Z_Array{
    Vector3D_XY_Z_Array():xy(NULL), z(NULL){}
    float2* xy;
    float * z;
};

#define CONTINOUS_ARRAY  1
/**
   GPU Vector field data structure with 3 separate arrays for each direction
   The AoS structure is used for efficiency
 */
struct cplVector3DArray{
    cplVector3DArray():m_memType(undef_mem), x(NULL), y(NULL), z(NULL){
    }

    explicit cplVector3DArray(float* buf, int n, mem_type type=gpu_mem);
    
    uint size()     const {return n; }
    uint capacity() const {return nAlign; }
    bool isContinous()      const {return (n == nAlign);}
    
    float *elementArray(int idx){
        if(idx == 0) return x;
        if(idx == 1) return y;
        if(idx == 2) return z;
        return NULL;
    }

    const float *elementArray(int idx) const {
        if(idx == 0) return x;
        if(idx == 1) return y;
        if(idx == 2) return z;
        return NULL;
    }

    mem_type m_memType;
    float *x, *y, *z;
    
    uint n;                        // the number of elements
    uint nAlign;                   // align size
};

/** @defgroup Alloc/Free group
 *  Allocate, deallocate the memory
 *  @{
 */
/** @brief Allocate the Vector field on Host  with the same structure on GPU*/
void allocateHostVector3DArray(cplVector3DArray& a, uint n);
/** @brief Allocate the Vector field on Host using pinned memory - high data transfer rate */
void allocatePinnedVector3DArray(cplVector3DArray& a, uint n);
/** @brief Allocate the Vector field on device */
void allocateDeviceVector3DArray(cplVector3DArray& a, uint n);

/** @brief Free pinned data on Host */
void freePinnedVector3DArray(cplVector3DArray& a);
/** @brief Free data on Host */
void freeHostVector3DArray(cplVector3DArray& a);
/** @brief Free data on Device */
void freeDeviceVector3DArray(cplVector3DArray& a);

/** @brief Free vector field - RECOMMENDED */
void freeVector3DArray(cplVector3DArray& a);
/** @} */ // end of memory alloc/dealloc group

/** @defgroup Data transfer
 *  Data transfer group
 *  @{
 */

/** @brief Copy from host to device */
void copyArrayToDevice(cplVector3DArray& d_o, cplVector3DArray& h_i, int nElems);

/** @brief Copy from device to host */
void copyArrayFromDevice(cplVector3DArray& h_o, cplVector3DArray& d_i, int nElems);

/** @brief Copy to device from device (full-array) */
void copyArrayDeviceToDevice(cplVector3DArray& d_o, const cplVector3DArray& d_i);

/** @brief Copy to device from device (partial of array)*/
void copyArrayDeviceToDevice(cplVector3DArray& d_o, const cplVector3DArray& d_i, uint nElems);

/** @brief Copy from host to device - asynchronous version */
void copyArrayToDeviceAsync(cplVector3DArray& d_o, cplVector3DArray& h_i, int nElems, cudaStream_t stream=NULL);

/** @brief Copy to host from device - asynchronous version */
void copyArrayFromDeviceAsync(cplVector3DArray& h_o, cplVector3DArray& d_i, int nElems, cudaStream_t stream=NULL);

/** @brief Copy to device from device - asynchronous version */
void copyArrayDeviceToDeviceAsync(cplVector3DArray& d_o, const cplVector3DArray& d_i, uint nElems, cudaStream_t stream=NULL);

/** @} */ // end of data transfer group

namespace cplVector3DOpers {
/**
 * Set all the value of the Vector field with single value Vector3f(v)
 */
    void SetMem(cplVector3DArray& d_o, const Vector3Df& v, uint n, cudaStream_t stream=NULL);
/**  
 * Set all the value of the Vector field with single float value (normaly used to zero out the memory)
 */
    void SetMem(cplVector3DArray& d_o, float c, uint n, cudaStream_t stream=NULL);

/**
 * Compute the magnitude array
 * d_o[i] = sqrt(d_i[i].x^2 + d_i[i].y^2 + d_i[i].z^2) 
 */
    void Magnitude(float* d_o, const cplVector3DArray& d_i, uint n, cudaStream_t stream=NULL);

/**
 * Compute the magnitude array
 * d_o[i] = d_i[i].x^2 + d_i[i].y^2 + d_i[i].z^2 
 */
    void SqrMagnitude(float* d_o, const cplVector3DArray& d_i, uint n,cudaStream_t stream=NULL);

/**
 * Compute the magnitude array
 * d_o[i] = d_a[i].x*d_b[i].x + d_a[i].y*d_b[i].y + d_a[i].z*d_b[i].z  
 */
    void DotProd(float *d_o, const cplVector3DArray& d_a, const cplVector3DArray& d_b, int n, cudaStream_t stream=NULL);
  
/** @defgroup operator
 * Basic operator on the Vector field
 * @{
 */
/** @brief d_o = d_i + c */
    void AddC(cplVector3DArray& d_o, const cplVector3DArray& d_i, const Vector3Df& c, uint n,cudaStream_t stream=NULL);
/** @brief d_o = d_i + c */
    void AddC_I(cplVector3DArray& d_o, const Vector3Df& c, uint n,cudaStream_t stream=NULL);
/** @brief d_o = d_a + d_b */
    void Add(cplVector3DArray& d_o, const cplVector3DArray& d_a, const cplVector3DArray& d_b, uint n,cudaStream_t stream=NULL);
/** @brief d_o += d_b */
    void Add_I(cplVector3DArray& d_o, const cplVector3DArray& d_b, uint n,cudaStream_t stream=NULL);

/** @brief d_o = d_a - d_b */
    void Sub(cplVector3DArray& d_o, const cplVector3DArray& d_a, const cplVector3DArray& d_b, uint n,cudaStream_t stream=NULL);
/** @brief d_o -= d_b */
    void Sub_I(cplVector3DArray& d_o, const cplVector3DArray& d_b, uint n,cudaStream_t stream=NULL);
/** @brief d_o = d_i - c */
    void SubC(cplVector3DArray& d_o, const cplVector3DArray& d_i, const Vector3Df& c, uint n,cudaStream_t stream=NULL);
/** @brief d_o -= c */
    void SubC_I(cplVector3DArray& d_o, const Vector3Df& c, uint n, cudaStream_t stream=NULL);

/** @brief d_o = d_a * d_b */
    void Mul(cplVector3DArray& d_o, const cplVector3DArray& d_a, const cplVector3DArray& d_b, uint n,cudaStream_t stream=NULL);
/** @brief d_o *= d_b */
    void Mul_I(cplVector3DArray& d_o, const cplVector3DArray& d_b, uint n,cudaStream_t stream=NULL);
/** @brief d_o = d_i * c */
    void MulC(cplVector3DArray& d_o, const cplVector3DArray& d_i, const Vector3Df& c, uint n,cudaStream_t stream=NULL);
/** @brief d_o = d_i * c */
    void MulC(cplVector3DArray& d_o, const cplVector3DArray& d_i, float c, uint n,cudaStream_t stream=NULL);
/** @brief d_o *= c */
    void MulC_I(cplVector3DArray& d_o, const Vector3Df& c, uint n, cudaStream_t stream=NULL);
/** @brief d_o *= c */
    void MulC_I(cplVector3DArray& d_o, float c, uint n,cudaStream_t stream=NULL);

/** @brief d_o = (d_i + d_a) * c */
    void AddMulC(cplVector3DArray& d_o, const cplVector3DArray& d_i, const cplVector3DArray& d_a, float c, uint n, cudaStream_t stream=NULL);
/** @brief d_o = (d_o + d_a) * c */
    void AddMulC_I(cplVector3DArray& d_o, const cplVector3DArray& d_a, float c, uint n, cudaStream_t stream=NULL);

/** @brief d_o.x = d_a.x + d_b, d_o.y = d_a.y + d_b, d_o.z = d_a.z + d_b */
    void Add(cplVector3DArray& d_o, const cplVector3DArray& d_a, const float* d_b, uint n,cudaStream_t stream=NULL);
/** @brief d_o.x += d_b, d_o.y += d_b, d_o.y += d_b, */
    void Add_I(cplVector3DArray& d_o, const float* d_b, uint n,cudaStream_t stream=NULL);

/** @brief d_o.x = d_a.x - d_b, d_o.y = d_a.y - d_b, d_o.z = d_a.z - d_b */
    void Sub(cplVector3DArray& d_o, const cplVector3DArray& d_a, const float* d_b, uint n,cudaStream_t stream=NULL);
/** @brief d_o.x -= d_b, d_o.y -= d_b, d_o.y -= d_b, */
    void Sub_I(cplVector3DArray& d_o, const float* d_b, uint n,cudaStream_t stream=NULL);

/** @brief d_o.x = d_a.x * d_b, d_o.y = d_a.y * d_b, d_o.z = d_a.z * d_b */
    void Mul(cplVector3DArray& d_o, const cplVector3DArray& d_a, const float* d_b, uint n,cudaStream_t stream=NULL);
/** @brief d_o.x *= d_b, d_o.y *= d_b, d_o.y *= d_b, */
    void Mul_I(cplVector3DArray& d_o, const float* d_b, uint n,cudaStream_t stream=NULL);

/** @brief d_o = d_a + d_b * c */
    void Add_MulC(cplVector3DArray& d_o, const cplVector3DArray& d_a, const cplVector3DArray& d_b, float c, uint n, cudaStream_t stream=NULL);
/** @brief d_o = d_o+ d_b * c */
    void Add_MulC_I(cplVector3DArray& d_o, const cplVector3DArray& d_b, float c, uint n, cudaStream_t stream=NULL);
/** @brief d_o = d_a + d_b * d_c */
    void Add_Mul(cplVector3DArray& d_o, const cplVector3DArray& d_a, const cplVector3DArray& d_b, const float* d_c, uint n, cudaStream_t stream=NULL);
/** @brief d_o = d_o + d_b * d_c */
    void Add_Mul_I(cplVector3DArray& d_o, const cplVector3DArray& d_b, const float* d_c, uint n, cudaStream_t stream=NULL);

/** @brief d_o = d_a * c + d_b */
    void MulCAdd(cplVector3DArray& d_o, const cplVector3DArray& d_a, float c, const cplVector3DArray& d_b, uint n,cudaStream_t stream=NULL);
/** @brief d_o = d_o * c + d_b */
    void MulCAdd_I(cplVector3DArray& d_o, float c, const cplVector3DArray& d_b, uint n,cudaStream_t stream=NULL);

/** @brief d_o = d_a * ca + d_b * cb */
    void MulC_Add_MulC(cplVector3DArray& d_o,
                       const cplVector3DArray& d_a, float ca,
                       const cplVector3DArray& d_b, float cb,
                       uint n,cudaStream_t stream=NULL);

/** @brief d_o = d_o * co + d_b * cb */
    void MulC_Add_MulC_I(cplVector3DArray& d_o, float co,
                         const cplVector3DArray& d_b, float cb,
                         uint n,cudaStream_t stream=NULL);

/** @brief d_o = (d_i + a) * b + c */
    void AddCMulCAddC(cplVector3DArray& d_o, const cplVector3DArray& d_i, const Vector3Df& a, const Vector3Df& b, const Vector3Df& c, int n, cudaStream_t stream=NULL);
/** @brief d_o = (d_o + a) * b + c */
    void AddCMulCAddC_I(cplVector3DArray& d_o, const Vector3Df& a, const Vector3Df& b, const Vector3Df& c, int n, cudaStream_t stream=NULL);
/** @brief d_o = (d_i + a) * b + c */
    void AddCMulCAddC(cplVector3DArray& d_o, const cplVector3DArray& d_i, float a, float b, float c, int n, cudaStream_t stream=NULL);
/** @brief d_o = (d_o + a) * b + c */
    void AddCMulCAddC_I(cplVector3DArray& d_o, float a, float b, float c, int n, cudaStream_t stream=NULL);

/** @brief d_o = (d_i * d_a) * c */
    void MulMulC(cplVector3DArray& d_o, const cplVector3DArray& d_i, const cplVector3DArray& d_a, float c, uint n, cudaStream_t stream=NULL);
/** @brief d_o = (d_o * d_a) * c */
    void MulMulC_I(cplVector3DArray& d_o, const cplVector3DArray& d_a, float c, uint n, cudaStream_t stream=NULL);
/** @brief d_o.x = (d_i.x * d_a) * c, d_o.y = (d_i.y * d_a) * c, d_o.z = (d_i.z * d_a) * c */
    void MulMulC(cplVector3DArray& d_o, const cplVector3DArray& d_i, const float* d_a, float c, uint n, cudaStream_t stream=NULL);
/** @brief d_o.x = (d_o.x * d_a) * c, d_o.y = (d_o.y * d_a) * c, d_o.z = (d_o.z * d_a) * c */
    void MulMulC_I(cplVector3DArray& d_o, const float* d_a, float c, uint n, cudaStream_t stream=NULL);

/** @brief d_o = d_a + d_b * d_c * d */
    void Add_MulMulC(cplVector3DArray& d_o, const cplVector3DArray& d_a, const cplVector3DArray& d_b, const float* d_c, float d, int n, cudaStream_t stream=NULL);
/** @brief d_o = d_o + d_b * d_c * d */
    void Add_MulMulC_I(const cplVector3DArray& d_a, const cplVector3DArray& d_b, const float* d_c, float d, int n, cudaStream_t stream=NULL);

/** @brief d_o = d_a + d_b * d_c * d */
    void Add_MulMulC(cplVector3DArray& d_o, const cplVector3DArray& d_a, const cplVector3DArray& d_b, const cplVector3DArray& d_c, float d, int n, cudaStream_t stream=NULL);
/** @brief d_o = d_o + d_b * d_c * d */
    void Add_MulMulC_I(cplVector3DArray& d_a, const cplVector3DArray& d_b, const cplVector3DArray& d_c, float d, int n, cudaStream_t stream=NULL);

/** @brief d_o = (d_i + a) * b */
    void AddCMulC(cplVector3DArray& d_o, const cplVector3DArray& d_i, const Vector3Df& a, const Vector3Df& b, int n, cudaStream_t stream=NULL);
/** @brief d_o = (d_o + a) * b */
    void AddCMulC_I(cplVector3DArray& d_o, const Vector3Df& a, const Vector3Df& b, int n, cudaStream_t stream=NULL);

/** @brief d_o = (d_i * a) + b */
    void MulCAddC(cplVector3DArray& d_o, const cplVector3DArray& d_i, const Vector3Df& a, const Vector3Df& b, int n, cudaStream_t stream=NULL);
/** @brief d_o = (d_o * a) + b */
    void MulCAddC_I(cplVector3DArray& d_o, const Vector3Df& a, const Vector3Df& b, int n, cudaStream_t stream=NULL);


/** @} */ // end of basic operator
    void FixedToFloating(cplVector3DArray& d_o, uint n, cudaStream_t stream=NULL);
    void FixedToFloatingNormalize(cplVector3DArray& d_o, uint n, cudaStream_t stream=NULL);

    /**
     * d_data -= d_var * eps 
     */
    void EpsUpdate(cplVector3DArray& d_data, cplVector3DArray d_var, float eps, int n, cudaStream_t stream=NULL);
}
#endif
