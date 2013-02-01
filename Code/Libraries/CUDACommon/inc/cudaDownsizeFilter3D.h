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

#ifndef __CUDA_DOWNSIZE_FILTER3D_H
#define __CUDA_DOWNSIZE_FILTER3D_H

#include <cutil_math.h>
#include <Vector3D.h>

class cplVector3DArray;

class cplDownsizeFilter3D {
public:
    cplDownsizeFilter3D(): m_isize(256, 256, 256), m_f(1, 1, 1){
        computeOutputSize();
    };
    cplDownsizeFilter3D(const Vector3Di& isize, const Vector3Di& f) : m_isize(isize), m_f(f){
        computeOutputSize();
    }

    void SetInputParams(const Vector3Di& isize, const Vector3Di& f){
        m_isize = isize;
        m_f     = f;
        computeOutputSize();
    }
    
    const Vector3Di&  GetInputImageSize() { return m_isize; };
    const Vector3Di&  GetOutputImageSize() { return m_osize; };
    const Vector3Di&  GetDownsampleFactor() { return m_f; };

    
    void filter(float* d_o, float* d_i, bool cache=true, cudaStream_t stream=NULL);
    void filter(cplVector3DArray& d_o, cplVector3DArray& d_i, bool cache=true, cudaStream_t stream=NULL);

    void printInfo();
private:
    void computeOutputSize();
    Vector3Di m_osize;
    Vector3Di m_isize;
    Vector3Di m_f;
};

#endif
