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

#ifndef __CUDA_RECURSIVE_GAUSSIAN_FILTER_H
#define __CUDA_RECURSIVE_GAUSSIAN_FILTER_H

#include <Vector3D.h>
#include <cuda_runtime.h>

class cplVector3DArray;

class cplRGFilter{
public:
    cplRGFilter(){};

    cplRGFilter(float sigma, int order){
        init(sigma, order);
    };
    
    void init(float sigma, int order);
    void filter(float* d_o, float* d_i,
                int sizeX, int sizeY, float* d_temp, cudaStream_t stream=NULL);

    void filter(float* d_o, float* d_i,
                const Vector3Di& size, float* d_temp, cudaStream_t stream=NULL);

    void filter(cplVector3DArray& d_o, cplVector3DArray& d_i,
                const Vector3Di& size, float* d_temp, cudaStream_t stream=NULL);

    int   GetOrder()       { return m_order; };
    float GetKernelWidth() { return m_sigma; };
    
protected:
    void filter_impl(float* d_o, float* d_i,
                     int sizeX, int sizeY, int sizeZ,
                     float* d_temp, cudaStream_t stream);
private:
    //a0-a3, b1, b2, coefp, coefn - filter parameters
    float m_sigma;
    int m_order;
    
    float a0, a1, a2, a3;
    float b1, b2;
    float coefp, coefn;
};

#endif
