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

#ifndef __CUDA_GAUSSIAN_FILTER__H
#define __CUDA_GAUSSIAN_FILTER__H

#include <Vector3D.h>
#include <cuda_runtime.h>
class cplVector3DArray;

class cplGaussianFilter{
public:
    cplGaussianFilter();

    void init(const Vector3Di& size, const Vector3Df& sigma, const Vector3Di& kRadius);
    void filter(float* d_o, const float* d_i, const Vector3Di& size, float* d_temp, cudaStream_t stream=NULL);
    void filter(cplVector3DArray& d_o, const cplVector3DArray& d_i, const Vector3Di& size, float* d_temp, cudaStream_t stream=NULL);

    const Vector3Df& getKernelWidth()  { return m_sigma; };
    const Vector3Di& getKernelRadius() { return m_kRadius; };
private:
    Vector3Df m_sigma;
    Vector3Di m_kRadius;
};

#endif
