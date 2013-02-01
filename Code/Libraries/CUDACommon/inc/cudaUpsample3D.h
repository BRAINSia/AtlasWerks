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

#ifndef __CUDA_UPSAMPLE3D_H
#define __CUDA_UPSAMPLE3D_H

#include <cutil.h>
#include <Vector3D.h>

#define UPSAMPLE_VOLUME_X 1
#define UPSAMPLE_VOLUME_Y 2
#define UPSAMPLE_VOLUME_Z 4
#define UPSAMPLE_VOLUME_2 8
#define UPSAMPLE_VOLUME_4 16

class cplVector3DArray;

class cplUpsampleFilter{
public:
    cplUpsampleFilter();
    ~cplUpsampleFilter();

    void setParams(const Vector3Di& isize, const Vector3Di& osize);
    void setInputSize(const Vector3Di& size);
    void setOutputSize(const Vector3Di& size);

    
    // Do filter on the volume (don't rescale) 
    void filter(float* d_o, float* d_i, cudaStream_t stream=NULL);

    // Do filter on the HField rescale on default
    void filter(float* d_o_x, float* d_o_y, float* d_oz,
                float* d_i_x, float* d_i_y, float* d_i_z,
                bool rescale = true, cudaStream_t stream=NULL);
    void filter(cplVector3DArray& d_o, cplVector3DArray& d_i,
                bool rescale = true, cudaStream_t stream=NULL);

    
    void filter(float2* d_o_xy, float* d_oz, float2* d_i_xy, float* d_i_z,
                bool rescale = true, cudaStream_t stream=NULL);
    void filter(float4* d_o, float4* d_i,
                bool rescale = true, cudaStream_t stream=NULL);


    void allocate(int mask);
    void release(int mask);

    void clean();

private:
    void filter_x(float* d_o, float* d_i, bool rescale = true, cudaStream_t stream =NULL);
    void filter_y(float* d_o, float* d_i, bool rescale = true, cudaStream_t stream =NULL);
    void filter_z(float* d_o, float* d_i, bool rescale = true, cudaStream_t stream =NULL);
    
    void copyToTexture(float* d_i, cudaStream_t stream=NULL);
    void copyToTexture(float* d_i_x, float* d_i_y, float *d_i_z, cudaStream_t stream=NULL);
    void copyToTexture(float4* d_i, cudaStream_t stream=NULL);
    void copyToTexture(float2* d_i_xy, float* d_i_z, cudaStream_t stream=NULL);
    void computeScale();

    Vector3Di m_osize, m_isize;
    Vector3Df m_r;
    
    cudaArray* d_volumeArray_x;
    cudaArray* d_volumeArray_y;
    cudaArray* d_volumeArray_z;
    cudaArray* d_volumeArray_xy;
    cudaArray* d_volumeArray_xyzw;

    cudaExtent volumeSize;
    int vol_mask;
};



#endif
