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

#include <cpl.h>
#include <cudaUpsample3D.h>
#include "cudaTrilerp.cu"

texture<float , 3, cudaReadModeElementType> com_tex3_float_x;
texture<float , 3, cudaReadModeElementType> com_tex3_float_y;
texture<float , 3, cudaReadModeElementType> com_tex3_float_z;
texture<float2, 3, cudaReadModeElementType> com_tex3_float_xy;
texture<float4, 3, cudaReadModeElementType> com_tex3_float_xyzw;

cplUpsampleFilter::cplUpsampleFilter(){
    d_volumeArray_x = NULL;
    d_volumeArray_y = NULL;
    d_volumeArray_z = NULL;
    d_volumeArray_xy = NULL;
    d_volumeArray_xyzw = NULL;

    m_isize = Vector3Di(1, 1, 1);
    m_osize = Vector3Di(1, 1, 1);
    m_r     = Vector3Df(1.f, 1.f, 1.f);
}

cplUpsampleFilter::~cplUpsampleFilter(){
    clean();
}


void cplUpsampleFilter::setParams(const Vector3Di& isize, const Vector3Di& osize)
{
    m_isize = isize;
    m_osize = osize;
    computeScale();
}

void cplUpsampleFilter::setInputSize(const Vector3Di& size){
    m_isize    = size;
    volumeSize = make_cudaExtent(m_isize.x, m_isize.y, m_isize.z);
    computeScale();
}
    
void cplUpsampleFilter::setOutputSize(const Vector3Di& size){
    m_osize      = size;
    computeScale();
}


void cplUpsampleFilter::computeScale(){
    m_r.x = (float) m_isize.x / (float) m_osize.x;
    m_r.y = (float) m_isize.y / (float) m_osize.y;
    m_r.z = (float) m_isize.z / (float) m_osize.z;
}


template<class T>
void allocate3DVolume(cudaArray** d_vol, cudaExtent& volumeSize){
    cudaChannelFormatDesc desc = cudaCreateChannelDesc<T>();
    cudaMalloc3DArray(d_vol, &desc, volumeSize);
}

void cplUpsampleFilter::allocate(int mask){
    if (mask & UPSAMPLE_VOLUME_X){
        fprintf(stderr, "Allocate the X volume\n");
        allocate3DVolume<float>(&d_volumeArray_x, volumeSize);
    }

    if (mask & UPSAMPLE_VOLUME_Y){
        fprintf(stderr, "Allocate the Y volume\n");
        allocate3DVolume<float>(&d_volumeArray_y, volumeSize);
    }

    if (mask & UPSAMPLE_VOLUME_Z){
        fprintf(stderr, "Allocate the Z volume\n");
        allocate3DVolume<float>(&d_volumeArray_z, volumeSize);
    }

    if (mask & UPSAMPLE_VOLUME_2){
        fprintf(stderr, "Allocate the XY volume\n");
        allocate3DVolume<float2>(&d_volumeArray_xy, volumeSize);
    }

    if (mask & UPSAMPLE_VOLUME_4){
        fprintf(stderr, "Allocate the XYZW volume\n");
        allocate3DVolume<float4>(&d_volumeArray_xyzw, volumeSize);
    }
    cutilCheckMsg("Allocate 3D texture memory");
}

void releaseVolume(cudaArray** d_vol){
    if (*d_vol != NULL)
        cudaFreeArray(*d_vol);
    *d_vol = NULL;
}

void cplUpsampleFilter::release(int mask){
    if (mask & UPSAMPLE_VOLUME_X)
        releaseVolume(&d_volumeArray_x);

    if (mask & UPSAMPLE_VOLUME_Y)
        releaseVolume(&d_volumeArray_y);
    
    if (mask & UPSAMPLE_VOLUME_Z)
        releaseVolume(&d_volumeArray_z);

    if (mask & UPSAMPLE_VOLUME_2)
        releaseVolume(&d_volumeArray_xy);
    
    if (mask & UPSAMPLE_VOLUME_4)
        releaseVolume(&d_volumeArray_xyzw);
}

void cplUpsampleFilter::clean(){
    release(UPSAMPLE_VOLUME_X | UPSAMPLE_VOLUME_Y | UPSAMPLE_VOLUME_Z
            | UPSAMPLE_VOLUME_2 | UPSAMPLE_VOLUME_4);
}

////////////////////////////////////////////////////////////////////////////////
//
////////////////////////////////////////////////////////////////////////////////
void cplUpsampleFilter::copyToTexture(float* d_i, cudaStream_t stream){
    if (d_volumeArray_x == NULL)
        allocate(UPSAMPLE_VOLUME_X);

    cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
    
    // Copy to the 3D volume 
    cudaMemcpy3DParms copyParams = {0};
    copyParams.extent   = volumeSize;
    copyParams.kind     = cudaMemcpyDeviceToDevice;

    copyParams.srcPtr   = make_cudaPitchedPtr((void*)d_i,
                                              m_isize.x *sizeof(float), m_isize.x, m_isize.y);
    copyParams.dstArray = d_volumeArray_x;
    cudaMemcpy3DAsync(&copyParams, stream);

    com_tex3_float_x.addressMode[0] = cudaAddressModeClamp;
    com_tex3_float_x.addressMode[1] = cudaAddressModeClamp;
    com_tex3_float_x.addressMode[2] = cudaAddressModeClamp;
    com_tex3_float_x.filterMode     = cudaFilterModeLinear; // Using 3D interpolation
    com_tex3_float_x.normalized     = false;

    cudaBindTextureToArray(com_tex3_float_x, d_volumeArray_x, desc);
}

void cplUpsampleFilter::copyToTexture(float* d_i_x, float* d_i_y, float* d_i_z, cudaStream_t stream){
    if (d_volumeArray_x == NULL)
        allocate(UPSAMPLE_VOLUME_X);

    if (d_volumeArray_y == NULL)
        allocate(UPSAMPLE_VOLUME_Y);

    if (d_volumeArray_z == NULL)
        allocate(UPSAMPLE_VOLUME_Z);

    // Copy to the 3D volume
    cudaMemcpy3DParms copyParams = {0};
    copyParams.extent   = volumeSize;
    copyParams.kind     = cudaMemcpyDeviceToDevice;

    copyParams.srcPtr   = make_cudaPitchedPtr((void*)d_i_x, m_isize.x *sizeof(float), m_isize.x, m_isize.y);
    copyParams.dstArray = d_volumeArray_x;
    cudaMemcpy3DAsync(&copyParams, stream);

    copyParams.srcPtr   = make_cudaPitchedPtr((void*)d_i_y, m_isize.x *sizeof(float), m_isize.x, m_isize.y);
    copyParams.dstArray = d_volumeArray_y;
    cudaMemcpy3DAsync(&copyParams, stream);

    copyParams.srcPtr   = make_cudaPitchedPtr((void*)d_i_z, m_isize.x *sizeof(float), m_isize.x, m_isize.y);
    copyParams.dstArray = d_volumeArray_z;
    cudaMemcpy3DAsync(&copyParams, stream);

    // Copy to the texture
    cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
    com_tex3_float_x.addressMode[0] = cudaAddressModeClamp;
    com_tex3_float_x.addressMode[1] = cudaAddressModeClamp;
    com_tex3_float_x.addressMode[2] = cudaAddressModeClamp;
    com_tex3_float_x.filterMode     = cudaFilterModeLinear; // Using 3D interpolation
    com_tex3_float_x.normalized     = false;
    cudaBindTextureToArray(com_tex3_float_x, d_volumeArray_x, desc);

    com_tex3_float_y.addressMode[0] = cudaAddressModeClamp;
    com_tex3_float_y.addressMode[1] = cudaAddressModeClamp;
    com_tex3_float_y.addressMode[2] = cudaAddressModeClamp;
    com_tex3_float_y.filterMode     = cudaFilterModeLinear; // Using 3D interpolation
    com_tex3_float_y.normalized     = false;
    cudaBindTextureToArray(com_tex3_float_y, d_volumeArray_y, desc);

    com_tex3_float_z.addressMode[0] = cudaAddressModeClamp;
    com_tex3_float_z.addressMode[1] = cudaAddressModeClamp;
    com_tex3_float_z.addressMode[2] = cudaAddressModeClamp;
    com_tex3_float_z.filterMode     = cudaFilterModeLinear; // Using 3D interpolation
    com_tex3_float_z.normalized     = false;
    cudaBindTextureToArray(com_tex3_float_z, d_volumeArray_z, desc);
    
}

void cplUpsampleFilter::copyToTexture(float2* d_i_xy, float* d_i_z, cudaStream_t stream){

    if (d_volumeArray_x == NULL)
        allocate(UPSAMPLE_VOLUME_X);
    
    if (d_volumeArray_xy == NULL)
        allocate(UPSAMPLE_VOLUME_2);

    cudaMemcpy3DParms copyParams = {0};
    copyParams.extent   = volumeSize;
    copyParams.kind     = cudaMemcpyDeviceToDevice;

    // upload z component 
    copyParams.srcPtr   = make_cudaPitchedPtr((void*)d_i_z, m_isize.x *sizeof(float), m_isize.x, m_isize.y);
    copyParams.dstArray = d_volumeArray_x;
    cudaMemcpy3DAsync(&copyParams, stream);

    // upload zy component 
    copyParams.srcPtr   = make_cudaPitchedPtr((void*)d_i_xy, m_isize.x *sizeof(float2), m_isize.x, m_isize.y);
    copyParams.dstArray = d_volumeArray_xy;
    cudaMemcpy3DAsync(&copyParams, stream);

    cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
    com_tex3_float_x.addressMode[0] = cudaAddressModeClamp;
    com_tex3_float_x.addressMode[1] = cudaAddressModeClamp;
    com_tex3_float_x.addressMode[2] = cudaAddressModeClamp;
    com_tex3_float_x.filterMode     = cudaFilterModeLinear; // Using 3D interpolation
    com_tex3_float_x.normalized     = false;
    cudaBindTextureToArray(com_tex3_float_x, d_volumeArray_x, desc);

    cudaChannelFormatDesc desc2 = cudaCreateChannelDesc<float2>();
    com_tex3_float_xy.addressMode[0] = cudaAddressModeClamp;
    com_tex3_float_xy.addressMode[1] = cudaAddressModeClamp;
    com_tex3_float_xy.addressMode[2] = cudaAddressModeClamp;
    com_tex3_float_xy.filterMode     = cudaFilterModeLinear; // Using 3D interpolation
    com_tex3_float_xy.normalized     = false;
    cudaBindTextureToArray(com_tex3_float_xy, d_volumeArray_xy, desc2);
}

void cplUpsampleFilter::copyToTexture(float4* d_i, cudaStream_t stream){

    if (d_volumeArray_xyzw == NULL)
        allocate(UPSAMPLE_VOLUME_4);

    cudaMemcpy3DParms copyParams = {0};
    copyParams.extent   = volumeSize;
    copyParams.kind     = cudaMemcpyDeviceToDevice;

    // upload z component 
    copyParams.srcPtr   = make_cudaPitchedPtr((void*)d_i, m_isize.x *sizeof(float4), m_isize.x, m_isize.y);
    copyParams.dstArray = d_volumeArray_xyzw;
    cudaMemcpy3DAsync(&copyParams, stream);

    cudaChannelFormatDesc desc4 = cudaCreateChannelDesc<float4>();
    com_tex3_float_xy.addressMode[0] = cudaAddressModeClamp;
    com_tex3_float_xy.addressMode[1] = cudaAddressModeClamp;
    com_tex3_float_xy.addressMode[2] = cudaAddressModeClamp;
    com_tex3_float_xy.filterMode     = cudaFilterModeLinear; // Using 3D interpolation
    com_tex3_float_xy.normalized     = false;
    cudaBindTextureToArray(com_tex3_float_xyzw, d_volumeArray_xyzw, desc4);
}


////////////////////////////////////////////////////////////////////////////////
//
////////////////////////////////////////////////////////////////////////////////
__global__ void cplUpsample3D_kernel(float* d_o,
                                      int osizeX, int osizeY, int osizeZ,
                                      int isizeX, int isizeY, int isizeZ,
                                      float rX, float rY, float rZ)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix < osizeX && iy < osizeY){
        int  oId   = ix + iy * osizeX;
        int  oPlane= osizeX * osizeY;

        float fX = ix * rX;
        float fY = iy * rY;

        for (int iz = 0; iz < osizeZ; ++iz, oId +=oPlane){
            float fZ = iz * rZ;
            d_o[oId] = triLerp_tex<BACKGROUND_STRATEGY_ID>(fX, fY, fZ, isizeX, isizeY, isizeZ);
        }
    }
}

__global__ void cplUpsample3D_kernel(float* d_o, float* d_i,
                                      int osizeX, int osizeY, int osizeZ,
                                      int isizeX, int isizeY, int isizeZ,
                                      float rX, float rY, float rZ)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix < osizeX && iy < osizeY){
        int  oId   = ix + iy * osizeX;
        int  oPlane= osizeX * osizeY;

        float fX = ix * rX;
        float fY = iy * rY;

        for (int iz = 0; iz < osizeZ; ++iz, oId +=oPlane){
            float fZ = iz * rZ;
            d_o[oId] = triLerp<BACKGROUND_STRATEGY_ID>(d_i, fX, fY, fZ, isizeX, isizeY, isizeZ);
        }
    }
}

void cplUpsampleFilter::filter(float* d_o, float* d_i, cudaStream_t stream){
    // perform the filter function (trilinear interpolation)
    dim3 threads(16,16);
    dim3 grids(iDivUp(m_osize.x, threads.x),iDivUp(m_osize.y, threads.y));

    cudaBindTexture(0, com_tex_float, d_i, m_isize.x * m_isize.y * m_isize.z * sizeof(float));
    cplUpsample3D_kernel<<<grids, threads, 0, stream>>>(d_o,
                                                         m_osize.x, m_osize.y, m_osize.z,
                                                         m_isize.x, m_isize.y,m_isize.z,
                                                         m_r.x, m_r.y, m_r.z);
}

////////////////////////////////////////////////////////////////////////////////
//
////////////////////////////////////////////////////////////////////////////////
template<bool rescale>
__global__ void cplUpsample3D_kernel(float* d_o_x, float* d_o_y, float* d_o_z,
                                      float* d_i_x, float* d_i_y, float* d_i_z,
                                      int osizeX, int osizeY, int osizeZ,
                                      int isizeX, int isizeY, int isizeZ,
                                      float rX, float rY, float rZ)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix < osizeX && iy < osizeY){
        int  oId   = ix + iy * osizeX;
        int  oPlane= osizeX * osizeY;

        float fX = (float) ix * rX;
        float fY = (float) iy * rY;

        for (int iz = 0; iz < osizeZ; ++iz, oId +=oPlane){
            float fZ = iz * rZ;
            float hx, hy, hz;
            
            triLerp<BACKGROUND_STRATEGY_ID>(hx, hy, hz, 
                                            d_i_x, d_i_y, d_i_z,
                                            fX, fY, fZ,
                                            isizeX, isizeY, isizeZ);
            if (rescale){
                hx /=rX;
                hy /=rY;
                hz /=rZ;
            }
            d_o_x[oId] = hx;
            d_o_y[oId] = hy;
            d_o_z[oId] = hz;
        }
    }

}

template<bool rescale>
__global__ void cplUpsample3D_tex_kernel(float* d_o_x, float* d_o_y, float* d_o_z,
                                          int osizeX, int osizeY, int osizeZ,
                                          int isizeX, int isizeY, int isizeZ,
                                          float rX, float rY, float rZ)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix < osizeX && iy < osizeY){
        int  oId   = ix + iy * osizeX;
        int  oPlane= osizeX * osizeY;

        float fX = (float) ix * rX;
        float fY = (float) iy * rY;

        for (int iz = 0; iz < osizeZ; ++iz, oId +=oPlane){
            float fZ = iz * rZ;
            float hx, hy, hz;
            
            triLerp_tex<BACKGROUND_STRATEGY_ID>(hx, hy, hz, 
                                                fX, fY, fZ,
                                                isizeX, isizeY, isizeZ);
            if (rescale){
                hx /=rX;
                hy /=rY;
                hz /=rZ;
            }
            d_o_x[oId] = hx;
            d_o_y[oId] = hy;
            d_o_z[oId] = hz;
        }
    }

}

void cplUpsampleFilter::filter(float* d_o_x, float* d_o_y, float* d_o_z,
                                float* d_i_x, float* d_i_y, float* d_i_z,
                                bool rescale, cudaStream_t stream){
    dim3 threads(16,16);
    dim3 grids(iDivUp(m_osize.x, threads.x),iDivUp(m_osize.y, threads.y));

    int memSize = m_isize.x * m_isize.y * m_isize.z * sizeof(float);

    cudaBindTexture(0, com_tex_float_x, d_i_x, memSize);
    cudaBindTexture(0, com_tex_float_y, d_i_y, memSize);
    cudaBindTexture(0, com_tex_float_z, d_i_z, memSize);

    
    if (rescale)
        cplUpsample3D_tex_kernel<true><<<grids, threads, 0, stream>>>(d_o_x, d_o_y, d_o_z,
                                                                       m_osize.x, m_osize.y, m_osize.z,
                                                                       m_isize.x, m_isize.y, m_isize.z,
                                                                       m_r.x, m_r.y, m_r.z);
    else
        cplUpsample3D_tex_kernel<false><<<grids, threads, 0, stream>>>(d_o_x, d_o_y, d_o_z,
                                                                        m_osize.x, m_osize.y, m_osize.z,
                                                                        m_isize.x, m_isize.y, m_isize.z,
                                                                        m_r.x, m_r.y, m_r.z);
}


void cplUpsampleFilter::filter(cplVector3DArray& d_o,
                                cplVector3DArray& d_i, bool rescale, cudaStream_t stream)
{
    filter(d_o.x, d_o.y, d_o.z,
           d_i.x, d_i.y, d_i.z, rescale, stream);
}

////////////////////////////////////////////////////////////////////////////////
//
////////////////////////////////////////////////////////////////////////////////
template<bool rescale>
__global__ void cplUpsample3D_kernel(float2* d_o_xy, float* d_o_z,
                                      int osizeX, int osizeY, int osizeZ,
                                      float rX, float rY, float rZ)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix < osizeX && iy < osizeY){
        int  oId   = ix + iy * osizeX;
        int  oPlane= osizeX * osizeY;

        float fX = ix * rX + 0.5f;
        float fY = iy * rY + 0.5f;
        
        for (int iz = 0; iz < osizeZ; ++iz, oId+= oPlane){
            float fZ = iz * rZ + 0.5f;

            if (!rescale) {
                d_o_xy[oId] = tex3D(com_tex3_float_xy, fX, fY, fZ);
                d_o_z[oId]  = tex3D(com_tex3_float_x , fX, fY, fZ);
            }else {
                float2 val  = tex3D(com_tex3_float_xy, fX, fY, fZ);
                d_o_xy[oId] = make_float2(val.x/rX, val.y/rY);
                d_o_z[oId]  = tex3D(com_tex3_float_x , fX, fY, fZ) / rZ;
            }
        }
    }
}

void cplUpsampleFilter::filter(float2* d_o_xy, float* d_o_z, float2* d_i_xy, float* d_i_z,
                                bool rescale, cudaStream_t stream)
{
    copyToTexture(d_i_xy, d_i_z, stream);
    
    // perform the filter function (trilinear interpolation)
    dim3 threads(16,16);
    dim3 grids(iDivUp(m_osize.x, threads.x),iDivUp(m_osize.y, threads.y));
    
    if (rescale)
        cplUpsample3D_kernel<true><<<grids, threads, 0, stream>>>(d_o_xy, d_o_z,
                                                                   m_osize.x, m_osize.y, m_osize.z,
                                                                   m_r.x, m_r.y, m_r.z);
    else
        cplUpsample3D_kernel<false><<<grids, threads, 0, stream>>>(d_o_xy, d_o_z,
                                                                    m_osize.x, m_osize.y, m_osize.z,
                                                                    m_r.x, m_r.y, m_r.z);

}


////////////////////////////////////////////////////////////////////////////////
//
////////////////////////////////////////////////////////////////////////////////
template<int channel, bool rescale>
__global__ void cplUpsample3D_Tex3D_kernel(float* d_o,
                                            int osizeX, int osizeY, int osizeZ,
                                            float rX, float rY, float rZ)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix < osizeX && iy < osizeY){
        int  oId   = ix + iy * osizeX;
        int  oPlane= osizeX * osizeY;

        float fX = ix * rX + 0.5f;
        float fY = iy * rY + 0.5f;

        for (int iz = 0; iz < osizeZ; ++iz, oId +=oPlane){
            float fZ = iz * rZ + 0.5f;

            float res = tex3D(com_tex3_float_x , fX, fY, fZ);

            if (!rescale)
                d_o[oId] = res ;
            else {
                float f;
                if (channel == 0) f = 1.f / rX;
                else if (channel == 1) f = 1.f / rY;
                else if (channel == 2) f = 1.f / rZ;
                d_o[oId] = res * f;
            }
        }
    }
}

void cplUpsampleFilter::filter_x(float* d_o, float* d_i, bool rescale, cudaStream_t stream)
{
    copyToTexture(d_i, stream);
    // perform the filter function (trilinear interpolation)
    dim3 threads(16,16);
    dim3 grids(iDivUp(m_osize.x, threads.x),iDivUp(m_osize.y, threads.y));

    if (rescale)
        cplUpsample3D_Tex3D_kernel<0, true><<<grids, threads, 0, stream>>>(d_o,
                                                                            m_osize.x, m_osize.y, m_osize.z,
                                                                            m_r.x, m_r.y, m_r.z);
    else
        cplUpsample3D_Tex3D_kernel<0, false><<<grids, threads, 0, stream>>>(d_o,
                                                                             m_osize.x, m_osize.y, m_osize.z,
                                                                             m_r.x, m_r.y, m_r.z);
}

void cplUpsampleFilter::filter_y(float* d_o, float* d_i, bool rescale, cudaStream_t stream)
{
    copyToTexture(d_i, stream);
    // perform the filter function (trilinear interpolation)
    dim3 threads(16,16);
    dim3 grids(iDivUp(m_osize.x, threads.x),iDivUp(m_osize.y, threads.y));

    if (rescale)
        cplUpsample3D_Tex3D_kernel<1, true><<<grids, threads, 0, stream>>>(d_o,
                                                                            m_osize.x, m_osize.y, m_osize.z,
                                                                            m_r.x, m_r.y, m_r.z);
    else
        cplUpsample3D_Tex3D_kernel<1, false><<<grids, threads, 0, stream>>>(d_o,
                                                                             m_osize.x, m_osize.y, m_osize.z,
                                                                             m_r.x, m_r.y, m_r.z);
}

void cplUpsampleFilter::filter_z(float* d_o, float* d_i, bool rescale, cudaStream_t stream)
{
    copyToTexture(d_i, stream);
    // perform the filter function (trilinear interpolation)
    dim3 threads(16,16);
    dim3 grids(iDivUp(m_osize.x, threads.x),iDivUp(m_osize.y, threads.y));

    if (rescale)
        cplUpsample3D_Tex3D_kernel<2, true><<<grids, threads, 0, stream>>>(d_o,
                                                                            m_osize.x, m_osize.y, m_osize.z,
                                                                            m_r.x, m_r.y, m_r.z);
    else
        cplUpsample3D_Tex3D_kernel<2, false><<<grids, threads, 0, stream>>>(d_o,
                                                                             m_osize.x, m_osize.y, m_osize.z,
                                                                             m_r.x, m_r.y, m_r.z);
}

////////////////////////////////////////////////////////////////////////////////
//
////////////////////////////////////////////////////////////////////////////////
template<bool rescale>
__global__ void cplUpsample3D_kernel(float4* d_o,
                                      int osizeX, int osizeY, int osizeZ,
                                      float rX, float rY, float rZ)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix < osizeX && iy < osizeY){
        int  oId   = ix + iy * osizeX;
        int  oPlane= osizeX * osizeY;

        float fX = ix * rX + 0.5f;
        float fY = iy * rY + 0.5f;

        for (int iz = 0; iz < osizeZ; ++iz, oId +=oPlane){
            float fZ = iz * rZ + 0.5f;
            float4 res = tex3D(com_tex3_float_xyzw, fX, fY, fZ);
            if (!rescale)
                d_o[oId] = res ;
            else
                d_o[oId] = make_float4(res.x / rX, res.y / rY, res.z / rZ, 1.f);
        }
    }
}

void cplUpsampleFilter::filter(float4* d_o, float4* d_i, bool rescale, cudaStream_t stream)
{
    copyToTexture(d_i, stream);
    
    // perform the filter function (trilinear interpolation)
    dim3 threads(16,16);
    dim3 grids(iDivUp(m_osize.x, threads.x),iDivUp(m_osize.y, threads.y));
    
    if (rescale)
        cplUpsample3D_kernel<true><<<grids, threads, 0, stream>>>(d_o,
                                                                   m_osize.x, m_osize.y, m_osize.z,
                                                                   m_r.x, m_r.y, m_r.z);
    else
        cplUpsample3D_kernel<false><<<grids, threads, 0, stream>>>(d_o,
                                                                    m_osize.x, m_osize.y, m_osize.z,
                                                                    m_r.x, m_r.y, m_r.z);

}

