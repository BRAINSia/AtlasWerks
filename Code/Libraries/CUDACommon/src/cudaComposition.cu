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

#include <cudaComposition.h>
#include <cpl.h>
#include "cudaTrilerp.cu"

//
// h(x) = f(x + delta*g(x))
//
template<int backgroundStrategy>
__global__ void cplBackwardMapping_kernel(float* d_hX, const float* d_fX, 
                                          const float* d_gX, const float* d_gY, const float* d_gZ,
                                          float delta,
                                          int sizeX, int sizeY, int sizeZ){

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i < sizeX && j < sizeY){
        int id = j * sizeX + i;
        for (int k=0; k< sizeZ; ++k, id+= sizeX*sizeY){
            float x = delta * d_gX[id] + (float)i;
            float y = delta * d_gY[id] + (float)j;
            float z = delta * d_gZ[id] + (float)k;

            d_hX[id] = triLerp<backgroundStrategy>(d_fX, x, y, z, sizeX, sizeY, sizeZ);
        }
    }
}

void cplBackwardMapping(float* d_hX, const float* d_fX, 
                        const float* d_gX, const float* d_gY, const float* d_gZ,
                        int sizeX, int sizeY, int sizeZ,
                        float delta,
                        int backgroundStrategy, cudaStream_t stream)
{
    dim3 threads(16,16);
    dim3 grids(iDivUp(sizeX, threads.x), iDivUp(sizeY, threads.y));
    
    if (backgroundStrategy == BACKGROUND_STRATEGY_ID)
        cplBackwardMapping_kernel<BACKGROUND_STRATEGY_ID><<<grids, threads, 0, stream>>>(d_hX, d_fX,
                                                                                         d_gX, d_gY, d_gZ,
                                                                                         delta, sizeX, sizeY, sizeZ);
    else if (backgroundStrategy == BACKGROUND_STRATEGY_PARTIAL_ID)
        cplBackwardMapping_kernel<BACKGROUND_STRATEGY_PARTIAL_ID><<<grids, threads,0 ,stream>>>(d_hX, d_fX,
                                                                                                d_gX, d_gY, d_gZ,
                                                                                                delta, sizeX, sizeY, sizeZ);
    else if (backgroundStrategy == BACKGROUND_STRATEGY_ZERO)
        cplBackwardMapping_kernel<BACKGROUND_STRATEGY_ZERO><<<grids, threads,0 ,stream>>>(d_hX, d_fX,
                                                                                          d_gX, d_gY, d_gZ,
                                                                                          delta, sizeX, sizeY, sizeZ);
    else if (backgroundStrategy == BACKGROUND_STRATEGY_CLAMP)
        cplBackwardMapping_kernel<BACKGROUND_STRATEGY_CLAMP><<<grids, threads,0 ,stream>>>(d_hX, d_fX,
                                                                                           d_gX, d_gY, d_gZ,
                                                                                           delta, sizeX, sizeY, sizeZ);
    else if (backgroundStrategy == BACKGROUND_STRATEGY_WRAP)
        cplBackwardMapping_kernel<BACKGROUND_STRATEGY_WRAP><<<grids, threads,0 ,stream>>>(d_hX, d_fX,
                                                                                          d_gX, d_gY, d_gZ,
                                                                                          delta, sizeX, sizeY, sizeZ);
}

void cplBackwardMapping(float* d_hX, const float* d_fX, 
                        const cplVector3DArray& d_g, const Vector3Di& size, 
                        float delta,
                        int backgroundStrategy, cudaStream_t stream)
{
    cplBackwardMapping(d_hX, d_fX,
                       d_g.x, d_g.y, d_g.z,
                       size.x, size.y, size.z, 
                       delta,
                       backgroundStrategy, stream);
}

template<int backgroundStrategy>
__global__ void cplBackwardMapping_kernel(float* d_hX, const float* d_fX, 
                                          const float* d_gX, const float* d_gY, const float* d_gZ,
                                          float delta,
                                          int sizeX, int sizeY, int sizeZ,
                                          float ispX, float ispY, float ispZ){

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    
    if (i < sizeX && j < sizeY){
        int id = j * sizeX + i;
        for (int k=0; k< sizeZ; ++k, id+= sizeX*sizeY){
            float x = delta * ispX * d_gX[id] + (float)i;
            float y = delta * ispY * d_gY[id] + (float)j;
            float z = delta * ispZ * d_gZ[id] + (float)k;

            d_hX[id] = triLerp<backgroundStrategy>(d_fX, x, y, z, sizeX, sizeY, sizeZ);
        }
    }
}


////////////////////////////////////////////////////////////////////////////////
//
////////////////////////////////////////////////////////////////////////////////
template<int backgroundStrategy>
__global__ void cplBackwardMapping_kernel(float* d_hX, float* d_hY, float* d_hZ,
                                          const float* d_fX, const float* d_fY, const float* d_fZ,
                                          const float* d_gX, const float* d_gY, const float* d_gZ,
                                          float delta,
                                          int sizeX, int sizeY, int sizeZ){

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i < sizeX && j < sizeY){
        int id = j * sizeX + i;
        for (int k=0; k< sizeZ; ++k, id+= sizeX*sizeY){
            float x = delta * d_gX[id] + (float)i;
            float y = delta * d_gY[id] + (float)j;
            float z = delta * d_gZ[id] + (float)k;

            float hx, hy, hz;

            triLerp<backgroundStrategy>(hx, hy, hz, d_fX,d_fY, d_fZ, x, y, z, sizeX, sizeY, sizeZ);

            d_hX[id] = hx;
            d_hY[id] = hy;
            d_hZ[id] = hz;
        }
    }
}


template<int backgroundStrategy>
__global__ void cplBackwardMapping_kernel(float* d_hX, float* d_hY, float* d_hZ,
                                          const float* d_fX, const float* d_fY, const float* d_fZ,
                                          const float* d_gX, const float* d_gY, const float* d_gZ,
                                          float delta,
                                          int sizeX, int sizeY, int sizeZ,
                                          float ispX, float ispY, float ispZ){

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i < sizeX && j < sizeY){
        int id = j * sizeX + i;
        for (int k=0; k< sizeZ; ++k, id+= sizeX*sizeY){
            float x = delta * ispX * d_gX[id] + (float)i;
            float y = delta * ispY * d_gY[id] + (float)j;
            float z = delta * ispZ * d_gZ[id] + (float)k;

            float hx, hy, hz;

            triLerp<backgroundStrategy>(hx, hy, hz, d_fX,d_fY, d_fZ, x, y, z, sizeX, sizeY, sizeZ);

            d_hX[id] = hx;
            d_hY[id] = hy;
            d_hZ[id] = hz;
        }
    }
}

void cplBackwardMapping(float* d_hX, float* d_hY, float* d_hZ,
                        const float* d_fX, const float* d_fY, const float* d_fZ,
                        const float* d_gX, const float* d_gY, const float* d_gZ,
                        int sizeX, int sizeY, int sizeZ,
                        float delta,
                        int backgroundStrategy, cudaStream_t stream)
{
    dim3 threads(16,16);
    dim3 grids(iDivUp(sizeX, threads.x), iDivUp(sizeY, threads.y));
    
    if (backgroundStrategy == BACKGROUND_STRATEGY_ID)
        cplBackwardMapping_kernel<BACKGROUND_STRATEGY_ID><<<grids, threads,0 ,stream>>>(d_hX, d_hY, d_hZ,
                                                                                        d_fX, d_fY, d_fZ,
                                                                                        d_gX, d_gY, d_gZ,
                                                                                        delta,
                                                                                        sizeX, sizeY, sizeZ);
    else if (backgroundStrategy == BACKGROUND_STRATEGY_PARTIAL_ID)
        cplBackwardMapping_kernel<BACKGROUND_STRATEGY_PARTIAL_ID><<<grids, threads,0 ,stream>>>(d_hX, d_hY, d_hZ,
                                                                                                d_fX, d_fY, d_fZ,
                                                                                                d_gX, d_gY, d_gZ,
                                                                                                delta,
                                                                                                sizeX, sizeY, sizeZ);
    else if (backgroundStrategy == BACKGROUND_STRATEGY_ZERO)
        cplBackwardMapping_kernel<BACKGROUND_STRATEGY_ZERO><<<grids, threads,0 ,stream>>>(d_hX, d_hY, d_hZ,
                                                                                          d_fX, d_fY, d_fZ,
                                                                                          d_gX, d_gY, d_gZ,
                                                                                          delta,
                                                                                          sizeX, sizeY, sizeZ);
    else if (backgroundStrategy == BACKGROUND_STRATEGY_CLAMP)
        cplBackwardMapping_kernel<BACKGROUND_STRATEGY_CLAMP><<<grids, threads,0 ,stream>>>(d_hX, d_hY, d_hZ,
                                                                                           d_fX, d_fY, d_fZ,
                                                                                           d_gX, d_gY, d_gZ,
                                                                                           delta,
                                                                                           sizeX, sizeY, sizeZ);
    else if (backgroundStrategy == BACKGROUND_STRATEGY_WRAP)
        cplBackwardMapping_kernel<BACKGROUND_STRATEGY_WRAP><<<grids, threads,0 ,stream>>>(d_hX, d_hY, d_hZ,
                                                                                          d_fX, d_fY, d_fZ,
                                                                                          d_gX, d_gY, d_gZ,
                                                                                          delta,
                                                                                          sizeX, sizeY, sizeZ);
    
}


void cplBackwardMapping(cplVector3DArray& d_h, const cplVector3DArray& d_f, const cplVector3DArray& d_g,
                        const Vector3Di& size,
                        float delta,
                        int backgroundStrategy, cudaStream_t stream)
{
    cplBackwardMapping(d_h.x, d_h.y, d_h.z,
                       d_f.x, d_f.y, d_f.z,
                       d_g.x, d_g.y, d_g.z,
                       size.x, size.y, size.z,
                       delta, backgroundStrategy, stream);
}

////////////////////////////////////////////////////////////////////////////////
//
////////////////////////////////////////////////////////////////////////////////
template<int backgroundStrategy>
__global__ void cplBackwardMapping_tex_kernel(float* d_hX, float* d_hY, float* d_hZ,
                                              const float* d_gX, const float* d_gY, const float* d_gZ,
                                              float delta,
                                              int sizeX, int sizeY, int sizeZ){

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i < sizeX && j < sizeY){
        int id = j * sizeX + i;
        for (int k=0; k< sizeZ; ++k, id+= sizeX*sizeY){
            float x = delta * d_gX[id] + (float)i;
            float y = delta * d_gY[id] + (float)j;
            float z = delta * d_gZ[id] + (float)k;

            float hx, hy, hz;

            triLerp_tex<backgroundStrategy>(hx, hy, hz, x, y, z, sizeX, sizeY, sizeZ);

            d_hX[id] = hx;
            d_hY[id] = hy;
            d_hZ[id] = hz;
        }
    }
}

template<int backgroundStrategy>
__global__ void cplBackwardMapping_tex_kernel(float* d_hX, float* d_hY, float* d_hZ,
                                              const float* d_gX, const float* d_gY, const float* d_gZ,
                                              float delta,
                                              int sizeX, int sizeY, int sizeZ,
                                              float ispX, float ispY, float ispZ
                                              ){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i < sizeX && j < sizeY){
        int id = j * sizeX + i;
        for (int k=0; k< sizeZ; ++k, id+= sizeX*sizeY){
            float x = delta * ispX * d_gX[id] + (float)i;
            float y = delta * ispY * d_gY[id] + (float)j;
            float z = delta * ispZ * d_gZ[id] + (float)k;

            float hx, hy, hz;

            triLerp_tex<backgroundStrategy>(hx, hy, hz, x, y, z, sizeX, sizeY, sizeZ);

            d_hX[id] = hx;
            d_hY[id] = hy;
            d_hZ[id] = hz;
        }
    }
}

void cplBackwardMapping_tex(float* d_hX, float* d_hY, float* d_hZ,
                            const float* d_fX, const float* d_fY, const float* d_fZ,
                            const float* d_gX, const float* d_gY, const float* d_gZ,
                            int sizeX, int sizeY, int sizeZ,
                            float delta,
                            int backgroundStrategy, cudaStream_t stream)
{
    dim3 threads(16,16);
    dim3 grids(iDivUp(sizeX, threads.x), iDivUp(sizeY, threads.y));

    int nP = sizeX * sizeY * sizeZ;
    
    cudaBindTexture(0, com_tex_float_x, d_fX, nP * sizeof(float));
    cudaBindTexture(0, com_tex_float_y, d_fY, nP * sizeof(float));
    cudaBindTexture(0, com_tex_float_z, d_fZ, nP * sizeof(float));

    if (backgroundStrategy == BACKGROUND_STRATEGY_ID)
        cplBackwardMapping_tex_kernel<BACKGROUND_STRATEGY_ID><<<grids, threads,0 ,stream>>>(d_hX, d_hY, d_hZ,
                                                                                            d_gX, d_gY, d_gZ,
                                                                                            delta,
                                                                                            sizeX, sizeY, sizeZ);
    else if (backgroundStrategy == BACKGROUND_STRATEGY_PARTIAL_ID)
        cplBackwardMapping_tex_kernel<BACKGROUND_STRATEGY_PARTIAL_ID><<<grids, threads,0 ,stream>>>(d_hX, d_hY, d_hZ,
                                                                                                    d_gX, d_gY, d_gZ,
                                                                                                    delta,
                                                                                                    sizeX, sizeY, sizeZ);
    else if (backgroundStrategy == BACKGROUND_STRATEGY_ZERO)
        cplBackwardMapping_tex_kernel<BACKGROUND_STRATEGY_ZERO><<<grids, threads,0 ,stream>>>(d_hX, d_hY, d_hZ,
                                                                                              d_gX, d_gY, d_gZ,
                                                                                              delta,
                                                                                              sizeX, sizeY, sizeZ);
    else if (backgroundStrategy == BACKGROUND_STRATEGY_CLAMP)
        cplBackwardMapping_tex_kernel<BACKGROUND_STRATEGY_CLAMP><<<grids, threads,0 ,stream>>>(d_hX, d_hY, d_hZ,
                                                                                               d_gX, d_gY, d_gZ,
                                                                                               delta,
                                                                                               sizeX, sizeY, sizeZ);
    else if (backgroundStrategy == BACKGROUND_STRATEGY_WRAP)
        cplBackwardMapping_tex_kernel<BACKGROUND_STRATEGY_WRAP><<<grids, threads,0 ,stream>>>(d_hX, d_hY, d_hZ,
                                                                                              d_gX, d_gY, d_gZ,
                                                                                              delta,
                                                                                              sizeX, sizeY, sizeZ);
    
}

void cplBackwardMapping_tex(cplVector3DArray& d_h, const cplVector3DArray& d_f, const cplVector3DArray& d_g,
                            const Vector3Di& size,
                            float delta,
                            int backgroundStrategy, cudaStream_t stream)
{
    cplBackwardMapping_tex(d_h.x, d_h.y, d_h.z,
                           d_f.x, d_f.y, d_f.z,
                           d_g.x, d_g.y, d_g.z,
                           size.x, size.y, size.z,
                           delta, backgroundStrategy, stream);
}
