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

#include "cudaTrilerp.cu"
#include <cudaHField3DUtils.h>
#include <cpl.h>
#include <AtlasWerksException.h>
#include <fstream>

namespace cudaHField3DUtils
{
//////////////////////////////////////////////////////////////////////// 
// Convert from regular hField to normalized one (unit spacing)
// i.e. h(x) = h(x) / sp
////////////////////////////////////////////////////////////////////////
    __global__ void normalize_kernel(float* d_hx, float* d_hy, float* d_hz,
                                     int w, int h, int l,
                                     float  ispX, float ispY, float ispZ){
        uint i     = blockIdx.x * blockDim.x + threadIdx.x;
        uint j     = blockIdx.y * blockDim.y + threadIdx.y;
        uint index = j * w + i;
        if (i < w && j < h){
            for (int k=0; k<l; ++k, index+=w*h){
                d_hx[index] *= ispX;
                d_hy[index] *= ispY;
                d_hz[index] *= ispZ;
            }
        }
    }

    void normalize(float* d_hx, float* d_hy, float* d_hz,
                   int w, int h, int l,
                   float spX, float spY, float spZ, cudaStream_t stream){
        dim3 threads(16,16);
        dim3 grids(iDivUp(w, threads.x), iDivUp(h, threads.y));
        normalize_kernel<<<grids, threads, 0, stream>>>(d_hx, d_hy, d_hz,
                                                        w, h, l, 1.f/spX, 1.f/spY, 1.f/spZ);
    }

//////////////////////////////////////////////////////////////////////// 
// Convert from normalize hField to regular one 
// i.e. h(x) = h(x) * sp
////////////////////////////////////////////////////////////////////////
    __global__ void restoreSP_kernel(float* d_hx, float* d_hy, float* d_hz,
                                     int w, int h, int l,
                                     float  spX, float spY, float spZ){
        uint i     = blockIdx.x * blockDim.x + threadIdx.x;
        uint j     = blockIdx.y * blockDim.y + threadIdx.y;
        uint index = j * w + i;
        if (i < w && j < h){
            for (int k=0; k<l; ++k, index+=w*h){
                d_hx[index] *= spX;
                d_hy[index] *= spY;
                d_hz[index] *= spZ;
            }
        }
    }

    void restoreSP(float* d_hx, float* d_hy, float* d_hz,
                   int w, int h, int l,
                   float spX, float spY, float spZ, cudaStream_t stream){
        dim3 threads(16,16);
        dim3 grids(iDivUp(w, threads.x), iDivUp(h, threads.y));
        restoreSP_kernel<<<grids, threads, 0, stream>>>(d_hx, d_hy, d_hz,
                                                        w, h, l, spX, spY, spZ);
    }
   
//////////////////////////////////////////////////////////////////////// 
// set hfield to identity (for regular hfield)
// i.e. h(x) = x
////////////////////////////////////////////////////////////////////////
    __global__ void setToIdentity_kernel(float* d_hx, float* d_hy, float* d_hz,
                                         int w, int h, int l,
                                         float spX, float spY, float spZ){
        uint i     = blockIdx.x * blockDim.x + threadIdx.x;
        uint j     = blockIdx.y * blockDim.y + threadIdx.y;
        uint index = j * w + i;
        if (i < w && j < h){
            for (int k=0; k<l; ++k, index+=w*h){
                d_hx[index] = i * spX;
                d_hy[index] = j * spY;
                d_hz[index] = k * spZ;
            }
        }
    }
    void setToIdentity(float* d_hx, float* d_hy, float* d_hz,
                       int w, int h, int l,
                       float spX, float spY, float spZ, 
                       cudaStream_t stream){
        dim3 threads(16,16);
        dim3 grids(iDivUp(w, threads.x), iDivUp(h, threads.y));
        setToIdentity_kernel<<<grids, threads, 0, stream>>>(d_hx, d_hy, d_hz,
                                                            w, h, l, spX, spY, spZ);
    }

//////////////////////////////////////////////////////////////////////// 
// Add identity to the current field
// i.e. h(x) = x + h(x)
////////////////////////////////////////////////////////////////////////
    __global__ void addIdentity_kernel(float* d_hx, float* d_hy, float* d_hz,
                                       int w, int h, int l,
                                       float spX, float spY, float spZ){
        uint i     = blockIdx.x * blockDim.x + threadIdx.x;
        uint j     = blockIdx.y * blockDim.y + threadIdx.y;
        uint index = j * w + i;

        if (i < w && j < h){
            for (int k=0; k<l; ++k, index +=w*h){
                d_hx[index] += i * spX;
                d_hy[index] += j * spY;
                d_hz[index] += k * spZ;
            }
        }
    }
    void addIdentity(float* d_hx, float* d_hy, float* d_hz,
                     int w, int h, int l, float spX, float spY, float spZ, 
                     cudaStream_t stream){
        dim3 threads(16,16);
        dim3 grids(iDivUp(w, threads.x), iDivUp(h, threads.y));
        addIdentity_kernel<<<grids, threads,0,stream>>>(d_hx, d_hy, d_hz, w, h, l, spX, spY, spZ);
    }

//////////////////////////////////////////////////////////////////////// 
// Convert velocity field to hfield spacing adjustment (in unnormalized space)
// i.e. h(x) = x * sp + v(x) 
////////////////////////////////////////////////////////////////////////
    __global__ void velocityToH_US_kernel(float* d_hx, float* d_hy, float* d_hz,
                                          const float* d_vx, const float* d_vy, const float* d_vz,
                                          int w, int h, int l,
                                          float spX, float spY, float spZ)
    {
        uint i     = blockIdx.x * blockDim.x + threadIdx.x;
        uint j     = blockIdx.y * blockDim.y + threadIdx.y;
        uint index = j * w + i;

        if (i < w && j < h){
            for (int k=0; k<l; ++k, index+=w*h){
                d_hx[index] = i * spX + d_vx[index];
                d_hy[index] = j * spY + d_vy[index];
                d_hz[index] = k * spZ + d_vz[index];
            }
        }
    }

    void velocityToH_US(float* d_hx, float* d_hy, float* d_hz,
                                  const float* d_vx, const float* d_vy, const float* d_vz,
                                  int w, int h, int l,
                                  float spX, float spY, float spZ, cudaStream_t stream)
    {
        dim3 threads(16,16);
        dim3 grids(iDivUp(w, threads.x), iDivUp(h, threads.y));
        velocityToH_US_kernel<<<grids, threads, 0, stream>>>(d_hx, d_hy, d_hz,
                                                                       d_vx, d_vy, d_vz,
                                                                       w, h, l,
                                                                       spX, spY, spZ);
    }

//////////////////////////////////////////////////////////////////////// 
// Convert velocity field to hfield spacing adjustment
// i.e. h(x) = x + v(x) * isp
////////////////////////////////////////////////////////////////////////
    __global__ void velocityToH_kernel(float* d_hx, float* d_hy, float* d_hz,
                                       const float* d_vx, const float* d_vy, const float* d_vz,
                                       int w, int h, int l,
                                       float ispX, float ispY, float ispZ)
    {
        uint i     = blockIdx.x * blockDim.x + threadIdx.x;
        uint j     = blockIdx.y * blockDim.y + threadIdx.y;
        uint index = j * w + i;

        if (i < w && j < h){
            for (int k=0; k<l; ++k, index+=w*h){
                d_hx[index] = i + ispX * d_vx[index];
                d_hy[index] = j + ispY * d_vy[index];
                d_hz[index] = k + ispZ * d_vz[index];
            }
        }
    }

    void velocityToH(float* d_hx, float* d_hy, float* d_hz,
                     const float* d_vx, const float* d_vy, const float* d_vz,
                     int w, int h, int l,
                     float spX, float spY, float spZ, cudaStream_t stream)
    {
        dim3 threads(16,16);
        dim3 grids(iDivUp(w, threads.x), iDivUp(h, threads.y));
        velocityToH_kernel<<<grids, threads, 0, stream>>>(d_hx, d_hy, d_hz,
                                                          d_vx, d_vy, d_vz,
                                                          w, h, l,
                                                          1.f/spX, 1.f/spY, 1.f/spZ);
    }

 //////////////////////////////////////////////////////////////////////// 
// Convert velocity field to hfield spacing adjustment
// i.e. h(x) = x + v(x) * isp * delta
////////////////////////////////////////////////////////////////////////
    __global__ void velocityToH_kernel(float* d_hx, float* d_hy, float* d_hz,
                                       const float* d_vx, const float* d_vy, const float* d_vz,
                                       float delta,
                                       int w, int h, int l,
                                       float ispX, float ispY, float ispZ)
    {
        uint i     = blockIdx.x * blockDim.x + threadIdx.x;
        uint j     = blockIdx.y * blockDim.y + threadIdx.y;
        uint index = j * w + i;

        if (i < w && j < h){
            for (int k=0; k<l; ++k, index+=w*h){
                d_hx[index] = i + delta * ispX * d_vx[index];
                d_hy[index] = j + delta * ispY * d_vy[index];
                d_hz[index] = k + delta * ispZ * d_vz[index];
            }
        }
    }

    void velocityToH(float* d_hx, float* d_hy, float* d_hz,
                     const float* d_vx, const float* d_vy, const float* d_vz,
                     float delta,
                     int w, int h, int l,
                     float spX, float spY, float spZ, cudaStream_t stream)
    {
        dim3 threads(16,16);
        dim3 grids(iDivUp(w, threads.x), iDivUp(h, threads.y));
        velocityToH_kernel<<<grids, threads, 0, stream>>>(d_hx, d_hy, d_hz,
                                                          d_vx, d_vy, d_vz,
                                                          delta, w, h, l,
                                                          1.f/spX, 1.f/spY, 1.f/spZ);
    }

    __global__ void velocityToH_I_kernel(float* d_hx, float* d_hy, float* d_hz,
                                         int w, int h, int l, float ispX, float ispY, float ispZ){
        uint i     = blockIdx.x * blockDim.x + threadIdx.x;
        uint j     = blockIdx.y * blockDim.y + threadIdx.y;
        uint index = j * w + i;

        if (i < w && j < h){
            for (int k=0; k<l; ++k, index+=w*h){
                d_hx[index] = float(i) + ispX * d_hx[index];
                d_hy[index] = float(j) + ispY * d_hy[index];
                d_hz[index] = float(k) + ispZ * d_hz[index];
            }
        }
    }
    
    void velocityToH_I(float* d_hx, float* d_hy, float* d_hz,
                       int w, int h, int l, float spX, float spY, float spZ,
                       cudaStream_t stream)
    {
        dim3 threads(16,16);
        dim3 grids(iDivUp(w, threads.x), iDivUp(h, threads.y));
        velocityToH_I_kernel<<<grids, threads, 0, stream>>>(d_hx, d_hy, d_hz,
                                                            w, h, l, 1.f/spX, 1.f/spY, 1.f/spZ);
    }

    __global__ void velocityToH_I_kernel(float* d_hx, float* d_hy, float* d_hz,
                                         float delta,
                                         int w, int h, int l, float ispX, float ispY, float ispZ){
        uint i     = blockIdx.x * blockDim.x + threadIdx.x;
        uint j     = blockIdx.y * blockDim.y + threadIdx.y;
        uint index = j * w + i;

        if (i < w && j < h){
            for (int k=0; k<l; ++k, index+=w*h){
                d_hx[index] = float(i) + delta * ispX * d_hx[index];
                d_hy[index] = float(j) + delta * ispY * d_hy[index];
                d_hz[index] = float(k) + delta * ispZ * d_hz[index];

            }
        }
    }
    void velocityToH_I(float* d_hx, float* d_hy, float* d_hz,
                       float delta,
                       int w, int h, int l, float spX, float spY, float spZ, cudaStream_t stream)
    {
        dim3 threads(16,16);
        dim3 grids(iDivUp(w, threads.x), iDivUp(h, threads.y));
        velocityToH_I_kernel<<<grids, threads, 0, stream>>>(d_hx, d_hy, d_hz,
                                                            delta, w, h, l, 1.f/spX, 1.f/spY, 1.f/spZ);
    }


    __global__ void hToVelocity_kernel(float* d_vx, float* d_vy, float* d_vz,
                                       const float* d_hx, const float* d_hy, const float* d_hz,
                                       int w, int h, int l,
                                       float spX, float spY, float spZ)
    {
        uint i     = blockIdx.x * blockDim.x + threadIdx.x;
        uint j     = blockIdx.y * blockDim.y + threadIdx.y;
        uint index = j * w + i;
        if (i < w && j < h){
            for (int k=0; k<l; ++k, index+=w*h){
                d_vx[index] = (d_hx[index] - i) * spX;
                d_vy[index] = (d_hy[index] - j) * spY;
                d_vz[index] = (d_hz[index] - k) * spZ;
            }
        }
    }
    
    void hToVelocity(float* d_vx, float* d_vy, float* d_vz,
                     const float* d_hx, const float* d_hy, const float* d_hz,
                     int w, int h, int l, float spX, float spY, float spZ, cudaStream_t stream)
    {
        dim3 threads(16,16);
        dim3 grids(iDivUp(w, threads.x), iDivUp(h, threads.y));
        hToVelocity_kernel<<<grids, threads, 0, stream>>>(d_vx, d_vy, d_vz, d_hx, d_hy, d_hz,
                                                          w, h, l, spX, spY, spZ);
    }

    __global__ void hToVelocity_I_kernel(float* d_vx, float* d_vy, float* d_vz,
                                         int w, int h, int l,
                                         float spX, float spY, float spZ)
    {
        uint i     = blockIdx.x * blockDim.x + threadIdx.x;
        uint j     = blockIdx.y * blockDim.y + threadIdx.y;
        uint index = j * w + i;
        if (i < w && j < h){
            for (int k=0; k<l; ++k, index+=w*h){
                d_vx[index] = (d_vx[index] - i) * spX;
                d_vy[index] = (d_vy[index] - j) * spY;
                d_vz[index] = (d_vz[index] - k) * spZ;
            }
        }
    }
    
    void hToVelocity_I(float* d_vx, float* d_vy, float* d_vz,
                       int w, int h, int l, float spX, float spY, float spZ, cudaStream_t stream)
    {
        dim3 threads(16,16);
        dim3 grids(iDivUp(w, threads.x), iDivUp(h, threads.y));
        hToVelocity_I_kernel<<<grids, threads, 0, stream>>>(d_vx, d_vy, d_vz,
                                                            w, h, l, spX, spY, spZ);
    }

    void HToDisplacement(float* d_ux, float* d_uy, float* d_uz,
                         const float* d_hx, const float* d_hy, const float* d_hz,
                         int w, int h, int l, float spX, float spY, float spZ, cudaStream_t stream){
        hToVelocity(d_ux, d_uy, d_uz, d_hx, d_hy, d_hz, w, h, l, spX, spY, spZ, stream);
    }

    void HToDisplacement_I(float* d_ux, float* d_uy, float* d_uz,
                           int w, int h, int l, float spX, float spY, float spZ, cudaStream_t stream){
        hToVelocity_I(d_ux, d_uy, d_uz, w, h, l, spX, spY, spZ, stream);
    }

    ////////////////////////////////////////////////////////////////////////////
    // compose a velocity and h field to get an hfield
    // h(x) = g(x) + v(g(x))
    ////////////////////////////////////////////////////////////////////////////
    template<int dir, BackgroundStrategy bg>
    __global__ void composeVH_kernel(float* d_hx, float* d_hy, float* d_hz,
                                     const float* d_vx, const float* d_vy, const float* d_vz,
                                     const float* d_gx, const float* d_gy, const float* d_gz,
                                     int w, int h, int l,
                                     float ispX, float ispY, float ispZ)
    {
        uint i = blockIdx.x * blockDim.x + threadIdx.x;
        uint j = blockIdx.y * blockDim.y + threadIdx.y;
        if (i < w && j < h){
            int id = i + w * j;
            for (int k=0; k < l; ++k, id+=w*h){

                float x = d_gx[id];
                float y = d_gy[id];
                float z = d_gz[id];
                
                float hx, hy, hz;
                triLerp<bg>(hx, hy, hz,
                            d_vx, d_vy, d_vz,
                            x, y, z, w, h, l);
                if (dir == 1){
                    d_hx[id] = x + ispX*hx;
                    d_hy[id] = y + ispY*hy;
                    d_hz[id] = z + ispZ*hz;
                }else if (dir == -1){
                    d_hx[id] = x - ispX*hx;
                    d_hy[id] = y - ispY*hy;
                    d_hz[id] = z - ispZ*hz;
                }
            }
        }
    }

    void composeVH(float* d_hx, float* d_hy, float* d_hz,
                   const float* d_vx, const float* d_vy, const float* d_vz,
                   const float* d_gx, const float* d_gy, const float* d_gz,
                   int w, int h, int l,
                   float spX, float spY, float spZ,
                   BackgroundStrategy bg, cudaStream_t stream)
    {
        dim3 threads(16,16);
        dim3 grids(iDivUp(w, threads.x), iDivUp(h, threads.y));
        if (bg == BACKGROUND_STRATEGY_ZERO)
            composeVH_kernel<1, BACKGROUND_STRATEGY_ZERO><<<grids, threads, 0, stream>>>(d_hx, d_hy, d_hz,
                                                                                         d_vx, d_vy, d_vz,
                                                                                         d_gx, d_gy, d_gz,
                                                                                         w, h, l,
                                                                                         1.f / spX, 1.f / spY, 1.f / spZ);
        else if (bg == BACKGROUND_STRATEGY_PARTIAL_ZERO)
            composeVH_kernel<1, BACKGROUND_STRATEGY_PARTIAL_ZERO><<<grids, threads, 0, stream>>>(d_hx, d_hy, d_hz,
                                                                                                 d_vx, d_vy, d_vz,
                                                                                                 d_gx, d_gy, d_gz,
                                                                                                 w, h, l,
                                                                                                 1.f / spX, 1.f / spY, 1.f / spZ);
        else
            throw AtlasWerksException(__FILE__,__LINE__,"Unsupported background strategy");
    }

    ////////////////////////////////////////////////////////////////////////////
    // compose a velocity and h field to get an hfield (spacing adjustment)
    // h(x) = g(x) - v(g(x))
    ////////////////////////////////////////////////////////////////////////////
    void composeVHInv(float* d_hx, float* d_hy, float* d_hz,
                      const float* d_vx, const float* d_vy, const float* d_vz,
                      const float* d_gx, const float* d_gy, const float* d_gz,
                      int w, int h, int l,
                      float spX, float spY, float spZ,
                      BackgroundStrategy bg, cudaStream_t stream)                  
    {
        dim3 threads(16,16);
        dim3 grids(iDivUp(w, threads.x), iDivUp(h, threads.y));
        if (bg == BACKGROUND_STRATEGY_ZERO)
            composeVH_kernel<-1, BACKGROUND_STRATEGY_ZERO><<<grids, threads, 0, stream>>>(d_hx, d_hy, d_hz,
                                                                                          d_vx, d_vy, d_vz,
                                                                                          d_gx, d_gy, d_gz,
                                                                                          w, h, l,
                                                                                          1.f / spX, 1.f / spY, 1.f / spZ);
        else if (bg == BACKGROUND_STRATEGY_PARTIAL_ZERO)
            composeVH_kernel<-1, BACKGROUND_STRATEGY_PARTIAL_ZERO><<<grids, threads, 0, stream>>>(d_hx, d_hy, d_hz,
                                                                                                  d_vx, d_vy, d_vz,
                                                                                                  d_gx, d_gy, d_gz,
                                                                                                  w, h, l,
                                                                                                  1.f / spX, 1.f / spY, 1.f / spZ);
        else
            throw AtlasWerksException(__FILE__,__LINE__,"Unsupported background strategy");
    }

    /**
     * compose a h field and a velocify field to get an hfield
     * h(x) = g(x+v(x))
     *
     * davisb 2007
     */
    template<bool fwd, BackgroundStrategy bg>
    __global__ void composeHV_kernel(float* d_hx, float* d_hy, float* d_hz,
                                     const float* d_gx, const float* d_gy, const float* d_gz,
                                     const float* d_vx, const float* d_vy, const float* d_vz,
                                     int w, int h, int l,
                                     float ispX, float ispY, float ispZ){
        
        uint i = blockIdx.x * blockDim.x + threadIdx.x;
        uint j = blockIdx.y * blockDim.y + threadIdx.y;
        if (i < w && j < h){
            int id = i + w * j;
            for (int k=0; k < l; ++k, id+=w*h){
                float x,y,z;

                if (fwd){
                    x = i + d_vx[id] * ispX;
                    y = j + d_vy[id] * ispY;
                    z = k + d_vz[id] * ispZ;
                } else {
                    x = i - d_vx[id] * ispX;
                    y = j - d_vy[id] * ispY;
                    z = k - d_vz[id] * ispZ;
                }

                float hx, hy, hz;
                triLerp<bg>(hx, hy, hz,
                            d_gx, d_gy, d_gz,
                            x, y, z, w, h, l);
                d_hx[id] = hx;
                d_hy[id] = hy;
                d_hz[id] = hz;
            }
        }
    }

    void composeHV(float* d_hx, float* d_hy, float* d_hz,
                   const float* d_gx, const float* d_gy, const float* d_gz,
                   const float* d_vx, const float* d_vy, const float* d_vz,
                   int w, int h, int l,
                   float spX, float spY, float spZ,
                   BackgroundStrategy bg, cudaStream_t stream)
    {
        dim3 threads(16,16);
        dim3 grids(iDivUp(w, threads.x), iDivUp(h, threads.y));
        if (bg == BACKGROUND_STRATEGY_ID)
            composeHV_kernel<true, BACKGROUND_STRATEGY_ID><<<grids, threads,0,stream>>>(d_hx, d_hy, d_hz,
                                                                                     d_gx, d_gy, d_gz,
                                                                                     d_vx, d_vy, d_vz,
                                                                                     w, h, l,
                                                                                     1.f / spX, 1.f / spY, 1.f / spZ);
        else if (bg == BACKGROUND_STRATEGY_PARTIAL_ID)
            composeHV_kernel<true, BACKGROUND_STRATEGY_PARTIAL_ID><<<grids, threads,0,stream>>>(d_hx, d_hy, d_hz,
                                                                                             d_gx, d_gy, d_gz,
                                                                                             d_vx, d_vy, d_vz,
                                                                                             w, h, l,
                                                                                             1.f / spX, 1.f / spY, 1.f / spZ);
        else
            throw AtlasWerksException(__FILE__,__LINE__,"Unsupported background strategy");
    }
    
    
    void composeHVInv(float* d_hx, float* d_hy, float* d_hz,
                      const float* d_gx, const float* d_gy, const float* d_gz,
                      const float* d_vx, const float* d_vy, const float* d_vz,
                      int w, int h, int l,
                      float spX, float spY, float spZ,
                      BackgroundStrategy bg, cudaStream_t stream)                  
    {
        dim3 threads(16,16);
        dim3 grids(iDivUp(w, threads.x), iDivUp(h, threads.y));
        if (bg == BACKGROUND_STRATEGY_ID)
            composeHV_kernel<false, BACKGROUND_STRATEGY_ID><<<grids, threads,0,stream>>>(d_hx, d_hy, d_hz,
                                                                                      d_gx, d_gy, d_gz,
                                                                                      d_vx, d_vy, d_vz,
                                                                                      w, h, l,
                                                                                      1.f / spX, 1.f / spY, 1.f / spZ);
        else if (bg == BACKGROUND_STRATEGY_PARTIAL_ID)
            composeHV_kernel<false, BACKGROUND_STRATEGY_PARTIAL_ID><<<grids, threads,0,stream>>>(d_hx, d_hy, d_hz,
                                                                                              d_gx, d_gy, d_gz,
                                                                                              d_vx, d_vy, d_vz,
                                                                                              w, h, l,
                                                                                              1.f / spX, 1.f / spY, 1.f / spZ);
        else
            throw AtlasWerksException(__FILE__,__LINE__,"Unsupported background strategy");
    }

    template<BackgroundStrategy bg>
    __global__ void composeHV_kernel(float* d_hx, float* d_hy, float* d_hz,
                                     const float* d_gx, const float* d_gy, const float* d_gz,
                                     const float* d_vx, const float* d_vy, const float* d_vz,
                                     float delta,
                                     int w, int h, int l,
                                     float ispX, float ispY, float ispZ){
        
        uint i = blockIdx.x * blockDim.x + threadIdx.x;
        uint j = blockIdx.y * blockDim.y + threadIdx.y;
        if (i < w && j < h){
            int id = i + w * j;
            for (int k=0; k < l; ++k, id+=w*h){
                float x,y,z;

                x = i + delta * d_vx[id] * ispX;
                y = j + delta * d_vy[id] * ispY;
                z = k + delta * d_vz[id] * ispZ;
                
                float hx, hy, hz;
                triLerp<bg>(hx, hy, hz,
                            d_gx, d_gy, d_gz,
                            x, y, z, w, h, l);
                d_hx[id] = hx;
                d_hy[id] = hy;
                d_hz[id] = hz;
            }
        }
    }

    void composeHV(float* d_hx, float* d_hy, float* d_hz,
                   const float* d_gx, const float* d_gy, const float* d_gz,
                   const float* d_vx, const float* d_vy, const float* d_vz, float delta,
                   int w, int h, int l,
                   float spX, float spY, float spZ,
                   BackgroundStrategy bg, cudaStream_t stream)
    {
        dim3 threads(16,16);
        dim3 grids(iDivUp(w, threads.x), iDivUp(h, threads.y));
        if (bg == BACKGROUND_STRATEGY_ID)
            composeHV_kernel<BACKGROUND_STRATEGY_ID><<<grids, threads,0,stream>>>(d_hx, d_hy, d_hz,
                                                                                  d_gx, d_gy, d_gz,
                                                                                  d_vx, d_vy, d_vz,
                                                                                  delta,
                                                                                  w, h, l,
                                                                                  1.f / spX, 1.f / spY, 1.f / spZ);
        else if (bg == BACKGROUND_STRATEGY_PARTIAL_ID)
            composeHV_kernel<BACKGROUND_STRATEGY_PARTIAL_ID><<<grids, threads,0,stream>>>(d_hx, d_hy, d_hz,
                                                                                          d_gx, d_gy, d_gz,
                                                                                          d_vx, d_vy, d_vz,
                                                                                          delta,
                                                                                          w, h, l,
                                                                                          1.f / spX, 1.f / spY, 1.f / spZ);
        else
            throw AtlasWerksException(__FILE__,__LINE__,"Unsupported background strategy");
    }
    
    
    void composeHVInv(float* d_hx, float* d_hy, float* d_hz,
                      const float* d_gx, const float* d_gy, const float* d_gz,
                      const float* d_vx, const float* d_vy, const float* d_vz,
                      float delta,
                      int w, int h, int l,
                      float spX, float spY, float spZ,
                      BackgroundStrategy bg, cudaStream_t stream)                  
    {
        composeHV(d_hx, d_hy, d_hz,
                  d_gx, d_gy, d_gz,
                  d_vx, d_vy, d_vz, - delta,
                  w , h, l, spX, spY, spZ, bg, stream);
    }

    /**
	 * apply uField to an image
	 * defImage(x) = image(x+u(x))
	 *
	 * trilerp by default but will use nearest neighbor if flag is set
	 * to true
	 *
	 * NOTE: this does not round for integer types
	 *
	 */
    template<BackgroundStrategy bg>
    __global__ void applyU_kernel(float* d_o, const float* d_i, 
                                  const float* d_uX, const float* d_uY, const float* d_uZ,
                                  int sizeX, int sizeY, int sizeZ,
                                  float iSpX, float iSpY, float iSpZ
                                  ){
        
        int i = threadIdx.x + blockIdx.x * blockDim.x;
        int j = threadIdx.y + blockIdx.y * blockDim.y;
        if (i < sizeX && j < sizeY){
            int id = j * sizeX + i;
            for (int k=0; k< sizeZ; ++k, id+= sizeX*sizeY){
                float x = iSpX * d_uX[id] + (float)i;
                float y = iSpY * d_uY[id] + (float)j;
                float z = iSpZ * d_uZ[id] + (float)k;
                d_o[id] = triLerp<bg>(d_i,  x, y, z, sizeX, sizeY, sizeZ);
            }
        }
    }


    void saveToAfile(int& sizeX, int& sizeY, int& sizeZ,
                     float& spX, float& spY, float& spZ,
                     float* h_i, float* h_px, float* h_py, float* h_pz)
    {
        std::ofstream fo("saveNew.dat", std::ofstream::binary);

        int nP = sizeX * sizeY * sizeZ;

        fo.write((char*)&sizeX, sizeof(int));
        fo.write((char*)&sizeY, sizeof(int));
        fo.write((char*)&sizeZ, sizeof(int));

        fo.write((char*)&spX, sizeof(float));
        fo.write((char*)&spY, sizeof(float));
        fo.write((char*)&spZ, sizeof(float));

        
        fo.write((char*)h_i,  nP * sizeof(float));
        fo.write((char*)h_px, nP * sizeof(float));
        fo.write((char*)h_py, nP * sizeof(float));
        fo.write((char*)h_pz, nP * sizeof(float));

        fo.close();
    }

    void applyU(float* d_o, const float* d_i,
                const float* d_ux, const float* d_uy, const float* d_uz,
                int sizeX, int sizeY, int sizeZ,
                float spX, float spY, float spZ, cudaStream_t stream)
    {
        dim3 threads(16,16);
        dim3 grids(iDivUp(sizeX, threads.x), iDivUp(sizeY, threads.y));

        {
            int size = sizeX * sizeY * sizeZ;
            float* h_i = new float [size];
            float* h_ux= new float [size];
            float* h_uy= new float [size];
            float* h_uz= new float [size];
//             saveToAfile(sizeX, sizeY, sizeZ,
//                         spX, spY, spZ,
//                         h_i, h_ux, h_uy, h_uz);
            delete []h_i;
            delete []h_ux;delete []h_uy;delete []h_uz;
        }
        
        applyU_kernel<BACKGROUND_STRATEGY_ZERO><<<grids, threads, 0, stream>>>(d_o, d_i,
                                                                               d_ux, d_uy, d_uz,
                                                                               sizeX, sizeY, sizeZ,
                                                                               1.f/spX, 1.f/spY, 1.f/spZ);
        cudaThreadSynchronize();
        cutilCheckMsg("applyU***");
    }
    void applyUInv(float* d_o, const float* d_i,
                   const float* d_ux, const float* d_uy, const float* d_uz,
                   int sizeX, int sizeY, int sizeZ,
                   float spX, float spY, float spZ, cudaStream_t stream)
    {
        dim3 threads(16,16);
        dim3 grids(iDivUp(sizeX, threads.x), iDivUp(sizeY, threads.y));
        applyU_kernel<BACKGROUND_STRATEGY_ZERO><<<grids, threads, 0, stream>>>(d_o, d_i,
                                                                              d_ux, d_uy, d_uz,
                                                                               sizeX, sizeY, sizeZ,
                                                                               -1.f/spX, -1.f/spY, -1.f/spZ);
    }
    
    template<BackgroundStrategy bg>
    __global__ void applyU_kernel(float* d_o, const float* d_i, 
                                  const float* d_uX, const float* d_uY, const float* d_uZ,
                                  float delta, 
                                  int sizeX, int sizeY, int sizeZ,
                                  float iSpX, float iSpY, float iSpZ
                                  ){
        
        int i = threadIdx.x + blockIdx.x * blockDim.x;
        int j = threadIdx.y + blockIdx.y * blockDim.y;
        if (i < sizeX && j < sizeY){
            int id = j * sizeX + i;
            for (int k=0; k< sizeZ; ++k, id+= sizeX*sizeY){
            
                float x = delta * iSpX * d_uX[id] + (float)i;
                float y = delta * iSpY * d_uY[id] + (float)j;
                float z = delta * iSpZ * d_uZ[id] + (float)k;

                d_o[id] = triLerp<bg>(d_i,  x, y, z, sizeX, sizeY, sizeZ);
            }
        }
    }

    void applyU(float* d_o, const float* d_i,
                const float* d_ux, const float* d_uy, const float* d_uz, float delta,
                int sizeX, int sizeY, int sizeZ,
                float spX, float spY, float spZ, cudaStream_t stream)
    {
        dim3 threads(16,16);
        dim3 grids(iDivUp(sizeX, threads.x), iDivUp(sizeY, threads.y));
        applyU_kernel<BACKGROUND_STRATEGY_ZERO><<<grids, threads,0,stream>>>(d_o, d_i,
                                                                             d_ux, d_uy, d_uz, delta,
                                                                             sizeX, sizeY, sizeZ,
                                                                             1.f/spX, 1.f/spY, 1.f/spZ);
    }

    void applyUInv(float* d_o, const float* d_i,
                   const float* d_ux, const float* d_uy, const float* d_uz, float delta,
                   int sizeX, int sizeY, int sizeZ, 
                   float spX, float spY, float spZ, cudaStream_t stream)
    {
        dim3 threads(16,16);
        dim3 grids(iDivUp(sizeX, threads.x), iDivUp(sizeY, threads.y));
        
        applyU_kernel<BACKGROUND_STRATEGY_ZERO><<<grids, threads, 0, stream>>>(d_o, d_i,
                                                                               d_ux, d_uy, d_uz,-delta,
                                                                               sizeX, sizeY, sizeZ,
                                                                               1.f/spX, 1.f/spY, 1.f/spZ);
    }

    template<int backgroundStrategy>
    __global__ void composeHV_tex_kernel(float* d_hX, float* d_hY, float* d_hZ,
                                         const float* d_vX, const float* d_vY, const float* d_vZ,
                                         float delta,
                                         int sizeX, int sizeY, int sizeZ,
                                         float spX, float spY, float spZ )
    {

        int i = threadIdx.x + blockIdx.x * blockDim.x;
        int j = threadIdx.y + blockIdx.y * blockDim.y;
        if (i < sizeX && j < sizeY){
            int id = j * sizeX + i;
            for (int k=0; k< sizeZ; ++k, id+= sizeX*sizeY){
                float x = delta * spX * d_vX[id] + (float)i;
                float y = delta * spY * d_vY[id] + (float)j;
                float z = delta * spZ * d_vZ[id] + (float)k;

                float hx, hy, hz;
                triLerp_tex<backgroundStrategy>(hx, hy, hz, x, y, z, sizeX, sizeY, sizeZ);

                d_hX[id] = hx;
                d_hY[id] = hy;
                d_hZ[id] = hz;
            }
        }
    }

    void composeHV_tex(float* d_hx, float* d_hy, float* d_hz,
                       const float* d_gx, const float* d_gy, const float* d_gz,
                       const float* d_vx, const float* d_vy, const float* d_vz,
                       float delta,
                       int w, int h, int l,
                       float spX, float spY, float spZ,
                       BackgroundStrategy bg, cudaStream_t stream)
    {
        dim3 threads(16,16);
        dim3 grids(iDivUp(w, threads.x), iDivUp(h, threads.y));

        cache_bind(d_gx, d_gy, d_gz);
        if (bg == BACKGROUND_STRATEGY_ID)
            composeHV_tex_kernel<BACKGROUND_STRATEGY_ID><<<grids, threads, 0, stream>>>(d_hx, d_hy, d_hz,
                                                                                        d_vx, d_vy, d_vz,
                                                                                        delta, w, h, l, spX, spY, spZ);
        else if (bg == BACKGROUND_STRATEGY_PARTIAL_ID)
            composeHV_tex_kernel<BACKGROUND_STRATEGY_PARTIAL_ID><<<grids, threads, 0, stream>>>(d_hx, d_hy, d_hz,
                                                                                                d_vx, d_vy, d_vz,
                                                                                                delta, w, h, l, spX, spY, spZ);
        else
            throw AtlasWerksException(__FILE__,__LINE__,"Unsupported background strategy");
    }

    void composeHVInv_tex(float* d_hx, float* d_hy, float* d_hz,
                          const float* d_gx, const float* d_gy, const float* d_gz,
                          const float* d_vx, const float* d_vy, const float* d_vz,
                          float delta,
                          int w, int h, int l,
                          float spX, float spY, float spZ,
                          BackgroundStrategy bg, cudaStream_t stream)
    {
        composeHV_tex(d_hx, d_hy, d_hz,
                      d_gx, d_gy, d_gz,
                      d_vx, d_vy, d_vz, -delta, 
                      w, h, l, spX, spY, spZ,
                      bg, stream);
    }
};

