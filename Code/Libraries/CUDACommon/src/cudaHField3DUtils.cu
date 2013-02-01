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
#include <cudaImage3D.h>

////////////////////////////////////////////////////////////////////////
// Processing on the HField 
//  1. Convert HField to unit spacing HField
//  2. Working on the unit spacing HField 
//  3. Convert result back to regular HField
////////////////////////////////////////////////////////////////////////

namespace cudaHField3DUtils
{
//////////////////////////////////////////////////////////////////////// 
// set hfield to identity
// i.e. h(x) = x
////////////////////////////////////////////////////////////////////////
    __global__ void setToIdentity_kernel(float* d_hx, float* d_hy, float* d_hz,
                                         int w, int h, int l){
        uint i     = blockIdx.x * blockDim.x + threadIdx.x;
        uint j     = blockIdx.y * blockDim.y + threadIdx.y;
        uint index = j * w + i;
        if (i < w && j < h){
            for (int k=0; k<l; ++k, index+=w*h){
                d_hx[index] = i;
                d_hy[index] = j;
                d_hz[index] = k;
            }
        }
    }
    void setToIdentity(float* d_hx, float* d_hy, float* d_hz, int w, int h, int l, cudaStream_t stream){
        dim3 threads(16,16);
        dim3 grids(iDivUp(w, threads.x), iDivUp(h, threads.y));
        setToIdentity_kernel<<<grids, threads, 0, stream>>>(d_hx, d_hy, d_hz, w, h, l);
    }

//////////////////////////////////////////////////////////////////////// 
// Add identity to the current field
// i.e. h(x) = x + h(x)
////////////////////////////////////////////////////////////////////////
    __global__ void addIdentity_kernel(float* d_hx, float* d_hy, float* d_hz, int w, int h, int l){
        uint i     = blockIdx.x * blockDim.x + threadIdx.x;
        uint j     = blockIdx.y * blockDim.y + threadIdx.y;
        uint index = j * w + i;

        if (i < w && j < h){
            for (int k=0; k<l; ++k, index +=w*h){
                d_hx[index] += i;
                d_hy[index] += j;
                d_hz[index] += k;
            }
        }
    }

    void addIdentity(float* d_hx, float* d_hy, float* d_hz, int w, int h, int l, cudaStream_t stream){
        dim3 threads(16,16);
        dim3 grids(iDivUp(w, threads.x), iDivUp(h, threads.y));
        addIdentity_kernel<<<grids, threads,0,stream>>>(d_hx, d_hy, d_hz, w, h, l);
    }

//////////////////////////////////////////////////////////////////////// 
// Convert velocity field to hfield
// i.e. h(x) = x + v(x) * delta
////////////////////////////////////////////////////////////////////////
    __global__ void velocityToH_kernel(float* d_hx, float* d_hy, float* d_hz,
                                       const float* d_vx, const float* d_vy, const float* d_vz,
                                       int w, int h, int l){
        uint i     = blockIdx.x * blockDim.x + threadIdx.x;
        uint j     = blockIdx.y * blockDim.y + threadIdx.y;
        uint index = j * w + i;

        if (i < w && j < h){
            for (int k=0; k<l; ++k, index+=w*h){
                d_hx[index] = float(i) + d_vx[index];
                d_hy[index] = float(j) + d_vy[index];
                d_hz[index] = float(k) + d_vz[index];
            }
        }
    }
    
    void velocityToH(float* d_hx, float* d_hy, float* d_hz,
                     const float* d_vx, const float* d_vy, const float* d_vz,
                     int w, int h, int l, cudaStream_t stream)
    {
        dim3 threads(16,16);
        dim3 grids(iDivUp(w, threads.x), iDivUp(h, threads.y));
        velocityToH_kernel<<<grids, threads, 0, stream>>>(d_hx, d_hy, d_hz,
                                                          d_vx, d_vy, d_vz,
                                                          w, h, l);
    }

    __global__ void velocityToH_kernel(float* d_hx, float* d_hy, float* d_hz,
                                       const float* d_vx, const float* d_vy, const float* d_vz,
                                       float delta,
                                       int w, int h, int l){
        uint i     = blockIdx.x * blockDim.x + threadIdx.x;
        uint j     = blockIdx.y * blockDim.y + threadIdx.y;
        uint index = j * w + i;

        if (i < w && j < h){
            for (int k=0; k<l; ++k, index+=w*h){
                d_hx[index] = float(i) + d_vx[index] * delta;
                d_hy[index] = float(j) + d_vy[index] * delta;
                d_hz[index] = float(k) + d_vz[index] * delta;
            }
        }
    }
    void velocityToH(float* d_hx, float* d_hy, float* d_hz,
                     const float* d_vx, const float* d_vy, const float* d_vz,
                     float delta,
                     int w, int h, int l, cudaStream_t stream)
    {
        dim3 threads(16,16);
        dim3 grids(iDivUp(w, threads.x), iDivUp(h, threads.y));
        velocityToH_kernel<<<grids, threads, 0, stream>>>(d_hx, d_hy, d_hz,
                                                          d_vx, d_vy, d_vz,
                                                          delta, w, h, l);
    }


//////////////////////////////////////////////////////////////////////// 
// Convert velocity field to hfield
// i.e. h(x) = x + v(x) * delta
////////////////////////////////////////////////////////////////////////
    __global__ void velocityToH_I_kernel(float* d_hx, float* d_hy, float* d_hz,
                                         int w, int h, int l){
        uint i     = blockIdx.x * blockDim.x + threadIdx.x;
        uint j     = blockIdx.y * blockDim.y + threadIdx.y;
        uint index = j * w + i;

        if (i < w && j < h){
            for (int k=0; k<l; ++k, index+=w*h){
                d_hx[index] = float(i) + d_hx[index];
                d_hy[index] = float(j) + d_hy[index];
                d_hz[index] = float(k) + d_hz[index];
            }
        }
    }
    void velocityToH_I(float* d_hx, float* d_hy, float* d_hz,
                       int w, int h, int l, cudaStream_t stream)
    {
        dim3 threads(16,16);
        dim3 grids(iDivUp(w, threads.x), iDivUp(h, threads.y));
        velocityToH_I_kernel<<<grids, threads, 0, stream>>>(d_hx, d_hy, d_hz,
                                                            w, h, l);
    }
    __global__ void velocityToH_I_kernel(float* d_hx, float* d_hy, float* d_hz,
                                         float delta,
                                         int w, int h, int l){
        uint i     = blockIdx.x * blockDim.x + threadIdx.x;
        uint j     = blockIdx.y * blockDim.y + threadIdx.y;
        uint index = j * w + i;

        if (i < w && j < h){
            for (int k=0; k<l; ++k, index+=w*h){
                d_hx[index] = float(i) + d_hx[index] * delta;
                d_hy[index] = float(j) + d_hy[index] * delta;
                d_hz[index] = float(k) + d_hz[index] * delta;
            }
        }
    }
    void velocityToH_I(float* d_hx, float* d_hy, float* d_hz,
                       float delta,
                       int w, int h, int l, cudaStream_t stream)
    {
        dim3 threads(16,16);
        dim3 grids(iDivUp(w, threads.x), iDivUp(h, threads.y));
        velocityToH_I_kernel<<<grids, threads, 0, stream>>>(d_hx, d_hy, d_hz,
                                                            delta, w, h, l);
    }

    __global__ void hToVelocity_kernel(float* d_vx, float* d_vy, float* d_vz,
                                       const float* d_hx, const float* d_hy, const float* d_hz,
                                       int w, int h, int l)
    {
        uint i     = blockIdx.x * blockDim.x + threadIdx.x;
        uint j     = blockIdx.y * blockDim.y + threadIdx.y;
        uint index = j * w + i;
        if (i < w && j < h){
            for (int k=0; k<l; ++k, index+=w*h){
                d_vx[index] = d_hx[index] - i;
                d_vy[index] = d_hy[index] - j;
                d_vz[index] = d_hz[index] - k;
            }
        }
    }

    void hToVelocity(float* d_vx, float* d_vy, float* d_vz,
                     const float* d_hx, const float* d_hy, const float* d_hz,
                     int w, int h, int l, cudaStream_t stream)
    {
        dim3 threads(16,16);
        dim3 grids(iDivUp(w, threads.x), iDivUp(h, threads.y));
        hToVelocity_kernel<<<grids, threads, 0, stream>>>(d_vx, d_vy, d_vz,
                                                          d_hx, d_hy, d_hz, w, h, l);
    }

    __global__ void hToVelocity_I_kernel(float* d_vx, float* d_vy, float* d_vz,
                                         int w, int h, int l)
    {
        uint i     = blockIdx.x * blockDim.x + threadIdx.x;
        uint j     = blockIdx.y * blockDim.y + threadIdx.y;
        uint index = j * w + i;
        if (i < w && j < h){
            for (int k=0; k<l; ++k, index+=w*h){
                d_vx[index] = d_vx[index] - i;
                d_vy[index] = d_vy[index] - j;
                d_vz[index] = d_vz[index] - k;
            }
        }
    }

    void hToVelocity_I(float* d_vx, float* d_vy, float* d_vz,
                       int w, int h, int l, cudaStream_t stream)
    {
        dim3 threads(16,16);
        dim3 grids(iDivUp(w, threads.x), iDivUp(h, threads.y));
        hToVelocity_I_kernel<<<grids, threads, 0, stream>>>(d_vx, d_vy, d_vz,
                                                            w, h, l);
    }

////////////////////////////////////////////////////////////////////////////
// Hfield to displacement 
// i.e. u(x) = h(x) - x
////////////////////////////////////////////////////////////////////////////
    void HToDisplacement(float* d_ux, float* d_uy, float* d_uz,
                         const float* d_hx, const float* d_hy, const float* d_hz,
                         int w, int h, int l, cudaStream_t stream){
        hToVelocity(d_ux, d_uy, d_uz, d_hx, d_hy, d_hz, w, h, l, stream);
    }

    void HToDisplacement_I(float* d_ux, float* d_uy, float* d_uz,
                           int w, int h, int l, cudaStream_t stream){
        hToVelocity_I(d_ux, d_uy, d_uz, w, h, l, stream);
    }

    ////////////////////////////////////////////////////////////////////////////
    // compose a velocity and hfield to get an hfield
    // h(x) = g(x) + v(g(x))
    ////////////////////////////////////////////////////////////////////////////
    template<int dir, BackgroundStrategy bg>
    __global__ void composeVH_kernel(float* d_hx, float* d_hy, float* d_hz,
                                     const float* d_vx, const float* d_vy, const float* d_vz,
                                     const float* d_gx, const float* d_gy, const float* d_gz,
                                     int w, int h, int l){

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
                if (dir == 1) {
                    d_hx[id] = x + hx;
                    d_hy[id] = y + hy;
                    d_hz[id] = z + hz;
                } else if (dir == -1){
                    d_hx[id] = x - hx;
                    d_hy[id] = y - hy;
                    d_hz[id] = z - hz;
                }
            }
        }
    }

    void composeVH(float* d_hx, float* d_hy, float* d_hz,
                   const float* d_vx, const float* d_vy, const float* d_vz,
                   const float* d_gx, const float* d_gy, const float* d_gz,
                   int w, int h, int l,   BackgroundStrategy bg, cudaStream_t stream){

        dim3 threads(16,16);
        dim3 grids(iDivUp(w, threads.x), iDivUp(h, threads.y));
        if (bg == BACKGROUND_STRATEGY_ZERO)
            composeVH_kernel<1, BACKGROUND_STRATEGY_ZERO><<<grids, threads, 0, stream>>>(d_hx, d_hy, d_hz,
                                                                                         d_vx, d_vy, d_vz,
                                                                                         d_gx, d_gy, d_gz,
                                                                                         w, h, l);
        else if (bg == BACKGROUND_STRATEGY_PARTIAL_ZERO)
            composeVH_kernel<1, BACKGROUND_STRATEGY_PARTIAL_ZERO><<<grids, threads, 0, stream>>>(d_hx, d_hy, d_hz,
                                                                                                 d_vx, d_vy, d_vz,
                                                                                                 d_gx, d_gy, d_gz,
                                                                                                 w, h, l);
        else
            throw AtlasWerksException(__FILE__,__LINE__,"Unsupported background strategy");
    }
    
    void composeVHInv(float* d_hx, float* d_hy, float* d_hz,
                      const float* d_vx, const float* d_vy, const float* d_vz,
                      const float* d_gx, const float* d_gy, const float* d_gz,
                      int w, int h, int l,
                      BackgroundStrategy bg, cudaStream_t stream){

        dim3 threads(16,16);
        dim3 grids(iDivUp(w, threads.x), iDivUp(h, threads.y));
        if (bg == BACKGROUND_STRATEGY_ZERO)
            composeVH_kernel<-1, BACKGROUND_STRATEGY_ZERO><<<grids, threads, 0, stream>>>(d_hx, d_hy, d_hz,
                                                                                          d_vx, d_vy, d_vz,
                                                                                          d_gx, d_gy, d_gz,
                                                                                          w, h, l);
        else if (bg == BACKGROUND_STRATEGY_PARTIAL_ZERO)
            composeVH_kernel<-1, BACKGROUND_STRATEGY_PARTIAL_ZERO><<<grids, threads, 0, stream>>>(d_hx, d_hy, d_hz,
                                                                                                  d_vx, d_vy, d_vz,
                                                                                                  d_gx, d_gy, d_gz,
                                                                                                  w, h, l);
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
                                     int w, int h, int l){
        
        uint i = blockIdx.x * blockDim.x + threadIdx.x;
        uint j = blockIdx.y * blockDim.y + threadIdx.y;
        if (i < w && j < h){
            int id = i + w * j;
            for (int k=0; k < l; ++k, id+=w*h){
                float x,y,z;

                if (fwd) {
                    x = i + d_vx[id];
                    y = j + d_vy[id];
                    z = k + d_vz[id];
                } else {
                    x = i - d_vx[id];
                    y = j - d_vy[id];
                    z = k - d_vz[id];
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
                   BackgroundStrategy bg, cudaStream_t stream)
    {
        dim3 threads(16,16);
        dim3 grids(iDivUp(w, threads.x), iDivUp(h, threads.y));
        if (bg == BACKGROUND_STRATEGY_ID)
            composeHV_kernel<true, BACKGROUND_STRATEGY_ID><<<grids, threads, 0, stream>>>(d_hx, d_hy, d_hz,
                                                                                          d_gx, d_gy, d_gz,
                                                                                          d_vx, d_vy, d_vz,
                                                                                          w, h, l);
        else if (bg == BACKGROUND_STRATEGY_PARTIAL_ID)
            composeHV_kernel<true, BACKGROUND_STRATEGY_PARTIAL_ID><<<grids, threads,0,stream>>>(d_hx, d_hy, d_hz,
                                                                                                d_gx, d_gy, d_gz,
                                                                                                d_vx, d_vy, d_vz,
                                                                                                w, h, l);
        else
            throw AtlasWerksException(__FILE__,__LINE__,"Unsupported background strategy");
    }

    void composeHVInv(float* d_hx, float* d_hy, float* d_hz,
                      const float* d_gx, const float* d_gy, const float* d_gz,
                      const float* d_vx, const float* d_vy, const float* d_vz,
                      int w, int h, int l,
                      BackgroundStrategy bg, cudaStream_t stream)
    {
        dim3 threads(16,16);
        dim3 grids(iDivUp(w, threads.x), iDivUp(h, threads.y));
        if (bg == BACKGROUND_STRATEGY_ID)
            composeHV_kernel<false, BACKGROUND_STRATEGY_ID><<<grids, threads, 0, stream>>>(d_hx, d_hy, d_hz,
                                                                                           d_gx, d_gy, d_gz,
                                                                                           d_vx, d_vy, d_vz,
                                                                                           w, h, l);
        else if (bg == BACKGROUND_STRATEGY_PARTIAL_ID)
            composeHV_kernel<false, BACKGROUND_STRATEGY_PARTIAL_ID><<<grids, threads, 0, stream>>>(d_hx, d_hy, d_hz,
                                                                                                   d_gx, d_gy, d_gz,
                                                                                                   d_vx, d_vy, d_vz,
                                                                                                   w, h, l);
        else
            throw AtlasWerksException(__FILE__,__LINE__,"Unsupported background strategy");
    }

    /**
     * compose a h field and a velocify field to get an hfield
     * h(x) = g(x+ delta * v(x))
     **/

    template<BackgroundStrategy bg>
    __global__ void composeHV_kernel(float* d_hx, float* d_hy, float* d_hz,
                                     const float* d_gx, const float* d_gy, const float* d_gz,
                                     const float* d_vx, const float* d_vy, const float* d_vz,
                                     float delta,
                                     int w, int h, int l){
        
        uint i = blockIdx.x * blockDim.x + threadIdx.x;
        uint j = blockIdx.y * blockDim.y + threadIdx.y;
        if (i < w && j < h){
            int id = i + w * j;
            for (int k=0; k < l; ++k, id+=w*h){
                float x,y,z;

                x = i + delta * d_vx[id];
                y = j + delta * d_vy[id];
                z = k + delta * d_vz[id];

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
                   float delta,
                   int w, int h, int l,
                   BackgroundStrategy bg, cudaStream_t stream)
    {
        dim3 threads(16,16);
        dim3 grids(iDivUp(w, threads.x), iDivUp(h, threads.y));
        if (bg == BACKGROUND_STRATEGY_ID)
            composeHV_kernel<BACKGROUND_STRATEGY_ID><<<grids, threads, 0, stream>>>(d_hx, d_hy, d_hz,
                                                                                    d_gx, d_gy, d_gz,
                                                                                    d_vx, d_vy, d_vz,
                                                                                    delta,
                                                                                    w, h, l);
        else if (bg == BACKGROUND_STRATEGY_PARTIAL_ID)
            composeHV_kernel<BACKGROUND_STRATEGY_PARTIAL_ID><<<grids, threads,0,stream>>>(d_hx, d_hy, d_hz,
                                                                                          d_gx, d_gy, d_gz,
                                                                                          d_vx, d_vy, d_vz,
                                                                                          delta,
                                                                                          w, h, l);
        else
            throw AtlasWerksException(__FILE__,__LINE__,"Unsupported background strategy");
    }

    /**
     * compose a h field and a velocify field to get an hfield
     * h(x) = g(x-delta *v(x))
     */

    void composeHVInv(float* d_hx, float* d_hy, float* d_hz,
                      const float* d_gx, const float* d_gy, const float* d_gz,
                      const float* d_vx, const float* d_vy, const float* d_vz,
                      float delta,
                      int w, int h, int l,
                      BackgroundStrategy bg, cudaStream_t stream)
    {
        composeHV(d_hx, d_hy, d_hz,
                  d_gx, d_gy, d_gz,
                  d_vx, d_vy, d_vz, - delta,
                  w , h, l, bg, stream);
    }

    ////////////////////////////////////////////////////////////////////////////
	// precompose h field with translation
	// creating h(x) = f(x + t)
	////////////////////////////////////////////////////////////////////////////
    __global__ void preComposeTranslation_kernel(float* d_hx, float* d_hy, float* d_hz,
                                                 const float* d_fx, const float* d_fy, const float* d_fz,
                                                 float tx, float ty, float tz,
                                                 int w, int h, int l)  {
        uint i = blockIdx.x * blockDim.x + threadIdx.x;
        uint j = blockIdx.y * blockDim.y + threadIdx.y;
        if (i < w && j < h){
            int id = i + w * j;
            for (int k=0; k < l; ++k, id+=w*h){
                float x,y,z;

                x = i + tx;
                y = j + ty;
                z = k + tz;
                
                float hx, hy, hz;
                triLerp<BACKGROUND_STRATEGY_ID>(hx, hy, hz,
                                                d_fx, d_fy, d_fz,
                                                x, y, z, w, h, l);
                d_hx[id] = hx;
                d_hy[id] = hy;
                d_hz[id] = hz;
            }
        }
    }

	void preComposeTranslation(float* d_hx, float* d_hy, float* d_hz,
                               const float* d_fx, const float* d_fy, const float* d_fz,
                               float tx, float ty, float tz,
                               int w, int h, int l, cudaStream_t stream){
        
        dim3 threads(16,16);
        dim3 grids(iDivUp(w, threads.x), iDivUp(h, threads.y));

        preComposeTranslation_kernel<<<grids, threads,0,stream>>>(d_hx, d_hy, d_hz,
                                                                  d_fx, d_fy, d_fz,
                                                                  tx, ty, tz, w, h, l);
    }

    ////////////////////////////////////////////////////////////////////////////
	// approximate the inverse of an incremental h field using according
	// to the following derivation
	//
	// hInv(x0) = x0 + d
	// x0 = h(x0 + d)
	// x0 = h(x0) + d // order zero expansion
	// d  = x0 - h(x0)
	//
	// hInv(x0) = x0 + x0 - h(x0)
	//
	////////////////////////////////////////////////////////////////////////////
    __global__ void computeInverseZerothOrder_kernel(float* d_hInvx, float* d_hInvy, float* d_hInvz,
                                                     const float* d_hx, const float* d_hy, const float* d_hz,
                                                     int w, int h, int l){
        uint i = blockIdx.x * blockDim.x + threadIdx.x;
        uint j = blockIdx.y * blockDim.y + threadIdx.y;
        if (i < w && j < h){
            int id = i + w * j;
            for (int k=0; k < l; ++k, id+=w*h){
                d_hInvx[id] = i + i - d_hx[id];
                d_hInvy[id] = j + j - d_hy[id];
                d_hInvz[id] = k + k - d_hz[id];
            }
        }

    }
    
    void computeInverseZerothOrder(float* d_hInvx, float* d_hInvy, float* d_hInvz,
                                   const float* d_hx, const float* d_hy, const float* d_hz,
                                   int w, int h, int l, cudaStream_t stream){
        dim3 threads(16,16);
        dim3 grids(iDivUp(w, threads.x), iDivUp(h, threads.y));
        computeInverseZerothOrder_kernel<<<grids, threads,0,stream>>>(d_hInvx, d_hInvy, d_hInvz,
                                                                      d_hx, d_hy, d_hz,
                                                                      w, h, l);
    }
    
    /////////////////////////////////////////////////////////////////////////////
	// apply hField to an image
	// defImage(x) = image(h(x))
	/////////////////////////////////////////////////////////////////////////////
    __global__ void  cudaHFieldApply_kernel(float* d_o, const float* d_i,
                                            const float* d_hx, const float* d_hy, const float* d_hz,
                                            int sizeX, int sizeY, int sizeZ)
    {
        int i = threadIdx.x + blockIdx.x * blockDim.x;
        int j = threadIdx.y + blockIdx.y * blockDim.y;
    
        if (i < sizeX && j < sizeY){
            int id = j * sizeX + i;
            for (int k=0; k< sizeZ; ++k, id+= sizeX*sizeY){
                float x = d_hx[id];
                float y = d_hy[id];
                float z = d_hz[id];
            
                d_o[id] = triLerp<BACKGROUND_STRATEGY_ZERO>(d_i, x, y, z, sizeX, sizeY, sizeZ);
            }
        }
    }

    void cudaHFieldApply(float* d_o, const float* d_i,
                         const float* d_hx, const float* d_hy, const float* d_hz,
                         int sizeX, int sizeY, int sizeZ, cudaStream_t stream)
    {
        dim3 threads(16,16);
        dim3 grids(iDivUp(sizeX, threads.x), iDivUp(sizeY, threads.y));
        cudaHFieldApply_kernel<<<grids, threads,0,stream>>>(d_o, d_i, d_hx, d_hy, d_hz, sizeX, sizeY, sizeZ);
    }

    __global__ void  cudaHFieldApply_tex_kernel(float* d_o,
                                                const float* d_hx, const float* d_hy, const float* d_hz,
                                                int sizeX, int sizeY, int sizeZ)
    {
        int i = threadIdx.x + blockIdx.x * blockDim.x;
        int j = threadIdx.y + blockIdx.y * blockDim.y;
            
        if (i < sizeX && j < sizeY){
            int id = j * sizeX + i;
            for (int k=0; k< sizeZ; ++k, id+= sizeX*sizeY){
                float x = d_hx[id];
                float y = d_hy[id];
                float z = d_hz[id];
                d_o[id] = triLerp_tex<BACKGROUND_STRATEGY_ZERO>(x, y, z, sizeX, sizeY, sizeZ);
            }
        }
    }
    
    void cudaHFieldApply_tex(float* d_o, const float* d_i,
                             const float* d_hx, const float* d_hy, const float* d_hz,
                             int sizeX, int sizeY, int sizeZ, cudaStream_t stream)
    {
        dim3 threads(16,16);
        dim3 grids(iDivUp(sizeX, threads.x), iDivUp(sizeY, threads.y));
        cudaBindTexture(0, com_tex_float, d_i, sizeX * sizeY * sizeZ * sizeof(float));
        cudaHFieldApply_tex_kernel<<<grids, threads,0,stream>>>(d_o, d_hx, d_hy, d_hz, sizeX, sizeY, sizeZ);
    }

    void apply(float* d_o, const float* d_i,
               const float* d_hx, const float* d_hy, const float* d_hz,
               int sizeX, int sizeY, int sizeZ, cudaStream_t stream){
        cudaHFieldApply_tex(d_o, d_i,
                            d_hx, d_hy, d_hz,
                            sizeX, sizeY, sizeZ, stream);
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
    template<bool fwd, BackgroundStrategy bg>
    __global__ void applyU_kernel(float* d_o, const float* d_i, 
                                  const float* d_uX, const float* d_uY, const float* d_uZ,
                                  int sizeX, int sizeY, int sizeZ){
        int i = threadIdx.x + blockIdx.x * blockDim.x;
        int j = threadIdx.y + blockIdx.y * blockDim.y;
        if (i < sizeX && j < sizeY){
            int id = j * sizeX + i;
            for (int k=0; k< sizeZ; ++k, id+= sizeX*sizeY){
                float x,y,z;
                if (fwd){
                    x = i + d_uX[id];
                    y = j + d_uY[id];
                    z = k + d_uZ[id];
                } else {
                    x = i - d_uX[id];
                    y = j - d_uY[id];
                    z = k - d_uZ[id];
                }
                d_o[id] = triLerp<bg>(d_i, x, y, z, sizeX, sizeY, sizeZ);
            }
        }
    }

    void applyU(float* d_o, const float* d_i,
                const float* d_ux, const float* d_uy, const float* d_uz,
                int sizeX, int sizeY, int sizeZ, cudaStream_t stream)
    {
        dim3 threads(16,16);
        dim3 grids(iDivUp(sizeX, threads.x), iDivUp(sizeY, threads.y));
        applyU_kernel<true, BACKGROUND_STRATEGY_ZERO><<<grids, threads, 0, stream>>>(d_o, d_i,
                                                                                     d_ux, d_uy, d_uz,
                                                                                     sizeX, sizeY, sizeZ);
    }
    void applyUInv(float* d_o, const float* d_i,
                   const float* d_ux, const float* d_uy, const float* d_uz,
                   int sizeX, int sizeY, int sizeZ, cudaStream_t stream)
    {
        dim3 threads(16,16);
        dim3 grids(iDivUp(sizeX, threads.x), iDivUp(sizeY, threads.y));
        applyU_kernel<false, BACKGROUND_STRATEGY_ZERO><<<grids, threads, 0, stream>>>(d_o, d_i,
                                                                                      d_ux, d_uy, d_uz,
                                                                                      sizeX, sizeY, sizeZ);
    }

    template<BackgroundStrategy bg>
    __global__ void applyU_kernel(float* d_o, const float* d_i, 
                                  const float* d_uX, const float* d_uY, const float* d_uZ,
                                  float delta,
                                  int sizeX, int sizeY, int sizeZ){
        int i = threadIdx.x + blockIdx.x * blockDim.x;
        int j = threadIdx.y + blockIdx.y * blockDim.y;
        if (i < sizeX && j < sizeY){
            int id = j * sizeX + i;
            for (int k=0; k< sizeZ; ++k, id+= sizeX*sizeY){
                float x,y,z;
                x = i + delta * d_uX[id];
                y = j + delta * d_uY[id];
                z = k + delta * d_uZ[id];
                d_o[id] = triLerp<bg>(d_i,  x, y, z, sizeX, sizeY, sizeZ);
            }
        }
    }

    void applyU(float* d_o, const float* d_i,
                const float* d_ux, const float* d_uy, const float* d_uz,
                float delta,
                int sizeX, int sizeY, int sizeZ, cudaStream_t stream)
    {
        dim3 threads(16,16);
        dim3 grids(iDivUp(sizeX, threads.x), iDivUp(sizeY, threads.y));
        applyU_kernel<BACKGROUND_STRATEGY_ZERO><<<grids, threads, 0, stream>>>(d_o, d_i,
                                                                               d_ux, d_uy, d_uz, delta,
                                                                               sizeX, sizeY, sizeZ);
    }

    void applyUInv(float* d_o, const float* d_i,
                   const float* d_ux, const float* d_uy, const float* d_uz,
                   float delta,
                   int sizeX, int sizeY, int sizeZ, cudaStream_t stream)
    {
        dim3 threads(16,16);
        dim3 grids(iDivUp(sizeX, threads.x), iDivUp(sizeY, threads.y));
        applyU_kernel<BACKGROUND_STRATEGY_ZERO><<<grids, threads, 0, stream>>>(d_o, d_i,
                                                                               d_ux, d_uy, d_uz, -delta,
                                                                               sizeX, sizeY, sizeZ);
    }
    ////////////////////////////////////////////////////////////////////////////////
    //
    ////////////////////////////////////////////////////////////////////////////////
    template<int rescale, BackgroundStrategy bg>
    __global__ void resample_kernel(float* d_ox, float* d_oy, float* d_oz,
                                    const float* d_ix, const float* d_iy, const float* d_iz,
                                    int osizeX, int osizeY, int osizeZ,
                                    int isizeX, int isizeY, int isizeZ)
    {
        uint x = blockIdx.x * blockDim.x + threadIdx.x;
        uint y = blockIdx.y * blockDim.y + threadIdx.y;

        float rX = (float)isizeX / (float)osizeX;
        float rY = (float)isizeY / (float)osizeY;
        float rZ = (float)isizeZ / (float)osizeZ;

        if (x < osizeX && y < osizeY){
            int id = x + osizeX * y;

            float i_x =  (rX - 1.f) / 2.f + x * rX;
            float i_y =  (rY - 1.f) / 2.f + y * rY;
            
            for (int z=0; z < osizeZ; ++z, id+=osizeX * osizeY){
                float i_z =  (rZ - 1.f) / 2.f + z * rZ;

                float ox, oy, oz;
                triLerp<bg>(ox, oy, oz,
                            d_ix, d_iy, d_iz,
                            i_x, i_y, i_z, isizeX, isizeY, isizeZ);
                
                if (rescale){
                    ox /= rX; oy /= rY; oz /= rZ;
                }
                
                d_ox[id] = ox;
                d_oy[id] = oy;
                d_oz[id] = oz;
            }
        }
    }

    void resample(cplVector3DArray& d_o,
                  const cplVector3DArray& d_i,
                  const Vector3Di& osize, const Vector3Di& isize,
                  BackgroundStrategy backgroundStrategy, bool rescaleVector, cudaStream_t stream)
    {
        if ((isize.x == osize.x) && (isize.y == osize.y) && (isize.z == osize.z)){
            copyArrayDeviceToDevice(d_o, d_i, isize.productOfElements());
            return;
        }
        dim3 threads(16,16);
        dim3 grids(iDivUp(osize.x, threads.x), iDivUp(osize.y, threads.y));
        if (rescaleVector){
            if (backgroundStrategy == BACKGROUND_STRATEGY_PARTIAL_ID)
                resample_kernel<1, BACKGROUND_STRATEGY_PARTIAL_ID><<<grids, threads,0,stream>>>(d_o.x, d_o.y, d_o.z,
                                                                                                d_i.x, d_i.y, d_i.z,
                                                                                                osize.x ,osize.y ,osize.z ,
                                                                                                isize.x ,isize.y ,isize.z);
            else if (backgroundStrategy == BACKGROUND_STRATEGY_ID)
                resample_kernel<1, BACKGROUND_STRATEGY_ID><<<grids, threads,0,stream>>>(d_o.x, d_o.y, d_o.z,
                                                                                        d_i.x, d_i.y, d_i.z,
                                                                                        osize.x ,osize.y ,osize.z ,
                                                                                        isize.x ,isize.y ,isize.z);
            else if (backgroundStrategy == BACKGROUND_STRATEGY_ZERO)
                resample_kernel<1, BACKGROUND_STRATEGY_ZERO><<<grids, threads,0,stream>>>(d_o.x, d_o.y, d_o.z,
                                                                                          d_i.x, d_i.y, d_i.z,
                                                                                          osize.x ,osize.y ,osize.z ,
                                                                                          isize.x ,isize.y ,isize.z);
            else if(backgroundStrategy == BACKGROUND_STRATEGY_CLAMP)
                resample_kernel<1, BACKGROUND_STRATEGY_CLAMP><<<grids, threads,0,stream>>>(d_o.x, d_o.y, d_o.z,
                                                                                           d_i.x, d_i.y, d_i.z,
                                                                                           osize.x ,osize.y ,osize.z ,
                                                                                           isize.x ,isize.y ,isize.z);
            else if(backgroundStrategy == BACKGROUND_STRATEGY_WRAP)
                resample_kernel<1, BACKGROUND_STRATEGY_WRAP><<<grids, threads,0,stream>>>(d_o.x, d_o.y, d_o.z,
                                                                                          d_i.x, d_i.y, d_i.z,
                                                                                          osize.x ,osize.y ,osize.z ,
                                                                                          isize.x ,isize.y ,isize.z);
        } else {
            if (backgroundStrategy == BACKGROUND_STRATEGY_PARTIAL_ID)
                resample_kernel<0, BACKGROUND_STRATEGY_PARTIAL_ID><<<grids, threads,0,stream>>>(d_o.x, d_o.y, d_o.z,
                                                                                                d_i.x, d_i.y, d_i.z,
                                                                                                osize.x ,osize.y ,osize.z ,
                                                                                                isize.x ,isize.y ,isize.z);
            else if (backgroundStrategy == BACKGROUND_STRATEGY_ID)
                resample_kernel<0, BACKGROUND_STRATEGY_ID><<<grids, threads,0,stream>>>(d_o.x, d_o.y, d_o.z,
                                                                                        d_i.x, d_i.y, d_i.z,
                                                                                        osize.x ,osize.y ,osize.z ,
                                                                                        isize.x ,isize.y ,isize.z);
            else if (backgroundStrategy == BACKGROUND_STRATEGY_ZERO)
                resample_kernel<0, BACKGROUND_STRATEGY_ZERO><<<grids, threads,0,stream>>>(d_o.x, d_o.y, d_o.z,
                                                                                          d_i.x, d_i.y, d_i.z,
                                                                                          osize.x ,osize.y ,osize.z ,
                                                                                          isize.x ,isize.y ,isize.z);
            else if(backgroundStrategy == BACKGROUND_STRATEGY_CLAMP)
                resample_kernel<0, BACKGROUND_STRATEGY_CLAMP><<<grids, threads,0,stream>>>(d_o.x, d_o.y, d_o.z,
                                                                                           d_i.x, d_i.y, d_i.z,
                                                                                           osize.x ,osize.y ,osize.z ,
                                                                                           isize.x ,isize.y ,isize.z);
            else if(backgroundStrategy == BACKGROUND_STRATEGY_WRAP)
                resample_kernel<0, BACKGROUND_STRATEGY_WRAP><<<grids, threads,0,stream>>>(d_o.x, d_o.y, d_o.z,
                                                                                          d_i.x, d_i.y, d_i.z,
                                                                                          osize.x ,osize.y ,osize.z ,
                                                                                          isize.x ,isize.y ,isize.z);
        }
    }

    template<int rescale, BackgroundStrategy bg>
    __global__ void resample_tex_kernel(float* d_ox, float* d_oy, float* d_oz,
                                        int osizeX, int osizeY, int osizeZ,
                                        int isizeX, int isizeY, int isizeZ)
    {
        uint x = blockIdx.x * blockDim.x + threadIdx.x;
        uint y = blockIdx.y * blockDim.y + threadIdx.y;

        float rX = (float)isizeX / (float)osizeX;
        float rY = (float)isizeY / (float)osizeY;
        float rZ = (float)isizeZ / (float)osizeZ;

        if (x < osizeX && y < osizeY){
            int id = x + osizeX * y;

            float i_x =  (rX - 1.f) / 2.f + x * rX;
            float i_y =  (rY - 1.f) / 2.f + y * rY;
            
            for (int z=0; z < osizeZ; ++z, id+=osizeX * osizeY){
                float i_z =  (rZ - 1.f) / 2.f + z * rZ;

                float ox, oy, oz;
                triLerp_tex<bg>(ox, oy, oz,
                                i_x, i_y, i_z, isizeX, isizeY, isizeZ);
                if (rescale){
                    ox /= rX; oy /= rY; oz /= rZ;
                }
                d_ox[id] = ox;
                d_oy[id] = oy;
                d_oz[id] = oz;
            }
        }
    }

    void resample_tex(cplVector3DArray& d_o,
                      const cplVector3DArray& d_i,
                      const Vector3Di& osize, const Vector3Di& isize,
                      BackgroundStrategy backgroundStrategy, bool rescaleVector, cudaStream_t stream)
    {
        if ((isize.x == osize.x) && (isize.y == osize.y) && (isize.z == osize.z)) {
            copyArrayDeviceToDevice(d_o, d_i, isize.productOfElements());
            return;
        }
        cache_bind(d_i.x, d_i.y, d_i.z);
        dim3 threads(16,16);
        dim3 grids(iDivUp(osize.x, threads.x), iDivUp(osize.y, threads.y));
        if (rescaleVector){
            if (backgroundStrategy == BACKGROUND_STRATEGY_PARTIAL_ID)
                resample_tex_kernel<1, BACKGROUND_STRATEGY_PARTIAL_ID><<<grids, threads, 0, stream>>>
                    (d_o.x, d_o.y, d_o.z,
                     osize.x ,osize.y ,osize.z ,
                     isize.x ,isize.y ,isize.z);
            else if (backgroundStrategy == BACKGROUND_STRATEGY_ID)
                resample_tex_kernel<1, BACKGROUND_STRATEGY_ID><<<grids, threads, 0, stream>>>
                    (d_o.x, d_o.y, d_o.z,
                     osize.x ,osize.y ,osize.z ,
                     isize.x ,isize.y ,isize.z);
            else if (backgroundStrategy == BACKGROUND_STRATEGY_ZERO)
                resample_tex_kernel<1, BACKGROUND_STRATEGY_ZERO><<<grids, threads, 0, stream>>>
                    (d_o.x, d_o.y, d_o.z,
                     osize.x ,osize.y ,osize.z ,
                     isize.x ,isize.y ,isize.z);
            else if(backgroundStrategy == BACKGROUND_STRATEGY_CLAMP)
                resample_tex_kernel<1, BACKGROUND_STRATEGY_CLAMP><<<grids, threads, 0, stream>>>
                    (d_o.x, d_o.y, d_o.z,
                     osize.x ,osize.y ,osize.z ,
                     isize.x ,isize.y ,isize.z);
            else if(backgroundStrategy == BACKGROUND_STRATEGY_WRAP)
                resample_tex_kernel<1, BACKGROUND_STRATEGY_WRAP><<<grids, threads, 0, stream>>>
                    (d_o.x, d_o.y, d_o.z,
                     osize.x ,osize.y ,osize.z ,
                     isize.x ,isize.y ,isize.z);
        } else {
            if (backgroundStrategy == BACKGROUND_STRATEGY_PARTIAL_ID)
                resample_tex_kernel<0, BACKGROUND_STRATEGY_PARTIAL_ID><<<grids, threads, 0, stream>>>
                    (d_o.x, d_o.y, d_o.z,
                     osize.x ,osize.y ,osize.z ,
                     isize.x ,isize.y ,isize.z);
            else if (backgroundStrategy == BACKGROUND_STRATEGY_ID)
                resample_tex_kernel<0, BACKGROUND_STRATEGY_ID><<<grids, threads, 0, stream>>>
                    (d_o.x, d_o.y, d_o.z,
                     osize.x ,osize.y ,osize.z ,
                     isize.x ,isize.y ,isize.z);
            else if (backgroundStrategy == BACKGROUND_STRATEGY_ZERO)
                resample_tex_kernel<0, BACKGROUND_STRATEGY_ZERO><<<grids, threads, 0, stream>>>
                    (d_o.x, d_o.y, d_o.z,
                     osize.x ,osize.y ,osize.z ,
                     isize.x ,isize.y ,isize.z);
            else if(backgroundStrategy == BACKGROUND_STRATEGY_CLAMP)
                resample_tex_kernel<0, BACKGROUND_STRATEGY_CLAMP><<<grids, threads, 0, stream>>>
                    (d_o.x, d_o.y, d_o.z,
                     osize.x ,osize.y ,osize.z ,
                     isize.x ,isize.y ,isize.z);
            else if(backgroundStrategy == BACKGROUND_STRATEGY_WRAP)
                resample_tex_kernel<0, BACKGROUND_STRATEGY_WRAP><<<grids, threads, 0, stream>>>
                    (d_o.x, d_o.y, d_o.z,
                     osize.x ,osize.y ,osize.z ,
                     isize.x ,isize.y ,isize.z);
        }
    }

  void divergence(float *d_o,
		  const cplVector3DArray& d_h,
		  cplVector3DArray& d_scratch,
		  const Vector3Di& size, 
		  const Vector3Df& sp,
		  bool wrap,
		  cudaStream_t stream)
  {
    uint nVox = size.productOfElements();
    cplVectorOpers::SetMem(d_o, 0.f, nVox);
    cplComputeGradient(d_scratch, d_h.x, size, sp, wrap, stream);
    cplVectorOpers::Add_I(d_o, d_scratch.x, nVox);
    cplComputeGradient(d_scratch, d_h.y, size, sp, wrap, stream);
    cplVectorOpers::Add_I(d_o, d_scratch.y, nVox);
    cplComputeGradient(d_scratch, d_h.z, size, sp, wrap, stream);
    cplVectorOpers::Add_I(d_o, d_scratch.z, nVox);
  }

    /**
     * compose a h field and a velocify field to get an hfield
     * h(x) = g(x+v(x))
     */
    template<int backgroundStrategy>
    __global__ void composeHV_tex_kernel(float* d_hX, float* d_hY, float* d_hZ,
                                         const float* d_vX, const float* d_vY, const float* d_vZ,
                                         float delta,
                                         int sizeX, int sizeY, int sizeZ){

        int i = threadIdx.x + blockIdx.x * blockDim.x;
        int j = threadIdx.y + blockIdx.y * blockDim.y;
        if (i < sizeX && j < sizeY){
            int id = j * sizeX + i;
            for (int k=0; k< sizeZ; ++k, id+= sizeX*sizeY){
                float x = delta * d_vX[id] + (float)i;
                float y = delta * d_vY[id] + (float)j;
                float z = delta * d_vZ[id] + (float)k;

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
                       int w, int h, int l, BackgroundStrategy bg, cudaStream_t stream)
    {
        dim3 threads(16,16);
        dim3 grids(iDivUp(w, threads.x), iDivUp(h, threads.y));

        cache_bind(d_gx, d_gy, d_gz);
        if (bg == BACKGROUND_STRATEGY_ID)
            composeHV_tex_kernel<BACKGROUND_STRATEGY_ID><<<grids, threads, 0, stream>>>(d_hx, d_hy, d_hz,
                                                                                        d_vx, d_vy, d_vz,
                                                                                        delta, w, h, l);
        else if (bg == BACKGROUND_STRATEGY_PARTIAL_ID)
            composeHV_tex_kernel<BACKGROUND_STRATEGY_PARTIAL_ID><<<grids, threads, 0, stream>>>(d_hx, d_hy, d_hz,
                                                                                                d_vx, d_vy, d_vz,
                                                                                                delta, w, h, l);
        else
            throw AtlasWerksException(__FILE__,__LINE__,"Unsupported background strategy");
    }

    void composeHVInv_tex(float* d_hx, float* d_hy, float* d_hz,
                          const float* d_gx, const float* d_gy, const float* d_gz,
                          const float* d_vx, const float* d_vy, const float* d_vz,
                          float delta,
                          int w, int h, int l, BackgroundStrategy bg, cudaStream_t stream)
    {
        composeHV_tex(d_hx, d_hy, d_hz,
                      d_gx, d_gy, d_gz,
                      d_vx, d_vy, d_vz, -delta, 
                      w, h, l, bg, stream);
    }
    
    ////////////////////////////////////////////////////////////////////////////////
    // Compute the Jacobian Determinent 
    ////////////////////////////////////////////////////////////////////////////////
    __device__ float det(float a00, float a01, float a02,
                         float a10, float a11, float a12,
                         float a20, float a21, float a22)
    {
        return a00 * a11 * a22 + a01 * a12 * a20 + a02 * a10 * a21 -
            a02 * a11 * a20 - a00 * a12 * a21 - a01 * a10 * a22;
    }
                     
    __global__ void jacobianDet_kernel(float* d_detJ,
                                       float* d_Xgx, float* d_Xgy, float* d_Xgz,
                                       float* d_Ygx, float* d_Ygy, float* d_Ygz,
                                       float* d_Zgx, float* d_Zgy, float* d_Zgz,
                                       int n)
    {
        uint blockId = blockIdx.y * gridDim.x + blockIdx.x;
        uint id      = blockId * blockDim.x + threadIdx.x;

        if (id < n){
            float a00 = d_Xgx[id], a01 = d_Xgy[id], a02 = d_Xgz[id];
            float a10 = d_Ygx[id], a11 = d_Ygy[id], a12 = d_Ygz[id];
            float a20 = d_Zgx[id], a21 = d_Zgy[id], a22 = d_Zgz[id];
        
            d_detJ[id] = det(a00, a01, a02,
                             a10, a11, a12,
                             a20, a21, a22);
        }
    }
                                       
    void jacobianDet(float* d_detJ,
                     cplVector3DArray& d_Xg,
                     cplVector3DArray& d_Yg,
                     cplVector3DArray& d_Zg,
                     int n, cudaStream_t stream){

        dim3 threads(256);
        dim3 grids(iDivUp(n, threads.x));
        checkConfig(grids);
        jacobianDet_kernel<<<grids, threads, 0, stream>>>(d_detJ,
                                                          d_Xg.x, d_Xg.y, d_Xg.z,
                                                          d_Yg.x, d_Yg.y, d_Yg.z,
                                                          d_Zg.x, d_Zg.y, d_Zg.z, n);
    }
    
    __global__ void jacobianDetV_kernel(float* d_detJ,
                                        float* d_Xgx, float* d_Xgy, float* d_Xgz,
                                        float* d_Ygx, float* d_Ygy, float* d_Ygz,
                                        float* d_Zgx, float* d_Zgy, float* d_Zgz,
                                        int n)
    {
        uint blockId = blockIdx.y * gridDim.x + blockIdx.x;
        uint id      = blockId * blockDim.x + threadIdx.x;

        if (id < n){
            float a00 = 1.f + d_Xgx[id], a01 = d_Xgy[id], a02 = d_Xgz[id];
            float a10 = d_Ygx[id], a11 = 1.f + d_Ygy[id], a12 = d_Ygz[id];
            float a20 = d_Zgx[id], a21 = d_Zgy[id], a22 = 1.f + d_Zgz[id];
        
            d_detJ[id] = det(a00, a01, a02,
                             a10, a11, a12,
                             a20, a21, a22);
        }
    }

    __global__ void jacobianDetVInv_kernel(float* d_detJ,
                                           float* d_Xgx, float* d_Xgy, float* d_Xgz,
                                           float* d_Ygx, float* d_Ygy, float* d_Ygz,
                                           float* d_Zgx, float* d_Zgy, float* d_Zgz,
                                           int n)
    {
        uint blockId = blockIdx.y * gridDim.x + blockIdx.x;
        uint id      = blockId * blockDim.x + threadIdx.x;

        if (id < n){
            float a00 = 1.f - d_Xgx[id], a01 = d_Xgy[id], a02 = d_Xgz[id];
            float a10 = d_Ygx[id], a11 = 1.f - d_Ygy[id], a12 = d_Ygz[id];
            float a20 = d_Zgx[id], a21 = d_Zgy[id], a22 = 1.f - d_Zgz[id];
            
            d_detJ[id] = det(a00, a01, a02,
                             a10, a11, a12,
                             a20, a21, a22);
        }
    }
    
    void jacobianDetV(float* d_detJ,
                      cplVector3DArray& d_Xg, cplVector3DArray& d_Yg, cplVector3DArray& d_Zg,
                      int n, cudaStream_t stream){

        dim3 threads(256);
        dim3 grids(iDivUp(n, threads.x));
        checkConfig(grids);
        jacobianDetV_kernel<<<grids, threads, 0, stream>>>(d_detJ,
                                                           d_Xg.x, d_Xg.y, d_Xg.z,
                                                           d_Yg.x, d_Yg.y, d_Yg.z,
                                                           d_Zg.x, d_Zg.y, d_Zg.z, n);
    }
    
    void jacobianDetVInv(float* d_detJ,
                         cplVector3DArray& d_Xg,
                         cplVector3DArray& d_Yg,
                         cplVector3DArray& d_Zg,
                         int n, cudaStream_t stream){
        dim3 threads(256);
        dim3 grids(iDivUp(n, threads.x));
        checkConfig(grids);
        jacobianDetVInv_kernel<<<grids, threads, 0, stream>>>(d_detJ,
                                                              d_Xg.x, d_Xg.y, d_Xg.z,
                                                              d_Yg.x, d_Yg.y, d_Yg.z,
                                                              d_Zg.x, d_Zg.y, d_Zg.z, n);
    }

    void jacobian(cplVector3DArray& d_Xg, cplVector3DArray& d_Yg, cplVector3DArray& d_Zg, 
                  cplVector3DArray& d_h,  const Vector3Di& size, const Vector3Df& sp, cudaStream_t stream)
    {
      cplComputeGradient(d_Xg, d_h.x, size, sp, stream);
      cplComputeGradient(d_Yg, d_h.y, size, sp, stream);
      cplComputeGradient(d_Zg, d_h.z, size, sp, stream);
    }

    void jacobianDetHField(float* d_J, cplVector3DArray& d_h,
                           cplVector3DArray &d_Xg, cplVector3DArray &d_Yg, cplVector3DArray &d_Zg,
                           const Vector3Di& size, const Vector3Df& sp, cudaStream_t stream)
    {
        jacobian(d_Xg, d_Yg, d_Zg, d_h, size, sp, stream);
        jacobianDet(d_J, d_Xg, d_Yg, d_Zg, size.productOfElements(), stream);
    }

    ////////////////////////////////////////////////////////////////////////////
// compose two h fields using trilinear interpolation
// h(x) = f(g(x))
////////////////////////////////////////////////////////////////////////////
	template<int backgroundStrategy>
    __global__ void compose_kernel(float* d_hx, float* d_hy, float* d_hz,
                                   const float* d_fx, const float* d_fy, const float* d_fz,
                                   const float* d_gx, const float* d_gy, const float* d_gz,
                                   int w, int h, int l){

        uint i = blockIdx.x * blockDim.x + threadIdx.x;
        uint j = blockIdx.y * blockDim.y + threadIdx.y;
        if (i < w && j < h){
            int id = i + w * j;
            for (int k=0; k < l; ++k, id+=w*h){

                float x = d_gx[id];
                float y = d_gy[id];
                float z = d_gz[id];
                
                float hx, hy, hz;
                
                triLerp<backgroundStrategy>(hx, hy, hz,
                                            d_fx, d_fy, d_fz,
                                            x, y, z,  w, h, l);
                d_hx[id] = hx;
                d_hy[id] = hy;
                d_hz[id] = hz;
            }
        }
    }

    void compose(float* d_hx, float* d_hy, float* d_hz,
                 const float* d_fx, const float* d_fy, const float* d_fz,
                 const float* d_gx, const float* d_gy, const float* d_gz,
                 int w, int h, int l, int backgroundStrategy, cudaStream_t stream){

        dim3 threads(16,16);
        dim3 grids(iDivUp(w, threads.x), iDivUp(h, threads.y));

        if (backgroundStrategy == BACKGROUND_STRATEGY_PARTIAL_ID)    
            compose_kernel<BACKGROUND_STRATEGY_PARTIAL_ID><<<grids, threads, 0, stream>>>(d_hx, d_hy, d_hz,
                                                                                          d_fx, d_fy, d_fz,
                                                                                          d_gx, d_gy, d_gz,
                                                                                          w, h, l);
        else if (backgroundStrategy == BACKGROUND_STRATEGY_ID)
            compose_kernel<BACKGROUND_STRATEGY_ID><<<grids, threads, 0, stream>>>(d_hx, d_hy, d_hz,
                                                                                  d_fx, d_fy, d_fz,
                                                                                  d_gx, d_gy, d_gz,
                                                                                  w, h, l);
        else if (backgroundStrategy == BACKGROUND_STRATEGY_ZERO)
            compose_kernel<BACKGROUND_STRATEGY_ZERO><<<grids, threads, 0, stream>>>(d_hx, d_hy, d_hz,
                                                                                    d_fx, d_fy, d_fz,
                                                                                    d_gx, d_gy, d_gz,
                                                                                    w, h, l);
        else if(backgroundStrategy == BACKGROUND_STRATEGY_CLAMP)
            compose_kernel<BACKGROUND_STRATEGY_CLAMP><<<grids, threads, 0, stream>>>(d_hx, d_hy, d_hz,
                                                                                     d_fx, d_fy, d_fz,
                                                                                     d_gx, d_gy, d_gz,
                                                                                     w, h, l);
        else if(backgroundStrategy == BACKGROUND_STRATEGY_WRAP)
            compose_kernel<BACKGROUND_STRATEGY_WRAP><<<grids, threads, 0, stream>>>(d_hx, d_hy, d_hz,
                                                                                    d_fx, d_fy, d_fz,
                                                                                    d_gx, d_gy, d_gz,
                                                                                    w, h, l);
        else
            throw AtlasWerksException(__FILE__,__LINE__,"Unsupported background strategy");
    }

};



