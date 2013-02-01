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
#include <cudaSORSolver.h>
#include <Vector3D.h>

/*----------------------------------------------------------------------
  PDE solver for hemlotz equation on the grids
  d_gv = L^{-1} (d_f) with L is the Helmlotz operator
  Inputs : d_gf        : the force field on the grids
           alpha, gamma: Helmlotz equation parametter (alpha = 1.0)
           w, h,l      : size of the grid (the domain will go from [0:w-1,0:h-1]
  Output :
           d_gv : the vector field on the grids
  ---------------------------------------------------------------------*/
texture<float , 1, cudaReadModeElementType> com_tex_float;
template<int color>
__global__ void
SORIHelmholtz3D_tex( float* g_U, float* g_f,
                     float* g_Un,
                     int w, int h, int l,
                     float alpha, float omega, float d, float h2) 
{
    const float oneOver = 1.f /d;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = 1 + ((j+i+color) & 1);
    const int wh = w * h;
    int index = i + j * w + k * wh ;
    float oldValue, newValue;
    // update the inside point only
    if ((i > 1) && (j > 1) &&  (i < w - 2) && (j < h - 2)){
        while (k < l-2){
            oldValue = tex1Dfetch(com_tex_float, index);

            newValue = tex1Dfetch(com_tex_float, index + 1);
            newValue+= tex1Dfetch(com_tex_float, index - 1);
            newValue+= tex1Dfetch(com_tex_float, index + w);
            newValue+= tex1Dfetch(com_tex_float, index - w);
            newValue+= tex1Dfetch(com_tex_float, index + wh);
            newValue+= tex1Dfetch(com_tex_float, index - wh);
            newValue*= alpha;
            
            newValue-= g_f[index] * h2;
            newValue*= oneOver;

            if (k < 2)
                g_Un[index] = 0;
            else             
                g_Un[index] =  omega * (newValue - oldValue) + oldValue;

            k+=2;
            index += 2* wh;
        }
    }
}

void cplSORSolver3D(float* d_gv, float* d_gf,
                     float alpha, float gamma,
                     int w, int h, int l, int nIters, float* d_temp, cudaStream_t stream)
{
    assert(d_gv != d_gf);
    int size    = w * h * l;
    int memSize = size * sizeof(float);
    int need_temp = (d_temp == NULL);
    if (need_temp)
        dmemAlloc(d_temp, w * h * l);

    const float h2   = 1.0;
    float pj    = 1.f/ 3 * (cos(M_PI / (w - 1)) + cos (M_PI / (h - 1)) + cos (M_PI / (l - 1)));
    float omega = 2 / ( 1 + sqrt(1 - pj * pj));
    float d     = 6 * alpha - gamma * h2;

    dim3 threads(16,16);
    dim3 grids(iDivUp(w, threads.x),iDivUp(h, threads.y));
    for (unsigned int i=0; i< nIters; i+=2){
        copyArrayDeviceToDeviceAsync(d_temp, d_gv, size, stream);
        cudaBindTexture(0, com_tex_float, d_gv, memSize);
        SORIHelmholtz3D_tex<0><<<grids, threads, 0, stream>>>(d_gv, d_gf, d_temp, w, h, l,
                                                              alpha, omega, d, h2);

        copyArrayDeviceToDeviceAsync(d_gv, d_temp, size, stream);
        cudaBindTexture(0, com_tex_float, d_temp, memSize);
        SORIHelmholtz3D_tex<1><<<grids, threads, 0, stream>>>(d_temp, d_gf, d_gv, w, h, l,
                                                              alpha, omega, d, h2);
    }
    if (need_temp)
        dmemFree(d_temp);
}


void cplSORSolver3D(float* d_gv, float* d_gf,
                     float alpha, float gamma, const Vector3Di& size, int nIter, float* d_temp, cudaStream_t stream)
{
    cplSORSolver3D(d_gv, d_gf, alpha, gamma, 
                    size.x, size.y, size.z, nIter, d_temp, stream);
}

void cplSORSolver3D(cplVector3DArray& d_gv, cplVector3DArray& d_gf,
                     float alpha, float gamma, const Vector3Di& size, int nIter, float* d_temp, cudaStream_t stream)
{
    cplSORSolver3D(d_gv.x, d_gf.x, alpha, gamma,
                    size.x, size.y, size.z, nIter, d_temp  , stream);
    cplSORSolver3D(d_gv.y, d_gf.y, alpha, gamma,
                    size.x, size.y, size.z, nIter, d_temp  , stream);
    cplSORSolver3D(d_gv.z, d_gf.z, alpha, gamma, 
                    size.x, size.y, size.z, nIter, d_temp  , stream);
}



////////////////////////////////////////////////////////////////////////////////
//   BLOCK SOR version on the shared mem
//   Step :
//   Add    next_next 
//   Update red_next
//   Update blue_current
//   Write prev - Remove prev
//
//   cur_red = cur_x cur_blue = cur_y when x + y + z = even
//   cur_red = cur_y cur_blue = cur_x when x + y + z = odd
////////////////////////////////////////////////////////////////////////////////

#define BLOCK_DIMX 10
#define BLOCK_DIMY 10
#define RADIUS 2

__device__ float helmholtz(float f, float U_xyz,
                           float U_xyzp, float U_xyzm, // z
                           float U_xypz, float U_xymz, // y
                           float U_xpyz, float U_xmyz, // x
                           float alpha, float omega, float d, float h2)
{
    const float oneOver = 1.f /d;
    float new_value = (U_xpyz + U_xmyz + U_xypz + U_xymz + U_xyzp + U_xyzm) * alpha - f * h2;
    new_value *= oneOver;
    return omega * (new_value - U_xyz) + U_xyz;
}

__device__ float2& At(float2* s, unsigned int j, unsigned int i)
{
    return s[(j + 1) * (BLOCK_DIMX + 2) + i];
}

/*
__global__ void SORHelmholtz3D_shared_kernel(float2* g_U, float2* g_F,
                                             float2* g_Un,
                                             const uint sizeX, const uint sizeY, const uint sizeZ,
                                             float alpha, float omega, float d, float h2) 
{
    const int BLOCK_SIZE = (BLOCK_DIMX + 2) * (BLOCK_DIMY + 2);
    __shared__ float2 s[4 * BLOCK_SIZE];
    float2* s_prev = s;
    float2* s_cur  = s_prev + BLOCK_SIZE;
    float2* s_next = s_cur  + BLOCK_SIZE;
    float2* s_next2= s_next + BLOCK_SIZE;
    
    uint ix  = blockIdx.x * (BLOCK_DIMX - 2) + threadIdx.x;
    uint iy  = blockIdx.y * (BLOCK_DIMY - 2) + threadIdx.y;
    
    uint iid = iy * sizeX  +  ix;
    uint oid = iid;

    uint planeSize = sizeX * sizeY;

    { // filling 0 for the shared mem
        float* s_temp = (float*) s;
        uint      tid = threadIdx.y * BLOCK_DIMX + threadIdx.x;
        while (tid < 4 * 2 * BLOCK_SIZE)
        {
            s_temp[tid] = 0;
            tid += BLOCK_DIMX * BLOCK_DIMY;
        }
    }

    __syncthreads();
    
    if ((ix >= sizeX) || (iy >= sizeY))
        return;
    
    // Initialize value with 0
    float2 next2U, next2F;
    float2 nextU=make_float2(0.f, 0.f), nextF=make_float2(0.f, 0.f);
    float2 curU =make_float2(0.f, 0.f), curF =make_float2(0.f, 0.f);
    float2 prevU=make_float2(0.f, 0.f), prevF=make_float2(0.f, 0.f);

    prevU = g_U[iid]; prevF = g_F[iid];
    At(s_prev,threadIdx.y,threadIdx.x) = prevU;
    if (threadIdx.y == 0)
        At(s_prev,threadIdx.y-1,threadIdx.x) = (iy > 0) ? g_U[iid-sizeX] : make_float2(0.f, 0.f);
    if (threadIdx.y == BLOCK_DIMY-1)
        At(s_prev,threadIdx.y+1,threadIdx.x) = (iy + 1 < sizeY) ? g_U[iid+sizeX] : make_float2(0.f, 0.f);
    iid += planeSize;

    curU = g_U[iid]; curF = g_F[iid];
    At(s_cur,threadIdx.y,threadIdx.x) = curU;
    if (threadIdx.y == 0)
        At(s_cur,threadIdx.y-1,threadIdx.x) = (iy > 0) ? g_U[iid-sizeX] : make_float2(0.f, 0.f);
    if (threadIdx.y == BLOCK_DIMY-1)
        At(s_cur,threadIdx.y+1,threadIdx.x) = (iy + 1 < sizeY) ? g_U[iid+sizeX] : make_float2(0.f, 0.f);
    iid += planeSize;

    nextU = g_U[iid]; nextF = g_F[iid];
    At(s_next,threadIdx.y,threadIdx.x) = nextU;
    if (threadIdx.y == 0)
        At(s_next,threadIdx.y-1,threadIdx.x) = (iy > 0) ? g_U[iid-sizeX] : make_float2(0.f, 0.f);
    if (threadIdx.y == BLOCK_DIMY-1)
        At(s_next,threadIdx.y+1,threadIdx.x) = (iy + 1 < sizeY) ? g_U[iid+sizeX] : make_float2(0.f, 0.f);
    iid += planeSize;
    
    __syncthreads();

    // update the red current
    uint flip = iy & 1;
    if (flip == 1){
        if (threadIdx.x < BLOCK_DIMX - 1)
            At(s_cur,threadIdx.y, threadIdx.x).y = curU.y = helmholtz(curF.y, curU.y,
                                                                      nextU.y, prevU.y,
                                                                      At(s_cur,threadIdx.y+1,threadIdx.x).y, At(s_cur,threadIdx.y-1,threadIdx.x).y, 
                                                                      At(s_cur,threadIdx.y,threadIdx.x+1).x, curU.x,
                                                                      alpha, omega,d, h2);
    }else {
        if (threadIdx.x > 0)
            At(s_cur,threadIdx.y, threadIdx.x).x = curU.x = helmholtz(curF.x, curU.x,
                                                                      nextU.x, prevU.x,
                                                                      At(s_cur,threadIdx.y+1,threadIdx.x).x, At(s_cur,threadIdx.y-1,threadIdx.x).x, 
                                                                      curU.y, At(s_cur,threadIdx.y,threadIdx.x-1).y,
                                                                      alpha, omega,d, h2);
    }

    __syncthreads();
    

    //preprocess

    for (int iz=0; iz < sizeZ-2; ++iz)
    {
        // 1. Add next2
        next2U=make_float2(0.f, 0.f), next2F=make_float2(0.f, 0.f);
        {
            next2U = g_U[iid]; next2F = g_F[iid];
            At(s_next2,threadIdx.y,threadIdx.x) = next2U;
            if (threadIdx.y == 0)
                At(s_next2,threadIdx.y-1,threadIdx.x) = (iy > 0) ? g_U[iid-sizeX] : make_float2(0.f, 0.f);
            if (threadIdx.y == BLOCK_DIMY-1)
                At(s_next2,threadIdx.y+1,threadIdx.x) = (iy + 1 < sizeY) ? g_U[iid+sizeX] : make_float2(0.f, 0.f);
            iid += planeSize;
        }
        if (flip == 0)
        {
            // 2. Update red_next
            if (threadIdx.x < BLOCK_DIMX - 1) {
                At(s_next,threadIdx.y, threadIdx.x).y = nextU.y = helmholtz(nextF.y, nextU.y,
                                                                            next2U.y, curU.y,
                                                                            At(s_next,threadIdx.y+1,threadIdx.x).y, At(s_next,threadIdx.y-1,threadIdx.x).y, 
                                                                            At(s_next,threadIdx.y,threadIdx.x+1).x, nextU.x,
                                                                            alpha, omega,d, h2);
            // 3. Update blue_current
                At(s_cur, threadIdx.y, threadIdx.x).y = curU.y = helmholtz(curF.y, curU.y,
                                                                           nextU.y, prevU.y,
                                                                           At(s_cur,threadIdx.y+1,threadIdx.x).y, At(s_cur,threadIdx.y-1,threadIdx.x).y,
                                                                           At(s_cur,threadIdx.y,threadIdx.x+1).x, curU.x,
                                                                           alpha, omega, d, h2);
            }
        }
        else {
            // 2. Update red_next
            if (threadIdx.x > 0) {
                At(s_next,threadIdx.y,threadIdx.x).x = nextU.x= helmholtz(nextF.x,nextU.x,
                                                                          next2U.x, curU.x,
                                                                          At(s_next,threadIdx.y+1,threadIdx.x).x, At(s_next,threadIdx.y-1,threadIdx.x).x, 
                                                                          nextU.y, At(s_next,threadIdx.y,threadIdx.x-1).y,
                                                                          alpha, omega, d, h2);
            // 3. Update blue_current
                At(s_cur,threadIdx.y,threadIdx.x).x = curU.x = helmholtz(curF.x, curU.x,
                                                                         nextU.x, prevU.x,
                                                                         At(s_cur,threadIdx.y+1,threadIdx.x).x, At(s_cur,threadIdx.y-1,threadIdx.x).x, 
                                                                         curU.y, At(s_cur,threadIdx.y,threadIdx.x-1).y,
                                                                         alpha, omega, d, h2);
            }
        }
        // 4.remove prev
        // 4.a write prev
        //__syncthreads();
        if ((threadIdx.x > 0) && (threadIdx.x < BLOCK_DIMX-1) && (threadIdx.y > 0) && (threadIdx.y < BLOCK_DIMY-1))
            g_Un[oid] = prevU;
        oid += planeSize;
                
        // 4.b circulate values
        prevU = curU  ; prevF= curF;
        curU  = nextU ; curF = nextF;
        nextU = next2U; nextF= next2F;

        // and the pointer
        float2* s_temp = s_prev;
        s_prev = s_cur;
        s_cur  = s_next;
        s_next = s_next2;
        s_next2= s_temp;

        flip = 1 - flip;
        
    }
    
    //3. Update blue_current
    if (flip == 0)
    {
        if (threadIdx.x < BLOCK_DIMX - 1) {
            // 3. Update blue_current
            At(s_cur, threadIdx.y, threadIdx.x).y = curU.y = helmholtz(curF.y, curU.y,
                                                                       nextU.y, prevU.y,
                                                                       At(s_cur,threadIdx.y+1,threadIdx.x).y, At(s_cur,threadIdx.y-1,threadIdx.x).y,
                                                                       At(s_cur,threadIdx.y,threadIdx.x+1).x, curU.x,
                                                                       alpha, omega, d, h2);
        }
    }
    else {
        if (threadIdx.x > 0) {
            // 3. Update blue_current
            At(s_cur,threadIdx.y,threadIdx.x).x = curU.x = helmholtz(curF.x, curU.x,
                                                                     nextU.x, prevU.x,
                                                                     At(s_cur,threadIdx.y+1,threadIdx.x).x, At(s_cur,threadIdx.y-1,threadIdx.x).x, 
                                                                     curU.y, At(s_cur,threadIdx.y,threadIdx.x-1).y,
                                                                     alpha, omega, d, h2);
        }
    }
    
    // post process
    // write the n - 2 layer
    if ((threadIdx.x > 0) && (threadIdx.x < BLOCK_DIMX-1) && (threadIdx.y > 0) && (threadIdx.y < BLOCK_DIMY-1)){
        g_Un[oid] = curU; oid += planeSize;
        g_Un[oid] = nextU; 
    }
}
*/

#define NO_CACHE 0
#if (!NO_CACHE)
texture<float2, 1> com_tex_float2_u;
texture<float2, 1> com_tex_float2_f;
#endif
__global__ void SORHelmholtz3D_shared_kernel(float2* g_U, float2* g_F,
                                             float2* g_Un,
                                             const uint sizeX, const uint sizeY, const uint sizeZ,
                                             float alpha, float omega, float d, float h2) 
{
    const int BLOCK_SIZE = (BLOCK_DIMX + 2) * (BLOCK_DIMY + 2);
    __shared__ float2 s[4 * BLOCK_SIZE];
    float2* s_prev = s;
    float2* s_cur  = s_prev + BLOCK_SIZE;
    float2* s_next = s_cur  + BLOCK_SIZE;
    float2* s_next2= s_next + BLOCK_SIZE;
    
    uint ix  = blockIdx.x * (BLOCK_DIMX - 2) + threadIdx.x;
    uint iy  = blockIdx.y * (BLOCK_DIMY - 2) + threadIdx.y + 1;
    
    uint iid = iy * sizeX  +  ix;
    uint oid = iid;

    uint planeSize = sizeX * sizeY;

    { // filling 0 for the shared mem
        float* s_temp = (float*) s;
        uint      tid = threadIdx.y * BLOCK_DIMX + threadIdx.x;
        while (tid < 4 * 2 * BLOCK_SIZE)
        {
            s_temp[tid] = 0;
            tid += BLOCK_DIMX * BLOCK_DIMY;
        }
    }

    __syncthreads();
    
    if ((ix >= sizeX) || (iy >= sizeY))
        return;

    bool in_bound = ((ix > 0) && (iy > 1 ) && (ix < sizeX - 1) && (iy < sizeY - 2));

    // Initialize value with 0
    float2 prevU=make_float2(0.f, 0.f), prevF=make_float2(0.f, 0.f); iid += planeSize;
    float2 curU =make_float2(0.f, 0.f), curF =make_float2(0.f, 0.f); iid += planeSize;
#if NO_CACHE
    float2 nextU = g_U[iid], nextF = g_F[iid];
    At(s_next,threadIdx.y,threadIdx.x) = nextU;
    if (threadIdx.y == 0)
        At(s_next,threadIdx.y-1,threadIdx.x) = (iy > 0) ? g_U[iid-sizeX] : make_float2(0.f, 0.f);
    if (threadIdx.y == BLOCK_DIMY-1)
        At(s_next,threadIdx.y+1,threadIdx.x) = (iy + 1 < sizeY) ? g_U[iid+sizeX] : make_float2(0.f, 0.f);
#else
    float2 nextU = tex1Dfetch(com_tex_float2_u, iid), nextF = tex1Dfetch(com_tex_float2_f, iid);
    At(s_next,threadIdx.y,threadIdx.x) = nextU;
    if (threadIdx.y == 0)
        At(s_next,threadIdx.y-1,threadIdx.x) = (iy > 0) ? tex1Dfetch(com_tex_float2_u, iid-sizeX) : make_float2(0.f, 0.f);
    if (threadIdx.y == BLOCK_DIMY-1)
        At(s_next,threadIdx.y+1,threadIdx.x) = (iy + 1 < sizeY) ? tex1Dfetch(com_tex_float2_u, iid+sizeX) : make_float2(0.f, 0.f);
#endif
    iid += planeSize;
    __syncthreads();
    
    //preprocess
    float2 next2U, next2F;
    uint flip = iy & 1;
    for (int iz=3; iz < sizeZ; ++iz)
    {
        // 1. Add next2
        next2U=make_float2(0.f, 0.f), next2F=make_float2(0.f, 0.f);
        {
#if NO_CACHE
            next2U = g_U[iid]; next2F = g_F[iid];
            At(s_next2,threadIdx.y,threadIdx.x) = next2U;
            if (threadIdx.y == 0)
                At(s_next2,threadIdx.y-1,threadIdx.x) = (iy > 0) ? g_U[iid-sizeX] : make_float2(0.f, 0.f);
            if (threadIdx.y == BLOCK_DIMY-1)
                At(s_next2,threadIdx.y+1,threadIdx.x) = (iy + 1 < sizeY) ? g_U[iid+sizeX] : make_float2(0.f, 0.f);
#else
            next2U = tex1Dfetch(com_tex_float2_u, iid); next2F = tex1Dfetch(com_tex_float2_f, iid);
            At(s_next2,threadIdx.y,threadIdx.x) = next2U;
            if (threadIdx.y == 0)
                At(s_next2,threadIdx.y-1,threadIdx.x) = (iy > 0) ? tex1Dfetch(com_tex_float2_u, iid-sizeX) : make_float2(0.f, 0.f);
            if (threadIdx.y == BLOCK_DIMY-1)
                At(s_next2,threadIdx.y+1,threadIdx.x) = (iy + 1 < sizeY) ? tex1Dfetch(com_tex_float2_u, iid+sizeX) : make_float2(0.f, 0.f);
#endif
            iid += planeSize;
        }
                        
        if (flip == 0)
        {
            // 2. Update red_next
            if (threadIdx.x < BLOCK_DIMX - 1) {
                bool cond = (in_bound && (iz < sizeZ-1));
                At(s_next,threadIdx.y, threadIdx.x).y = nextU.y = helmholtz(nextF.y, nextU.y,
                                                                            next2U.y, curU.y,
                                                                            At(s_next,threadIdx.y+1,threadIdx.x).y, At(s_next,threadIdx.y-1,threadIdx.x).y, 
                                                                            At(s_next,threadIdx.y,threadIdx.x+1).x, nextU.x,
                                                                            alpha, omega,d, h2) * (float) cond;
                
                // 3. Update blue_current
                cond = (in_bound && (iz > 3));
                At(s_cur, threadIdx.y, threadIdx.x).y = curU.y = helmholtz(curF.y, curU.y,
                                                                           nextU.y, prevU.y,
                                                                           At(s_cur,threadIdx.y+1,threadIdx.x).y, At(s_cur,threadIdx.y-1,threadIdx.x).y,
                                                                           At(s_cur,threadIdx.y,threadIdx.x+1).x, curU.x,
                                                                           alpha, omega, d, h2) * float(cond);
            }
        }
        else {
            // 2. Update red_next
            if (threadIdx.x > 0) {
                bool cond = (in_bound && (iz < sizeZ-1));
                At(s_next,threadIdx.y,threadIdx.x).x = nextU.x= helmholtz(nextF.x,nextU.x,
                                                                          next2U.x, curU.x,
                                                                          At(s_next,threadIdx.y+1,threadIdx.x).x, At(s_next,threadIdx.y-1,threadIdx.x).x, 
                                                                          nextU.y, At(s_next,threadIdx.y,threadIdx.x-1).y,
                                                                          alpha, omega, d, h2) * float(cond); ;
            // 3. Update blue_current
                cond = (in_bound && (iz > 3));
                At(s_cur,threadIdx.y,threadIdx.x).x = curU.x = helmholtz(curF.x, curU.x,
                                                                         nextU.x, prevU.x,
                                                                         At(s_cur,threadIdx.y+1,threadIdx.x).x, At(s_cur,threadIdx.y-1,threadIdx.x).x, 
                                                                         curU.y, At(s_cur,threadIdx.y,threadIdx.x-1).y,
                                                                         alpha, omega, d, h2) * float(cond);;
            }
        }

        __syncthreads();
        
        // 4.remove prev
        // 4.a write prev
        if (iz >= 5)
            if ((threadIdx.x > 0) && (threadIdx.x < BLOCK_DIMX-1) && (threadIdx.y > 0) && (threadIdx.y < BLOCK_DIMY-1)
                && (in_bound))
                //&& (iy > 1 ) && (ix < sizeX - 1) && (iy < sizeY - 2))
                g_Un[oid] = prevU;
        oid += planeSize;
                
        // 4.b circulate values
        prevU = curU  ; prevF= curF;
        curU  = nextU ; curF = nextF;
        nextU = next2U; nextF= next2F;

        // and the pointer
        float2* s_temp = s_prev;
        s_prev = s_cur;
        s_cur  = s_next;
        s_next = s_next2;
        s_next2= s_temp;

        flip = 1 - flip;

    }
    
    // post process
    // write the n - 2 layer
    if ((threadIdx.x > 0) && (threadIdx.x < BLOCK_DIMX-1) && (threadIdx.y > 0) && (threadIdx.y < BLOCK_DIMY-1)
        && (in_bound))
        //&& (iy > 1) && (ix < sizeX - 1) && (iy < sizeY - 2))
        g_Un[oid] = prevU;
}

void SORHelmholtz3D_shared(float* g_U, float* g_F,
                           float* g_Un,
                           const uint sizeX, const uint sizeY, const uint sizeZ,
                           float alpha, float omega, float d, float h2,  bool initMem, cudaStream_t stream) 
{
    float2* g_U2  = (float2*) g_U;
    float2* g_F2  = (float2*) g_F;
    float2* g_Un2 = (float2*) g_Un;

    if (initMem)
        cplVectorOpers::SetMem(g_Un, 0.f , sizeX * sizeY * sizeZ, stream);
    int sizeX2 = sizeX / 2;
    dim3 threads(BLOCK_DIMX, BLOCK_DIMY);
    dim3 grids(iDivUp(sizeX2 - 2, BLOCK_DIMX - 2), iDivUp(sizeY - 4, BLOCK_DIMY - 2));

#if (!NO_CACHE)
    cudaBindTexture(NULL, com_tex_float2_u, g_U2);
    cudaBindTexture(NULL, com_tex_float2_f, g_F2);
#endif    
    SORHelmholtz3D_shared_kernel<<<grids, threads, 0, stream>>>(g_U2, g_F2, g_Un2, sizeX2, sizeY, sizeZ, alpha, omega, d, h2);
}

void SORHelmholtz3D_shared(float* g_U, float* g_F, float* g_Un,
                           const uint sizeX, const uint sizeY, const uint sizeZ,
                           float alpha, float gamma, int nIters, cudaStream_t stream) 
{
    const float h2    = 1.0;
    float pj    = 1.f/ 3 * (cos(M_PI / (sizeX - 1)) + cos (M_PI / (sizeY - 1)) + cos (M_PI / (sizeZ - 1)));
    float omega = 2 / ( 1 + sqrt(1 - pj * pj));
    float d     = 6 * alpha - gamma * h2;

    cplVectorOpers::SetMem(g_Un, 0.f , sizeX * sizeY * sizeZ, stream);
    for (int i=0; i< nIters; ++i){
        SORHelmholtz3D_shared(g_U ,g_F,g_Un, sizeX, sizeY, sizeZ, alpha, omega, d, h2, 0, stream);
        SORHelmholtz3D_shared(g_Un,g_F,g_U , sizeX, sizeY, sizeZ, alpha, omega, d, h2, 0, stream);
    }
}

void cplSORSolver3D_shared(float* d_gv, float* d_gf,
                            float alpha, float gamma,
                            const Vector3Di& size, int nIters, float* d_temp, cudaStream_t stream)
{
    SORHelmholtz3D_shared(d_gv, d_gf, d_temp,
                          size.x, size.y, size.z, alpha, gamma, nIters / 2, stream);
}


bool testBound(int x, int y, int z,
               int sizeX, int sizeY, int sizeZ,
               int b)
{
    return ((x < b) || (y < b) || (z < b) || (x >= sizeX - b) || (y >= sizeY - b) || (z >= sizeZ - b));
}

void printHostVolume3D(float* h_i, int sizeX, int sizeY, int sizeZ, const char* name)
{
    fprintf(stdout, "\n Volume %s \n",name);
    for (int iz=0; iz < sizeZ; ++iz){
        fprintf(stdout, "Slice %d \n\n",iz);
        for (int iy=0; iy < sizeY; ++iy){
            for (int ix=0; ix < sizeX; ++ix)
                fprintf(stdout, "%5.2f ",h_i[ ix + iy * sizeX + iz * sizeX * sizeY]);
            fprintf(stdout, "\n");
        }
    }
}

void printDeviceVolume3D(float* d_i, int sizeX, int sizeY, int sizeZ, const char* name)
{
    int nElems = sizeX * sizeY * sizeZ;
    float* h_i = new float [nElems];
    copyArrayFromDevice(h_i, d_i, nElems);
    printHostVolume3D(h_i, sizeX, sizeY, sizeZ, name);
    delete []h_i;
}

#define BOUND 2
void setIdentity(float* h, int sizeX, int sizeY, int sizeZ)
{
    int id = 0;
    for (int iz = 0; iz < sizeZ; ++iz)
        for (int iy = 0; iy < sizeY; ++iy)
            for (int ix= 0; ix < sizeX; ++ix, ++id)
                if (!testBound(ix, iy, iz, sizeZ, sizeY, sizeZ, BOUND))
                    h[id] = id;
                else
                    h[id] = 0;
}

void setRandom(float* h, int sizeX, int sizeY, int sizeZ)
{
    int id = 0;
    for (int iz = 0; iz < sizeZ; ++iz)
        for (int iy = 0; iy < sizeY; ++iy)
            for (int ix= 0; ix < sizeX; ++ix, ++id)
                if (testBound(ix, iy, iz, sizeX, sizeY, sizeZ, BOUND))
                    h[id] = 0;
                else
                    h[id] = rand() % 0xFF;
}

void testSOR(int sizeX, int sizeY, int sizeZ) 
{
    fprintf(stderr, "Problem size %d %d %d \n", sizeX, sizeY, sizeZ);
    int nElems = sizeX * sizeY * sizeZ;
    float* h_f = new float [nElems];
    setRandom(h_f, sizeX, sizeY, sizeZ);
    
    float* d_f;
    dmemAlloc(d_f, nElems);
    copyArrayToDevice(d_f, h_f, nElems);
    
    float* d_U, *d_Un, *d_U1;

    dmemAlloc(d_U, nElems);
    dmemAlloc(d_U1, nElems);
    dmemAlloc(d_Un, nElems);
    
    setRandom(h_f, sizeX, sizeY, sizeZ);

    copyArrayToDevice(d_U, h_f, nElems);
    copyArrayDeviceToDevice(d_U1, d_U, nElems);
    
    const float h2    = 1.0;
    const float alpha = 0.01;
    const float gamma = 0.001;
    
//    float pj    = 1.f/ 3 * (cos(M_PI / (sizeX - 1)) + cos (M_PI / (sizeY - 1)) + cos (M_PI / (sizeZ - 1)));
//    float omega = 2 / ( 1 + sqrt(1 - pj * pj));
    float omega = 1.5;
    float d     = 6 * alpha - gamma * h2;

    dim3 threads(16,16);
    dim3 grids(iDivUp(sizeX, threads.x),iDivUp(sizeY, threads.y));

    cudaEventTimer timer;
    timer.start();

    //printDeviceVolume3D(d_U, sizeX, sizeY, sizeZ, "Input");
    cudaMemcpy(d_Un, d_U, nElems * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaBindTexture(0, com_tex_float, d_U, nElems * sizeof(float));
    SORIHelmholtz3D_tex<0><<<grids, threads>>>(d_U, d_f, d_Un,
                                               sizeX, sizeY, sizeZ, 
                                               alpha, omega, d, h2);

    cudaMemcpy(d_U, d_Un, nElems * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaBindTexture(0, com_tex_float, d_Un, nElems * sizeof(float));
    SORIHelmholtz3D_tex<1><<<grids, threads>>>(d_Un, d_f, d_U,
                                               sizeX, sizeY, sizeZ, 
                                               alpha, omega, d, h2);
    cudaThreadSynchronize();
    timer.stop();
    timer.printTime("Tex1D fetch");
    cplVectorOpers::SetMem(d_Un, 0.f, nElems);
    timer.reset();
    SORHelmholtz3D_shared(d_U1, d_f, d_Un,
                          sizeX, sizeY, sizeZ, 
                          alpha, omega, d, h2, 0,
                          NULL);
    cudaThreadSynchronize();
    timer.stop();
    timer.printTime("Shared mem");

    // printDeviceVolume3D(d_U, sizeX, sizeY, sizeZ, "Ground Truth");
//     printDeviceVolume3D(d_Un, sizeX, sizeY, sizeZ, "Output");
    
    cplVectorOpers::Sub_I(d_Un, d_U, sizeX * sizeY * sizeZ);
    cplReduce rd;
    rd.init();

    if (rd.MaxAbs(d_Un, sizeX * sizeY * sizeZ) < 1.e-5)
        fprintf(stderr, "Test PASSES");
    else
        fprintf(stderr, "Test FAILED");

    dmemFree(d_U);
    dmemFree(d_U1);
    dmemFree(d_Un);
}
