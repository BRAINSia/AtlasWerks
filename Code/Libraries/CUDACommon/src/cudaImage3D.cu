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
#include <cpl.h>
#include <cudaImage3D.h>
#include <cpuImage3D.h>
#include <AtlasWerksException.h>

texture<float , 3, cudaReadModeElementType> com_tex3_float;
////////////////////////////////////////////////////////////////////////////////
// Compute the gradient of an image (a simple version with 0 boundary condition)
//   Input : the input imge float
//   Ouput : 3-channel gradient
////////////////////////////////////////////////////////////////////////////////
__global__ void cplGradient3D_32f_kernel(const float* src,
                                          float* d_gdx, float* d_gdy, float* d_gdz,  
                                          uint sizeX, uint sizeY, uint sizeZ)
{
    uint i = threadIdx.x + blockIdx.x * blockDim.x;
    uint j = threadIdx.y + blockIdx.y * blockDim.y;
    const uint planeSize = sizeX * sizeY;
    
    uint index = i + j * sizeX;
    float x0yz, x1yz, xy0z, xy1z, xyz0, xyz1;

    if (i<sizeX && j < sizeY)
        for (uint k=0 ; k < sizeZ; ++k, index += planeSize){
            xyz1 = ( k == sizeZ - 1) ? 0 : src[index + planeSize];
            xy1z = ( j == sizeY - 1) ? 0 : src[index + sizeX];
            x1yz = ( i == sizeX - 1) ? 0 : src[index + 1];

            xyz0 = ( k == 0) ? 0 : src[index - planeSize];
            xy0z = ( j == 0) ? 0 : src[index - sizeX];
            x0yz = ( i == 0) ? 0 : src[index - 1];

            d_gdx[index] = 0.5f * (x1yz - x0yz);
            d_gdy[index] = 0.5f * (xy1z - xy0z);
            d_gdz[index] = 0.5f * (xyz1 - xyz0);
        }
}

void cplGradient3D_32f(float * d_gdx, float * d_gdy, float * d_gdz, const float* src,
                        uint sizeX, uint sizeY, uint sizeZ, cudaStream_t stream)
{
    dim3 threads(16,16);
    dim3 grids(iDivUp(sizeX, threads.x), iDivUp(sizeY, threads.y));
    cplGradient3D_32f_kernel<<<grids, threads, 0, stream>>>(src, d_gdx, d_gdy, d_gdz, 
                                                             sizeX, sizeY, sizeZ);
}

////////////////////////////////////////////////////////////////////////////////
// Compute the gradient of an image (a simple version with 0 boundary condition)
//   Input : the input imge float
//   Ouput : 3-channel gradient
////////////////////////////////////////////////////////////////////////////////

__global__ void cplGradient3D_32f_tex_linear_kernel(float * d_gdx, float * d_gdy, float * d_gdz,  
                                                     uint sizeX, uint sizeY, uint sizeZ)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    const int planeSize = sizeX * sizeY;
    
    int index = i + j * sizeX;
    float x0yz, x1yz, xy0z, xy1z, xyz0, xyz1;

    if (i<sizeX && j < sizeY)
        for (int k=0 ; k < sizeZ; ++k, index += planeSize){
            // Slice 1
            xyz1 = ( k == sizeZ - 1) ? 0 : tex1Dfetch(com_tex_float,index + planeSize);
            xy1z = ( j == sizeY - 1) ? 0 : tex1Dfetch(com_tex_float,index + sizeX);
            x1yz = ( i == sizeX  - 1) ? 0 : tex1Dfetch(com_tex_float,index + 1);

            xyz0 = ( k == 0) ? 0 : tex1Dfetch(com_tex_float,index - planeSize);
            xy0z = ( j == 0) ? 0 : tex1Dfetch(com_tex_float,index - sizeX);
            x0yz = ( i == 0) ? 0 : tex1Dfetch(com_tex_float,index - 1);

            d_gdx[index] = 0.5f * (x1yz - x0yz);
            d_gdy[index] = 0.5f * (xy1z - xy0z);
            d_gdz[index] = 0.5f * (xyz1 - xyz0);
        }
}

void cplGradient3D_32f_linear(float * d_gdx, float * d_gdy, float * d_gdz,
                               const float* src,
                               uint sizeX, uint sizeY, uint sizeZ, cudaStream_t stream)
{
    dim3 threads(16,16);
    dim3 grids(iDivUp(sizeX, threads.x), iDivUp(sizeY, threads.y));
    
    cudaBindTexture(0, com_tex_float, src, sizeX * sizeY * sizeZ * sizeof(float));
    cplGradient3D_32f_tex_linear_kernel<<<grids, threads, 0, stream>>>(d_gdx, d_gdy,d_gdz, 
                                                                        sizeX, sizeY, sizeZ);
}

////////////////////////////////////////////////////////////////////////////////
// Compute the gradient of an image (a simple version with 0 boundary condition)
//   Input : the input imge float
//   Ouput : 3-channel gradient
////////////////////////////////////////////////////////////////////////////////
__global__ void cplGradient3D_32f_tex_kernel(float * d_gdx, float * d_gdy, float * d_gdz,
                                              int sizeX, int sizeY, int sizeZ)
{
    int i        = threadIdx.x + blockIdx.x * blockDim.x;
    int j        = threadIdx.y + blockIdx.y * blockDim.y;
    int index    = i + j * sizeX;
    int planeSize= sizeX * sizeY;
    
    float xyz, x0yz, x1yz, xy0z, xy1z, xyz0, xyz1;
    float x, y;

    if ( i < sizeX && j < sizeY){
        x = i;
        y = j;

        xyz  = 0;
        xyz1 = tex3D(com_tex3_float, x, y, 0);

        for (int k=0 ; k < sizeZ; ++k, index +=planeSize){

            x1yz = tex3D(com_tex3_float, x + 1.f, y, k);
            x0yz = tex3D(com_tex3_float, x - 1.f, y, k);
            
            xy1z = tex3D(com_tex3_float, x, y + 1.f, k);
            xy0z = tex3D(com_tex3_float, x, y - 1.f, k);

            xyz0 = xyz;
            xyz  = xyz1;
            xyz1 = tex3D(com_tex3_float, x, y, k + 1);
            
            d_gdx[index] = 0.5f * (x1yz - x0yz);
            d_gdy[index] = 0.5f * (xy1z - xy0z);
            d_gdz[index] = 0.5f * (xyz1 - xyz0);
        }
    }
}

void cplGradient3D_32f_tex(float * d_gdx, float * d_gdy, float * d_gdz,
                            const float* src,
                            uint w, uint h, uint l, cudaStream_t stream)
{
    //create 3D array 
    cudaArray * d_volumeArray = 0;
    cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
    cudaExtent volumeSize      = make_cudaExtent(w, h, l);
    cudaMalloc3DArray(&d_volumeArray, &desc, volumeSize);

    // copy data to 3D array
    cudaMemcpy3DParms copyParams = {0};
    copyParams.srcPtr   = make_cudaPitchedPtr((void*)src, w *sizeof(float), w, h);
    copyParams.dstArray = d_volumeArray;
    copyParams.extent   = volumeSize;
    copyParams.kind     = cudaMemcpyDeviceToDevice;

    cudaMemcpy3D(&copyParams);

    com_tex3_float.addressMode[0] = cudaAddressModeClamp;
    com_tex3_float.addressMode[1] = cudaAddressModeClamp;
    com_tex3_float.addressMode[2] = cudaAddressModeClamp;
    
    com_tex3_float.filterMode     = cudaFilterModePoint;
    com_tex3_float.normalized     = false;
    cudaBindTextureToArray(com_tex3_float, d_volumeArray, desc);

    dim3 threads(16,16);
    dim3 grids(iDivUp(w, threads.x), iDivUp(h, threads.y));
    cplGradient3D_32f_tex_kernel<<<grids, threads, 0, stream>>>(d_gdx, d_gdy,d_gdz, 
                                                                 w, h, l);
    cudaFreeArray(d_volumeArray);
}

/*-----------------------------------------------------------------------
  Gradient function of the grids (this is the old function)
  with the assumption of we assume the dx, dy = 1
  Inputs : d_gu scalar fields of the grids
  w, h : size of the grid (the domain will go from [0:w-1,0:h-1]
  Ouput  : d_ggx gradient on x direction 
  d_ggy gradient on y direction
  d_ggz gradient on y direction
  d_gg  gradient on both x,y,z
           
  -----------------------------------------------------------------------*/
void gradient3D(float* d_ggx, float* d_ggy, float* d_ggz, float* d_gu,
                int w, int h, int l, cudaStream_t stream)
{
    cplGradient3D_32f_linear(d_ggx, d_ggy, d_ggz, d_gu, w, h, l, stream);
}

void gradient3D(cplVector3DArray& d_g, float* d_gu,
                int w, int h, int l, cudaStream_t stream)
{
    cplGradient3D_32f_linear(d_g.x, d_g.y, d_g.z, d_gu, w, h, l, stream);
}

////////////////////////////////////////////////////////////////////////////////
// Function with Vector3D_XY_Z_Array format (xy, z)
////////////////////////////////////////////////////////////////////////////////
__global__ void cplGradient3D_32f_tex_linear_kernel(float2* d_gdxy, float * d_gdz,  
                                                     uint sizeX, uint sizeY, uint sizeZ)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    const int planeSize = sizeX * sizeY;
    
    int index = i + j * sizeX;
    float x0yz, x1yz, xy0z, xy1z, xyz0, xyz1;

    if (i<sizeX && j < sizeY)
        for (int k=0 ; k < sizeZ; ++k, index += planeSize){
            // Slice 1
            xyz1 = ( k == sizeZ - 1) ? 0 : tex1Dfetch(com_tex_float,index + planeSize);
            xy1z = ( j == sizeY - 1) ? 0 : tex1Dfetch(com_tex_float,index + sizeX);
            x1yz = ( i == sizeX  - 1) ? 0 : tex1Dfetch(com_tex_float,index + 1);

            xyz0 = ( k == 0) ? 0 : tex1Dfetch(com_tex_float,index - planeSize);
            xy0z = ( j == 0) ? 0 : tex1Dfetch(com_tex_float,index - sizeX);
            x0yz = ( i == 0) ? 0 : tex1Dfetch(com_tex_float,index - 1);

            float2 xyDev;
            xyDev.x = 0.5f * (x1yz - x0yz);
            xyDev.y = 0.5f * (xy1z - xy0z);
            d_gdxy[index] = xyDev;
            d_gdz [index] = 0.5f * (xyz1 - xyz0);
        }
}

void gradient3D(Vector3D_XY_Z_Array& d_gg, float* d_gu, int sizeX, int sizeY, int sizeZ, cudaStream_t stream)
{
    dim3 threads(16,16);
    dim3 grids(iDivUp(sizeX, threads.x), iDivUp(sizeY, threads.y));
    cudaBindTexture(0, com_tex_float, d_gu, sizeX * sizeY * sizeZ * sizeof(float));
    cplGradient3D_32f_tex_linear_kernel<<<grids, threads, 0, stream>>>(d_gg.xy,d_gg.z, sizeX, sizeY, sizeZ);
}

/*----------------------------------------------------------------------
  Trilinear interpolation from the grids to the points
  Inputs : d_gu grid scalar fields
  w, h,l size of the grid (the domain will go from [0:w-1,0:h-1, 0:l-1]
  d_px position of the particles
  nP   number of particles
  Output : d_pu the scalar value on the grid           
  ---------------------------------------------------------------------*/
__global__ void cpliTrilerp_kernel(float* pv, float2* px_xy, float* px_z, int n, 
                                   float* gu, int w, int h, int l){ 
    uint blockId= blockIdx.x  + blockIdx.y * gridDim.x;
    int id      = threadIdx.x + blockDim.x * blockId;
    const int wh= w * h;
    
    if (id < n){
        volatile float2 xy = px_xy[id];
        float  z  = px_z[id];
        float  x  = xy.x;
        float  y  = xy.y;
                
        if ( x >= 0 && x < w - 1 &&
             y >= 0 && y < h - 1 &&
             z >= 0 && z < l - 1 )
        {
            pv[id] = simple_triLerpTex_kernel(gu, x,y,z, w, wh);
        }
        else
            pv[id] = 0;
    }
}


void triLerp(float* d_pu, Vector3D_XY_Z_Array& d_px, int nP,
             float* d_gu, int w, int h, int l, cudaStream_t stream){

    dim3 threads(256);
    dim3 grids(iDivUp(nP, threads.x));
    checkConfig(grids);

    cudaBindTexture(0, com_tex_float, d_gu, w * h * l * sizeof(float));
    cpliTrilerp_kernel<<<grids, threads, 0, stream>>>(d_pu, d_px.xy, d_px.z, nP,
                                                      d_gu, w, h, l);
}

/*----------------------------------------------------------------------
  Trilinear interpolation from the grids to the points
  Inputs : d_gu grid scalar fields
  w, h, l size of the grid (the domain will go from [0:w-1,0:h-1, 0:l-1]
  d_px position of the particles
  nP   number of particles
  Output : d_pu the scalar value on the grid           
  ---------------------------------------------------------------------*/
__global__ void cpliTrilerp_kernel(float* pu, float* p_x, float* p_y, float* p_z, int n,
                                   float* gu, int w, int h, int l)
{ 
    uint blockId= blockIdx.x  + blockIdx.y * gridDim.x;
    int id      = threadIdx.x + blockDim.x * blockId;

    const int wh= w * h;
    if (id < n){
        float x = p_x[id], y= p_y[id], z = p_z[id];

        if ( x >= 0 && x < w - 1 &&
             y >= 0 && y < h - 1 &&
             z >= 0 && z < l - 1 )
        {
            pu[id] = simple_triLerpTex_kernel(gu, x,y,z, w, wh);
        }
        else
            pu[id] = 0;
    }
}


void triLerp(float* d_pu, cplVector3DArray& d_px, int nP,
             float* d_gu, int w, int h, int l, cudaStream_t stream)
{
    dim3 threads(256);
    dim3 grids(iDivUp(nP, threads.x));
    
    cudaBindTexture(0, com_tex_float, d_gu, w * h * l * sizeof(float));
    cpliTrilerp_kernel<<<grids, threads, 0, stream>>>(d_pu, d_px.x, d_px.y, d_px.z, nP,
                                                      d_gu,  w, h, l);
}

////////////////////////////////////////////////////////////////////////////////
// Compute the gradient of an image full version with boundary condition
//   warp = 0 : no boundary use forward or backward different 
//        > 0 : cyclic boundary
//
//   Input : d_i        : the input image float 
//           w, h, l    : size of the image
//           sx, sy, sz : scalars (axes's unit values)
//   Ouput : d_gdx, d_gdy, d_gdz : 3-channel gradient
//           
////////////////////////////////////////////////////////////////////////////////

template<int wrap>
__global__ void cudaComputeGradient_kernel(float * d_gdx, float * d_gdy, float * d_gdz,
                                           const float * d_i,
                                           int w, int h, int l,
                                           float sx, float sy, float sz)
                                           
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    
    int wh = w * h;
    int id = j * w + i;
    
    if (i<w && j < h)
        for (int k=0 ; k < l; ++k, id += wh){

            if (i == 0)
                d_gdx[id] = (wrap) ? (d_i[id+1] - d_i[id + w - 1]) / (2.f * sx) : (d_i[id+1] - d_i[id]) / sx;
            else if (i == w - 1) 
                d_gdx[id] = (wrap) ? (d_i[id-(w-1)] - d_i[id - 1]) / (2.f * sx) : (d_i[id] - d_i[id-1]) / sx;
            else
                d_gdx[id] = (d_i[id+1] - d_i[id-1]) / (2.f * sx);

            if (j == 0)
                d_gdy[id] = (wrap) ? (d_i[id+w] - d_i[id + (h - 1)* w]) / (2.f * sy) : (d_i[id+w] - d_i[id]) / sy;
            else if (j == h - 1) 
                d_gdy[id] = (wrap) ? (d_i[id-(h-1)*w] - d_i[id - w]) / (2.f * sy) : (d_i[id] - d_i[id-w]) / sy;
            else
                d_gdy[id] = (d_i[id+w] - d_i[id-w]) / (2.f * sy);

            if (k == 0)
                d_gdz[id] = (wrap) ? (d_i[id+wh] - d_i[id + (l - 1)* wh]) / (2.f * sz) : (d_i[id+wh] - d_i[id]) / sz;
            else if (k == l - 1) 
                d_gdz[id] = (wrap) ? (d_i[id-(l-1)*wh] - d_i[id - wh]) / (2.f * sz) : (d_i[id] - d_i[id-wh]) / sz;
            else
                d_gdz[id] = (d_i[id+wh] - d_i[id-wh]) / (2.f * sz);
        }
}

template<int wrap>
__global__ void cudaComputeGradient_shared_kernel(float * d_gdx, float * d_gdy, float * d_gdz,
                                                  const float * d_i,
                                                  int w, int h, int l,
                                                  float sx, float sy, float sz)
                                           
{
    const int radius = 1;

    __shared__ float s_data[16 + radius * 2][16 + radius * 2];

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    
    if (i < w && j < h){
        int planeSize = w* h;
        int id        = j * w + i;

        float back    = 0;
        float current = 0;
        float front   = d_i[id];

        for (int k=0; k< l; ++k){
            back    = current;
            current = front;
            front   = getTexVal(id + planeSize);
            //front   = d_i[id + planeSize];
            
            __syncthreads();

            int tx = threadIdx.x + radius;
            int ty = threadIdx.y + radius;

            if (threadIdx.x < radius){
                s_data[ty][threadIdx.x]               = d_i[id - radius];
                s_data[ty][threadIdx.x + 16 + radius] = d_i[id + 16];
            }

            if (threadIdx.y < radius){
                s_data[threadIdx.y][tx]               = d_i[id - radius * w];
                s_data[threadIdx.y + 16 + radius][tx] = d_i[id + 16 * w];
            }

            s_data[ty][tx] = current;
            
            __syncthreads();

            if (i == 0)
                d_gdx[id] = (s_data[ty][tx+1] -s_data[ty][tx]) / sx;
            else if (i == w - 1)
                d_gdx[id] = (s_data[ty][tx] -s_data[ty][tx-1]) / sx;
            else 
                d_gdx[id] = (s_data[ty][tx+1] -s_data[ty][tx-1])/ (2.f * sx);

            if (j == 0)
                d_gdy[id] = (s_data[ty+1][tx] -s_data[ty][tx]) /sy;
            else if (j == h - 1)
                d_gdy[id] = (s_data[ty][tx] -s_data[ty-1][tx]) /sy;
            else
                d_gdy[id] = (s_data[ty+1][tx] -s_data[ty-1][tx])/(2.f * sy);

            if (k==0)
                d_gdz[id] = (front - current)/sz;
            else if (k == l - 1)
                d_gdz[id] = (current - back) /sz;
            else
                d_gdz[id] = (front - back) /(2.f * sz);
            
            id += planeSize;
        }
    }
}

void cudaComputeGradient_shared(float * d_gdx, float * d_gdy, float * d_gdz, // out put gradient
                                const float * d_i,                                 // input image
                                uint w, uint h, uint l,                      // size of image  
                                float sx, float sy, float sz,   // spacing 
                                bool warp, cudaStream_t stream)                             // warp or not
{
    dim3 threads(16,16);
    dim3 grids(iDivUp(w, threads.x), iDivUp(h, threads.y));
    cudaBindTexture(0, com_tex_float, d_i, w * h * l * sizeof(float));
    int share = 18 * 18 * sizeof(float);

    if (warp == false)
        cudaComputeGradient_shared_kernel<0><<<grids, threads, share, stream>>>(d_gdx, d_gdy, d_gdz,
                                                                                d_i,
                                                                                w, h, l,
                                                                                sx, sy, sz);
    else
        cudaComputeGradient_shared_kernel<1><<<grids, threads, share, stream>>>(d_gdx, d_gdy, d_gdz,
                                                                                d_i,
                                                                                w, h, l,
                                                                                sx, sy, sz);
}

/*
  template<int wrap>
  __global__ void cudaComputeGradient_shared32x32_kernel(float * d_gdx, float * d_gdy, float * d_gdz,
  float * d_i,
  int w, int h, int l,
  float sx, float sy, float sz)
                                           
  {
  const int radius  = 1;
    
  const int BLOCK_X = 16;
  const int BLOCK_Y = 8;
    
  __shared__ float s_data[2*BLOCK_Y + radius * 2][BLOCK_X + radius * 2];

  int i = threadIdx.x + blockIdx.x * BLOCK_X;
  int j = threadIdx.y + blockIdx.y * BLOCK_Y *2;

    
  if (i < w && j < h){
  int planeSize = w* h;
  int id        = j * w + i;

  float back    = 0;
  float current = 0;
  float front   = d_i[id];

  float backH    = 0;
  float currentH = 0;
  float frontH   = d_i[id+BLOCK_Y*w];

  int inside = (j + BLOCK_Y < h);
        
  __syncthreads();
        
        
                
  for (int k=0; k< l; ++k){
  back    = current;
  current = front;
  front   = getTexVal(id + planeSize);

  backH   = currentH;
  currentH= frontH;
  frontH  = getTexVal(id + planeSize + BLOCK_Y *w);

  __syncthreads();
            
  int tx = threadIdx.x + radius;
  int ty = threadIdx.y + radius;

  if (threadIdx.x < radius){
  s_data[ty][threadIdx.x]               = d_i[id - radius];
  s_data[ty][threadIdx.x + BLOCK_X + radius] = d_i[id + BLOCK_X];
  }

  if (threadIdx.y < radius){
  s_data[threadIdx.y][tx]               = d_i[id - radius * w];
  s_data[threadIdx.y + 2*BLOCK_Y + radius][tx] = d_i[id + 2*BLOCK_Y * w];
  }

  s_data[ty][tx] = current;
  s_data[ty+BLOCK_Y][tx] = currentH;
            
  __syncthreads();

  if (i == 0){
  d_gdx[id] = (s_data[ty][tx+1] -s_data[ty][tx]) / sx;
  if (inside)
  d_gdx[id+BLOCK_Y*w] = (s_data[ty+BLOCK_Y][tx+1] -s_data[ty+BLOCK_Y][tx]) / sx;
  }
  else if (i == w - 1){
  d_gdx[id] = (s_data[ty][tx] -s_data[ty][tx-1]) / sx;
  if (inside)
  d_gdx[id+BLOCK_Y*w] = (s_data[ty+BLOCK_Y][tx] -s_data[ty+BLOCK_Y][tx-1]) / sx;
  }
  else {
  d_gdx[id] = (s_data[ty][tx+1] -s_data[ty][tx-1])/ (2 * sx);
  if (inside)
  d_gdx[id+BLOCK_Y*w] = (s_data[ty+BLOCK_Y][tx+1] -s_data[ty+BLOCK_Y][tx-1])/ (2 * sx);
  }
  /            
  if (j == 0)
  d_gdy[id] = (s_data[ty+1][tx] -s_data[ty][tx]) /sy;
  else if (j == h - 1)
  d_gdy[id] = (s_data[ty][tx] -s_data[ty-1][tx]) /sy;
  else
  d_gdy[id] = (s_data[ty+1][tx] -s_data[ty-1][tx])/(2 * sy);

  if (inside){
  if (j + BLOCK_Y == h-1 )
  d_gdy[id+BLOCK_Y*w] = (s_data[ty+BLOCK_Y][tx] -s_data[ty-1+BLOCK_Y][tx]) /sy;
  else 
  d_gdy[id+BLOCK_Y*w] = (s_data[ty+1+BLOCK_Y][tx] -s_data[ty-1+BLOCK_Y][tx])/(2 * sy);
  }

  if (k==0){
  d_gdz[id] = (front - current)/sz;
  if (inside)
  d_gdz[id+BLOCK_Y*w] = (frontH - currentH)/sz;
  }
  else if (k == l - 1){
  d_gdz[id] = (current - back) /sz;
  if (inside)
  d_gdz[id+BLOCK_Y*w] = (currentH - backH) /sz;
  }
  else {
  d_gdz[id] = (front - back) /(2 * sz);
  if (inside)
  d_gdz[id+BLOCK_Y*w] = (frontH - backH) /(2 * sz);
  }
  id += planeSize;
  }
  }

  }

  void cudaComputeGradient_shared32x32(float * d_gdx, float * d_gdy, float * d_gdz, // out put gradient
  float * d_i,                                 // input image
  uint w, uint h, uint l,                      // size of image  
  float sx, float sy, float sz,   // spacing 
  bool warp)                             // warp or not
  {
  dim3 threads(16,8);
  dim3 grids(iDivUp(w,threads.x), iDivUp(h,2*threads.y));
  cudaBindTexture(0, com_tex_float, d_i, w * h * l * sizeof(float));

  if (warp == false)
  cudaComputeGradient_shared32x32_kernel<0><<<grids, threads, 0, stream>>>(d_gdx, d_gdy, d_gdz,
  d_i,
  w, h, l,
  sx, sy, sz);
  else
  cudaComputeGradient_shared32x32_kernel<1><<<grids, threads, 0, stream>>>(d_gdx, d_gdy, d_gdz,
  d_i,
  w, h, l,
  sx, sy, sz);
  }
*/


template<int wrap>
__global__ void cudaComputeGradient_kernel(float * d_gdx, float * d_gdy, float * d_gdz,
                                           int w, int h, int l,
                                           float sx, float sy, float sz)
                                           
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    
    int wh = w * h;
    int id = j * w + i;
    
    if (i<w && j < h)
        for (int k=0 ; k < l; ++k, id += wh){

            if (i == 0)
                d_gdx[id] = (wrap) ? (getTexVal(id+1) - getTexVal(id + w - 1)) / (2.f * sx) : (getTexVal(id+1) - getTexVal(id)) / sx;
            else if (i == w - 1) 
                d_gdx[id] = (wrap) ? (getTexVal(id-(w-1)) - getTexVal(id - 1)) / (2.f * sx) : (getTexVal(id) - getTexVal(id-1)) / sx;
            else
                d_gdx[id] = (getTexVal(id+1) - getTexVal(id-1)) / (2.f * sx);

            if (j == 0)
                d_gdy[id] = (wrap) ? (getTexVal(id+w) - getTexVal(id + (h - 1)* w)) / (2.f * sy) : (getTexVal(id+w) - getTexVal(id)) / sy;
            else if (j == h - 1) 
                d_gdy[id] = (wrap) ? (getTexVal(id-(h-1)*w) - getTexVal(id - w)) / (2.f * sy) : (getTexVal(id) - getTexVal(id-w)) / sy;
            else
                d_gdy[id] = (getTexVal(id+w) - getTexVal(id-w)) / (2.f * sy);

            if (k == 0)
                d_gdz[id] = (wrap) ? (getTexVal(id+wh) - getTexVal(id + (l - 1)* wh)) / (2.f * sz) : (getTexVal(id+wh) - getTexVal(id)) / sz;
            else if (k == l - 1) 
                d_gdz[id] = (wrap) ? (getTexVal(id-(l-1)*wh) - getTexVal(id - wh)) / (2.f * sz) : (getTexVal(id) - getTexVal(id-wh)) / sz;
            else
                d_gdz[id] = (getTexVal(id+wh) - getTexVal(id-wh)) / (2.f * sz);
        }
}


void cudaComputeGradient(float * d_gdx, float * d_gdy, float * d_gdz, // out put gradient
                         const float * d_i,                                 // input image
                         uint w, uint h, uint l,                      // size of image  
                         float sx, float sy, float sz,   // spacing 
                         bool warp, cudaStream_t stream)                             // warp or not
{
    dim3 threads(16,16);
    dim3 grids(iDivUp(w, threads.x), iDivUp(h, threads.y));
    if (warp == false)
        cudaComputeGradient_kernel<0><<<grids, threads, 0, stream>>>(d_gdx, d_gdy, d_gdz,
                                                                     d_i,
                                                                     w, h, l,
                                                                     sx, sy, sz);
    else
        cudaComputeGradient_kernel<1><<<grids, threads, 0, stream>>>(d_gdx, d_gdy, d_gdz,
                                                                     d_i,
                                                                     w, h, l,
                                                                     sx, sy, sz);
}

void cudaComputeGradient_tex(float * d_gdx, float * d_gdy, float * d_gdz, // out put gradient
                             float * d_i,                                 // input image
                             uint w, uint h, uint l,                      // size of image  
                             float sx, float sy, float sz,   // spacing 
                             bool warp, cudaStream_t stream)                             // warp or not
{
    dim3 threads(16,16);
    dim3 grids(iDivUp(w, threads.x), iDivUp(h, threads.y));
    cudaBindTexture(0, com_tex_float, d_i, w * h * l * sizeof(float));
    if (warp == false)
        cudaComputeGradient_kernel<0><<<grids, threads, 0, stream>>>(d_gdx, d_gdy, d_gdz,
                                                                     w, h, l,
                                                                     sx, sy, sz);
    else
        cudaComputeGradient_kernel<1><<<grids, threads, 0, stream>>>(d_gdx, d_gdy, d_gdz,
                                                                     w, h, l,
                                                                     sx, sy, sz);
}

void cplComputeGradient(float * d_gdx, float * d_gdy, float * d_gdz, // out put gradient
                         const float * d_i,                                 // input image
                         uint w, uint h, uint l,                      // size of image  
                         float sx, float sy, float sz,   // spacing 
                         bool wrap, cudaStream_t stream)                             // wrap or not
{
    //cudaComputeGradient_tex(d_gdx, d_gdy, d_gdz, d_i, w, h, l, sx, sy, sz, wrap);
    //cudaComputeGradient(d_gdx, d_gdy, d_gdz, d_i, w, h, l, sx, sy, sz, wrap);
  //cudaComputeGradient_shared(d_gdx, d_gdy, d_gdz, d_i, w, h, l, sx, sy, sz, wrap, stream);
  cudaComputeGradient(d_gdx, d_gdy, d_gdz, d_i, w, h, l, sx, sy, sz, wrap, stream);
}

void cplComputeGradient(cplVector3DArray& d_gd,                     // output gradient
                         const float * d_i,                                 // input image
                         uint w, uint h, uint l,                      // size of image  
                         float sx, float sy, float sz,                // spacing 
                         bool warp, cudaStream_t stream){
    cplComputeGradient(d_gd.x, d_gd.y, d_gd.z, d_i, w, h, l, sx, sy, sz, warp, stream);
}


void cplComputeGradient(cplVector3DArray& d_gd,
                         const float * d_i,             
                         const Vector3Di& size,            // size of image  
                         const Vector3Df& sp,              // spacing 
                         bool warp, cudaStream_t stream)
{
    cplComputeGradient(d_gd.x, d_gd.y, d_gd.z, d_i,
                        size.x, size.y, size.z,
                        sp.x, sp.y, sp.z, warp, stream);
}


void testGradient3D(float* h_img, uint w, uint h, uint l){
    float* h_devX, * h_devY, *h_devZ;
    float* d_devX, * d_devY, *d_devZ;
    float *d_img;

    int size = w * h * l;
    h_devX = new float [size];
    h_devY = new float [size];
    h_devZ = new float [size];

    dmemAlloc(d_img, size );
    
    dmemAlloc(d_devX, size );
    dmemAlloc(d_devY, size );
    dmemAlloc(d_devZ, size );
    
    gradient_cpu3D(h_devX, h_devY, h_devZ, h_img, w, h, l);

    cudaMemcpy(d_img, h_img, sizeof(float) * size, cudaMemcpyHostToDevice);
    
    uint timer;
    uint nIter = 1000;

    CUT_SAFE_CALL( cutCreateTimer( &timer));

    CUT_SAFE_CALL( cutResetTimer( timer));
    CUT_SAFE_CALL( cutStartTimer( timer));
    for (uint i=0; i < nIter; ++i)
        cplGradient3D_32f(d_devX, d_devY, d_devZ, d_img, w, h, l, 0);
    
    cudaThreadSynchronize();
    CUT_SAFE_CALL( cutStopTimer( timer));
    printf( "Processing time: %f (ms)\n", cutGetTimerValue(timer)/nIter);

    CUT_SAFE_CALL( cutResetTimer( timer));
    CUT_SAFE_CALL( cutStartTimer( timer));
    for (uint i=0; i < nIter; ++i)
        cplGradient3D_32f_linear(d_devX, d_devY, d_devZ, d_img, w, h, l, 0);
    
    cudaThreadSynchronize();
    CUT_SAFE_CALL( cutStopTimer( timer));
    printf( "Processing time: %f (ms)\n", cutGetTimerValue(timer)/nIter);
    
    CUT_SAFE_CALL( cutResetTimer( timer));
    CUT_SAFE_CALL( cutStartTimer( timer));
    for (uint i=0; i < nIter; ++i)
        cplGradient3D_32f_tex(d_devX, d_devY, d_devZ, d_img, w, h, l, 0);
    
    cudaThreadSynchronize();
    CUT_SAFE_CALL( cutStopTimer( timer));
    printf( "Processing time: %f (ms)\n", cutGetTimerValue(timer)/nIter);

    testError(h_devX, d_devX, 1e-6, size, "X Grad");
    testError(h_devY, d_devY, 1e-6, size, "Y Grad");
    testError(h_devZ, d_devZ, 1e-6, size, "Z Grad");

    cudaFree(d_img);
    cudaFree(d_devX);
    cudaFree(d_devY);
    cudaFree(d_devZ);
    
    delete []h_devY;
    delete []h_devX;
    delete []h_devZ;
}

void testGradient3D(float* h_img,
                    uint w, uint h, uint l,
                    float sx, float sy, float sz){
    float* h_devX, * h_devY, *h_devZ;
    float* d_devX, * d_devY, *d_devZ;
    float *d_img;

    int size = w * h * l;
    h_devX = new float [size];
    h_devY = new float [size];
    h_devZ = new float [size];

    int wrap = 0;

    fprintf(stderr,"cpuComputeGradient has been removed due to conflicting name!!\n");
    exit(-1);
    //cpuComputeGradient(h_devX, h_devY, h_devZ, h_img, w, h, l, sx, sy, sz, wrap);
        
    dmemAlloc(d_img, size );
    dmemAlloc(d_devX, size );
    dmemAlloc(d_devY, size );
    dmemAlloc(d_devZ, size );

    cudaMemcpy(d_img, h_img, sizeof(float) * size, cudaMemcpyHostToDevice);
    
    uint timer;
    uint nIter = 1000;

    CUT_SAFE_CALL( cutCreateTimer( &timer));
    for (uint i=0; i < nIter; ++i)
        cplComputeGradient(d_devX, d_devY, d_devZ, d_img, w, h, l, sx, sy, sz, wrap);
    cudaThreadSynchronize();
    CUT_SAFE_CALL( cutStopTimer( timer));
    printf( "Processing time: %f (ms)\n", cutGetTimerValue(timer)/nIter);
    
    CUT_SAFE_CALL( cutResetTimer( timer));
    CUT_SAFE_CALL( cutStartTimer( timer));
    for (uint i=0; i < nIter; ++i)
        cudaComputeGradient_tex(d_devX, d_devY, d_devZ, d_img, w, h, l, sx, sy, sz, wrap,0);
    cudaThreadSynchronize();
    CUT_SAFE_CALL( cutStopTimer( timer));
    printf( "Processing time: %f (ms)\n", cutGetTimerValue(timer)/nIter);

    testError(h_devX, d_devX, 1e-6, size, "X Grad");
    testError(h_devY, d_devY, 1e-6, size, "Y Grad");
    testError(h_devY, d_devY, 1e-6, size, "Z Grad");

    cudaFree(d_img);
    cudaFree(d_devX);
    cudaFree(d_devY);
    cudaFree(d_devZ);
    
    delete []h_devY;
    delete []h_devX;
    delete []h_devZ;
}


/*--------------------------------------------------------------------------------------------
  Backward mapping function with zero boundary.
  Only use this function with input and output is an image (NOT deformation field)
  since this require zero boundary
  
  Inputs : d_src                 : source image
  d_vx, dvy, dvz        : velocity fields
  w , h, l              : size of the volume
  Output :
  d_result              : output
  id = i + j * w  + k * w * h;
  d_result[id] = d_src[i+vx[id], j+vy[id], k+vz[id]] 
  --------------------------------------------------------------------------------------------*/

__global__ void cplBackwardMapping3D_32f_C1_kernel(float * result, float * image,
                                                    float * vx, float * vy, float *vz,
                                                    int w, int h, int l)
{
    float x, y, z;
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    uint j = blockIdx.y * blockDim.y + threadIdx.y;
    uint k;
    
    uint wh = w * h;
    uint index = i + w * j;
    
    for (k = 0; k < l; ++k, index+=wh)
        if ( i < w && j < h){
            x = (float)(i) + vx[index];
            y = (float)(j) + vy[index];
            z = (float)(k) + vz[index];

            if (x >= 0 && x < w - 1 &&
                y >= 0 && y < h - 1 &&
                z >= 0 && z < l - 1)
                result[index] = simple_triLerp_kernel(image, x, y, z, w, wh );
            else
                result[index]=0;
        }
}

void cplBackwardMapping3D_32f_C1(float * result, float * image, float * vx, float * vy, float *vz,
                                  uint w, uint h, uint l, cudaStream_t stream){
    dim3 threads(16,16);
    dim3 grids(iDivUp(w, threads.x), iDivUp(h, threads.y));
    cplBackwardMapping3D_32f_C1_kernel<<<grids, threads, 0, stream>>>(result, image, vx, vy,vz, w, h, l);
}

__global__ void cplBackwardMapping3D_tex_linear_kernel(float * result, float * image,
                                                        float * vx, float * vy, float *vz,
                                                        int w, int h, int l)
{
    float x, y, z;
    
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    uint j = blockIdx.y * blockDim.y + threadIdx.y;
    uint k;
    
    uint wh = w * h;
    uint index = i + w * j;
    
    for (k = 0; k < l; ++k, index+=wh)
        if ( (i < w) && (j < h)){
            x = (float)(i) + vx[index];
            y = (float)(j) + vy[index];
            z = (float)(k) + vz[index];

            if ((x >= 0) && (x < (w - 1)) &&
                (y >= 0) && (y < (h - 1)) &&
                (z >= 0) && (z < (l - 1)))
            {
                result[index] = simple_triLerpTex_kernel(image, x, y, z, w, wh);
            }
            else
                result[index] = 0.f;
        }
}

void cplBackwardMapping3D_32f_C1_TexLinear(float * result, float * image, float * vx, float * vy, float *vz,
                                            uint w, uint h, uint l, cudaStream_t stream){
    dim3 threads(16,16);
    dim3 grids(iDivUp(w, threads.x), iDivUp(h, threads.y));
    cudaBindTexture(0, com_tex_float, image, w * h * l * sizeof(float));
    cplBackwardMapping3D_tex_linear_kernel<<<grids, threads, 0, stream>>>(result, image, vx, vy,vz, w, h, l);
}

__global__ void cplBackwardMapping3D_tex_linear_kernel(float * result, float * image,
                                                        float * vx, float * vy, float *vz,
                                                        float delta,
                                                        int w, int h, int l)
{
    float x, y, z;
    
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    uint j = blockIdx.y * blockDim.y + threadIdx.y;
    uint k;
    
    uint wh = w * h;
    uint index = i + w * j;
    
    for (k = 0; k < l; ++k, index+=wh)
        if ( i < w && j < h){
            x = (float)(i) + vx[index] * delta;
            y = (float)(j) + vy[index] * delta;
            z = (float)(k) + vz[index] * delta;

            if (x >= 0 && x < w - 1 &&
                y >= 0 && y < h - 1 &&
                z >= 0 && z < l - 1){
                result[index] = simple_triLerpTex_kernel(image, x, y, z, w, wh);
            }
            else
                result[index] = 0.f;
        }
}

void cplBackwardMapping3D_32f_C1_TexLinear(float * result, float * image,
                                            float * vx, float * vy, float *vz,
                                            float delta,
                                            uint w, uint h, uint l, cudaStream_t stream){
    dim3 threads(16,16);
    dim3 grids(iDivUp(w, threads.x), iDivUp(h, threads.y));

    cudaBindTexture(0, com_tex_float, image, w * h * l * sizeof(float));
    cplBackwardMapping3D_tex_linear_kernel<<<grids, threads, 0, stream>>>(result, image, vx, vy,vz, delta, w, h, l);
}


//--------------------------------------------------------------------------
//
//
//
//
//--------------------------------------------------------------------------

__global__ void cplBackwardMapping3D_tex3D_kernel(float * result,
                                                   float * vx, float * vy, float *vz,
                                                   uint w, uint h, uint l)
{
    int i        = threadIdx.x + blockIdx.x * blockDim.x;
    int j        = threadIdx.y + blockIdx.y * blockDim.y;
    int index    = i + j * w;
    int wh       = w * h;
    
    float x, y, z;
    if ( i < w && j < h){
        for (int k=0 ; k < l; ++k, index +=wh){
            x = (float)(i) + vx[index];
            y = (float)(j) + vy[index];
            z = (float)(k) + vz[index];
            
            if (x >= 0 && x < w - 1 &&
                y >= 0 && y < h - 1 &&
                z >= 0 && z < l - 1){
                result[index] = tex3D(com_tex3_float, x + 0.5f, y + 0.5f, z + 0.5f);
            }
            else
                result[index] = 0.f;
        }
    }
}

//--------------------------------------------------------------------------
//
//
//
//
//--------------------------------------------------------------------------

void cplBackwardMapping3D_32f_C1_tex(float * result,
                                      float * src, 
                                      float * vx, float * vy, float *vz,
                                      uint w, uint h, uint l, cudaStream_t stream)
{
    cudaArray * d_volumeArray = 0;
    cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
    cudaExtent volumeSize      = make_cudaExtent(w, h, l);
    cudaMalloc3DArray(&d_volumeArray, &desc, volumeSize);

    cudaMemcpy3DParms copyParams = {0};
    copyParams.srcPtr   = make_cudaPitchedPtr((void*)src, w * sizeof (float), w, h);
    copyParams.dstArray = d_volumeArray;
    copyParams.extent   = volumeSize;
    copyParams.kind     = cudaMemcpyDeviceToDevice;
    cudaMemcpy3D(&copyParams);

    com_tex3_float.addressMode[0] = cudaAddressModeClamp;
    com_tex3_float.addressMode[1] = cudaAddressModeClamp;
    com_tex3_float.addressMode[2] = cudaAddressModeClamp;
    com_tex3_float.filterMode     = cudaFilterModeLinear; // Using 3D interpolation
    com_tex3_float.normalized     = false;
    cudaBindTextureToArray(com_tex3_float, d_volumeArray, desc);

    dim3 threads(16,16);
    dim3 grids(iDivUp(w, threads.x), iDivUp(h, threads.y));
    cplBackwardMapping3D_tex3D_kernel<<<grids, threads, 0, stream>>>(result, vx, vy, vz, w, h, l);

    cudaFreeArray(d_volumeArray);
}


void cplBackwardMapping3D_32f_C1_tex(float * result,
                                      float * src,
                                      cudaArray* d_volumeArray,
                                      float * vx, float * vy, float *vz,
                                      uint w, uint h, uint l, cudaStream_t stream)
{
    cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
    cudaExtent volumeSize      = make_cudaExtent(w, h, l);

    cudaMemcpy3DParms copyParams = {0};
    copyParams.srcPtr   = make_cudaPitchedPtr((void*)src, w *sizeof(float), w, h);
    copyParams.dstArray = d_volumeArray;
    copyParams.extent   = volumeSize;
    copyParams.kind     = cudaMemcpyDeviceToDevice;
    cudaMemcpy3D(&copyParams);

    com_tex3_float.addressMode[0] = cudaAddressModeClamp;
    com_tex3_float.addressMode[1] = cudaAddressModeClamp;
    com_tex3_float.addressMode[2] = cudaAddressModeClamp;
    com_tex3_float.filterMode     = cudaFilterModeLinear; // Using 3D interpolation
    com_tex3_float.normalized     = false;
    cudaBindTextureToArray(com_tex3_float, d_volumeArray, desc);
    
    dim3 threads(16,16);
    dim3 grids(iDivUp(w, threads.x), iDivUp(h, threads.y));
    cplBackwardMapping3D_tex3D_kernel<<<grids, threads, 0, stream>>>(result, vx, vy, vz, w, h, l);
}

/*--------------------------------------------------------------------------------------------
  Reverse mapping function
  Inputs : d_src                 : source image
  d_vx, dvy, dvz        : velocity fields
  w , h, l              : size of the volume
  Output :
  d_result              : output
  id = i + j * w  + k * w * h;
  d_result[id] = d_src[i+vx[id], j+vy[id], k+vz[id]] 
  --------------------------------------------------------------------------------------------*/
void cplBackwardMapping3D(float * d_result,
                           float * d_src, 
                           float * d_vx, float *d_vy, float *d_vz,
                           uint w, uint h, uint l, cudaStream_t stream){

    //cplBackwardMapping3D_32f_C1_TexLinear(d_result, d_src, d_vx, d_vy, d_vz, w, h, l, stream);
    cplBackwardMapping3D_32f_C1(d_result, d_src, d_vx, d_vy, d_vz, w, h, l, stream);
    
    //  If you want to have a faster version but less accuracy
    //cplBackwardMapping3D_32f_C1_tex(d_result, d_src, d_vx, d_vy, d_vz, w, h, l);
}

void cplBackwardMapping3D(float * d_result,
                           float * d_src, 
                           float * d_vx, float *d_vy, float *d_vz,
                           float delta,
                           uint w, uint h, uint l, cudaStream_t stream){

    cplBackwardMapping3D_32f_C1_TexLinear(d_result, d_src, d_vx, d_vy, d_vz,
                                           delta,
                                           w, h, l, stream);
    //  If you want to have a faster version but less accuracy
    //cplBackwardMapping3D_32f_C1_tex(d_result, d_src, d_vx, d_vy, d_vz, w, h, l);
}

void testBackwardMapping3D(float* h_iImg,
                           float* h_vx, float* h_vy, float* h_vz,
                           uint w, uint h, int l){

    int size = w * h * l;
    float* h_oImg;
    h_oImg = new float [size];

    float* d_oImg, *d_iImg, *d_vx, *d_vy, *d_vz;

    dmemAlloc(d_iImg, size );
    dmemAlloc(d_oImg, size );
        
    dmemAlloc(d_vx, size );
    dmemAlloc(d_vy, size );
    dmemAlloc(d_vz, size );

    cudaMemcpy(d_iImg, h_iImg, sizeof(float) * size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vx, h_vx, sizeof(float) * size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vy, h_vy, sizeof(float) * size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vz, h_vz, sizeof(float) * size, cudaMemcpyHostToDevice);
    
    uint timer;
    uint nIter = 1000;

    cpuBackwardMapping3D(h_oImg, h_iImg, h_vx, h_vy, h_vz, w, h, l);
    
    CUT_SAFE_CALL( cutCreateTimer( &timer));

    CUT_SAFE_CALL( cutResetTimer( timer));
    CUT_SAFE_CALL( cutStartTimer( timer));

    for (uint i=0; i < nIter; ++i)
        cplBackwardMapping3D_32f_C1_tex(d_oImg, d_iImg, d_vx, d_vy, d_vz, w, h, l, 0);
    cudaThreadSynchronize();
    CUT_SAFE_CALL( cutStopTimer( timer));
    printf( "Processing time: %f (ms)\n", cutGetTimerValue(timer)/nIter);

    testError(h_oImg, d_oImg, 0.5 * 1e-4, size, "Flow image");
        
    CUT_SAFE_CALL( cutResetTimer( timer));
    CUT_SAFE_CALL( cutStartTimer( timer));
    for (uint i=0; i < nIter; ++i)
        cplBackwardMapping3D_32f_C1_TexLinear(d_oImg, d_iImg, d_vx, d_vy, d_vz, w, h, l, 0);
            
    cudaThreadSynchronize();
    CUT_SAFE_CALL( cutStopTimer( timer));
    printf( "Processing time: %f (ms)\n", cutGetTimerValue(timer)/nIter);
    testError(h_oImg, d_oImg, 0.5 * 1e-4, size, "Flow image");
    
    
    CUT_SAFE_CALL( cutResetTimer( timer));
    CUT_SAFE_CALL( cutStartTimer( timer));
    for (uint i=0; i < nIter; ++i)
        cplBackwardMapping3D_32f_C1(d_oImg, d_iImg, d_vx, d_vy, d_vz, w, h, l, 0);
    cudaThreadSynchronize();
    CUT_SAFE_CALL( cutStopTimer( timer));
    printf( "Processing time: %f (ms)\n", cutGetTimerValue(timer)/nIter);

    testError(h_oImg, d_oImg, 0.5 * 1e-4, size, "Flow image");
    
    cudaFree(d_oImg);
    cudaFree(d_iImg);
    cudaFree(d_vy);
    cudaFree(d_vx);
    cudaFree(d_vz);

    delete []h_oImg;
}


/*----------------------------------------------------------------------------------------------------
  These functions use an indirect presentation of the transformation field (increment form) in that
  the transformation is seperated into 2 part : the identity (x) and the change
  H(x). So actually for this technique we store the change h(x) instead of the
  transformation itself

  Assume h_k = x + H_k(x)
  From   h_(k+1) = h_k( x + delta * v_x)
  We have x + H_k+1 = (x + delta * v_x) + H_k( x + delta * v_x)
  then   H_k+1 = (delta * v_x) + H_k( x + delta * v_x) (compose function)
  From   I_k = I_0(h_k) = I_0(x + H_k)  (backward mapping function) 

  One nice properties of the h_x is that when we perform the multi-scale approach
  we can use the resample and intepolation to compute H_k from it lower scale version

  DO NOT use these functions with regular deformation field since it does not have the zero boundary 
  ----------------------------------------------------------------------------------------------------*/

void cplBackwardMapping3D(float* d_hPb, float* d_h, cplVector3DArray& d_v, float delta, uint w, uint h, uint l, cudaStream_t stream){
    cplBackwardMapping3D(d_hPb, d_h, d_v.x, d_v.y, d_v.z, delta, w, h, l, stream);
}

void cplBackwardMapping3D(float* d_hPb, float* d_h, cplVector3DArray& d_v, uint w, uint h, uint l, cudaStream_t stream){
    cplBackwardMapping3D(d_hPb, d_h, d_v.x, d_v.y, d_v.z, w, h, l, stream);
}

void cplBackwardMapping3D(cplVector3DArray& d_hPb, cplVector3DArray& d_h, cplVector3DArray& d_v, float delta, uint w, uint h, uint l, cudaStream_t stream){
    cplBackwardMapping3D(d_hPb.x, d_h.x, d_v, delta, w, h, l, stream);
    cplBackwardMapping3D(d_hPb.y, d_h.y, d_v, delta, w, h, l, stream);
    cplBackwardMapping3D(d_hPb.z, d_h.z, d_v, delta, w, h, l, stream);
}

void cplBackwardMapping3D(cplVector3DArray& d_hPb, cplVector3DArray& d_h, cplVector3DArray& d_v, uint w, uint h, uint l, cudaStream_t stream){
    cplBackwardMapping3D(d_hPb.x, d_h.x, d_v, w, h, l, stream);
    cplBackwardMapping3D(d_hPb.y, d_h.y, d_v, w, h, l, stream);
    cplBackwardMapping3D(d_hPb.z, d_h.z, d_v, w, h, l, stream);
}

void composeTransformation(cplVector3DArray& d_H, cplVector3DArray& d_v,
                           cplVector3DArray& d_HPb,
                           float delta, uint w, uint h, uint l, cudaStream_t stream){
    
    // Pullback the field
    cplBackwardMapping3D(d_HPb, d_H, d_v, delta, w, h, l, stream);
    
    // Then push forward
    uint n = w * h * l;
    cplVector3DOpers::MulCAdd(d_H, d_v, delta, d_HPb, n, stream);
}


/*--------------------------------------------------------------------------------------------
  Reverse mapping function (simple version with zero boundary condition, will be replaced
  in the future)
  Inputs : d_src                 : source image
  d_hx, d_hy, d_hz      : deformation field
  w , h, l              : size of the volume
  Output :
  d_result              : output
  id = i + j * w  + k * w * h;
  d_result[id] = d_src[hx[id], hy[id], hz[id]] 
  --------------------------------------------------------------------------------------------*/
__global__ void cplReverseMap3D_kernel(float * result, float * image,
                                        float * hx, float * hy, float *hz,
                                        int w, int h, int l)
{
    float x, y, z;
    
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    uint j = blockIdx.y * blockDim.y + threadIdx.y;
    uint k;
    
    uint wh = w * h;
    uint index = i + w * j;
    
    for (k = 0; k < l; ++k, index+=wh)
        if ( i < w && j < h){
            x = hx[index];
            y = hy[index];
            z = hz[index];

            if (x >= 0 && x < w - 1 &&
                y >= 0 && y < h - 1 &&
                z >= 0 && z < l - 1){
                result[index] = simple_triLerpTex_kernel(image, x, y, z, w, wh);
            }
            else
                result[index] = 0.f;
        }
}

void cplReverseMap3D(float * result, float * image,
                      float * hx, float * hy, float *hz,
                      uint w, uint h, uint l, cudaStream_t stream){
    dim3 threads(16,16);
    dim3 grids(iDivUp(w, threads.x), iDivUp(h, threads.y));
    cudaBindTexture(0, com_tex_float, image, w * h * l * sizeof(float));
    cplReverseMap3D_kernel<<<grids, threads, 0, stream>>>(result, image, hx, hy,hz, w, h, l);
}

void cplReverseMap3D(float* d_I0t, float* d_I0, cplVector3DArray& d_h, uint w, uint h, uint l, cudaStream_t stream){
    cplReverseMap3D(d_I0t, d_I0, d_h.x, d_h.y, d_h.z, w, h, l, stream);
}

/*--------------------------------------------------------------------------------------------

  Upsampling function float* d_oImg, float* d_iImg, int w, int h, int l){
  Inputs : d_iImg                : source image
  w , h, l              : size of the volume
  Output :
  d_oImg              : output
  with size 2* (w, h, l)
  --------------------------------------------------------------------------------------------*/
__global__ void cplUpsampling_tex_kernel(float * d_resut, int w, int h, int l)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    int k;
    float x, y, z;

    int i1 = (i - 1) >> 1;
    int j1 = (j - 1) >> 1;

    x = (i & 1) ? i1 + 0.25f : i1 + 0.75f;
    y = (j & 1) ? j1 + 0.25f : j1 + 0.75f;
    
    int index = i + w * j;
    int wh    = w * h;
    if ( i < w  && j < h) {
        for (k = 0; k < l; ++k)
        {
            int k1 = (k - 1) >> 1;
            float comp = 0.f;
            if (i1 <0 || j1 < 0 || k1 < 0 || i > w - 2 || j > h - 2 || k  > l - 2)
                comp = 0.f;
            else {
                z      = (k & 1) ? k1 + 0.25f : k1 + 0.75f;
                comp   = tex3D(com_tex3_float, x + 0.5f, y + 0.5f, z + 0.5f);
            }
            d_resut[index] = 2 * comp;
            index+= wh;
        }
    }
}

// ATENTION : w,h,l is the size of the original image
// The size of the upsampling image is 2*w, 2*h, 2*l

void cplUpsampling_tex(float* d_oImg, float* d_iImg, int w, int h, int l, cudaStream_t stream){

    cudaArray * d_volumeArray = 0;
    cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
    cudaExtent volumeSize      = make_cudaExtent(w, h, l);
    cudaMalloc3DArray(&d_volumeArray, &desc, volumeSize);

    cudaMemcpy3DParms copyParams = {0};
    copyParams.srcPtr   = make_cudaPitchedPtr((void*)d_iImg, w *sizeof(float), w, h);
    copyParams.dstArray = d_volumeArray;
    copyParams.extent   = volumeSize;
    copyParams.kind     = cudaMemcpyDeviceToDevice;
    cudaMemcpy3D(&copyParams);

    com_tex3_float.addressMode[0] = cudaAddressModeClamp;
    com_tex3_float.addressMode[1] = cudaAddressModeClamp;
    com_tex3_float.addressMode[2] = cudaAddressModeClamp;
    com_tex3_float.filterMode     = cudaFilterModeLinear; // Using 3D interpolation
    com_tex3_float.normalized     = false;
    cudaBindTextureToArray(com_tex3_float, d_volumeArray, desc);

    dim3 threads(16,16);
    dim3 grids(iDivUp(w * 2, threads.x), iDivUp(h * 2, threads.y));
    cplUpsampling_tex_kernel<<<grids, threads, 0, stream>>>(d_oImg, 2 * w, 2 * h, 2 * l);

    cudaFreeArray(d_volumeArray);
}

void cplUpsampling_tex(float* d_oImg,
                        float* d_iImg,
                        cudaArray * d_volumeArray,
                        int w, int h, int l, cudaStream_t stream){

    cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
    cudaExtent volumeSize      = make_cudaExtent(w, h, l);

    cudaMemcpy3DParms copyParams = {0};
    copyParams.srcPtr   = make_cudaPitchedPtr((void*)d_iImg, w *sizeof(float), w, h);
    copyParams.dstArray = d_volumeArray;
    copyParams.extent   = volumeSize;
    copyParams.kind     = cudaMemcpyDeviceToDevice;

    cudaMemcpy3D(&copyParams);

    com_tex3_float.addressMode[0] = cudaAddressModeClamp;
    com_tex3_float.addressMode[1] = cudaAddressModeClamp;
    com_tex3_float.addressMode[2] = cudaAddressModeClamp;
    com_tex3_float.filterMode     = cudaFilterModeLinear; // Using 3D interpolation
    com_tex3_float.normalized     = false;

    cudaBindTextureToArray(com_tex3_float, d_volumeArray, desc);
    dim3 threads(16,16);
    dim3 grids(iDivUp(w * 2, threads.x), iDivUp(h * 2, threads.y));
    cplUpsampling_tex_kernel<<<grids, threads, 0, stream>>>(d_oImg, 2 * w, 2 * h, 2 * l);
}


////////////////////////////////////////////////////////////////////////////////
// Compute the L2NormSquare \sum_i ||v||^2
////////////////////////////////////////////////////////////////////////////////

float cplL2NormSqr(cplReduce* p_Rd, float* d_i, int N){
    return p_Rd->Sum2(d_i, N);
}

float cplL2NormSqr(cplReduce* p_Rd, cplImage3D& d_i)
{
    return p_Rd->Sum2(d_i.getDataPtr(), d_i.getNElems()) * d_i.getVoxelVol();
}

////////////////////////////////////////////////////////////////////////////////
// Compute the dot product of 2 images
////////////////////////////////////////////////////////////////////////////////

float cplL2DotProd(cplReduce* p_Rd, float* d_i, float* d_i1, int N){
    return p_Rd->Dot(d_i, d_i1, N);
}

float cplL2DotProd(cplReduce* p_Rd, cplImage3D& d_i, cplImage3D& d_i1){
    return p_Rd->Dot(d_i.getDataPtr(), d_i1.getDataPtr(), d_i.getNElems()) * d_i.getVoxelVol();
}


template<bool useOriginOffset, BackgroundStrategy bg>
__global__ void cplResampling_kernel(float* d_o, const float* d_i,
                                     int osizeX, int osizeY, int osizeZ,
                                     int isizeX, int isizeY, int isizeZ)
{
    uint x = blockIdx.x * blockDim.x + threadIdx.x;
    uint y = blockIdx.y * blockDim.y + threadIdx.y;

    float rX = (float)isizeX / (float)osizeX;
    float rY = (float)isizeY / (float)osizeY;
    float rZ = (float)isizeZ / (float)osizeZ;

    float offX, offY, offZ = 0.f;

    if(useOriginOffset){
        offX = (rX-1.f)/2.f;
        offY = (rY-1.f)/2.f;
        offZ = (rZ-1.f)/2.f;
    }

    if (x < osizeX && y < osizeY){
        int id = x + osizeX * y;

        float i_x =  x * rX + offX;
        float i_y =  y * rY + offY;

        for (int z=0; z < osizeZ; ++z, id += osizeX * osizeY){
            float i_z = z * rZ + offZ;
            d_o[id] = triLerp<bg>(d_i,
                                  i_x, i_y, i_z,
                                  isizeX, isizeY, isizeZ);
        }
    }
}

void cplResample(float* d_o, const float* d_i,
                  const Vector3Di& oSize, const Vector3Di& iSize,
                  BackgroundStrategy backgroundStrategy, bool useOriginOffset,
                  cudaStream_t stream)
{
    if ((iSize.x == oSize.x) && (iSize.y == oSize.y) && (iSize.z == oSize.z))
        copyArrayDeviceToDevice(d_o, d_i, iSize.x*iSize.y*iSize.z);
    else {
        dim3 threads(16,16);
        dim3 grids(iDivUp(oSize.x, threads.x), iDivUp(oSize.y, threads.y));
        if (useOriginOffset){
            if (backgroundStrategy == BACKGROUND_STRATEGY_PARTIAL_ID)
                cplResampling_kernel<true, BACKGROUND_STRATEGY_PARTIAL_ID><<<grids, threads, 0, stream>>>(d_o, d_i,
                                                                                                           oSize.x ,oSize.y ,oSize.z ,
                                                                                                           iSize.x ,iSize.y ,iSize.z);
            else if (backgroundStrategy == BACKGROUND_STRATEGY_ID)
                cplResampling_kernel<true, BACKGROUND_STRATEGY_ID><<<grids, threads, 0, stream>>>(d_o, d_i,
                                                                                                   oSize.x ,oSize.y ,oSize.z ,
                                                                                                   iSize.x ,iSize.y ,iSize.z);
            else if (backgroundStrategy == BACKGROUND_STRATEGY_ZERO)
                cplResampling_kernel<true, BACKGROUND_STRATEGY_ZERO><<<grids, threads, 0, stream>>>(d_o, d_i,
                                                                                                     oSize.x ,oSize.y ,oSize.z ,
                                                                                                     iSize.x ,iSize.y ,iSize.z);
            else if (backgroundStrategy == BACKGROUND_STRATEGY_CLAMP)
                cplResampling_kernel<true, BACKGROUND_STRATEGY_CLAMP><<<grids, threads, 0, stream>>>(d_o, d_i,
                                                                                                      oSize.x ,oSize.y ,oSize.z ,
                                                                                                      iSize.x ,iSize.y ,iSize.z);
            else if (backgroundStrategy == BACKGROUND_STRATEGY_WRAP)
                cplResampling_kernel<true, BACKGROUND_STRATEGY_WRAP><<<grids, threads, 0, stream>>>(d_o, d_i,
                                                                                                     oSize.x ,oSize.y ,oSize.z ,
                                                                                                     iSize.x ,iSize.y ,iSize.z);
            else
                throw AtlasWerksException(__FILE__, __LINE__, "Error, unknown background strategy");
        } else {
            if (backgroundStrategy == BACKGROUND_STRATEGY_PARTIAL_ID)
                cplResampling_kernel<false, BACKGROUND_STRATEGY_PARTIAL_ID><<<grids, threads, 0, stream>>>(d_o, d_i,
                                                                                                            oSize.x ,oSize.y ,oSize.z ,
                                                                                                            iSize.x ,iSize.y ,iSize.z);
            else if (backgroundStrategy == BACKGROUND_STRATEGY_ID)
                cplResampling_kernel<false, BACKGROUND_STRATEGY_ID><<<grids, threads, 0, stream>>>(d_o, d_i,
                                                                                                    oSize.x ,oSize.y ,oSize.z ,
                                                                                                    iSize.x ,iSize.y ,iSize.z);
            else if (backgroundStrategy == BACKGROUND_STRATEGY_ZERO)
                cplResampling_kernel<false, BACKGROUND_STRATEGY_ZERO><<<grids, threads, 0, stream>>>(d_o, d_i,
                                                                                                      oSize.x ,oSize.y ,oSize.z ,
                                                                                                      iSize.x ,iSize.y ,iSize.z);
            else if (backgroundStrategy == BACKGROUND_STRATEGY_CLAMP)
                cplResampling_kernel<false, BACKGROUND_STRATEGY_CLAMP><<<grids, threads, 0, stream>>>(d_o, d_i,
                                                                                                       oSize.x ,oSize.y ,oSize.z ,
                                                                                                       iSize.x ,iSize.y ,iSize.z);
            else if (backgroundStrategy == BACKGROUND_STRATEGY_WRAP)
                cplResampling_kernel<false, BACKGROUND_STRATEGY_WRAP><<<grids, threads, 0, stream>>>(d_o, d_i,
                                                                                                      oSize.x ,oSize.y ,oSize.z ,
                                                                                                      iSize.x ,iSize.y ,iSize.z);
            else
                throw AtlasWerksException(__FILE__, __LINE__, "Error, unknown background strategy");
        }
    }
}




void cplUpsample(float* d_o, const float* d_i,
                 const Vector3Di& osize, const Vector3Di& isize,
                 BackgroundStrategy backgroundStrategy,
                 bool useOriginOffset, cudaStream_t stream)
{
    assert((isize.x <=  osize.x) && (isize.y <=  osize.y) && (isize.z <=  osize.z));
    cplResample(d_o, d_i, osize, isize, backgroundStrategy, useOriginOffset, stream);
}
