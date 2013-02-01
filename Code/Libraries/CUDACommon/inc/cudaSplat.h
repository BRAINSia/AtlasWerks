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

#ifndef __CUDA_SPLAT_H
#define __CUDA_SPLAT_H

#include <cuda_runtime.h>
#include <Vector3D.h>

class cplReduce;
class cplVector3DArray;
struct Vector3D_XY_Z_Array;

/*--------------------------------------------------------------------------------------------
  Splating 
  Inputs : d_src                 : source image
           d_vx, dvy, dvz        : velocity fields
           w , h, l              : size of the volume
  Output :
           d_dst                 : output intergrate from the input
 --------------------------------------------------------------------------------------------*/
void cplSplating(float* d_dst, float* d_src,
                  float* d_vx , float* d_vy, float* d_vz,
                  uint w, uint h, uint l);

void cplSplating(float* d_dst, float* d_src,
                  float* d_dst1, float* d_src1,
                  float* d_vx , float* d_vy, float* d_vz,
                  uint w, uint h, uint l);

// This the version for large input if the size of the image can not handle 
void cplSplating2(float* d_dst, float* d_src,
                   float* d_vx , float* d_vy, float* d_vz,
                   uint w, uint h, uint l);

void splatingCPU(float* h_dst, float* h_src,
                 float* vx, float* vy, float* vz,
                 int w, int h, int l);


/*-----------------------------------------------------------------------
  Splat function from the point to the grids
  Inputs : d_pu scalar value of the point
           d_px position of the point
           w, h, l : size of the grid (the domain will go from [0:w-1,0:h-1, 0:l-1]
           we assume the dx, dy, dz = 1
  Ouput  : d_gu the scalar field of the grid 
-----------------------------------------------------------------------*/
void splat3D(float* d_gu, float* d_pu, Vector3D_XY_Z_Array& d_px,
             int w, int h, int l,
             int nP);

void splat3D(float4* d_gu, float4* d_pu, Vector3D_XY_Z_Array& d_px,
             int w, int h, int l,
             int nP);

////////////////////////////////////////////////////////////////////////////////
/// 
/// Regular splat function 
/// 
////////////////////////////////////////////////////////////////////////////////

void cplSplat3D(float* d_dst, uint sizeX, uint sizeY, uint sizeZ,
                 float* d_src, float* d_px , float* d_py, float* d_pz, uint nP, cudaStream_t stream=NULL);

////////////////////////////////////////////////////////////////////////////////
/// 
/// Regular splat function for Vector field
/// 
////////////////////////////////////////////////////////////////////////////////

void cplSplat3D(cplVector3DArray& d_o, uint sizeX, uint sizeY, uint sizeZ, 
                 cplVector3DArray& d_i, cplVector3DArray& d_pos, uint nP, cudaStream_t stream=NULL);

void cplSplat3D(cplVector3DArray& d_o, const Vector3Di& size,
                 cplVector3DArray& d_i, cplVector3DArray& d_pos, uint nP, cudaStream_t stream=NULL);

////////////////////////////////////////////////////////////////////////////////
/// 
/// Splat both the function d_src and the distance to the neighbor point 
/// to the grid d_dst = sum d_src * distance d_w = sum d_distance
////////////////////////////////////////////////////////////////////////////////

void cplSplat3D(float* d_dst, float* d_w, uint sizeX, uint sizeY, uint sizeZ,
                 float* d_src, float* d_px , float* d_py, float* d_pz, uint nP, cudaStream_t stream=NULL);

void cplSplat3D(float* d_dst, float* d_w, const Vector3Di& size,
                 float* d_src, cplVector3DArray& d_pos, uint nP, cudaStream_t stream=NULL);

////////////////////////////////////////////////////////////////////////////////
/// 
/// Splat function that normalize the input first then splat
/// and return the result back to the range
////////////////////////////////////////////////////////////////////////////////

void cplSplat3D(cplReduce& rd,
                 float* d_dst, uint sizeX, uint sizeY, uint sizeZ,
                 float* d_src, float* d_px , float* d_py, float* d_pz, uint nP, cudaStream_t stream=NULL);

void cplSplat3D(cplReduce& rd,
                 float* d_dst, const Vector3Di& size,
                 float* d_src, cplVector3DArray& d_pos, uint nP, cudaStream_t stream=NULL);

void cpuSplat3D(float* h_dst, uint w, uint h, uint l,
                float* h_src, float* h_px , float* h_py, float* h_pz, uint nP);


// Old name cplSplatingAtomicUnsigned
void cplSplat3DH(float* d_dst, float* d_src,
                 float* d_px , float* d_py, float* d_pz,
                 uint sizeX, uint sizeY, uint sizeZ, cudaStream_t stream=NULL );

void cplSplat3DH(cplReduce& rd, float* d_dst, float* d_src,
                 float* d_px , float* d_py, float* d_pz,
                 uint sizeX, uint sizeY, uint sizeZ, cudaStream_t stream=NULL);

void cplSplat3DH(float* d_dst, float* d_src,
                 cplVector3DArray& d_p, const Vector3Di& size, cudaStream_t stream=NULL);

void cplSplat3DH(cplReduce& rd, float* d_dst, float* d_src,
                 cplVector3DArray& d_p, const Vector3Di& size, cudaStream_t stream=NULL);


void cplSplat3DV(float* d_dst, float* d_src,
                 float* d_vx , float* d_vy, float* d_vz,
                 uint sizeX, uint sizeY, uint sizeZ, cudaStream_t stream=NULL);

void cplSplat3DV(float* d_wd, float* d_w,
                 float* d_vx, float* d_vy, float* d_vz,
                 uint sizeX, uint sizeY, uint sizeZ,
                 float spX, float spY, float spZ, cudaStream_t stream=NULL);

void cplSplat3DV(float* d_dst, float* d_src,
                 cplVector3DArray& d_v, const Vector3Di& size, cudaStream_t stream=NULL);

void cplSplat3DV(float* d_dst, float* d_src,
                 cplVector3DArray& d_v, const Vector3Di& size, const Vector3Df& sp, cudaStream_t stream=NULL);



//cplSplatingAtomicUnsigned_shared
void cplSplat3DV_shared(float* d_dst, float* d_src,
                         float* d_vx , float* d_vy, float* d_vz,
                         uint sizeX, uint sizeY, uint sizeZ, cudaStream_t stream=NULL);


void testSplating(uint w, uint h, int l);
void testSplating2(uint w, uint h, int l);

    
#endif
