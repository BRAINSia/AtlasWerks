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

#ifndef __CUDA_IMAGE3D_H
#define __CUDA_IMAGE3D_H

#include <Vector3D.h>
#include <cuda_runtime.h>
#include <libDefine.h>

class cplReduce;
class cplVector3DArray;
class Vector3D_XY_Z_Array;

class cplImage3D{
public:
    cplImage3D(const Vector3Di& size,
                const Vector3Df& org = Vector3Df(0.f,0.f,0.f),
                const Vector3Df& sp  = Vector3Df(1.f,1.f,1.f));


    float* getDataPtr() const {return d_data;}
    
    int    getNElems()  const {return m_nVox; };
    float  getVoxelVol()const {return m_voxVol; }

    void   copyFromHost(float* h_data);
    void   copyToHost(float* h_data);
    void   copyFromDevice(float* d_data);
    void   copyToDevice(float* d_data);

    void   allocate();
    void   clean();
private:
    Vector3Di m_vSize;    // Size of the image 
    Vector3Df m_vOrg;     // Origin of the image
    Vector3Df m_vSp;      // Spacing of the image 

    uint      m_nVox;     // Number of voxel 
    float     m_voxVol;   // Voxel volume  

    float* d_data;        // device image data 
};

/**
 * Gradient function of the grids (old version will be removed in the future)
 * Inputs : d_gu scalar fields of the grids
 *          w, h, l: size of the grid
 *          we assume the dx, dy = 1
 * Ouput  : d_gdx gradient on x direction 
 *          d_gdy gradient on y direction
 *          d_gdz gradient on y direction
 *          d_g   gradient on all x,y,z
 */
void cplGradient3D_32f(float * d_gdx, float * d_gdy, float * d_gdz, const float* src,
                        uint sizeX, uint sizeY, uint sizeZ, cudaStream_t stream=NULL);

void cplGradient3D_32f_linear(float * d_gdx, float * d_gdy, float * d_gdz, const float* src,
                               uint sizeX, uint sizeY, uint sizeZ, cudaStream_t stream=NULL);

void cplGradient3D_32f_tex(float * d_gdx, float * d_gdy, float * d_gdz,
                            const float* src,
                            uint w, uint h, uint l, cudaStream_t stream=NULL);


void gradient3D(float* d_gdx, float* d_gdy, float* d_gdz,
                const float* d_gu, int w, int h, int l, cudaStream_t stream=NULL);

void gradient3D(cplVector3DArray& d_g, const float* d_gu, int w, int h, int l, cudaStream_t stream=NULL);
void gradient3D(Vector3D_XY_Z_Array& d_g, const float* d_gu, int w, int h, int l, cudaStream_t stream=NULL);

/**
 *Gradient function of the grids (full version - SHOULD use this instead)
 * Inputs : d_gu scalar fields of the grids
 *          w, h, l    : size of the grid
 *          sx, sy, sz : spacing units (normaly use sx = sy = sz = 1)
 *          wrap: boundary condition (cyclic or not)
 *             0: use central difference on the inside, one side difference for the boundary 
 *             1: use central difference every where + cyclic boundary
 *          When do not know use wrap = 0;
 * Ouput  : d_gdx gradient on x direction 
 *          d_gdy gradient on y direction
 *          d_gdz gradient on y direction
 */
void cplComputeGradient(float * d_gdx, float * d_gdy, float * d_gdz, // output gradient
                         const float * d_i,                                 // input image
                         uint w, uint h, uint l,                      // size of image  
                         float sx = 1.f, float sy = 1.f, float sz = 1.f,// spacing 
                         bool warp=false, cudaStream_t stream=NULL);

void cplComputeGradient(cplVector3DArray& d_gd,                     // output gradient
                         const float * d_i,                                 // input image
                         uint w, uint h, uint l,                      // size of image  
                         float sx = 1.f, float sy=1.f, float sz=1.f,  // spacing 
                         bool warp=false, cudaStream_t stream=NULL);

void cplComputeGradient(cplVector3DArray& d_gd,                     // output gradient
                         const float * d_i,                                 // input image
                         const Vector3Di& size,                       // size of image  
                         const Vector3Df& sp=Vector3Df(1.f, 1.f, 1.f),// spacing 
                         bool warp=false, cudaStream_t stream=NULL);

void testGradient3D(float* h_img, uint w, uint h, uint l);
void testGradient3D(float* h_img, uint w, uint h, uint l, float sx, float sy, float sz);

/**
 * Backward mapping function with zero boundary.
 * Only use this function with input and output is an image (NOT deformation field)
 * since this require zero boundary
 * 
 * Inputs : d_src                 : source image
 *          d_vx, dvy, dvz        : velocity fields
 *          w , h, l              : size of the volume
 * Output :
 *          d_result              : output
 *          id = i + j * w  + k * w * h;
 *          d_result[id] = d_src[i+vx[id], j+vy[id], k+vz[id]] 
 */

void cplBackwardMapping3D(float * d_result, float * d_src, 
                           float * d_vx, float *d_vy, float *d_vz,
                           uint w, uint h, uint l, cudaStream_t stream=NULL);

void cplBackwardMapping3D(float * d_result, float * d_src, 
                           float * d_vx, float *d_vy, float *d_vz,
                           float delta,
                           uint w, uint h, uint l, cudaStream_t stream=NULL);

void cplBackwardMapping3D(float* d_hPb, float* d_h, cplVector3DArray& d_v, float delta, uint w, uint h, uint l, cudaStream_t stream=NULL);
void cplBackwardMapping3D(float* d_hPb, float* d_h, cplVector3DArray& d_v, uint w, uint h, uint l, cudaStream_t stream=NULL);

void cplBackwardMapping3D(cplVector3DArray& d_hPb, cplVector3DArray& d_h, cplVector3DArray& d_v, float delta, uint w, uint h, uint l, cudaStream_t stream=NULL);
void cplBackwardMapping3D(cplVector3DArray& d_hPb, cplVector3DArray& d_h, cplVector3DArray& d_v, uint w, uint h, uint l, cudaStream_t stream=NULL);

void cplBackwardMapping(float* d_hX, float* d_fX, 
                         float* d_gX, float* d_gY, float* d_gZ,
                         int sizeX, int sizeY, int sizeZ,
                         float delta,
                         int backgroundStrategy);

void cplBackwardMapping(float* d_hX, float* d_hY, float* d_hZ,
                         float* d_fX, float* d_fY, float* d_fZ,
                         float* d_gX, float* d_gY, float* d_gZ,
                         int sizeX, int sizeY, int sizeZ,
                         float delta,
                         int backgroundStrategy);


void composeTransformation(cplVector3DArray& d_H, cplVector3DArray& d_v, cplVector3DArray& d_HPb,
                           float delta, uint w, uint h, uint l, cudaStream_t stream=NULL);


/*--------------------------------------------------------------------------------------------
  Reverse mapping function (simple version with zero boundary condition, will be replaced
  in the future)
  Only use this function with input and output is an image (NOT deformation field)
  since this require zero boundary

  Inputs : d_src                 : source image
  d_hx, d_hy, d_hz      : deformation field
  w , h, l              : size of the volume
  Output :
  d_result              : output
  id = i + j * w  + k * w * h;
  d_result[id] = d_src[hx[id], hy[id], hz[id]] 
  --------------------------------------------------------------------------------------------*/

void cplReverseMap3D(float * d_result, float * d_src, 
                      float * d_hx, float *d_hy, float *d_hz,
                      uint w, uint h, uint l, cudaStream_t stream=NULL);

void cplReverseMap3D(float* d_I0t, float* d_I0, cplVector3DArray& d_h, uint w, uint h, uint l, cudaStream_t stream=NULL);

/*----------------------------------------------------------------------
  Trilinear interpolation from the grids to the points
  Inputs : d_gu grid scalar fields
  w, h,l size of the grid (the domain will go from [0:w-1,0:h-1, 0:l-1]
  d_px position of the particles
  nP   number of particles
  Output : d_pu the scalar value on the grid           
  ---------------------------------------------------------------------*/
void triLerp(float* d_pu, Vector3D_XY_Z_Array& d_px, int nP, float* d_gu, int w, int h, int l, cudaStream_t stream=NULL);
void triLerp(float* d_pu, cplVector3DArray&   d_px, int nP, float* d_gu, int w, int h, int l, cudaStream_t stream=NULL);

void testBackwardMapping3D(float* h_iImg,
                           float* h_vx, float* h_vy, float* h_vz,
                           uint w, uint h, int l);

void cplUpsampling_tex(float* d_oImg, float* d_iImg, int w, int h, int l, cudaStream_t stream=NULL);
void cplUpsampling_tex(float* d_oImg, float* d_iImg, cudaArray * d_volumeArray, int w, int h, int l, cudaStream_t stream=NULL);


float cplL2NormSqr(cplReduce* p_Rd, float* d_i, int N);
float cplL2DotProd(cplReduce* p_Rd, float* d_i, float* d_i1, int N);

float cplL2NormSqr(cplReduce* p_Rd, cplImage3D& d_i);
float cplL2DotProd(cplReduce* p_Rd, cplImage3D& d_i, cplImage3D& d_i1);


/**
 * Up sample d_i onto the grid d_o
 */
void cplResample(float* d_o, const float* d_i,
                 const Vector3Di& osize, const Vector3Di& isize,
                 BackgroundStrategy backgroundStrategy,
                 bool useOriginOffset=false, cudaStream_t stream=NULL);

void cplUpsample(float* d_o, const float* d_i,
                 const Vector3Di& oSize, const Vector3Di& iSize,
                 BackgroundStrategy backgroundStrategy, bool useOriginOffset=false,
                 cudaStream_t stream=NULL);


#endif
