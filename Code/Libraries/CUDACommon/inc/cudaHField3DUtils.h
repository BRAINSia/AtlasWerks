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

#ifndef __CUDA__HFIELD3D_UTILS__H
#define __CUDA__HFIELD3D_UTILS__H

#include <Vector3D.h>
#include <libDefine.h>
#include <cudaVector3DArray.h>

namespace cudaHField3DUtils
{
    void normalize(float* d_hx, float* d_hy, float* d_hz,
                   int w, int h, int l,
                   float spX, float spY, float spZ, cudaStream_t stream = NULL);

    void normalize(cplVector3DArray& d_h, const Vector3Di& size, const Vector3Df sp, cudaStream_t stream=NULL);


    void restoreSP(float* d_hx, float* d_hy, float* d_hz,
                   int w, int h, int l,
                   float spX, float spY, float spZ, cudaStream_t stream = NULL);

    void restoreSP(cplVector3DArray& d_h, const Vector3Di& size, const Vector3Df sp, cudaStream_t stream=NULL);
    

    /**
     * set hfield to identity
     * i.e. h(x) = x
     */
    void setToIdentity(float* d_hx, float* d_hy, float* d_hz, int w, int h, int l, cudaStream_t stream=NULL);
    void setToIdentity(cplVector3DArray& d_h, const Vector3Di& size, cudaStream_t stream=NULL);
    void setToIdentity(float* d_hx, float* d_hy, float* d_hz, int w, int h, int l, float spX, float spY, float spZ, cudaStream_t stream=NULL);
    void setToIdentity(cplVector3DArray& d_h, const Vector3Di& size, const Vector3Df& sp, cudaStream_t stream=NULL);

    /**
     * Add identity to the current field
     * i.e. h(x) = x + h(x)
     */
    void addIdentity(float* d_hx, float* d_hy, float* d_hz, int w, int h, int l, cudaStream_t stream=NULL);
    void addIdentity(cplVector3DArray& d_h, const Vector3Di& size, cudaStream_t stream=NULL);
    void addIdentity(float* d_hx, float* d_hy, float* d_hz, int w, int h, int l, float spX, float spY, float spZ, cudaStream_t stream=NULL);
    void addIdentity(cplVector3DArray& d_h, const Vector3Di& size, const Vector3Df& sp, cudaStream_t stream=NULL);

    /**
     * Convert velocity field to hfield
     * i.e. h(x) = x + v(x) * delta
     */
    void velocityToH(float* d_hx, float* d_hy, float* d_hz, const float* d_vx, const float* d_vy, const float* d_vz,
                     int w, int h, int l, cudaStream_t stream=NULL);
    void velocityToH(cplVector3DArray& d_h, const cplVector3DArray& d_v,
                     const Vector3Di& size, cudaStream_t stream=NULL);

    void velocityToH(float* d_hx, float* d_hy, float* d_hz, const float* d_vx, const float* d_vy, const float* d_vz,
                     float delta, int w, int h, int l, cudaStream_t stream=NULL);
    void velocityToH(cplVector3DArray& d_h, const cplVector3DArray& d_v,
                     float delta, const Vector3Di& size, cudaStream_t stream=NULL);

    void velocityToH(float* d_hx, float* d_hy, float* d_hz, const float* d_vx, const float* d_vy, const float* d_vz,
                     int w, int h, int l,  float spX, float spY, float spZ, cudaStream_t stream=NULL);
    void velocityToH(cplVector3DArray& d_h, const cplVector3DArray& d_v,
                     const Vector3Di& size, const Vector3Df& sp, cudaStream_t stream=NULL);

    void velocityToH_US(float* d_hx, float* d_hy, float* d_hz, const float* d_vx, const float* d_vy, const float* d_vz,
                     int w, int h, int l,  float spX, float spY, float spZ, cudaStream_t stream=NULL);
    void velocityToH_US(cplVector3DArray& d_h, const cplVector3DArray& d_v,
                        const Vector3Di& size, const Vector3Df& sp, cudaStream_t stream=NULL);

    void velocityToH(float* d_hx, float* d_hy, float* d_hz,  const float* d_vx, const float* d_vy, const float* d_vz,
                     float delta, int w, int h, int l, float spX, float spY, float spZ, cudaStream_t stream=NULL);
    void velocityToH(cplVector3DArray& d_h, const cplVector3DArray& d_v,
                     float delta, const Vector3Di& size, const Vector3Df& sp, cudaStream_t stream=NULL);


    void velocityToH_I(float* d_hx, float* d_hy, float* d_hz, int w, int h, int l, cudaStream_t stream=NULL);
    void velocityToH_I(cplVector3DArray& d_h, const Vector3Di& size, cudaStream_t stream=NULL);
    void velocityToH_I(float* d_hx, float* d_hy, float* d_hz, float delta, int w, int h, int l, cudaStream_t stream=NULL);
    void velocityToH_I(cplVector3DArray& d_h, float delta, const Vector3Di& size, cudaStream_t stream=NULL);

    void velocityToH_I(float* d_hx, float* d_hy, float* d_hz,  int w, int h, int l,  float spX, float spY, float spZ, cudaStream_t stream=NULL);
    void velocityToH_I(cplVector3DArray& d_h, const Vector3Di& size, const Vector3Df& sp, cudaStream_t stream=NULL);
    void velocityToH_I(float* d_hx, float* d_hy, float* d_hz, float delta, int w, int h, int l, float spX, float spY, float spZ, cudaStream_t stream=NULL);
    void velocityToH_I(cplVector3DArray& d_h, float delta, const Vector3Di& size, const Vector3Df& sp, cudaStream_t stream=NULL);


    void hToVelocity(float* d_vx, float* d_vy, float* d_vz, const float* d_hx, const float* d_hy, const float* d_hz,
                     int w, int h, int l, cudaStream_t stream=NULL);
    void hToVelocity(float* d_vx, float* d_vy, float* d_vz, const float* d_hx, const float* d_hy, const float* d_hz,
                     int w, int h, int l, float spX, float spY, float spZ, cudaStream_t stream=NULL);
    void hToVelocity(cplVector3DArray& d_v, const cplVector3DArray& d_h,
                     const Vector3Di& size, cudaStream_t stream=NULL);
    void hToVelocity(cplVector3DArray& d_v, const cplVector3DArray& d_h,
                     const Vector3Di& size, const Vector3Df& sp, cudaStream_t stream=NULL);

    void hToVelocity_I(float* d_vx, float* d_vy, float* d_vz,
                     int w, int h, int l, cudaStream_t stream=NULL);
    void hToVelocity_I(float* d_vx, float* d_vy, float* d_vz,
                     int w, int h, int l, float spX, float spY, float spZ, cudaStream_t stream=NULL);
    void hToVelocity_I(cplVector3DArray& d_v, const Vector3Di& size, cudaStream_t stream=NULL);
    void hToVelocity_I(cplVector3DArray& d_v, const Vector3Di& size, const Vector3Df& sp, cudaStream_t stream=NULL);


    void HToDisplacement(float* d_ux, float* d_uy, float* d_uz, const float* d_hx, const float* d_hy, const float* d_hz,
                     int w, int h, int l, cudaStream_t stream=NULL);
    void HToDisplacement(float* d_ux, float* d_uy, float* d_uz, const float* d_hx, const float* d_hy, const float* d_hz,
                     int w, int h, int l, float spX, float spY, float spZ, cudaStream_t stream=NULL);
    void HToDisplacement(cplVector3DArray& d_u, const cplVector3DArray& d_h,
                     const Vector3Di& size, cudaStream_t stream=NULL);
    void HToDisplacement(cplVector3DArray& d_u, const cplVector3DArray& d_h,
                     const Vector3Di& size, const Vector3Df& sp, cudaStream_t stream=NULL);

    void HToDisplacement_I(float* d_ux, float* d_uy, float* d_uz,
                     int w, int h, int l, cudaStream_t stream=NULL);
    void HToDisplacement_I(float* d_ux, float* d_uy, float* d_uz,
                     int w, int h, int l, float spX, float spY, float spZ, cudaStream_t stream=NULL);
    void HToDisplacement_I(cplVector3DArray& d_u, const Vector3Di& size, cudaStream_t stream=NULL);
    void HToDisplacement_I(cplVector3DArray& d_u, const Vector3Di& size, const Vector3Df& sp, cudaStream_t stream=NULL);

    
    /**
     * compose two h fields using trilinear interpolation
     * h(x) = f(g(x))
     */
    void compose(float* d_hx, float* d_hy, float* d_hz,
                 const float* d_fx, const float* d_fy, const float* d_fz,
                 const float* d_gx, const float* d_gy, const float* d_gz,
                 int w, int h, int l, int backgroundStrategy=BACKGROUND_STRATEGY_ZERO, cudaStream_t stream=NULL);
    void compose(cplVector3DArray& d_h, const cplVector3DArray& d_f, const cplVector3DArray& d_g,
                 const Vector3Di& size, int backgroundStrategy=BACKGROUND_STRATEGY_ZERO, cudaStream_t stream=NULL);


    /**
     * compose a velocity and h field to get an hfield
     * h(x) = g(x) + v(g(x))
     */
    void composeVH(float* d_hx, float* d_hy, float* d_hz,
                   const float* d_vx, const float* d_vy, const float* d_vz,
                   const float* d_gx, const float* d_gy, const float* d_gz,
                   int w, int h, int l, BackgroundStrategy bg=BACKGROUND_STRATEGY_ZERO, cudaStream_t stream=NULL);
    
    void composeVH(cplVector3DArray& d_h, const cplVector3DArray& d_v, const cplVector3DArray& d_g,
                   const Vector3Di& size, BackgroundStrategy bg=BACKGROUND_STRATEGY_ZERO, cudaStream_t stream=NULL);
    
    /**
     * compose a velocity and h field to get an hfield
     * h(x) = g(x) - v(g(x))
     */
    void composeVHInv(float* d_hx, float* d_hy, float* d_hz,
                      const float* d_vx, const float* d_vy, const float* d_vz,
                      const float* d_gx, const float* d_gy, const float* d_gz,
                      int w, int h, int l, BackgroundStrategy bg, cudaStream_t stream=NULL);
    
    void composeVHInv(cplVector3DArray& d_h, const cplVector3DArray& d_v, const cplVector3DArray& d_g,
                      const Vector3Di& size, BackgroundStrategy bg, cudaStream_t stream=NULL);

    /**
     * compose a velocity and h field to get an hfield (consider of spacing change)
     * h(x) = g(x) + v(g(x))
     */
    void composeVH(float* d_hx, float* d_hy, float* d_hz,
                   const float* d_vx, const float* d_vy, const float* d_vz,
                   const float* d_gx, const float* d_gy, const float* d_gz,
                   int w, int h, int l,
                   float spX, float spY, float spZ, BackgroundStrategy bg, 
                   cudaStream_t stream=NULL);
    void composeVH(cplVector3DArray& d_h, const cplVector3DArray& d_v, const cplVector3DArray& d_g,
                   const Vector3Di& size, const Vector3Df& sp, BackgroundStrategy bg, 
                   cudaStream_t stream=NULL);
    /**
     * compose a velocity and h field to get an hfield (consider of spacing change)
     * h(x) = g(x) - v(g(x))
     */
    void composeVHInv(float* d_hx, float* d_hy, float* d_hz,
                      const float* d_vx, const float* d_vy, const float* d_vz,
                      const float* d_gx, const float* d_gy, const float* d_gz,
                      int w, int h, int l,
                      float spX, float spY, float spZ, BackgroundStrategy bg, 
                      cudaStream_t stream=NULL);
    void composeVHInv(cplVector3DArray& d_h, const cplVector3DArray& d_v, const cplVector3DArray& d_g,
                      const Vector3Di& size, const Vector3Df& sp, BackgroundStrategy bg, 
                      cudaStream_t stream=NULL);

    /**
     * compose a h field and a velocify field to get an hfield
     * h(x) = g(x+v(x))
     */
    void composeHV(float* d_hx, float* d_hy, float* d_hz,
                   const float* d_gx, const float* d_gy, const float* d_gz,
                   const float* d_vx, const float* d_vy, const float* d_vz,
                   int w, int h, int l,
                   BackgroundStrategy bg = BACKGROUND_STRATEGY_ID, cudaStream_t stream=NULL);
    void composeHV(cplVector3DArray& d_h, const cplVector3DArray& d_g, const cplVector3DArray& d_v,
                   const Vector3Di& size,
                   BackgroundStrategy bg = BACKGROUND_STRATEGY_ID, cudaStream_t stream=NULL);

    /**
     * compose a h field and a inverse velocify field to get an hfield
     * h(x) = g(x-v(x))
     */
    void composeHVInv(float* d_hx, float* d_hy, float* d_hz,
                      const float* d_gx, const float* d_gy, const float* d_gz,
                      const float* d_vx, const float* d_vy, const float* d_vz,
                      int w, int h, int l,
                      BackgroundStrategy bg = BACKGROUND_STRATEGY_ID, cudaStream_t stream=NULL);

    void composeHVInv(cplVector3DArray& d_h, const cplVector3DArray& d_g, const cplVector3DArray& d_v,
                      const Vector3Di& size, 
                      BackgroundStrategy bg = BACKGROUND_STRATEGY_ID, cudaStream_t stream=NULL);

    /**
     * compose a h field and a velocify field to get an hfield
     * h(x) = g(x+v(x))
     */
    void composeHV(float* d_hx, float* d_hy, float* d_hz,
                   const float* d_gx, const float* d_gy, const float* d_gz,
                   const float* d_vx, const float* d_vy, const float* d_vz,
                   int w, int h, int l,
                   float spX, float spY, float spZ,
                   BackgroundStrategy bg = BACKGROUND_STRATEGY_ID, cudaStream_t stream=NULL);
    void composeHV(cplVector3DArray& d_h, const cplVector3DArray& d_g, const cplVector3DArray& d_v,
                   const Vector3Di& size,  const Vector3Df& sp,
                   BackgroundStrategy bg = BACKGROUND_STRATEGY_ID, cudaStream_t stream=NULL);

    /**
     * compose a h field and a inverse velocify field to get an hfield
     * h(x) = g(x-v(x))
     */
    void composeHVInv(float* d_hx, float* d_hy, float* d_hz,
                      const float* d_gx, const float* d_gy, const float* d_gz,
                      const float* d_vx, const float* d_vy, const float* d_vz,
                      int w, int h, int l,
                      float spX, float spY, float spZ,
                      BackgroundStrategy bg = BACKGROUND_STRATEGY_ID, cudaStream_t stream=NULL);
    void composeHVInv(cplVector3DArray& d_h, const cplVector3DArray& d_g, const cplVector3DArray& d_v,
                      const Vector3Di& size, const Vector3Df& sp,
                      BackgroundStrategy bg = BACKGROUND_STRATEGY_ID, cudaStream_t stream=NULL);


    /**
     * compose a h field and a velocify field to get an hfield
     * h(x) = g(x + delta  * v(x))
     */
    void composeHV(float* d_hx, float* d_hy, float* d_hz,
                   const float* d_gx, const float* d_gy, const float* d_gz,
                   const float* d_vx, const float* d_vy, const float* d_vz,
                   float delta,
                   int w, int h, int l,
                   BackgroundStrategy bg = BACKGROUND_STRATEGY_ID, cudaStream_t stream=NULL);
    void composeHV(cplVector3DArray& d_h, const cplVector3DArray& d_g, const cplVector3DArray& d_v,
                   float delta,
                   const Vector3Di& size,
                   BackgroundStrategy bg = BACKGROUND_STRATEGY_ID, cudaStream_t stream=NULL);

    /**
     * compose a h field and a inverse velocify field to get an hfield
     * h(x) = g(x-v(x))
     */
    void composeHVInv(float* d_hx, float* d_hy, float* d_hz,
                      const float* d_gx, const float* d_gy, const float* d_gz,
                      const float* d_vx, const float* d_vy, const float* d_vz,
                      float delta,
                      int w, int h, int l,
                      BackgroundStrategy bg = BACKGROUND_STRATEGY_ID, cudaStream_t stream=NULL);

    void composeHVInv(cplVector3DArray& d_h, const cplVector3DArray& d_g, const cplVector3DArray& d_v,
                      float delta, const Vector3Di& size, 
                      BackgroundStrategy bg = BACKGROUND_STRATEGY_ID, cudaStream_t stream=NULL);

    /**
     * compose a h field and a velocify field to get an hfield
     * h(x) = g(x+v(x))
     */
    void composeHV(float* d_hx, float* d_hy, float* d_hz,
                   const float* d_gx, const float* d_gy, const float* d_gz,
                   const float* d_vx, const float* d_vy, const float* d_vz,
                   float delta,
                   int w, int h, int l,
                   float spX, float spY, float spZ,
                   BackgroundStrategy bg = BACKGROUND_STRATEGY_ID, cudaStream_t stream=NULL);
    void composeHV(cplVector3DArray& d_h, const cplVector3DArray& d_g, const cplVector3DArray& d_v,
                   float delta,
                   const Vector3Di& size,  const Vector3Df& sp,
                   BackgroundStrategy bg = BACKGROUND_STRATEGY_ID, cudaStream_t stream=NULL);

    /**
     * compose a h field and a inverse velocify field to get an hfield
     * h(x) = g(x-v(x))
     */
    void composeHVInv(float* d_hx, float* d_hy, float* d_hz,
                      const float* d_gx, const float* d_gy, const float* d_gz,
                      const float* d_vx, const float* d_vy, const float* d_vz,
                      float delta,
                      int w, int h, int l,
                      float spX, float spY, float spZ,
                      BackgroundStrategy bg = BACKGROUND_STRATEGY_ID, cudaStream_t stream=NULL);
    void composeHVInv(cplVector3DArray& d_h, const cplVector3DArray& d_g, const cplVector3DArray& d_v,
                      float delta,
                      const Vector3Di& size, const Vector3Df& sp,
                      BackgroundStrategy bg = BACKGROUND_STRATEGY_ID, cudaStream_t stream=NULL);
    /**
     * precompose h field with translation
     * creating h(x) = f(x + t)
     */
    void preComposeTranslation(float* d_hx, float* d_hy, float* d_hz,
                               const float* d_fx, const float* d_fy, const float* d_fz,
                               float tx, float ty, float tz,
                               int w, int h, int l, cudaStream_t stream=NULL);

    void preComposeTranslation(cplVector3DArray& d_h,  const cplVector3DArray& d_f,
                               const Vector3Df& t,
                               const Vector3Di& size, cudaStream_t stream=NULL);


  /**
   * approximate the inverse of an incremental h field using according
   * to the following derivation
   *
   * hInv(x0) = x0 + d
   * x0 = h(x0 + d)
   * x0 = h(x0) + d * order zero expansion
   * d  = x0 - h(x0)
   *
   * hInv(x0) = x0 + x0 - h(x0)
   *
   */
	void computeInverseZerothOrder(float* d_hInvx, float* d_hInvy, float* d_hInvz,
                                   const float* d_hx, const float* d_hy, const float* d_hz,
                                   int w, int h, int l, cudaStream_t stream=NULL);
    
    void computeInverseZerothOrder(cplVector3DArray& d_hInv,
                                   const cplVector3DArray& d_h,
                                   const Vector3Di& size, cudaStream_t stream=NULL);
    /**
     * apply hField to an image
     * defImage(x) = image(h(x))
     */
    void apply(float* d_o, const float* d_i,
               const float* d_hx, const float* d_hy, const float* d_hz,
               int w, int h, int l, cudaStream_t stream=NULL);
    
    void apply(float* d_o, const float* d_i,
               const cplVector3DArray& d_h,
               const Vector3Di& size, cudaStream_t stream=NULL);

    /**
	 * apply uField to an image
	 * defImage(x) = image(x+u(x))
	 *
	 */
    void applyU(float* d_o, const float* d_i,
                const float* d_ux, const float* d_uy, const float* d_uz,
                int sizeX, int sizeY, int sizeZ, cudaStream_t stream=NULL);
    void applyU(float* d_o, const float* d_i,
                const cplVector3DArray& d_u,
                const Vector3Di& size, cudaStream_t stream=NULL);
    /**
	 * apply uField to an image
	 * defImage(x) = image(x-u(x))
	 *
	 */
    void applyUInv(float* d_o, const float* d_i,
                   const float* d_ux, const float* d_uy, const float* d_uz,
                   int sizeX, int sizeY, int sizeZ, cudaStream_t stream=NULL);
    void applyUInv(float* d_o, const float* d_i,
                   const cplVector3DArray& d_u,
                   const Vector3Di& size, cudaStream_t stream=NULL);

	/**
	 * apply uField to an image
	 * defImage(x) = image(x+u(x))
	 *
	 */
    void applyU(float* d_o, const float* d_i,
                const float* d_ux, const float* d_uy, const float* d_uz,
                int sizeX, int sizeY, int sizeZ,
                float spX, float spY, float spZ, cudaStream_t stream=NULL);
    void applyU(float* d_o, const float* d_i,
                const cplVector3DArray& d_u,
                const Vector3Di& size, const Vector3Df& sp, cudaStream_t stream=NULL);
    
    /**
	 * apply uField to an image
	 * defImage(x) = image(x-u(x))
	 *
	 */
    void applyUInv(float* d_o, const float* d_i,
                   const float* d_ux, const float* d_uy, const float* d_uz, 
                   int sizeX, int sizeY, int sizeZ,
                   float spX, float spY, float spZ, cudaStream_t stream=NULL);
    void applyUInv(float* d_o, const float* d_i,
                   const cplVector3DArray& d_u,
                   const Vector3Di& size, const Vector3Df& sp, cudaStream_t stream=NULL);

    /**
	 * apply uField to an image
	 * defImage(x) = image(x+delta*u(x))
	 *
	 */
    void applyU(float* d_o, const float* d_i,
                const float* d_ux, const float* d_uy, const float* d_uz,  float delta,
                int sizeX, int sizeY, int sizeZ, cudaStream_t stream=NULL);
    void applyU(float* d_o, const float* d_i,
                const cplVector3DArray& d_u,  float delta,
                const Vector3Di& size, cudaStream_t stream=NULL);

    /**
	 * apply uField to an image
	 * defImage(x) = image(x-delta*u(x))
	 *
	 */

    void applyUInv(float* d_o, const float* d_i,
                const float* d_ux, const float* d_uy, const float* d_uz,  float delta,
                int sizeX, int sizeY, int sizeZ, cudaStream_t stream=NULL);
    void applyUInv(float* d_o, const float* d_i,
                const cplVector3DArray& d_u, float delta,
                const Vector3Di& size, cudaStream_t stream=NULL);

	/**
	 * apply uField to an image
	 * defImage(x) = image(x+delta*u(x))
	 *
	 */
    void applyU(float* d_o, const float* d_i,
                const float* d_ux, const float* d_uy, const float* d_uz, float delta,
                int sizeX, int sizeY, int sizeZ,
                float spX, float spY, float spZ, cudaStream_t stream=NULL);
    void applyU(float* d_o, const float* d_i,
                const cplVector3DArray& d_u, float delta,
                const Vector3Di& size, const Vector3Df& sp, cudaStream_t stream=NULL);
    
    /**
	 * apply uField to an image
	 * defImage(x) = image(x-delta*u(x))
	 *
	 */
    void applyUInv(float* d_o, const float* d_i,
                   const float* d_ux, const float* d_uy, const float* d_uz, float delta,
                   int sizeX, int sizeY, int sizeZ,
                   float spX, float spY, float spZ, cudaStream_t stream=NULL);
    void applyUInv(float* d_o, const float* d_i,
                   const cplVector3DArray& d_u, float delta,
                   const Vector3Di& size, const Vector3Df& sp, cudaStream_t stream=NULL);

  void resample(cplVector3DArray& d_o,
		const cplVector3DArray& d_i,
		const Vector3Di& osize, const Vector3Di& isize,
		BackgroundStrategy bg=BACKGROUND_STRATEGY_CLAMP, bool rs= true, cudaStream_t stream=NULL);
  
  void divergence(float *d_o,
		  const cplVector3DArray& d_h,
		  cplVector3DArray& d_scratch,
		  const Vector3Di& size, 
		  const Vector3Df& sp=Vector3Df(1.f, 1.f, 1.f),
		  bool wrap=false,
		  cudaStream_t stream=NULL);

    void composeHV_tex(float* d_hx, float* d_hy, float* d_hz,
                       const float* d_gx, const float* d_gy, const float* d_gz,
                       const float* d_vx, const float* d_vy, const float* d_vz,
                       float delta, int w, int h, int l,
                       BackgroundStrategy bg=BACKGROUND_STRATEGY_ID, cudaStream_t stream=NULL);
    void composeHV_tex(cplVector3DArray& d_h, const cplVector3DArray& d_g, const cplVector3DArray& d_v,
                       float delta, const Vector3Di& size,
                       BackgroundStrategy bg=BACKGROUND_STRATEGY_ID, cudaStream_t stream=NULL);

    void composeHVInv_tex(float* d_hx, float* d_hy, float* d_hz,
                          const float* d_gx, const float* d_gy, const float* d_gz,
                          const float* d_vx, const float* d_vy, const float* d_vz,
                          float delta, int w, int h, int l,
                          BackgroundStrategy bg=BACKGROUND_STRATEGY_ID, cudaStream_t stream=NULL);
    
    void composeHVInv_tex(cplVector3DArray& d_h, const cplVector3DArray& d_g, const cplVector3DArray& d_v,
                          float delta, const Vector3Di& size,
                          BackgroundStrategy bg=BACKGROUND_STRATEGY_ID, cudaStream_t stream=NULL);

    void composeHV_tex(float* d_hx, float* d_hy, float* d_hz,
                       const float* d_gx, const float* d_gy, const float* d_gz,
                       const float* d_vx, const float* d_vy, const float* d_vz,
                       float delta, int w, int h, int l, float spX, float spY, float spZ,
                       BackgroundStrategy bg=BACKGROUND_STRATEGY_ID, cudaStream_t stream=NULL);
    void composeHV_tex(cplVector3DArray& d_h, const cplVector3DArray& d_g, const cplVector3DArray& d_v,
                       float delta, const Vector3Di& size,  const Vector3Df& sp,
                       BackgroundStrategy bg=BACKGROUND_STRATEGY_ID, cudaStream_t stream=NULL);
    void composeHVInv_tex(float* d_hx, float* d_hy, float* d_hz,
                          const float* d_gx, const float* d_gy, const float* d_gz,
                          const float* d_vx, const float* d_vy, const float* d_vz,
                          float delta, int w, int h, int l, float spX, float spY, float spZ,
                          BackgroundStrategy bg=BACKGROUND_STRATEGY_ID, cudaStream_t stream=NULL);
    void composeHVInv_tex(cplVector3DArray& d_h, const cplVector3DArray& d_g, const cplVector3DArray& d_v,
                          float delta, const Vector3Di& size, const Vector3Df& sp,
                          BackgroundStrategy bg=BACKGROUND_STRATEGY_ID, cudaStream_t stream=NULL);
    
    /**
     * dGX, dGY, dGZ are scratch variables used to hold gradients
     * calculated during jacobian computation
     */
    void jacobianDetHField(float* dJ, cplVector3DArray& dH,
                           cplVector3DArray &dGX, cplVector3DArray &dGY, cplVector3DArray &dGZ,
                           const Vector3Di& size, const Vector3Df& sp, cudaStream_t stream=NULL);

    void jacobian(cplVector3DArray& d_Xg, cplVector3DArray& d_Yg, cplVector3DArray& d_Zg, 
                  cplVector3DArray& d_h,  const Vector3Di& size, const Vector3Df& sp, cudaStream_t stream=NULL);
    
    void jacobianDet(float* d_detJ,
                     cplVector3DArray& d_Xg,cplVector3DArray& d_Yg, cplVector3DArray& d_Zg,
                     int n, cudaStream_t stream=NULL);
    void jacobianDetV(float* d_detJ,
                      cplVector3DArray& d_Xg,cplVector3DArray& d_Yg, cplVector3DArray& d_Zg,
                      int n, cudaStream_t stream=NULL);
    void jacobianDetVInv(float* d_detJ,
                         cplVector3DArray& d_Xg,cplVector3DArray& d_Yg, cplVector3DArray& d_Zg,
                         int n, cudaStream_t stream=NULL);

};// namespace cudaHField3DUtils
#endif // __CUDA_HFIELD3D_UTILS_H__
