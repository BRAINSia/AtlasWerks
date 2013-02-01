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

#include <cudaHField3DUtils.h>
#include <libDefine.h>
#include <cpl.h>

namespace cudaHField3DUtils
{
    void normalize(cplVector3DArray& d_h, const Vector3Di& size, const Vector3Df& sp, cudaStream_t stream){
        normalize( d_h.x,  d_h.y,  d_h.z, size.x, size.y, size.z, sp.x, sp.y, sp.z, stream);
    };

    void restoreSP(cplVector3DArray& d_h, const Vector3Di& size, const Vector3Df& sp, cudaStream_t stream){
        restoreSP( d_h.x,  d_h.y,  d_h.z, size.x, size.y, size.z, sp.x, sp.y, sp.z, stream);
    };


    //////////////////////////////////////////////////////////////////////// 
    // set hfield to identity
    // i.e. h(x) = x
    ////////////////////////////////////////////////////////////////////////
    void setToIdentity(cplVector3DArray& d_h, const Vector3Di& size, cudaStream_t stream){
        setToIdentity(d_h.x, d_h.y, d_h.z, size.x, size.y, size.z, stream);
    };
    void setToIdentity(cplVector3DArray& d_h, const Vector3Di& size, const Vector3Df& sp, cudaStream_t stream){
        setToIdentity(d_h.x, d_h.y, d_h.z, size.x, size.y, size.z, sp.x, sp.y, sp.z, stream);
    };

//////////////////////////////////////////////////////////////////////// 
// Add identity to the current field
// i.e. h(x) = x + h(x)
////////////////////////////////////////////////////////////////////////
    void addIdentity(cplVector3DArray& d_h, const Vector3Di& size, cudaStream_t stream){
        addIdentity(d_h.x, d_h.y, d_h.z, size.x, size.y, size.z, stream);
    }

    void addIdentity(cplVector3DArray& d_h, const Vector3Di& size, const Vector3Df& sp, cudaStream_t stream){
        addIdentity(d_h.x, d_h.y, d_h.z, size.x, size.y, size.z, sp.x, sp.y, sp.z, stream);
    }

//////////////////////////////////////////////////////////////////////// 
// Convert velocity field to hfield
// i.e. h(x) = x + v(x)
////////////////////////////////////////////////////////////////////////
    void velocityToH(cplVector3DArray& d_h, const cplVector3DArray& d_v,
                     const Vector3Di& size, cudaStream_t stream){
        velocityToH(d_h.x, d_h.y, d_h.z,
                    d_v.x, d_v.y, d_v.z,
                    size.x, size.y, size.z, stream);
    }

//////////////////////////////////////////////////////////////////////// 
// Convert velocity field to hfield
// i.e. h(x) = x + v(x) * delta
////////////////////////////////////////////////////////////////////////
    void velocityToH(cplVector3DArray& d_h, const cplVector3DArray& d_v, float delta,
                     const Vector3Di& size, cudaStream_t stream){
        velocityToH(d_h.x, d_h.y, d_h.z,
                    d_v.x, d_v.y, d_v.z, delta,
                    size.x, size.y, size.z, stream);
    }
//////////////////////////////////////////////////////////////////////// 
// Convert velocity field to hfield (spacing adjustment)
// i.e. h(x) = x + v(x) * isp
////////////////////////////////////////////////////////////////////////
    void velocityToH(cplVector3DArray& d_h, const cplVector3DArray& d_v,
                     const Vector3Di& size, const Vector3Df& sp, cudaStream_t stream)
    {
        velocityToH(d_h.x, d_h.y, d_h.z,
                    d_v.x, d_v.y, d_v.z,
                    size.x, size.y, size.z,
                    sp.x, sp.y, sp.z, stream);
    }

    void velocityToH_US(cplVector3DArray& d_h, const cplVector3DArray& d_v,
                        const Vector3Di& size, const Vector3Df& sp, cudaStream_t stream)
    {
        velocityToH_US(d_h.x, d_h.y, d_h.z,
                       d_v.x, d_v.y, d_v.z,
                       size.x, size.y, size.z,
                       sp.x, sp.y, sp.z, stream);
    }
    
//////////////////////////////////////////////////////////////////////// 
// Convert velocity field to hfield (spacing adjustment)
// i.e. h(x) = x + v(x) * delta * isp
////////////////////////////////////////////////////////////////////////
    void velocityToH(cplVector3DArray& d_h, const cplVector3DArray& d_v,
                     float delta, const Vector3Di& size, const Vector3Df& sp, cudaStream_t stream)
    {
        velocityToH(d_h.x, d_h.y, d_h.z,
                    d_v.x, d_v.y, d_v.z, delta,
                    size.x, size.y, size.z,
                    sp.x  , sp.y  , sp.z, stream);
    }

//////////////////////////////////////////////////////////////////////// 
// Convert velocity field to hfield (inplace)
// i.e. h(x) = x + h(x) * delta
////////////////////////////////////////////////////////////////////////
    void velocityToH_I(cplVector3DArray& d_h, const Vector3Di& size, cudaStream_t stream){
        velocityToH_I(d_h.x, d_h.y, d_h.z, size.x, size.y, size.z, stream);
    }

    void velocityToH_I(cplVector3DArray& d_h, float delta, const Vector3Di& size, cudaStream_t stream){
        velocityToH_I(d_h.x, d_h.y, d_h.z, delta, size.x, size.y, size.z, stream);
    }

////////////////////////////////////////////////////////////////////////////
// Convert velocity field to hfield (space adjustment - inplace version)
// i.e. h(x) = x + h(x) * delta 
////////////////////////////////////////////////////////////////////////////
    void velocityToH_I(cplVector3DArray& d_h, const Vector3Di& size, const Vector3Df& sp, cudaStream_t stream){
        velocityToH_I( d_h.x , d_h.y , d_h.z ,
                      size.x, size.y, size.z,
                        sp.x  , sp.y  , sp.z  , stream);
    }

    void velocityToH_I(cplVector3DArray& d_h, float delta,
                       const Vector3Di& size, const Vector3Df& sp, cudaStream_t stream)
    {
        velocityToH_I(d_h.x , d_h.y , d_h.z , delta,
                      size.x, size.y, size.z,
                      sp.x  , sp.y  , sp.z  , stream);
    }

    void hToVelocity(cplVector3DArray& d_v, const cplVector3DArray& d_h,
                     const Vector3Di& size, cudaStream_t stream)
    {
        hToVelocity(d_v.x, d_v.y, d_v.z,
                    d_h.x, d_h.y, d_h.z, size.x, size.y, size.z, stream);
    }
    void hToVelocity(cplVector3DArray& d_v, const cplVector3DArray& d_h,
                     const Vector3Di& size, const Vector3Df& sp, cudaStream_t stream)
    {
        hToVelocity(d_v.x, d_v.y, d_v.z,
                    d_h.x, d_h.y, d_h.z, size.x, size.y, size.z, sp.x, sp.y, sp.z, stream);
    }
    void hToVelocity_I(cplVector3DArray& d_v,
                       const Vector3Di& size, cudaStream_t stream)
    {
        hToVelocity_I(d_v.x, d_v.y, d_v.z, size.x, size.y, size.z, stream);
    }
    void hToVelocity_I(cplVector3DArray& d_v, 
                       const Vector3Di& size, const Vector3Df& sp, cudaStream_t stream)
    {
        hToVelocity_I(d_v.x, d_v.y, d_v.z, size.x, size.y, size.z, sp.x, sp.y, sp.z, stream);
    }

    ////////////////////////////////////////////////////////////////////////////
    // Hfield to displacement 
    // i.e. u(x) = h(x) - x
    ////////////////////////////////////////////////////////////////////////////
    void HToDisplacement(cplVector3DArray& d_u, const cplVector3DArray& d_h,
                         const Vector3Di& size, cudaStream_t stream)
    {
        HToDisplacement(d_u.x, d_u.y, d_u.z,
                        d_h.x, d_h.y, d_h.z, size.x, size.y, size.z, stream);
    }
    void HToDisplacement(cplVector3DArray& d_u, const cplVector3DArray& d_h,
                         const Vector3Di& size, const Vector3Df& sp, cudaStream_t stream)
    {
        HToDisplacement(d_u.x, d_u.y, d_u.z,
                        d_h.x, d_h.y, d_h.z, size.x, size.y, size.z, sp.x, sp.y, sp.z, stream);
    }
    void HToDisplacement_I(cplVector3DArray& d_u,
                           const Vector3Di& size, cudaStream_t stream)
    {
        HToDisplacement_I(d_u.x, d_u.y, d_u.z, size.x, size.y, size.z, stream);
    }
    void HToDisplacement_I(cplVector3DArray& d_u, 
                           const Vector3Di& size, const Vector3Df& sp, cudaStream_t stream)
    {
        HToDisplacement_I(d_u.x, d_u.y, d_u.z, size.x, size.y, size.z, sp.x, sp.y, sp.z, stream);
    }
    
////////////////////////////////////////////////////////////////////////////
// compose two h fields using trilinear interpolation
// h(x) = f(g(x))
////////////////////////////////////////////////////////////////////////////
    void compose(cplVector3DArray& d_h,
                 const cplVector3DArray& d_f, const cplVector3DArray& d_g,
                 const Vector3Di& size, int backgroundStrategy, cudaStream_t stream)
    {
        compose(d_h.x, d_h.y, d_h.z,
                d_f.x, d_f.y, d_f.z,
                d_g.x, d_g.y, d_g.z,
                size.x, size.y, size.z, backgroundStrategy, stream);
    }

    ////////////////////////////////////////////////////////////////////////////
    // compose a velocity and h field to get an hfield
    // h(x) = g(x) + v(g(x))
    ////////////////////////////////////////////////////////////////////////////
    void composeVH(cplVector3DArray& d_h,
                   const cplVector3DArray& d_v, const cplVector3DArray& d_g,
                   const Vector3Di& size,
                   BackgroundStrategy bg, cudaStream_t stream)
    {
        composeVH(d_h.x, d_h.y, d_h.z,
                  d_v.x, d_v.y, d_v.z,
                  d_g.x, d_g.y, d_g.z,
                  size.x, size.y, size.z, bg, stream);
    }

    ////////////////////////////////////////////////////////////////////////////
    // compose a velocity and h field to get an hfield
    // h(x) = g(x) - v(g(x))
    ////////////////////////////////////////////////////////////////////////////
    void composeVHInv(cplVector3DArray& d_h,
                      const cplVector3DArray& d_v, const cplVector3DArray& d_g,
                      const Vector3Di& size,
                      BackgroundStrategy bg, cudaStream_t stream)
    {
        composeVHInv(d_h.x, d_h.y, d_h.z,
                     d_v.x, d_v.y, d_v.z,
                     d_g.x, d_g.y, d_g.z,
                     size.x, size.y, size.z, bg, stream);
    }

    ////////////////////////////////////////////////////////////////////////////
    // compose a velocity and h field to get an hfield (spacing adjustment)
    // h(x) = g(x) + v(g(x))
    ////////////////////////////////////////////////////////////////////////////
    void composeVH(cplVector3DArray& d_h, 
                   const cplVector3DArray& d_v, const cplVector3DArray& d_g,
                   const Vector3Di& size, const Vector3Df& sp,
                   BackgroundStrategy bg, cudaStream_t stream)
    {
        composeVH(d_h.x, d_h.y, d_h.z,
                  d_v.x, d_v.y, d_v.z,
                  d_g.x, d_g.y, d_g.z,
                  size.x, size.y, size.z,
                  sp.x, sp.y, sp.z, bg, stream);
    }
    ////////////////////////////////////////////////////////////////////////////
    // compose a velocity and h field to get an hfield (spacing adjustment)
    // h(x) = g(x) + v(g(x))
    ////////////////////////////////////////////////////////////////////////////
    void composeVHInv(cplVector3DArray& d_h, 
                      const cplVector3DArray& d_v, const cplVector3DArray& d_g,
                      const Vector3Di& size, const Vector3Df& sp,
                      BackgroundStrategy bg, cudaStream_t stream)
    {
        composeVHInv(d_h.x, d_h.y, d_h.z,
                     d_v.x, d_v.y, d_v.z,
                     d_g.x, d_g.y, d_g.z,
                     size.x, size.y, size.z,
                     sp.x, sp.y, sp.z, bg, stream);
    }

    //
	// compose a h field and a velocify field to get an hfield
	// h(x) = g(x+v(x))
	//
    void composeHV(cplVector3DArray& d_h, const cplVector3DArray& d_g, const cplVector3DArray& d_v,
                   const Vector3Di& size, BackgroundStrategy bg, cudaStream_t stream)
    {
        composeHV(d_h.x, d_h.y, d_h.z,
                  d_g.x, d_g.y, d_g.z,
                  d_v.x, d_v.y, d_v.z,
                  size.x, size.y, size.z, bg, stream);
    }

    //
	// compose a h field and a velocify field to get an hfield
	// h(x) = g(x-v(x))
	//
    void composeHVInv(cplVector3DArray& d_h, const cplVector3DArray& d_g, const cplVector3DArray& d_v,
                      const Vector3Di& size, BackgroundStrategy bg, cudaStream_t stream)
    {
        composeHVInv(d_h.x, d_h.y, d_h.z,
                     d_g.x, d_g.y, d_g.z,
                     d_v.x, d_v.y, d_v.z,
                     size.x, size.y, size.z, bg, stream);
    }
    //
	// compose a h field and a velocify field to get an hfield
	// h(x) = g(x+v(x))
	//
    void composeHV(cplVector3DArray& d_h, const cplVector3DArray& d_g, const cplVector3DArray& d_v,
                   const Vector3Di& size, const Vector3Df& sp,
                   BackgroundStrategy bg, cudaStream_t stream)
    {
        composeHV(d_h.x, d_h.y, d_h.z,
                  d_g.x, d_g.y, d_g.z,
                  d_v.x, d_v.y, d_v.z,
                  size.x, size.y, size.z,
                  sp.x, sp.y, sp.z, bg, stream);
    }
    //
	// compose a h field and a velocify field to get an hfield
	// h(x) = g(x-v(x))
	//
    void composeHVInv(cplVector3DArray& d_h, const cplVector3DArray& d_g, const cplVector3DArray& d_v,
                      const Vector3Di& size, const Vector3Df& sp,
                      BackgroundStrategy bg, cudaStream_t stream)
    {
        composeHVInv(d_h.x, d_h.y, d_h.z,
                     d_g.x, d_g.y, d_g.z,
                     d_v.x, d_v.y, d_v.z,
                     size.x, size.y, size.z,
                     sp.x, sp.y, sp.z, bg, stream);
    }

    //
	// compose a h field and a velocify field to get an hfield
	// h(x) = g(x+v(x))
	//
    void composeHV(cplVector3DArray& d_h, const cplVector3DArray& d_g, const cplVector3DArray& d_v, float delta,
                   const Vector3Di& size, BackgroundStrategy bg, cudaStream_t stream)
    {
        composeHV(d_h.x, d_h.y, d_h.z,
                  d_g.x, d_g.y, d_g.z,
                  d_v.x, d_v.y, d_v.z, delta,
                  size.x, size.y, size.z, bg, stream);
    }

    //
	// compose a h field and a velocify field to get an hfield
	// h(x) = g(x-v(x))
	//
    void composeHVInv(cplVector3DArray& d_h, const cplVector3DArray& d_g, const cplVector3DArray& d_v, float delta,
                      const Vector3Di& size, BackgroundStrategy bg, cudaStream_t stream)
    {
        composeHVInv(d_h.x, d_h.y, d_h.z,
                     d_g.x, d_g.y, d_g.z,
                     d_v.x, d_v.y, d_v.z, delta,
                     size.x, size.y, size.z, bg, stream);
    }
    //
	// compose a h field and a velocify field to get an hfield
	// h(x) = g(x+v(x))
	//
    void composeHV(cplVector3DArray& d_h, const cplVector3DArray& d_g, const cplVector3DArray& d_v,float delta,
                   const Vector3Di& size, const Vector3Df& sp,
                   BackgroundStrategy bg, cudaStream_t stream)
    {
        composeHV(d_h.x, d_h.y, d_h.z,
                  d_g.x, d_g.y, d_g.z,
                  d_v.x, d_v.y, d_v.z,  delta,
                  size.x, size.y, size.z,
                  sp.x, sp.y, sp.z, bg, stream);
    }
    //
	// compose a h field and a velocify field to get an hfield
	// h(x) = g(x-v(x))
	//
    void composeHVInv(cplVector3DArray& d_h, const cplVector3DArray& d_g, const cplVector3DArray& d_v,float delta,
                      const Vector3Di& size, const Vector3Df& sp,
                      BackgroundStrategy bg, cudaStream_t stream)
    {
        composeHVInv(d_h.x, d_h.y, d_h.z,
                     d_g.x, d_g.y, d_g.z,
                     d_v.x, d_v.y, d_v.z, delta,
                     size.x, size.y, size.z, 
                     sp.x, sp.y, sp.z, bg, stream);
    }

    ////////////////////////////////////////////////////////////////////////////
	// precompose h field with translation
	// creating h(x) = f(x + t)
	////////////////////////////////////////////////////////////////////////////
    void preComposeTranslation(cplVector3DArray& d_h, const cplVector3DArray& d_f,
                               const Vector3Df& t, const Vector3Di& size, cudaStream_t stream)
    {
        preComposeTranslation(d_h.x, d_h.y, d_h.z,
                              d_f.x, d_f.y, d_f.z,
                              t.x, t.y, t.z,
                              size.x, size.y, size.z, stream);
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
    void computeInverseZerothOrder(cplVector3DArray& d_hInv,
                                   const cplVector3DArray& d_h,
                                   const Vector3Di& size, cudaStream_t stream)
    {
        computeInverseZerothOrder(d_hInv.x, d_hInv.y, d_hInv.z,
                                  d_h.x, d_h.y, d_h.z,
                                  size.x, size.y, size.z, stream);

    }

    /////////////////////////////////////////////////////////////////////////////
	// apply hField to an image
	// defImage(x) = image(h(x))
	/////////////////////////////////////////////////////////////////////////////
    void apply(float* d_o, const float* d_i,
               const cplVector3DArray& d_h,
               const Vector3Di& size, cudaStream_t stream){
        apply(d_o, d_i,
              d_h.x, d_h.y, d_h.z,
              size.x, size.y, size.z, stream);
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
    void applyU(float* d_o, const float* d_i,
                const cplVector3DArray& d_u,
                const Vector3Di& size, cudaStream_t stream)
    {
        applyU(d_o, d_i,
               d_u.x, d_u.y, d_u.z,
               size.x, size.y, size.z, stream);
    }

    /**
	 * apply uField inverse to an image
	 * defImage(x) = image(x-u(x))
	 *
	 * trilerp by default but will use nearest neighbor if flag is set
	 * to true
	 *
	 * NOTE: this does not round for integer types
	 *
	 */
    void applyUInv(float* d_o, const float* d_i,
                const cplVector3DArray& d_u,
                const Vector3Di& size, cudaStream_t stream)
    {
        applyUInv(d_o, d_i,
                  d_u.x, d_u.y, d_u.z,
                  size.x, size.y, size.z, stream);
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
    void applyU(float* d_o, const float* d_i,
                const cplVector3DArray& d_u,
                const Vector3Di& size, const Vector3Df& sp, cudaStream_t stream)
    {
        applyU(d_o, d_i,
               d_u.x, d_u.y, d_u.z,
               size.x, size.y, size.z,
               sp.x, sp.y, sp.z, stream);
    }

    /**
	 * apply uField to an image
	 * defImage(x) = image(x-u(x))
	 *
	 * trilerp by default but will use nearest neighbor if flag is set
	 * to true
	 *
	 * NOTE: this does not round for integer types
	 *
	 */
    void applyUInv(float* d_o, const float* d_i,
                   const cplVector3DArray& d_u,
                   const Vector3Di& size, const Vector3Df& sp, cudaStream_t stream)
    {
        applyUInv(d_o, d_i,
                  d_u.x, d_u.y, d_u.z,
                  size.x, size.y, size.z,
                  sp.x, sp.y, sp.z, stream);
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
    void applyU(float* d_o, const float* d_i,
                const cplVector3DArray& d_u, float delta,
                const Vector3Di& size, cudaStream_t stream)
    {
        applyU(d_o, d_i,
               d_u.x, d_u.y, d_u.z, delta,
               size.x, size.y, size.z, stream);
    }

    /**
	 * apply uField inverse to an image
	 * defImage(x) = image(x-u(x))
	 *
	 * trilerp by default but will use nearest neighbor if flag is set
	 * to true
	 *
	 * NOTE: this does not round for integer types
	 *
	 */
    void applyUInv(float* d_o, const float* d_i,
                   const cplVector3DArray& d_u, float delta, const Vector3Di& size, cudaStream_t stream)
    {
        applyUInv(d_o, d_i,
                  d_u.x, d_u.y, d_u.z, delta,
                  size.x, size.y, size.z, stream);
    }

    /**
	 * apply uField to an image
	 * defImage(x) = image(x+u(x))
	 *
	 * trilerp by default but will use nearest neighbor if flag is set
	 * to true
	 *
	 */
    void applyU(float* d_o, const float* d_i,
                const cplVector3DArray& d_u, float delta,
                const Vector3Di& size, const Vector3Df& sp, cudaStream_t stream)
    {
        applyU(d_o, d_i,
               d_u.x, d_u.y, d_u.z, delta,
               size.x, size.y, size.z,
               sp.x, sp.y, sp.z, stream);
    }

    /**
	 * apply uField to an image
	 * defImage(x) = image(x-u(x))
	 *
	 * trilerp by default but will use nearest neighbor if flag is set
	 * to true
	 *
	 */
    void applyUInv(float* d_o, const float* d_i,
                   const cplVector3DArray& d_u, float delta,
                   const Vector3Di& size, const Vector3Df& sp, cudaStream_t stream)
    {
        applyUInv(d_o, d_i,
                  d_u.x,  d_u.y,  d_u.z, delta,
                  size.x, size.y, size.z,
                  sp.x,   sp.y,   sp.z, stream);
    }

    void composeHV_tex(cplVector3DArray& d_h, const cplVector3DArray& d_g, const cplVector3DArray& d_v,
                       float delta, const Vector3Di& size, BackgroundStrategy bg, cudaStream_t stream)
    {
        composeHV_tex(d_h.x, d_h.y, d_h.z,
                      d_g.x, d_g.y, d_g.z,
                      d_v.x, d_v.y, d_v.z, delta,
                      size.x, size.y, size.z, bg, stream);
    }

    void composeHV_tex(cplVector3DArray& d_h, const cplVector3DArray& d_g, const cplVector3DArray& d_v,
                       float delta, const Vector3Di& size, const Vector3Df& sp, BackgroundStrategy bg, cudaStream_t stream)
    {
        composeHV_tex(d_h.x, d_h.y, d_h.z,
                      d_g.x, d_g.y, d_g.z,
                      d_v.x, d_v.y, d_v.z, delta,
                      size.x, size.y, size.z, sp.x, sp.y, sp.z, bg, stream);
    }

    void composeHVInv_tex(cplVector3DArray& d_h, const cplVector3DArray& d_g, const cplVector3DArray& d_v,
                          float delta, const Vector3Di& size, BackgroundStrategy bg, cudaStream_t stream)
    {
        composeHVInv_tex(d_h.x, d_h.y, d_h.z,
                         d_g.x, d_g.y, d_g.z,
                         d_v.x, d_v.y, d_v.z, delta,
                         size.x, size.y, size.z, bg, stream);
    }

    void composeHVInv_tex(cplVector3DArray& d_h, const cplVector3DArray& d_g, const cplVector3DArray& d_v,
                          float delta, const Vector3Di& size, const Vector3Df& sp, BackgroundStrategy bg, cudaStream_t stream)
    {
        composeHVInv_tex(d_h.x, d_h.y, d_h.z,
                         d_g.x, d_g.y, d_g.z,
                         d_v.x, d_v.y, d_v.z, delta,
                         size.x, size.y, size.z, sp.x, sp.y, sp.z, bg, stream);
    }
};// End of cudaHField3DUtils name space

