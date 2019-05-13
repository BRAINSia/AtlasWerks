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

#ifndef __HFIELD3D_UTILS_H__
#define __HFIELD3D_UTILS_H__

#ifndef SWIG

#include <cassert>
#include <limits>
#include <cmath>
#include <queue>
#include <fftw3.h>

#include "Vector3D.h"
#include "Array3D.h"
#include "Array3DUtils.h"
#include "DataTypes/Image.h"
#include "ROI.h"
#include "Surface.h"
#include "Matrix3D.h"
#include "Array3DIO.h"
#include "AtlasWerksException.h"

#ifdef max
#undef max
#endif

#endif // SWIG

class HField3DUtils
{
public:
  enum VectorBackgroundStrategy { BACKGROUND_STRATEGY_PARTIAL_ID,
				  BACKGROUND_STRATEGY_ID,
				  BACKGROUND_STRATEGY_PARTIAL_ZERO,
				  BACKGROUND_STRATEGY_ZERO,
				  BACKGROUND_STRATEGY_CLAMP,
				  BACKGROUND_STRATEGY_WRAP };

  /**
   * set hfield to identity
   * i.e. h(x) = x
   *
   * If spacing is included, scale by spacing for "world space"
   * identity
   * i.e. h(x) = x*spacing
   *
   * davisb 2003
   */
  template <class T>
  static
  void 
  setToIdentity(Array3D<Vector3D<T> >& hField, Vector3D<double> spacing = Vector3D<double>(1.0, 1.0, 1.0));

#ifdef SWIG
  %template(setToIdentity) setToIdentity<float>;
#endif // SWIG
  
  /**
   * add identity to current field
   * i.e. h(x) = x + h(x)
   *
   * jsp 2009
   */
  template <class T>
  static
  void 
  addIdentity(Array3D<Vector3D<T> >& hField);

#ifdef SWIG
  %template(addIdentity) addIdentity< float >;
#endif // SWIG

  /**
   * convert a velocity field to an h field
   * h(x) = x + v(x) * delta
   *
   * bcd 2004
   */
  template <class T>
  static
  void 
  velocityToH(Array3D<Vector3D<T> >& hField, const T& delta);

  /**
   * convert a world-unit velocity field to a voxel-space h field
   * h(x) = x + v(x) / spacing
   *
   * jsp 2009
   */
  template <class T>
  static
  void 
  velocityToH(Array3D<Vector3D<T> >& hField, 
	      const Vector3D<double>& spacing = Vector3D<double>(1.0, 1.0, 1.0));

#ifdef SWIG
   %template(velocityToH) velocityToH< float >;
#endif // SWIG

  /**
   * convert a voxel-space h field to a world-space velocity field 
   * v(x) = (h(x) - x) * spacing
   *
   * jsp 2009
   */
  template <class T>
  static
  void 
  hToVelocity(Array3D<Vector3D<T> >& hField, 
	      const Vector3D<double>& spacing = Vector3D<T>(1.0, 1.0, 1.0));

#ifdef SWIG
   %template(hToVelocity) hToVelocity< float >;
#endif // SWIG

  /**
   * convert an h field to a displacement
   * u(x) = h(x) - x
   *
   * bcd 2004
   */
  template <class T>
  static
  void 
  HtoDisplacement(Array3D<Vector3D<T> >& hField);

#ifdef SWIG
   %template(HtoDisplacement) HtoDisplacement< float >;
#endif // SWIG

  /**
   * Computes the Lie group exponential map of a vector field.  This
   * is just a stationary flow along a single vector field.  This uses
   * the iterative scaling and squaring (ISS) method.  So specify time
   * to flow, along with n=log2(number of time steps).  For more
   * details see Arsigny2006 or Ashburner2007.
   *
   * jdh 2010
   */
  template <class T>
  static
  void
  GroupExponential(const Array3D<Vector3D<T> >& vField, double t, unsigned int n, Array3D<Vector3D<T> >& h,
	  VectorBackgroundStrategy backgroundStrategy=BACKGROUND_STRATEGY_ID);

#ifdef SWIG
   %template(GroupExponential) GroupExponential< float >; 
#endif //SWIG

  /**
   * Test to see if this is an hField or displacement field -- just
   * computes norm(vf)^2 and norm(vf-x)^2.  If norm(vf)^2 is smaller, assume
   * this is a displacement field.  If norm(vf-x)^2 is smaller, assume
   * this is an hField.  Not foolproof, but should work in most
   * circumstances.
   *
   * jsp 2009
   */
  template <class T>
  static
  bool
  IsHField(Array3D<Vector3D<T> >& vf);

#ifdef SWIG
   %template(IsHField) IsHField< float >;
#endif // SWIG

  template <class T>
  static 
  inline
  Vector3D<T>  
  sumTest(Array3D<Vector3D<T> >& hField);

  /**
   * Compute the dot product of two vector fields, appropriately scaled by spacing
   */
  template <class T>
  static 
  double 
  l2DotProd(const Array3D<Vector3D<T> > &v1, 
	    const Array3D<Vector3D<T> > &v2, 
	    const Vector3D<double> spacing = Vector3D<double>(1.0, 1.0, 1.0));

#ifdef SWIG
   %template(l2DotProd) l2DotProd< float >;
#endif // SWIG


  /**
   * Compute the voxel-wise dot product of two vector fields, appropriately scaled by spacing
   */
  template <class T>
  static 
  void 
  pointwiseL2DotProd(const Array3D<Vector3D<T> > &v1, 
	    const Array3D<Vector3D<T> > &v2, 
		     Array3D<T>& dotProdI, 
	    const Vector3D<double> spacing = Vector3D<double>(1.0, 1.0, 1.0));



  /**
   * Initialize with an affine transform
   * h(x) = A*Id
   *
   * jsp 2010
   */
  template <class T>
  static 
  void
  initializeFromAffine(Array3D<Vector3D<T> >&h, 
		       const AffineTransform3D<T> &aff,
		       const Vector3D<double> origin=Vector3D<double>(0.f,0.f,0.f),
		       const Vector3D<double> spacing=Vector3D<double>(1.f,1.f,1.f));
  
  /**
   * Initialize with the inverse of the given affine transform
   * h(x) = A^(-1)*Id
   *
   * jsp 2010
   */
  template <class T>
  static 
  void
  initializeFromAffineInv(Array3D<Vector3D<T> >&h, 
			  const AffineTransform3D<T> &aff,
			  const Vector3D<double> origin=Vector3D<double>(0.f,0.f,0.f),
			  const Vector3D<double> spacing=Vector3D<double>(1.f,1.f,1.f));
  
  /**
   * Initialize with an affine transform
   * h(x) = A*Id
   *
   * jsp 2010
   */
  template <class T>
  static 
  void
  initializeFromAffine(Array3D<Vector3D<T> >&h, 
		       const AffineTransform3D<T> &aff,
		       bool invertAff,
		       const Vector3D<double> origin=Vector3D<double>(0.f,0.f,0.f),
		       const Vector3D<double> spacing=Vector3D<double>(1.f,1.f,1.f));
  
#ifdef SWIG
   %template(initializeFromAffine) initializeFromAffine< float >;
#endif // SWIG

  /**
   * compose two h fields using trilinear interpolation
   * h(x) = f(g(x))
   *
   * davisb 2003
   */
  template <class T>
  static 
  void
  compose(const Array3D<Vector3D<T> >& f,
          const Array3D<Vector3D<T> >& g,
          Array3D<Vector3D<T> >& h,
	  VectorBackgroundStrategy backgroundStrategy=BACKGROUND_STRATEGY_ID);

  /**
   * compose two h fields with different origins/spacings, sampling at
   * the size/origin/spacing of the output hField (h)
   * h(x) = f(g(x))
   *
   * davisb 2003
   */
  template <class T>
  static 
  void
  compose(const Array3D<Vector3D<T> >& f,
	  const Vector3D<double> fOrigin,
	  const Vector3D<double> fSpacing,
          const Array3D<Vector3D<T> >& g,
	  const Vector3D<double> gOrigin,
	  const Vector3D<double> gSpacing,
          Array3D<Vector3D<T> >& h,
	  const Vector3D<double> hOrigin,
	  const Vector3D<double> hSpacing,
	  VectorBackgroundStrategy backgroundStrategy=BACKGROUND_STRATEGY_ID);

#ifdef SWIG
   %template(compose) compose< float >;
#endif // SWIG

  /**
   * compose a velocity and h field to get an hfield
   * h(x) = g(x) + v(g(x))
   *
   * Fixed 9-15-08: add g(x) + v(g(x)) instead of x + v(g(x)) --JDH
   *
   * davisb 2007
   */
  template <class T>
  static 
  void
  composeVH(const Array3D<Vector3D<T> >& v,
            const Array3D<Vector3D<T> >& g,
            Array3D<Vector3D<T> >& h,
	    const Vector3D<double>& spacing = Vector3D<double>(1.0,1.0,1.0),
	    VectorBackgroundStrategy backgroundStrategy = BACKGROUND_STRATEGY_ZERO);

#ifdef SWIG
   %template(composeVH) composeVH< float >;
#endif // SWIG

  /**
   * invert and compose velocity with h field to get an hfield
   * h(x) = g(x) - v(g(x))
   *
   * jsp 2010
   */
  template <class T>
  static 
  void
  composeVHInv(const Array3D<Vector3D<T> >& v,
	       const Array3D<Vector3D<T> >& g,
	       Array3D<Vector3D<T> >& h,
	       const Vector3D<double>& spacing = Vector3D<double>(1.0,1.0,1.0),
	       VectorBackgroundStrategy backgroundStrategy = BACKGROUND_STRATEGY_ZERO);

#ifdef SWIG
   %template(composeVHInv) composeVHInv< float >;
#endif // SWIG

  /**
   * compose a h field and a velocify field to get an hfield
   * h(x) = g(x+v(x))
   *
   * davisb 2007
   */
  template <class T>
  static 
  void
  composeHV(const Array3D<Vector3D<T> >& g,
            const Array3D<Vector3D<T> >& v,
            Array3D<Vector3D<T> >& h,
	    const Vector3D<double>& spacing = Vector3D<double>(1.0,1.0,1.0),
	    VectorBackgroundStrategy backgroundStrategy = BACKGROUND_STRATEGY_ID);

#ifdef SWIG
   %template(composeHV) composeHV< float >;
#endif // SWIG

  /**
   * compose a h field and a velocify field to get an hfield
   * h(x) = g(x-v(x))
   *
   * davisb 2007
   */
  template <class T>
  static 
  void
  composeHVInv(const Array3D<Vector3D<T> >& g,
	       const Array3D<Vector3D<T> >& v,
	       Array3D<Vector3D<T> >& h,
	       const Vector3D<double>& spacing = Vector3D<double>(1.0,1.0,1.0),
	       VectorBackgroundStrategy backgroundStrategy = BACKGROUND_STRATEGY_ID);

#ifdef SWIG
   %template(composeHVInv) composeHVInv< float >;
#endif // SWIG

  /**
   * iteratively compute \phi_{t+1,0} given \phi{t,0}, \phi_{0,t}, 
   * and v_t
   *
   * jsp 2010
   */
  template <class T>
  static 
  void
  composeHVInvIterative(const Array3D<Vector3D<T> >& phi_t0,
			const Array3D<Vector3D<T> >& phi_0t,
			const Array3D<Vector3D<T> >& v_t,
			Array3D<Vector3D<T> >& h,
			unsigned int nIters = 5,
			const Vector3D<double>& spacing = Vector3D<double>(1.0,1.0,1.0),
			VectorBackgroundStrategy backgroundStrategy = BACKGROUND_STRATEGY_PARTIAL_ID,
			bool debug=false);

#ifdef SWIG
   %template(composeHVInvIterative) composeHVInvIterative< float >;
#endif // SWIG

  /**
   * compose h field with translation
   * creating h(x) = f(x) + t
   *
   * davisb 2003
   */
  template <class T>
  static
  void
  composeTranslation(const Array3D<Vector3D<T> >& f,
		     const Vector3D<T>& t,
		     Array3D<Vector3D<T> >& h);

  /**
   * compose h field with translation
   * creating h(x) = f(x) + t
   *
   * davisb 2003
   */
  template <class T>
  static
  void
  composeTranslation(const Array3D<Vector3D<T> >& f,
		     const Vector3D<T>& t,
		     Array3D<Vector3D<T> >& h,
                     const ROI<int,unsigned int>& roi);

#ifdef SWIG
   %template(composeTranslation) composeTranslation< float >;
#endif // SWIG

  /**
   * precompose h field with translation
   * creating h(x) = f(x + t)
   *
   * davisb 2003
   */
  template <class T>
  static
  void
  preComposeTranslation(const Array3D<Vector3D<T> >& f,
			const Vector3D<T>& t,
			Array3D<Vector3D<T> >& h);

#ifdef SWIG
   %template(preComposeTranslation) preComposeTranslation< float >;
#endif // SWIG

  /**  
   *  approximate the inverse of an incremental h field using according
   *  to the following derivation
   *  
   *  hInv(x0) = x0 + d
   *  x0 = h(x0 + d)  
   *  x0 = h(x0) + d // order zero expansion
   *  d  = x0 - h(x0)
   *  
   *  hInv(x0) = x0 + x0 - h(x0)
   *  
   *  
   *  bcd 2004
   */
  template <class T>
  static
  void
  computeInverseZerothOrder(const Array3D<Vector3D<T> >& h,
			    Array3D<Vector3D<T> >& hinv);

#ifdef SWIG
   %template(computeInverseZerothOrder) computeInverseZerothOrder< float >;
#endif // SWIG

  template <class T>
  static
  void
  computeInverseConsistencyError(const Array3D<Vector3D<T> >& h,
				 const Array3D<Vector3D<T> >& hinv,
				 double& minError, double& maxError,
				 double& meanError, double& stdDevError);

  template <class T>
  static
  void
  computeInverseConsistencyError(const Array3D<Vector3D<T> >& h,
				 const Array3D<Vector3D<T> >& hinv,
				 double& hhinvMinError, 
				 double& hhinvMaxError,
				 double& hhinvMeanError, 
				 double& hhinvStdDevError,
				 double& hinvhMinError, 
				 double& hinvhMaxError,
				 double& hinvhMeanError, 
				 double& hinvhStdDevError);

  template <class T>
  static
  void
  reportInverseConsistencyError(const Array3D<Vector3D<T> >& h,
				const Array3D<Vector3D<T> >& hinv);

#ifdef SWIG
   %template(reportInverseConsistencyError) reportInverseConsistencyError< float >;
#endif // SWIG

  /**
   * apply hField to an image 
   * defImage(x) = image(h(x))
   *
   * trilerp by default but will use nearest neighbor if flag is set
   * to true.  Output image is at the resolution of the defImage
   * passed in.
   *
   * NOTE: this does not round for integer types
   *
   * davisb 2003
   */
  template <class T, class U, 
	    Array3DUtils::ScalarBackgroundStrategyT BackgroundStrategy,
	    Array3DUtils::InterpT InterpMethod>
  static
  void
  apply(const Array3D<T>& image,
	const Array3D<Vector3D<U> >& hField,
	Array3D<T>& defImage,
	const T& background = 0);

  template <class T, class U>
  static
  void
  apply(const Array3D<T>& image,
	const Array3D<Vector3D<U> >& hField,
	Array3D<T>& defImage,
	const T& background = 0)
  {
    apply<T,U,
      DEFAULT_SCALAR_BACKGROUND_STRATEGY,
      DEFAULT_SCALAR_INTERP>
      (image, hField, defImage, background);
  }

  // Apply, but use BACKGROUND_STRATEGY_WRAP when lerping
  // jdh 2011
  template <class T, class U, 
	    Array3DUtils::ScalarBackgroundStrategyT BackgroundStrategy,
	    Array3DUtils::InterpT InterpMethod>
  static
  void
  applyPeriodic(const Array3D<T>& image,
	const Array3D<Vector3D<U> >& hField,
	Array3D<T>& defImage);

  template <class T, class U>
  static
  void
  applyPeriodic(const Array3D<T>& image,
	const Array3D<Vector3D<U> >& hField,
	Array3D<T>& defImage)

  {
    applyPeriodic<T,U,
      DEFAULT_SCALAR_BACKGROUND_STRATEGY,
      DEFAULT_SCALAR_INTERP>
      (image, hField, defImage);
  }

  /**
   * apply hField to the roi of an image 
   * defImage(x) = image(h(x))
   *
   * trilerp by default but will use nearest neighbor if flag is set
   * to true
   *
   * NOTE: this does not round for integer types
   *
   * davisb 2004
   */
  template <class T, class U, 
	    Array3DUtils::ScalarBackgroundStrategyT BackgroundStrategy,
	    Array3DUtils::InterpT InterpMethod>
  static
  void
  apply(const Array3D<T>& image,
	const Array3D<Vector3D<U> >& hField,
	Array3D<T>& defImage,
	int hFieldStartX,
	int hFieldStartY,
	int hFieldStartZ,
	const T& background = 0);

  template <class T, class U>
  static
  void
  apply(const Array3D<T>& image,
	const Array3D<Vector3D<U> >& hField,
	Array3D<T>& defImage,
	int hFieldStartX,
	int hFieldStartY,
	int hFieldStartZ,
	const T& background = 0)

  {
    apply<T,U,
      DEFAULT_SCALAR_BACKGROUND_STRATEGY,
      DEFAULT_SCALAR_INTERP>
      (image, hField, defImage, hFieldStartX, hFieldStartY, hFieldStartZ, background);
  }

  /**  
   *  Apply hField to an roi of an image, taking origin and spacing
   *  into account.  This separately considers the origin and spacing
   *  of the target image (specified in defImage), the moving image
   *  ('image'), and the hField.  The roi is determined by hFieldOrigin
   *  and hFieldSpacing, rather than an ROI object.  Use this if the
   *  origin and spacing of the image being deformed are different from
   *  those of the images used to generate the hfield.  The entries in
   *  hField are assumed to be given in voxel coordinates, so they must
   *  be scaled and shifted by hFieldSpacing and hFieldOrigin
   *  respectively to get them in physical coordinates.
   *  
   *  defImage(x) = image(h(x))
   *  
   *  trilerp only
   *  
   *  NOTE: this does not round for T and U if they're integer types
   *  
   *  foskey 2004
   */
  template <class T, class U, 
	    Array3DUtils::ScalarBackgroundStrategyT BackgroundStrategy,
	    Array3DUtils::InterpT InterpMethod>
  static
  void 
  applyOldVersion(const Image<T>& image,
		  const Array3D< Vector3D<U> >& hField,
		  Image<T>& defImage,
		  Vector3D<double> hFieldOrigin,
		  Vector3D<double> hFieldSpacing,
		  const T& background = 0);

  template <class T, class U>
  static
  void 
  applyOldVersion(const Image<T>& image,
		  const Array3D< Vector3D<U> >& hField,
		  Image<T>& defImage,
		  Vector3D<double> hFieldOrigin,
		  Vector3D<double> hFieldSpacing,
		  const T& background = 0)
  {
    applyOldVersion<T,U,
      DEFAULT_SCALAR_BACKGROUND_STRATEGY,
      DEFAULT_SCALAR_INTERP>
      (image, hField, defImage, hFieldOrigin, hFieldSpacing, background);
  }

#ifdef SWIG
  %template(applyOldVersion) applyOldVersion< float, float, DEFAULT_SCALAR_BACKGROUND_STRATEGY, DEFAULT_SCALAR_INTERP >;
#endif // SWIG

  /**  
   *  Apply hField to an roi of an image, taking origin and spacing
   *  into account.  The roi is determined by hFieldOrigin and
   *  hFieldSpacing, rather than an ROI object.  The size of the
   *  resulting image is the size of the hField, which equals the size
   *  of the ROI.  Use this if the origin and spacing of the image
   *  being deformed are different from those of the images used to
   *  generate the hfield.  The entries in hField are assumed to be
   *  given in voxel coordinates, so they must be scaled and shifted by
   *  hFieldSpacing and hFieldOrigin respectively to get them in
   *  physical coordinates.
   *  
   *  defImage(x) = image(h(x))
   *  
   *  trilerp only
   *  
   *  NOTE: this does not round for T and U if they're integer types
   *  
   *  foskey 2004
   */  
  template <class T, class U, 
	    Array3DUtils::ScalarBackgroundStrategyT BackgroundStrategy,
	    Array3DUtils::InterpT InterpMethod>
  static
  void 
  apply(const Image<T>& image,
	const Array3D< Vector3D<U> >& hField,
	Image<T>& defImage,
        Vector3D<double> hFieldOrigin,
        Vector3D<double> hFieldSpacing,
	const T& background = 0);

  template <class T, class U>
  static
  void 
  apply(const Image<T>& image,
	const Array3D< Vector3D<U> >& hField,
	Image<T>& defImage,
        Vector3D<double> hFieldOrigin,
        Vector3D<double> hFieldSpacing,
	const T& background = 0)
  {
    apply<T,U,
      DEFAULT_SCALAR_BACKGROUND_STRATEGY,
      DEFAULT_SCALAR_INTERP>
      (image, hField, defImage, hFieldOrigin, hFieldSpacing, background);
  }

#ifdef SWIG
  %template(apply) apply< float, float, DEFAULT_SCALAR_BACKGROUND_STRATEGY, DEFAULT_SCALAR_INTERP >;
#endif // SWIG

  /**  
   *  apply hField to an image by using a mask 
   *  defImage(x) = image(h(x))
   *  
   *  The mask allows you to avoid applying the algorithm on certain
   *  parts of the image, like for example the bone.  When the value of
   *  the mask is false, the displacement is null.
   *  
   *  trilerp by default but will use nearest neighbor if flag is set
   *  to true
   *  
   *  NOTE: this does not round for integer types
   *  
   *  dprigent 2004
   */  
  template <class T, class U, 
	    Array3DUtils::ScalarBackgroundStrategyT BackgroundStrategy,
	    Array3DUtils::InterpT InterpMethod>
  static
  void
  applyWithMask(const Array3D<T>& image,
		Array3D<Vector3D<U> > hField,
		Array3D<T>& defImage,
		Array3D<bool>& mask,
		const T& background = 0);

  template <class T, class U>
  static
  void
  applyWithMask(const Array3D<T>& image,
		Array3D<Vector3D<U> > hField,
		Array3D<T>& defImage,
		Array3D<bool>& mask,
		const T& background = 0)
  {
    applyWithMask<T,U,
      DEFAULT_SCALAR_BACKGROUND_STRATEGY,
      DEFAULT_SCALAR_INTERP>
      (image, hField, defImage, mask, background);
  }
  
#ifdef SWIG
   %template(applyWithMask) applyWithMask< float, float, DEFAULT_SCALAR_BACKGROUND_STRATEGY, DEFAULT_SCALAR_INTERP >;
#endif // SWIG

  /**
   * Compute defImage = image(hField), where defImage has the size,
   * origin, and spacing of image.
   */
  template <class T, class U, 
	    Array3DUtils::ScalarBackgroundStrategyT BackgroundStrategy,
	    Array3DUtils::InterpT InterpMethod>
  static
  void 
  applyAtImageResolution(const Image<T>& image,
			 const Array3D< Vector3D<U> >& hField,
			 Image<T>& defImage,
			 Vector3D<double> hFieldOrigin,
			 Vector3D<double> hFieldSpacing,
			 const T& background = 0);

  template <class T, class U>
  static
  void 
  applyAtImageResolution(const Image<T>& image,
			 const Array3D< Vector3D<U> >& hField,
			 Image<T>& defImage,
			 Vector3D<double> hFieldOrigin,
			 Vector3D<double> hFieldSpacing,
			 const T& background = 0)
  {
    applyAtImageResolution<T,U,
      DEFAULT_SCALAR_BACKGROUND_STRATEGY,
      DEFAULT_SCALAR_INTERP>
      (image, hField, defImage, hFieldOrigin, hFieldSpacing, background);
  }

#ifdef SWIG
  %template(applyAtImageResolution) applyAtImageResolution< float, float, DEFAULT_SCALAR_BACKGROUND_STRATEGY, DEFAULT_SCALAR_INTERP >;
#endif // SWIG
  
  /**
   * Compute defImage = image(hField), where defImage defines the
   * output origin, resoution and spacing.
   */
  template <class T, class U, 
	    Array3DUtils::ScalarBackgroundStrategyT BackgroundStrategy,
	    Array3DUtils::InterpT InterpMethod>
  static
  void 
  applyAtNewResolution(const Image<T>& image,
		       const Array3D< Vector3D<U> >& hField,
		       Image<T>& defImage,
		       Vector3D<double> hFieldOrigin,
		       Vector3D<double> hFieldSpacing,
		       const T& background);

  template <class T, class U>
  static
  void 
  applyAtNewResolution(const Image<T>& image,
		       const Array3D< Vector3D<U> >& hField,
		       Image<T>& defImage,
		       Vector3D<double> hFieldOrigin,
		       Vector3D<double> hFieldSpacing,
		       const T& background)

  {
    applyAtNewResolution<T,U,
      DEFAULT_SCALAR_BACKGROUND_STRATEGY,
      DEFAULT_SCALAR_INTERP>
      (image, hField, defImage, hFieldOrigin, hFieldSpacing, background);
  }
  
#ifdef SWIG
  %template(applyAtNewResolution) applyAtNewResolution< float, float, DEFAULT_SCALAR_BACKGROUND_STRATEGY, DEFAULT_SCALAR_INTERP >;
#endif // SWIG


  /**
   * apply uField to an image 
   * defImage(x) = image(x+u(x))
   *
   * trilerp by default but will use nearest neighbor if flag is set
   * to true
   *
   * NOTE: this does not round for integer types
   *
   * jsp 2009
   */
  template <class T, class U, 
	    Array3DUtils::ScalarBackgroundStrategyT BackgroundStrategy,
	    Array3DUtils::InterpT InterpMethod>
  static
  void
  applyU(const Array3D<T>& image,
	 const Array3D<Vector3D<U> >& uField,
	 Array3D<T>& defImage,
	 const Vector3D<double>& spacing = Vector3D<double>(1.0,1.0,1.0),
	 const T& background = 0);

  template <class T, class U>
  static
  void
  applyU(const Array3D<T>& image,
	 const Array3D<Vector3D<U> >& uField,
	 Array3D<T>& defImage,
	 const Vector3D<double>& spacing = Vector3D<double>(1.0,1.0,1.0),
	 const T& background = 0)
  {
    applyU<T,U,
      DEFAULT_SCALAR_BACKGROUND_STRATEGY,
      DEFAULT_SCALAR_INTERP>
      (image, uField, defImage, spacing, background);
  }

#ifdef SWIG
   %template(applyU) applyU< float, float, DEFAULT_SCALAR_BACKGROUND_STRATEGY, DEFAULT_SCALAR_INTERP >;
#endif // SWIG

  //
  // note: you should not call this function with &image == &defImage
  //
  template <class T, class U>
  static
  void
  forwardApply(const Array3D<T>& image,
               const Array3D<Vector3D<U> >& hField,
               Array3D<T>& defImage,
               int hFieldStartX,
               int hFieldStartY,
               int hFieldStartZ,
               const T& background = 0,
	       bool normalize = true);

  template <class T, class U>
  static
  void
  forwardApply(const Array3D<T>& image,
               const Array3D<Vector3D<U> >& hField,
               Array3D<T>& defImage,
               const T& background = 0,
	       bool normalize = true);

  // Same splatting as in forwardApply, but look up coordinates by
  // wrapping
  // jdh 2011
  template <class T, class U>
  static
  void
  forwardApplyPeriodic(const Array3D<T>& image,
                       const Array3D<Vector3D<U> >& hField,
                       Array3D<T>& defImage,
                       bool normalize = true);


#ifdef SWIG
   %template(forwardApply) forwardApply< float, float >;
#endif // SWIG

  /**  
   *  apply to a region of interest in an image
   *  defImage(x+riox) = image(h(x))
   *  
   *  trilerp by default but will use nearest neighbor
   *  if flag is set
   *  
   *  
   *  I DONT THINK THIS IS WHAT WE WANT!!! bcd
   *  
   */  
  template <class T, class U, 
	    Array3DUtils::ScalarBackgroundStrategyT BackgroundStrategy,
	    Array3DUtils::InterpT InterpMethod>
  static
  void
  applyWithROI(const Array3D<T>& image,
  	       const ROI<int, unsigned int>& roi,
  	       const Array3D<Vector3D<U> >& hField,
  	       Array3D<T>& defImage,
  	       const T& background = 0);

  template <class T, class U>
  static
  void
  applyWithROI(const Array3D<T>& image,
  	       const ROI<int, unsigned int>& roi,
  	       const Array3D<Vector3D<U> >& hField,
  	       Array3D<T>& defImage,
  	       const T& background = 0)
  {
    applyWithROI<T,U,
      DEFAULT_SCALAR_BACKGROUND_STRATEGY,
      DEFAULT_SCALAR_INTERP>
      (image, roi, hField, defImage, background);
  }

#ifdef SWIG
  %template(applyWithROI) applyWithROI< float, float, DEFAULT_SCALAR_BACKGROUND_STRATEGY, DEFAULT_SCALAR_INTERP >;
#endif // SWIG

  /**  
   *  apply to surface
   *  vertexi = h(vertexi)
   *  
   *  assumes that surface is in image index coordinates
   *  
   *  davisb 2003
   */  
  template <class T>
  static
  void
  apply(Surface& surface,
	const Array3D<Vector3D<T> >& h);

#ifdef SWIG
  %template(apply) apply< float >;
#endif // SWIG

  /**  
   *  applyWithROI to surface
   *  vertexi = h(vertexi)
   *  
   *  assumes that surface is in image index
   *  coordinates
   */  
  template <class T>
  static
  void
  applyWithROI(Surface& surface,
               const ROI<int, unsigned int>& roi,
	       const Array3D<Vector3D<T> >& h);

#ifdef SWIG
   %template(applyWithROI) applyWithROI< float >;
#endif // SWIG

  /**  
   *  inverse apply to surface
   *  vertexi = hinv(vertexi)
   *  
   *  assumes that surface is in image index coordinates
   *  
   *  davisb 2004
   */  
  template <class T>
  static
  void
  inverseApply(Surface& surface,
               const Array3D<Vector3D<T> >& h);

  template <class T>
  static
  void
  inverseApply(Surface& surface,
               const Array3D<Vector3D<T> >& h,
               const ROI<int, unsigned int>& roi);
               
  template <class T>
  static
  void
  inverseApply(Surface& surface,
               const Array3D<Vector3D<T> >& h,
               int hFieldStartX,
               int hFieldStartY,
               int hFieldStartZ);


#ifdef SWIG
  %template(inverseApply) inverseApply< float >;
#endif // SWIG

  /**
   *
   * Upsample via sinc interpolation.  Currently using full complex
   * FFT, may want to go to r2c/c2r in the future to save
   * memory. Also, this version only works for hfields with an even
   * number of pixels in each dimension.
   *
   * jsp 2009
   */
  template <class T>
  static
  void
  sincUpsample(const Array3D<Vector3D<T> >& inHField, 
	       Array3D<Vector3D<T> >& outHField,
	       unsigned int factor);

  /**
   *
   * Upsample via sinc interpolation.  Currently using full complex
   * FFT, may want to go to r2c/c2r in the future to save
   * memory. Also, this version only works for hfields with an even
   * number of pixels in each dimension.
   *
   * jsp 2009
   */
  template <class T>
  static
  void
  sincUpsample(const Array3D<Vector3D<T> >& inHField, 
	       Array3D<Vector3D<T> >& outHField,
	       Vector3D<unsigned int> &newSize);

#ifdef SWIG
   %template(sincUpsample) sincUpsample< float >;
#endif // SWIG

  /**
   *
   * Modified version of resample, properly centers image
   *
   * jsp 2009
   */
  template<class U>
  static 
  void 
  resampleNew(const Array3D<Vector3D<U> >& inHField, 
	      Array3D<Vector3D<U> >& outHField,
	      const unsigned int outputSizeX,
	      const unsigned int outputSizeY,
	      const unsigned int outputSizeZ,
	      VectorBackgroundStrategy backgroundStrategy = 
	      BACKGROUND_STRATEGY_CLAMP,
	      bool rescaleVectors = true);


  template<class U>
  static 
  void 
  resampleNew(const Array3D<Vector3D<U> >& inHField, 
	      Array3D<Vector3D<U> >& outHField, 
	      const Vector3D<unsigned int>& outputSize,
	      VectorBackgroundStrategy backgroundStrategy = BACKGROUND_STRATEGY_CLAMP,
	      bool rescaleVectors = true);
  
#ifdef SWIG
   %template(resampleNew) resampleNew< float >;
#endif // SWIG

  /**  
   *  create an h field of another size
   *  outHField becomes inHField scaled to 
   *  outputSizeX x outputSizeY x outputSizeZ
   *  
   *  davisb 2003
   */  
  template<class U>
  static 
  void 
  resample(const Array3D<Vector3D<U> >& inHField, 
	   Array3D<Vector3D<U> >& outHField,
	   const unsigned int outputSizeX,
	   const unsigned int outputSizeY,
	   const unsigned int outputSizeZ,
           VectorBackgroundStrategy backgroundStrategy = BACKGROUND_STRATEGY_ID);

  template<class U>
  static 
  void 
  resample(const Array3D<Vector3D<U> >& inHField, 
	   Array3D<Vector3D<U> >& outHField, 
	   const Vector3D<unsigned int>& outputSize,
           VectorBackgroundStrategy backgroundStrategy = BACKGROUND_STRATEGY_ID);

#ifdef SWIG
   %template(resample) resample< float >;
#endif // SWIG

  /**  
   *  trilerp into h field
   *  
   *  hx, hy, hz gets h(x, y, z)
   *  
   *  this code has been optimized
   *  speed over beauty...
   *  
   *  davisb 2003
   */  
  template <class T>
  static
  void
  trilerp(const Array3D<Vector3D<T> >& h,
	  const T& x, const T& y, const T& z,
	  T& hx, T& hy, T& hz,
	  VectorBackgroundStrategy backgroundStrategy = BACKGROUND_STRATEGY_ID);
  
#ifdef SWIG
  // trilerp added via 'extend' in UtilitiesDatatypes.i
  //   %template(trilerp) trilerp< float >;
#endif // SWIG

  /**  
   *  HField3DUtils::divergence
   *  
   *  Computes the divergence, "Nabla-Dot", of vector field
   *  (Sometimes known as the trace of the Jacobi matrix)
   *  
   *           del h1   del h2   del h3
   *  div h =  ------ + ------ + ------
   *           del x1   del x2   del x3
   *  
   *  Where h(x) = [h1 h2 h3] with x = [x1 x2 x3] and 'del' 
   *  indicates a partial derivative. 
   *  
   *  This is intended to be a diagnostic rather than real-time
   *  method.
   *  
   *  Input
   *    hField          - 3D transformation field
   *  
   *  Input/Output
   *    divergenceImage - 3D scalar divergence image
   *  
   *  Output
   *    <none>
   *  
   *  P Lorenzen (2005)
   */  
  template <class T, class U>
  static
  void
    divergence(const Array3D<Vector3D<U> >& hField, Array3D<T>& divergence, Vector3D<double> spacing = Vector3D<double>(1.0,1.0,1.0),
		       bool wrap=false);

#ifdef SWIG
  %template(divergence) divergence< float, float >;
#endif // SWIG

  /**  
   *  Compute Jacobian but respect spacing
   *  jdh 2008
   */  
  template <class T, class U>
  static
  void
  jacobian(const Array3D<Vector3D<U> >& hField,
	   Array3D<T>& jacobian,
           Vector3D<double> spacing = Vector3D<double>(1.0f,1.0f,1.0f));

#ifdef SWIG
  %template(jacobian) jacobian< float, float >;
#endif // SWIG

  /**
   * Create an Array3D containing the norm of each element in the
   * input hField
   *
   * jsp 2009
   */
  template <class T>
  static
  void
  pointwiseL2Norm(const Array3D<Vector3D<T> >& hField,
		  Array3D<T> &norms);

#ifdef SWIG
  %template(pointwiseL2Norm) pointwiseL2Norm< float >;
#endif // SWIG

  /**
   * compute the minimum and maximum deformation distance.  That is,
   * the min and max of:
   *
   * norm(v(x) - x)
   *
   * for any x
   *
   * bcd 2004
   */
  template <class T>
  static
  void
  minMaxDeformationL2Norm(const Array3D<Vector3D<T> >& hField,
			  double& min, double& max);

#ifdef SWIG
   %template(minMaxDeformationL2Norm) minMaxDeformationL2Norm< float >;
#endif // SWIG

  /**
   * compute the minimum and maximum velocity.  That is, the min and
   * max of norm(v(x)) for any x.
   *
   * bcd 2004
   */
  template <class T>
  static
  void
  minMaxVelocityL2Norm(const Array3D<Vector3D<T> >& hField,
		       double& min, double& max);

#ifdef SWIG
   %template(minMaxVelocityL2Norm) minMaxVelocityL2Norm< float >;
#endif // SWIG

  /**  
   *  experimental, compute jacobian at non-grid point
   *  
   *  bcd 2004
   */  
  template <class T>
  static
  void
  jacobian(const Array3D<Vector3D<T> >& h,
	   const T& x, const T& y, const T& z,
	   double* const J);
  
#ifdef SWIG
  %template(jacobian) jacobian< float >;
#endif // SWIG

  /**  
   *  compute the Jacobian of the transformation at a grid point
   *  
   *  bcd 2004
   */  
  template <class T>
  static
  void
  jacobianAtGridpoint(const Array3D<Vector3D<T> >& h,
		      int x, int y, int z,
		      double* const J);

#ifdef SWIG
   %template(jacobianAtGridpoint) jacobianAtGridpoint< float >;
#endif // SWIG

  /**  
   *  experimental
   *  bcd 2004
   *  dont worry about dxdy = dydx right now
   */  
  template <class T>
  static
  void
  hessianAtGridpoint(const Array3D<Vector3D<T> >& h,
		     int x, int y, int z,
		     double* const H);

#ifdef SWIG
   %template(hessianAtGridpoint) hessianAtGridpoint< float >;
#endif // SWIG

  template <class T>
  static
  bool
  inverseOfPoint(const Array3D<Vector3D<T> >& h,
                 const T& x, const T& y, const T& z,
                 T& hinvx, T& hinvy, T& hinvz,
                 float thresholdDistance = 50.0f);

#ifdef SWIG
   %template(inverseOfPoint) inverseOfPoint< float >;
#endif // SWIG

  template <class T>
  static
  float 
  inverseClosestPoint(const Array3D<Vector3D<T> >& h,
                      const T& x, const T& y, const T& z,
                      int& hinvx, int& hinvy, int& hinvz);

#ifdef SWIG
   %template(inverseClosestPoint) inverseClosestPoint< float >;
#endif // SWIG

  /**  
   *  find hinvx such that h(hinvx) = x.  near guess x0.
   */  
  template <class T>
  static
  bool
  inverseOfPointRefine(const Array3D<Vector3D<T> >& h,
                       const T& x, const T& y, const T& z, 
                       const int& x0, const int& y0, const int& z0,
                       T& hinvx, T& hinvy, T& hinvz);

  /**  The function 'func' should take three coordinates representing a
   *  position in world coordinates, and return a vector representing
   *  an offset (essentially a velocity) in voxel coordinates.  The
   *  function 'fillByFunction' produces an hfield 'h' that corresponds
   *  to the given deformation formula and has the specified origin and
   *  spacing.
   */
  static void
  fillByFunction(Array3D< Vector3D<float> >& h,
                 Vector3D<double> origin,
                 Vector3D<double> spacing,
                 Vector3D<float> (*func)(double, double, double));

};

#ifndef SWIG
#include "HField3DUtils.txx"
#endif // SWIG

#endif // __HFIELD3D_UTILS_H__
