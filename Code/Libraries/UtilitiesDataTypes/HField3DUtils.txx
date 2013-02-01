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


//#include "HField3DUtils.h"
// for safe_mod
#include "Arithmetic.h"

template <class T>
inline
void 
HField3DUtils::
setToIdentity(Array3D<Vector3D<T> >& hField, Vector3D<double> spacing)
{
  Vector3D<unsigned int> size = hField.getSize();
  for (unsigned int z = 0; z < size.z; ++z) {
    for (unsigned int y = 0; y < size.y; ++y) {
      for (unsigned int x = 0; x < size.x; ++x) {
	hField(x, y, z).set(x*((T)spacing.x), 
			    y*((T)spacing.y), 
			    z*((T)spacing.z));
      }
    }
  }
}

template <class T>
inline
void 
HField3DUtils::
addIdentity(Array3D<Vector3D<T> >& hField)
{
  Vector3D<unsigned int> size = hField.getSize();
  for (unsigned int z = 0; z < size.z; ++z) {
    for (unsigned int y = 0; y < size.y; ++y) {
      for (unsigned int x = 0; x < size.x; ++x) {
	hField(x, y, z) = hField(x,y,z) + Vector3D<T>(x, y, z);
      }
    }
  }
}

template <class T>
inline
void 
HField3DUtils::
velocityToH(Array3D<Vector3D<T> >& hField, const T& delta)
{
  Vector3D<unsigned int> size = hField.getSize();
  for (unsigned int z = 0; z < size.z; ++z) {
    for (unsigned int y = 0; y < size.y; ++y) {
      for (unsigned int x = 0; x < size.x; ++x) {
	hField(x, y, z).x = x + hField(x,y,z).x * delta;
	hField(x, y, z).y = y + hField(x,y,z).y * delta;
	hField(x, y, z).z = z + hField(x,y,z).z * delta;
      }
    }
  }
}




template <class T>
inline
void 
HField3DUtils::
velocityToH(Array3D<Vector3D<T> >& hField, 
	    const Vector3D<double>& spacing)
{
  Vector3D<unsigned int> size = hField.getSize();
  for (unsigned int z = 0; z < size.z; ++z) {
    for (unsigned int y = 0; y < size.y; ++y) {
      for (unsigned int x = 0; x < size.x; ++x) {
	hField(x, y, z).x = x + hField(x,y,z).x / spacing.x;
	hField(x, y, z).y = y + hField(x,y,z).y / spacing.y;
	hField(x, y, z).z = z + hField(x,y,z).z / spacing.z;
      }
    }
  }
}

template <class T>
inline
void 
HField3DUtils::
hToVelocity(Array3D<Vector3D<T> >& hField, 
	    const Vector3D<double>& spacing)
{
  Vector3D<unsigned int> size = hField.getSize();
  for (unsigned int z = 0; z < size.z; ++z) {
    for (unsigned int y = 0; y < size.y; ++y) {
      for (unsigned int x = 0; x < size.x; ++x) {
	hField(x, y, z).x = (hField(x,y,z).x - x) * spacing.x;
	hField(x, y, z).y = (hField(x,y,z).y - y) * spacing.y;
	hField(x, y, z).z = (hField(x,y,z).z - z) * spacing.z;
      }
    }
  }
}

template <class T>
inline
void 
HField3DUtils::
HtoDisplacement(Array3D<Vector3D<T> >& hField)
{
  Vector3D<unsigned int> size = hField.getSize();
  for (unsigned int z = 0; z < size.z; ++z) {
    for (unsigned int y = 0; y < size.y; ++y) {
      for (unsigned int x = 0; x < size.x; ++x) {
	hField(x, y, z).x = hField(x,y,z).x - x;
	hField(x, y, z).y = hField(x,y,z).y - y;
	hField(x, y, z).z = hField(x,y,z).z - z;
      }
    }
  }
}

template <class T>
void
HField3DUtils::
GroupExponential(const Array3D<Vector3D<T> >& vField, double t, unsigned int n, Array3D<Vector3D<T> >& h, VectorBackgroundStrategy backgroundStrategy)
{
  // Initialize hshort to vField so we can turn it into an hfield in
  // the next step
  Array3D<Vector3D<T> > hshort(vField);

  // Initialize to t/2^n*vField so that when we square n times we get
  // a flow for time t
  // note: 1<<n=2^n
  HField3DUtils::velocityToH(hshort, static_cast<float>(t/static_cast<double>(1 << n)));

  Array3D<Vector3D<T> > *hcurr = &hshort, *hnext = &h;
  for (unsigned int i = 0; i < n; ++i)
    { // Square hshort
    std::cout << "Composing, step " << i+1 << " of " << n << "..." << std::endl;
    HField3DUtils::compose(*hcurr, *hcurr, *hnext, backgroundStrategy);
    // Swap pointers
    std::swap(hcurr, hnext);
    }

  if (hcurr == &hshort)
    { // We need to be careful here, if we square an even number of
      // times the result ends up back in hshort and we have to copy
      // it over into h
    h.setData(*hcurr);
    }
}

template <class T>
inline
bool
HField3DUtils::
IsHField(Array3D<Vector3D<T> >& vf)
{
  double vFieldNorm = 0.0;
  double hFieldNorm = 0.0;
  Vector3D<unsigned int> size = vf.getSize();
  for (unsigned int z = 0; z < size.z; ++z) {
    for (unsigned int y = 0; y < size.y; ++y) {
      for (unsigned int x = 0; x < size.x; ++x) {
	Vector3D<T> &v = vf(x,y,z);
	vFieldNorm += 
	  (v.x) * (v.x) +
	  (v.y) * (v.y) +
	  (v.z) * (v.z);
	hFieldNorm += 
	  (v.x-x) * (v.x-x) +
	  (v.y-y) * (v.y-y) +
	  (v.z-z) * (v.z-z);
      }
    }
  }
  return (hFieldNorm < vFieldNorm);
}

template <class T>
inline
double 
HField3DUtils::
l2DotProd(const Array3D<Vector3D<T> > &v1, 
	  const Array3D<Vector3D<T> > &v2, 
	  const Vector3D<double> spacing)
{
  Array3D<Vector3D<T> > vf = v1;
  vf.pointwiseMultiplyBy(v2);
  Vector3D<T> sum = Array3DUtils::sumElements(vf);
  return spacing.productOfElements()*(sum.x + sum.y + sum.z);
}

template <class T>
inline
void 
HField3DUtils::
pointwiseL2DotProd(const Array3D<Vector3D<T> > &v1, 
	           const Array3D<Vector3D<T> > &v2, 
		   Array3D<T>& dotProdI, 
	           const Vector3D<double> spacing){
  Vector3D<unsigned int> size = v1.getSize();
  Vector3D<unsigned int> size2 = v2.getSize();

  if (!( (size.x==size2.x) &&  (size.y==size2.y) &&  (size.z==size2.z) )){
    std::cerr << "Size mismatch of two vector fields for computing pointwise inner product";
  }
  dotProdI.resize(size);
  unsigned int numElements = v1.getNumElements();

  for (unsigned int i = 0; i < numElements; i++){
    dotProdI(i) = v1(i).dot(v2(i));
  }
}


template <class T>
inline
void
HField3DUtils::
initializeFromAffine(Array3D<Vector3D<T> >&h, 
		     const AffineTransform3D<T> &aff,
		     const Vector3D<double> origin,
		     const Vector3D<double> spacing)
{
  initializeFromAffine(h, aff, false, origin, spacing);
}

template <class T>
inline
void
HField3DUtils::
initializeFromAffineInv(Array3D<Vector3D<T> >&h, 
			const AffineTransform3D<T> &aff,
			const Vector3D<double> origin,
			const Vector3D<double> spacing)
{
  initializeFromAffine(h, aff, true, origin, spacing);
}

template <class T>
inline
void
HField3DUtils::
initializeFromAffine(Array3D<Vector3D<T> >&h, 
		     const AffineTransform3D<T> &aff_in,
		     bool invertAff,
		     const Vector3D<double> origin,
		     const Vector3D<double> spacing)
{
  Vector3D<unsigned int> size = h.getSize();
    
  AffineTransform3D<T> aff = aff_in;
  if(invertAff){
    if(!aff.invert()){
      throw AtlasWerksException(__FILE__,__LINE__,"Error, could not invert affine transform");
    }
  }

  for (unsigned int z = 0; z < size.z; ++z) 
    {
      for (unsigned int y = 0; y < size.y; ++y) 
	{
	  for (unsigned int x = 0; x < size.x; ++x) 
	    {
	      //itk::Point<float, 3> p, tp, tinvp;
	      Vector3D<float> p,tp;
	      p[0] = x * spacing[0] + origin[0];
	      p[1] = y * spacing[1] + origin[1];
	      p[2] = z * spacing[2] + origin[2];
	      aff.transformVector(p,tp);
		
	      h(x,y,z).set((tp[0] - origin[0]) /spacing[0],
			   (tp[1] - origin[1]) /spacing[1],
			   (tp[2] - origin[2]) /spacing[2]);
	    }
	}
    }
}
  
  
template <class T>
inline
void
HField3DUtils::
compose(const Array3D<Vector3D<T> >& f,
	const Array3D<Vector3D<T> >& g,
	Array3D<Vector3D<T> >& h,
	VectorBackgroundStrategy backgroundStrategy)
{
  Vector3D<unsigned int> size = h.getSize();
  for (unsigned int z = 0; z < size.z; ++z) {
    for (unsigned int y = 0; y < size.y; ++y) {
      for (unsigned int x = 0; x < size.x; ++x) {
	trilerp(f, 
		g(x,y,z).x, g(x,y,z).y, g(x,y,z).z, 
		h(x,y,z).x, h(x,y,z).y, h(x,y,z).z,
		backgroundStrategy);
      }
    }
  }    
}

// h = f(g)
template <class T>
inline
void
HField3DUtils::
compose(const Array3D<Vector3D<T> >& f,
	const Vector3D<double> fOrigin,
	const Vector3D<double> fSpacing,
	const Array3D<Vector3D<T> >& g,
	const Vector3D<double> gOrigin,
	const Vector3D<double> gSpacing,
	Array3D<Vector3D<T> >& h,
	const Vector3D<double> hOrigin,
	const Vector3D<double> hSpacing,
	VectorBackgroundStrategy backgroundStrategy)
{
  Vector3D<double> hSize = h.getSize();

  // scale for moving point from h index space to g index space
  Vector3D<double> HToGScale = hSpacing / gSpacing;
  Vector3D<double> HToGOffset = (hOrigin-gOrigin) / gSpacing;

  // scale for moving point from g index space to f index space 
  Vector3D<double> GToFScale = gSpacing / fSpacing;
  Vector3D<double> GToFOffset = (gOrigin-fOrigin) / fSpacing;
  
  // scale for moving point from f index space back to h index space 
  Vector3D<double> FToHScale = fSpacing / hSpacing;
  Vector3D<double> FToHOffset = (fOrigin-hOrigin) / hSpacing;
  
  for (unsigned int z = 0; z < hSize.z; ++z) {
    for (unsigned int y = 0; y < hSize.y; ++y) {
      for (unsigned int x = 0; x < hSize.x; ++x) {
	// convert from h index space to g index space
	Vector3D<T> gPt(x,y,z);
	gPt = gPt*HToGScale + HToGOffset;
	Vector3D<T> fPt;
	// look up value in g
	trilerp(g,
		gPt.x, gPt.y, gPt.z,
		fPt.x, fPt.y, fPt.z,
		backgroundStrategy);
	// convert from g index space to f index space
	fPt = fPt*GToFScale + GToFOffset;
	// look up value in f, assign to h
	Vector3D<T> hPt;
	trilerp(f,
		fPt.x, fPt.y, fPt.z,
		hPt.x, hPt.y, hPt.z,
		backgroundStrategy);
	// convert back to h index space
	hPt = hPt * FToHScale + FToHOffset;
	h(x,y,z) = hPt;
      }
    }
  }
}

template <class T>
inline
void
HField3DUtils::
composeVH(const Array3D<Vector3D<T> >& v,
	  const Array3D<Vector3D<T> >& g,
	  Array3D<Vector3D<T> >& h,
	  const Vector3D<double>& spacing,
	  VectorBackgroundStrategy backgroundStrategy)
{
  Vector3D<unsigned int> size = h.getSize();
  for (unsigned int z = 0; z < size.z; ++z) {
    for (unsigned int y = 0; y < size.y; ++y) {
      for (unsigned int x = 0; x < size.x; ++x) {
	// compute h(x) = v(g(x))
	trilerp(v, 
		g(x,y,z).x, g(x,y,z).y, g(x,y,z).z, 
		h(x,y,z).x, h(x,y,z).y, h(x,y,z).z,
		backgroundStrategy);
	h(x,y,z).x /= ((T)spacing.x);
	h(x,y,z).y /= ((T)spacing.y);
	h(x,y,z).z /= ((T)spacing.z);
	// compute h(x) = g(x) + v(g(x))
	h(x,y,z).x += g(x,y,z).x;
	h(x,y,z).y += g(x,y,z).y;
	h(x,y,z).z += g(x,y,z).z;         
      }
    }
  }
}

template <class T>
inline
void
HField3DUtils::
composeVHInv(const Array3D<Vector3D<T> >& v,
	     const Array3D<Vector3D<T> >& g,
	     Array3D<Vector3D<T> >& h,
	     const Vector3D<double>& spacing,
	     VectorBackgroundStrategy backgroundStrategy)
{
  Vector3D<unsigned int> size = h.getSize();
  for (unsigned int z = 0; z < size.z; ++z) {
    for (unsigned int y = 0; y < size.y; ++y) {
      for (unsigned int x = 0; x < size.x; ++x) {
	// compute h(x) = v(g(x))
	trilerp(v, 
		g(x,y,z).x, g(x,y,z).y, g(x,y,z).z, 
		h(x,y,z).x, h(x,y,z).y, h(x,y,z).z,
		backgroundStrategy);
	h(x,y,z).x /= -((T)spacing.x);
	h(x,y,z).y /= -((T)spacing.y);
	h(x,y,z).z /= -((T)spacing.z);
	// compute h(x) = g(x) + v(g(x))
	h(x,y,z).x += g(x,y,z).x;
	h(x,y,z).y += g(x,y,z).y;
	h(x,y,z).z += g(x,y,z).z;         
      }
    }
  }
}

template <class T>
inline
void
HField3DUtils::
composeHV(const Array3D<Vector3D<T> >& g,
	  const Array3D<Vector3D<T> >& v,
	  Array3D<Vector3D<T> >& h,
	  const Vector3D<double>& spacing,
	  VectorBackgroundStrategy backgroundStrategy)
{
  Vector3D<unsigned int> size = h.getSize();
  for (unsigned int z = 0; z < size.z; ++z) {
    for (unsigned int y = 0; y < size.y; ++y) {
      for (unsigned int x = 0; x < size.x; ++x) {
	trilerp(g, 
		x+v(x,y,z).x/((T)spacing.x), 
		y+v(x,y,z).y/((T)spacing.y), 
		z+v(x,y,z).z/((T)spacing.z), 
		h(x,y,z).x, h(x,y,z).y, h(x,y,z).z,
		backgroundStrategy);
      }
    }
  }    
}

template <class T>
inline
void
HField3DUtils::
composeHVInv(const Array3D<Vector3D<T> >& g,
	     const Array3D<Vector3D<T> >& v,
	     Array3D<Vector3D<T> >& h,
	     const Vector3D<double>& spacing,
	     VectorBackgroundStrategy backgroundStrategy)
{
  Vector3D<unsigned int> size = h.getSize();
  for (unsigned int z = 0; z < size.z; ++z) {
    for (unsigned int y = 0; y < size.y; ++y) {
      for (unsigned int x = 0; x < size.x; ++x) {
	trilerp(g, 
		x-v(x,y,z).x/((T)spacing.x), 
		y-v(x,y,z).y/((T)spacing.y), 
		z-v(x,y,z).z/((T)spacing.z), 
		h(x,y,z).x, h(x,y,z).y, h(x,y,z).z,
		backgroundStrategy);
      }
    }
  }    
}

template <class T>
inline
void
HField3DUtils::
composeHVInvIterative(const Array3D<Vector3D<T> >& phi_t0,
		      const Array3D<Vector3D<T> >& phi_0t,
		      const Array3D<Vector3D<T> >& v_t,
		      Array3D<Vector3D<T> >& h,
		      unsigned int nIters,
		      const Vector3D<double>& spacing,
		      VectorBackgroundStrategy backgroundStrategy,
		      bool debug)
{
  // compute initial estimate
  Array3D<Vector3D<T> > h_prev(phi_t0.getSize());
  Array3D<Vector3D<T> > scratchV(phi_t0.getSize());
  // bootstrap with old method
  composeHVInv(phi_t0, v_t, h_prev, 
	       spacing, backgroundStrategy);
  h = h_prev;
  if(debug){
    std::cout << "Diff Norm: ";
  }
  // iterate
  for(unsigned int iter=0;iter<nIters;iter++){
    // h = \phi_{0,t} \circ \phi_{t+1,0}
    compose(phi_0t,h_prev,h,
	    backgroundStrategy);
    // scratchV = v_t \circ \phi_{0,t} \circ \phi_{t+1,0}
    compose(v_t,h,scratchV,
	    backgroundStrategy);
    // h = \phi_{t,0}( x - v_t \circ \phi_{0,t} \circ \phi_{t+1,0} )
    composeHVInv(phi_t0,scratchV,h,
		 spacing, backgroundStrategy);
    // compute difference ( for debugging )
    if(debug){
      scratchV = h;
      scratchV.pointwiseSubtract(h_prev);
      std::cout << "[" << iter << ": " << l2DotProd(scratchV, scratchV) << "] ";
    }
    // update as average of this computation and prev. computation
    //       h.pointwiseAdd(h_prev);
    //       h.scale(0.5);
    h_prev = h;
  }
  if(debug){
    std::cout << std::endl;
  }
}

template <class T>
inline
void
HField3DUtils::
composeTranslation(const Array3D<Vector3D<T> >& f,
		   const Vector3D<T>& t,
		   Array3D<Vector3D<T> >& h)
{
  unsigned int numElements = f.getNumElements();
  for (unsigned int i = 0; i < numElements; ++i)
    {
      h(i) = f(i) + t;
    }
}

template <class T>
inline
void
HField3DUtils::
composeTranslation(const Array3D<Vector3D<T> >& f,
		   const Vector3D<T>& t,
		   Array3D<Vector3D<T> >& h,
		   const ROI<int,unsigned int>& roi)
{
  for (int z = roi.getStartZ(); z <= roi.getStopZ(); ++z) {
    for (int y = roi.getStartY(); y <= roi.getStopY(); ++y) {
      for (int x = roi.getStartX(); x <= roi.getStopX(); ++x) {
	h(x,y,z) = f(x,y,z) + t;
      }
    }
  }
}

template <class T>
inline
void
HField3DUtils::
preComposeTranslation(const Array3D<Vector3D<T> >& f,
		      const Vector3D<T>& t,
		      Array3D<Vector3D<T> >& h)
{
  Vector3D<unsigned int> size = f.getSize();
  for (unsigned int z = 0; z < size.z; ++z) {
    for (unsigned int y = 0; y < size.y; ++y) {
      for (unsigned int x = 0; x < size.x; ++x) {
	trilerp(f, 
		x + t.x, y + t.y, z + t.z, 
		h(x, y, z).x, h(x, y, z).y, h(x, y, z).z);
      }
    }
  }
}

template <class T>
inline
void
HField3DUtils::
computeInverseZerothOrder(const Array3D<Vector3D<T> >& h,
			  Array3D<Vector3D<T> >& hinv)
{
  assert(h.getSize() == hinv.getSize());

  Vector3D<unsigned int> size = h.getSize();
  for (unsigned int z = 0; z < size.z; ++z) {    
    for (unsigned int y = 0; y < size.y; ++y) {    
      for (unsigned int x = 0; x < size.x; ++x) {    
	hinv(x,y,z).x = x + x - h(x,y,z).x;
	hinv(x,y,z).y = y + y - h(x,y,z).y;
	hinv(x,y,z).z = z + z - h(x,y,z).z;
      }
    }
  }
}
template <class T>
inline
void
HField3DUtils::
computeInverseConsistencyError(const Array3D<Vector3D<T> >& h,
			       const Array3D<Vector3D<T> >& hinv,
			       double& minError, double& maxError,
			       double& meanError, double& stdDevError)
{
  Vector3D<unsigned int> size = hinv.getSize();
  assert(h.getSize() == hinv.getSize());
  assert(h.getNumElements() > 0);

  // compute hinvh(x) = hinv(h(x))
  Array3D<Vector3D<T> > hinvh(size);
  compose(hinv, h, hinvh);

  // compute statistics on hinvh
  minError  = std::numeric_limits<double>::max();
  maxError  = 0;
  meanError = 0;

  //
  // compute min, max, and mean
  //
  //unsigned int numElements = hinvh.getNumElements();
  unsigned int z;
  for (z = 30; z < size.z-30; ++z) {    
    for (unsigned int y = 30; y < size.y-30; ++y) {    
      for (unsigned int x = 30; x < size.x-30; ++x) {    
	double error = 
	  sqrt((hinvh(x,y,z).x - x) * (hinvh(x,y,z).x - x) +
	       (hinvh(x,y,z).y - y) * (hinvh(x,y,z).y - y) +
	       (hinvh(x,y,z).z - z) * (hinvh(x,y,z).z - z));
	if (error > maxError) maxError = error;
	if (error < minError) minError = error;
	meanError += error;
      }
    }
  }
  meanError /= (size.x-60) * (size.y-60) * (size.z-60);

  //
  // now compute standard deviation
  // 
  stdDevError = 0;
  for (z = 30; z < size.z-30; ++z) {    
    for (unsigned int y = 30; y < size.y-30; ++y) {    
      for (unsigned int x = 30; x < size.x-30; ++x) {    
	double error = 
	  sqrt((hinvh(x,y,z).x - x) * (hinvh(x,y,z).x - x) +
	       (hinvh(x,y,z).y - y) * (hinvh(x,y,z).y - y) +
	       (hinvh(x,y,z).z - z) * (hinvh(x,y,z).z - z));
	stdDevError += (error - meanError) * (error - meanError);
      }
    }
  }
  stdDevError /= ((size.x-60) * (size.y-60) * (size.z-60) - 1);
  stdDevError = sqrt(stdDevError);
}

template <class T>
inline
void
HField3DUtils::
computeInverseConsistencyError(const Array3D<Vector3D<T> >& h,
			       const Array3D<Vector3D<T> >& hinv,
			       double& hhinvMinError, 
			       double& hhinvMaxError,
			       double& hhinvMeanError, 
			       double& hhinvStdDevError,
			       double& hinvhMinError, 
			       double& hinvhMaxError,
			       double& hinvhMeanError, 
			       double& hinvhStdDevError)
{
  computeInverseConsistencyError(hinv, h, hhinvMinError, hhinvMaxError,
				 hhinvMeanError, hhinvStdDevError);
  computeInverseConsistencyError(h, hinv, hinvhMinError, hinvhMaxError,
				 hinvhMeanError, hinvhStdDevError);
}

template <class T>
inline
void
HField3DUtils::
reportInverseConsistencyError(const Array3D<Vector3D<T> >& h,
			      const Array3D<Vector3D<T> >& hinv)
{
  double 
    hhinvMinError    = 0, 
    hhinvMaxError    = 0, 
    hhinvMeanError   = 0, 
    hhinvStdDevError = 0;
  double 
    hinvhMinError    = 0, 
    hinvhMaxError    = 0, 
    hinvhMeanError   = 0, 
    hinvhStdDevError = 0;

  computeInverseConsistencyError(h,
				 hinv,
				 hhinvMinError, 
				 hhinvMaxError, 
				 hhinvMeanError, 
				 hhinvStdDevError,
				 hinvhMinError, 
				 hinvhMaxError,
				 hinvhMeanError,
				 hinvhStdDevError);

  std::cerr << "\t|h(hinv(x)) - x|L2" << std::endl
	    << "\tMin      : " << hhinvMinError << std::endl
	    << "\tMax      : " << hhinvMaxError << std::endl	
	    << "\tMean     : " << hhinvMeanError << std::endl	
	    << "\tSt. Dev. : " << hhinvStdDevError << std::endl	
	    << "\t|hinv(h(x)) - x|L2" << std::endl
	    << "\tMin      : " << hinvhMinError << std::endl
	    << "\tMax      : " << hinvhMaxError << std::endl	
	    << "\tMean     : " << hinvhMeanError << std::endl	
	    << "\tSt. Dev. : " << hinvhStdDevError << std::endl;
}

template <class T, class U, 
	  Array3DUtils::ScalarBackgroundStrategyT BackgroundStrategy,
	  Array3DUtils::InterpT InterpMethod>
inline
void
HField3DUtils::
apply(const Array3D<T>& image,
      const Array3D<Vector3D<U> >& hField,
      Array3D<T>& defImage,
      const T& background)
{
  Vector3D<unsigned int> size = defImage.getSize();
  
  for (unsigned int z = 0; z < size.z; ++z) {
    for (unsigned int y = 0; y < size.y; ++y) {
      for (unsigned int x = 0; x < size.x; ++x) {
	defImage(x, y, z) = 
	  Array3DUtils::
	  interp<T, BackgroundStrategy, InterpMethod>(image,
						      hField(x, y, z),
						      background);
      }
    }
  }
}

template <class T, class U, 
	  Array3DUtils::ScalarBackgroundStrategyT BackgroundStrategy,
	  Array3DUtils::InterpT InterpMethod>
inline
void
HField3DUtils::
applyPeriodic(const Array3D<T>& image,
              const Array3D<Vector3D<U> >& hField,
              Array3D<T>& defImage)
{  
  Vector3D<unsigned int> size = defImage.getSize();

  for (unsigned int z = 0; z < size.z; ++z) {
    for (unsigned int y = 0; y < size.y; ++y) {
      for (unsigned int x = 0; x < size.x; ++x) {
	defImage(x, y, z) = 
	  Array3DUtils::
	  interp<T, BackgroundStrategy, InterpMethod>(image,
						      hField(x, y, z).x,
						      hField(x, y, z).y,
						      hField(x, y, z).z,
						      Array3DUtils::BACKGROUND_STRATEGY_WRAP,
						      defImage(x,y,z));
      }
    }
  } 
}


template <class T, class U, 
	  Array3DUtils::ScalarBackgroundStrategyT BackgroundStrategy,
	  Array3DUtils::InterpT InterpMethod>
inline
void
HField3DUtils::
apply(const Array3D<T>& image,
      const Array3D<Vector3D<U> >& hField,
      Array3D<T>& defImage,
      int hFieldStartX,
      int hFieldStartY,
      int hFieldStartZ,
      const T& background)
{
  Vector3D<unsigned int> imageSize = image.getSize();
  Vector3D<unsigned int> hFieldSize = hField.getSize();

  // copy entire image
  defImage = image;
  
  // ammend in region of interest
  for (unsigned int z = 0; 
       z < hFieldSize.z && z + hFieldStartZ < imageSize.z; ++z) {
    for (unsigned int y = 0; 
	 y < hFieldSize.y && y + hFieldStartY < imageSize.y; ++y) {
      for (unsigned int x = 0; 
	   x < hFieldSize.x && x + hFieldStartX < imageSize.x; ++x) {
	defImage(x+hFieldStartX, y+hFieldStartY, z+hFieldStartZ) = 
	  Array3DUtils::
	  interp<T, BackgroundStrategy, InterpMethod>(image,
						      hField(x, y, z).x + hFieldStartX,
						      hField(x, y, z).y + hFieldStartY,
						      hField(x, y, z).z + hFieldStartZ,
						      background);
      }
    }
  }
}

template <class T, class U, 
	  Array3DUtils::ScalarBackgroundStrategyT BackgroundStrategy,
	  Array3DUtils::InterpT InterpMethod>
inline
void 
HField3DUtils::
applyOldVersion(const Image<T>& image,
		const Array3D< Vector3D<U> >& hField,
		Image<T>& defImage,
		Vector3D<double> hFieldOrigin,
		Vector3D<double> hFieldSpacing,
		const T& background)
{
  Vector3D<unsigned int> defImageSize = defImage.getSize();
  Vector3D<unsigned int> hFieldSize = hField.getSize();

  Vector3D<double> imageScaleHtoI = hFieldSpacing / image.getSpacing();
  Vector3D<double> defImageScaleHtoI = hFieldSpacing / defImage.getSpacing();

  Vector3D<double> imageRoiStart = (hFieldOrigin - image.getOrigin()) /
    image.getSpacing();

  Vector3D<double> defImageRoiStart = (hFieldOrigin - defImage.getOrigin()) /
    defImage.getSpacing();

  Vector3D<double> defImageRoiSize = Vector3D<double>(hFieldSize) *
    defImageScaleHtoI;

  for (unsigned int i = 0; i < 3; ++i) {
    if (defImageRoiStart[i] < 0) defImageRoiStart[i] = 0;
    if (defImageRoiSize[i] + defImageRoiStart[i] + 0.5 > defImageSize[i]) {
      defImageRoiStart[i] = defImageSize[i] - defImageRoiStart[0] - 0.6;
    }
  }

  unsigned int hFieldStartX = (unsigned int)(defImageRoiStart[0] + 0.5);
  unsigned int hFieldStartY = (unsigned int)(defImageRoiStart[1] + 0.5);
  unsigned int hFieldStartZ = (unsigned int)(defImageRoiStart[2] + 0.5);

  unsigned int hFieldSizeX = (unsigned int)(defImageRoiSize[0] + 0.5);
  unsigned int hFieldSizeY = (unsigned int)(defImageRoiSize[1] + 0.5);
  unsigned int hFieldSizeZ = (unsigned int)(defImageRoiSize[2] + 0.5);

  for (unsigned int z = 0; z < hFieldSizeZ; ++z) {
    for (unsigned int y = 0; y < hFieldSizeY; ++y) {
      for (unsigned int x = 0; x < hFieldSizeX; ++x) {

	Vector3D<float> imagePoint = hField(x, y, z) * imageScaleHtoI + 
	  imageRoiStart;

	defImage(x+hFieldStartX, y+hFieldStartY, z+hFieldStartZ) = 
	  static_cast<T>(Array3DUtils::
			 interp<T, BackgroundStrategy, InterpMethod>(image, 
								     imagePoint,
								     background));
	
      }
    }
  }
}

template <class T, class U, 
	  Array3DUtils::ScalarBackgroundStrategyT BackgroundStrategy,
	  Array3DUtils::InterpT InterpMethod>
inline
void 
HField3DUtils::
apply(const Image<T>& image,
      const Array3D< Vector3D<U> >& hField,
      Image<T>& defImage,
      Vector3D<double> hFieldOrigin,
      Vector3D<double> hFieldSpacing,
      const T& background)
{
  Vector3D<unsigned int> hFieldSize = hField.getSize();

  Vector3D<double> scaleHtoI = hFieldSpacing / image.getSpacing();

  Vector3D<double> roiStart = (hFieldOrigin - image.getOrigin()) /
    image.getSpacing();

  defImage.resize(hFieldSize);
  defImage.setOrigin(hFieldOrigin);
  defImage.setSpacing(hFieldSpacing);

  for (unsigned int z = 0; z < hFieldSize.z; ++z) {
    for (unsigned int y = 0; y < hFieldSize.y; ++y) {
      for (unsigned int x = 0; x < hFieldSize.x; ++x) {

	Vector3D<float> imagePoint = hField(x, y, z) * scaleHtoI + 
	  roiStart;

	defImage(x, y, z) = 
	  static_cast<T>(Array3DUtils::
			 interp<T, BackgroundStrategy, InterpMethod>(image,
								     imagePoint,
								     background));
      }
    }
  }
}

template <class T, class U, 
	  Array3DUtils::ScalarBackgroundStrategyT BackgroundStrategy,
	  Array3DUtils::InterpT InterpMethod>
inline
void
HField3DUtils::
applyWithMask(const Array3D<T>& image,
	      Array3D<Vector3D<U> > hField,
	      Array3D<T>& defImage,
	      Array3D<bool>& mask,
	      const T& background)
{
  Vector3D<unsigned int> size = image.getSize();
    
  for (unsigned int z = 0; z < size.z; ++z) {
    for (unsigned int y = 0; y < size.y; ++y) {
      for (unsigned int x = 0; x < size.x; ++x) {
	if(mask(x,y,z))
	  {
	    defImage(x, y, z) = 
	      Array3DUtils::
	      interp<T, BackgroundStrategy, InterpMethod>(image,
							  hField(x, y, z),
							  background);
	  }
	else
	  {
	    hField(x, y, z).x=x;
	    hField(x, y, z).y=y;
	    hField(x, y, z).z=z;
            
	  }
      }
    }
  }
}
  
template <class T, class U, 
	  Array3DUtils::ScalarBackgroundStrategyT BackgroundStrategy,
	  Array3DUtils::InterpT InterpMethod>
inline
void 
HField3DUtils::
applyAtImageResolution(const Image<T>& image,
		       const Array3D< Vector3D<U> >& hField,
		       Image<T>& defImage,
		       Vector3D<double> hFieldOrigin,
		       Vector3D<double> hFieldSpacing,
		       const T& background)
{
  defImage.resize(image.getSize());
  defImage.setOrigin(image.getOrigin());
  defImage.setSpacing(image.getSpacing());
  applyAtNewResolution<T, U, BackgroundStrategy, InterpMethod>(image, hField, defImage, hFieldOrigin, hFieldSpacing, background);
}

template <class T, class U, 
	  Array3DUtils::ScalarBackgroundStrategyT BackgroundStrategy,
	  Array3DUtils::InterpT InterpMethod>
inline
void 
HField3DUtils::
applyAtNewResolution(const Image<T>& image,
		     const Array3D< Vector3D<U> >& hField,
		     Image<T>& defImage,
		     Vector3D<double> hFieldOrigin,
		     Vector3D<double> hFieldSpacing,
		     const T& background)
{
  Vector3D<double> hFieldSize = hField.getSize();

  // output size/origin/spacing from defImage
  Vector3D<double> size = defImage.getSize();
  Vector3D<double> origin = defImage.getOrigin();
  Vector3D<double> spacing = defImage.getSpacing();

  // scale for moving point from defImage index space to HField index space
  Vector3D<double> IToHScale = spacing / hFieldSpacing;
  Vector3D<double> IToHOffset = (origin-hFieldOrigin) / hFieldSpacing;
  
  // scale for moving point from HField index space to initial image index space 
  Vector3D<double> HToIScale = hFieldSpacing / image.getSpacing();
  Vector3D<double> HToIOffset = (hFieldOrigin-image.getOrigin()) / image.getSpacing();
  
  for (unsigned int z = 0; z < size.z; ++z) {
    for (unsigned int y = 0; y < size.y; ++y) {
      for (unsigned int x = 0; x < size.x; ++x) {
	
	Vector3D<U> hFieldPt(x,y,z);
	hFieldPt = hFieldPt*IToHScale + IToHOffset;
	Vector3D<U> imagePt;
	trilerp(hField,
		hFieldPt.x, hFieldPt.y, hFieldPt.z,
		imagePt.x, imagePt.y, imagePt.z);
	imagePt = imagePt*HToIScale + HToIOffset;
	
	defImage(x, y, z) = 
	  static_cast<T>(Array3DUtils::
			 interp<T, BackgroundStrategy, InterpMethod>(image, imagePt, background));
	
      }
    }
  }
}

template <class T, class U, 
	  Array3DUtils::ScalarBackgroundStrategyT BackgroundStrategy,
	  Array3DUtils::InterpT InterpMethod>
inline
void
HField3DUtils::
applyU(const Array3D<T>& image,
       const Array3D<Vector3D<U> >& uField,
       Array3D<T>& defImage,
       const Vector3D<double>& spacing,
       const T& background)
{
  Vector3D<unsigned int> size = defImage.getSize();
  
  for (unsigned int z = 0; z < size.z; ++z) {
    for (unsigned int y = 0; y < size.y; ++y) {
      for (unsigned int x = 0; x < size.x; ++x) {
	Vector3D<U> coord = uField(x,y,z)/spacing + Vector3D<U>(x,y,z);
	defImage(x, y, z) = 
	  Array3DUtils::
	  interp<T, BackgroundStrategy, InterpMethod>(image,coord,background);
      }
    }
  } 
}

template <class T, class U>
inline
void
HField3DUtils::
forwardApply(const Array3D<T>& image,
	     const Array3D<Vector3D<U> >& hField,
	     Array3D<T>& defImage,
	     int hFieldStartX,
	     int hFieldStartY,
	     int hFieldStartZ,
	     const T& background,
	     bool normalize)
{
  Vector3D<unsigned int> imageSize = image.getSize();
  Vector3D<unsigned int> hFieldSize = hField.getSize();
  Array3D<float> count(defImage.getSize());
  count.fill(0);

  // copy entire image
  defImage = image;

  // fill roi with background
  for (unsigned int z = 0; 
       z < hFieldSize.z && z + hFieldStartZ < imageSize.z; ++z) {
    for (unsigned int y = 0; 
	 y < hFieldSize.y && y + hFieldStartY < imageSize.y; ++y) {
      for (unsigned int x = 0; 
	   x < hFieldSize.x && x + hFieldStartX < imageSize.x; ++x) {
	defImage(x+hFieldStartX,y+hFieldStartY,z+hFieldStartZ) = background;
      }
    }
  }

  // trilinear weights
  float w000, w001, w010, w011, w100, w101, w110, w111;

  T ix;
    
  // floor index and residuals
  U hx, hy, hz;
  int fx, fy, fz;
  float rx, ry, rz;

  for (unsigned int z = 0; 
       z < hFieldSize.z && z + hFieldStartZ < imageSize.z; ++z) {
    for (unsigned int y = 0; 
	 y < hFieldSize.y && y + hFieldStartY < imageSize.y; ++y) {
      for (unsigned int x = 0; 
	   x < hFieldSize.x && x + hFieldStartX < imageSize.x; ++x) {

	// get intensity value from source image
	ix = image(x + hFieldStartX,
		   y + hFieldStartY,
		   z + hFieldStartZ);
	  
	// get h field value---where this intensity should go
	hx = hField(x,y,z).x + hFieldStartX;
	hy = hField(x,y,z).y + hFieldStartY;
	hz = hField(x,y,z).z + hFieldStartZ;

	// a fast version of the floor function
	fx = static_cast<int>(hx);
	fy = static_cast<int>(hy);
	fz = static_cast<int>(hz);
	if (hx < 0 && hx != static_cast<int>(hx)) --fx;
	if (hy < 0 && hy != static_cast<int>(hy)) --fy;
	if (hz < 0 && hz != static_cast<int>(hz)) --fz;

	if (hx > -1 && hx < (int) imageSize.x && // inside vol w/ 1 px border
	    hy > -1 && hy < (int) imageSize.y &&
	  hz > -1 && hz < (int) imageSize.z)
	  {
	    // compute trilinear weights
	    rx = hx - fx;
	    ry = hy - fy;
	    rz = hz - fz;
	    w000 = (1.0 - rx) * (1.0 - ry) * (1.0 - rz);
	    w001 = (1.0 - rx) * (1.0 - ry) * (rz);
	    w010 = (1.0 - rx) * (ry)       * (1.0 - rz);
	    w011 = (1.0 - rx) * (ry)       * (rz);
	    w100 = (rx)       * (1.0 - ry) * (1.0 - rz);
	    w101 = (rx)       * (1.0 - ry) * (rz);
	    w110 = (rx)       * (ry)       * (1.0 - rz);
	    w111 = (rx)       * (ry)       * (rz);

	    // see which corners of cube are valid
	    bool
	      floorXIn = (fx >= 0), ceilXIn = (fx < (int) imageSize.x - 1),
	      floorYIn = (fy >= 0), ceilYIn = (fy < (int) imageSize.y - 1),
	      floorZIn = (fz >= 0), ceilZIn = (fz < (int) imageSize.z - 1);

	    if (floorXIn && floorYIn && floorZIn)
	      {
		defImage(fx, fy, fz)       += w000 * ix;
		count(fx, fy, fz)          += w000;
	      }
	    if (floorXIn && floorYIn && ceilZIn)
	      {
		defImage(fx, fy, fz+1)     += w001 * ix;
		count(fx, fy, fz+1)        += w001;
	      }
	    if (floorXIn && ceilYIn && floorZIn)
	      {
		defImage(fx, fy+1, fz)     += w010 * ix;
		count(fx, fy+1, fz)        += w010;
	      }
	    if (floorXIn && ceilYIn && ceilZIn)
	      {
		defImage(fx, fy+1, fz+1)   += w011 * ix;
		count(fx, fy+1, fz+1)      += w011;
	      }
	    if (ceilXIn && floorYIn && floorZIn)
	      {
		defImage(fx+1, fy, fz)     += w100 * ix;
		count(fx+1, fy, fz)        += w100;
	      }
	    if (ceilXIn && floorYIn && ceilZIn)
	      {
		defImage(fx+1, fy, fz+1)   += w101 * ix;
		count(fx+1, fy, fz+1)      += w101;
	      }
	    if (ceilXIn && ceilYIn && floorZIn)
	      {
		defImage(fx+1, fy+1, fz)   += w110 * ix;
		count(fx+1, fy+1, fz)      += w110;
	      }
	    if (ceilXIn && ceilYIn && ceilZIn)
	      {
		defImage(fx+1, fy+1, fz+1) += w111 * ix;
		count(fx+1, fy+1, fz+1)    += w111;
	      }
	  }          
      }
    }
  }    
  
  // find holes
  //     float zeta = 1.0/20.0;
  //     std::queue<Vector3D<unsigned int> > holes;
  //     for (unsigned int z = 0; 
  //          z < hFieldSize.z && z + hFieldStartZ < imageSize.z; ++z) {
  //       for (unsigned int y = 0; 
  //            y < hFieldSize.y && y + hFieldStartY < imageSize.y; ++y) {
  //         for (unsigned int x = 0; 
  //              x < hFieldSize.x && x + hFieldStartX < imageSize.x; ++x) {
  //           if (count(x+hFieldStartX, y+hFieldStartY, z+hFieldStartZ) < zeta) {
  //             holes.push(Vector3D<unsigned int>(x+hFieldStartX,
  //                                               y+hFieldStartY,
  //                                               z+hFieldStartZ));
  //           }
  //         }
  //       }
  //     }

  // fill in holes with average of 6-neighbors
  //     int maxIters = 10;
  //     int iter = 0;
  //     float tau = 1.0/3.0;
  //     while (holes.size() && iter++ < maxIters) {
  //       std::cerr << "filling " << holes.size() << " holes: " 
  //                 << iter << std::endl;
  //       for (int i = 0; i < (int) holes.size(); ++i) {
  //         // pop coords from front of queue
  //         unsigned int x = holes.front().x;
  //         unsigned int y = holes.front().y;
  //         unsigned int z = holes.front().z;
  //         holes.pop();

  //         if (x+1 < imageSize.x && count(x+1,y,z) > tau) {
  //           defImage(x,y,z) += defImage(x+1,y,z);
  //           count(x,y,z)    += count(x+1,y,z);
  //         }

  //         if (x-1 > 0 && count(x-1,y,z) > tau) { 
  //           defImage(x,y,z) += defImage(x-1,y,z);
  //           count(x,y,z)    += count(x-1,y,z);
  //         }

  //         if (y+1 < imageSize.y && count(x,y+1,z) > tau) {
  //           defImage(x,y,z) += defImage(x,y+1,z);
  //           count(x,y,z)    += count(x,y+1,z);
  //         }

  //         if (y-1 > 0 && count(x,y-1,z) > tau) { 
  //           defImage(x,y,z) += defImage(x,y-1,z);
  //           count(x,y,z)    += count(x,y-1,z);
  //         }

  //         if (z+1 < imageSize.z && count(x,y,z+1) > tau) {
  //           defImage(x,y,z) += defImage(x,y,z+1);
  //           count(x,y,z)    += count(x,y,z+1);
  //         }

  //         if (z-1 > 0 && count(x,y,z-1) > tau) { 
  //           defImage(x,y,z) += defImage(x,y,z-1);
  //           count(x,y,z)    += count(x,y,z-1);
  //         }
        
  //         // add back to queue if did'nt find any neighbors
  //         if (count(x,y,z) < zeta) {
  //           holes.push(Vector3D<unsigned int>(x,y,z));
  //         }
  //       }
  //     }


  // normalize 
  if(normalize){
    for (unsigned int z = 0; 
	 z < hFieldSize.z && z + hFieldStartZ < imageSize.z; ++z) {
      for (unsigned int y = 0; 
	   y < hFieldSize.y && y + hFieldStartY < imageSize.y; ++y) {
	for (unsigned int x = 0; 
	     x < hFieldSize.x && x + hFieldStartX < imageSize.x; ++x) {
	  defImage(x+hFieldStartX, y+hFieldStartY, z+hFieldStartZ) =
	    count(x+hFieldStartX, y+hFieldStartY, z+hFieldStartZ) > 0
	    ? defImage(x+hFieldStartX, y+hFieldStartY, z+hFieldStartZ)
	    / count(x+hFieldStartX, y+hFieldStartY, z+hFieldStartZ)
	    : background;
	} // end loop x
      } // end loop y
    } // end loop z
  } // end if normalize

}

template <class T, class U>
inline
void
HField3DUtils::
forwardApply(const Array3D<T>& image,
	     const Array3D<Vector3D<U> >& hField,
	     Array3D<T>& defImage,
	     const T& background,
	     bool normalize)
{
  Vector3D<unsigned int> size = image.getSize();
  defImage.fill(0.0f);
  Array3D<float> count(defImage.getSize());
  count.fill(0.0f);

  // trilinear weights
  float w000, w001, w010, w011, w100, w101, w110, w111;

  T ix;
    
  // floor index and residuals
  U hx, hy, hz;
  int fx, fy, fz;
  float rx, ry, rz;
    
  for (unsigned int z = 0; z < size.z; ++z) {
    for (unsigned int y = 0; y < size.y; ++y) {
      for (unsigned int x = 0; x < size.x; ++x) {
	// get intensity value
	ix = image(x,y,z);
	  
	// get h field value---where this intensity should go
	hx = hField(x,y,z).x;
	hy = hField(x,y,z).y;
	hz = hField(x,y,z).z;

	// this is a fast version of the floor function
	fx = static_cast<int>(hx);
	fy = static_cast<int>(hy);
	fz = static_cast<int>(hz);
	if (hx < 0 && hx != static_cast<int>(hx)) --fx;
	if (hy < 0 && hy != static_cast<int>(hy)) --fy;
	if (hz < 0 && hz != static_cast<int>(hz)) --fz;

	if (hx > -1 && hx < (int) size.x &&  // inside vol with 1 pix border
	    hy > -1 && hy < (int) size.y &&
	  hz > -1 && hz < (int) size.z)
	  {
	    // compute trilinear weights
	    rx = hx - fx;
	    ry = hy - fy;
	    rz = hz - fz;
	    w000 = (1.0 - rx) * (1.0 - ry) * (1.0 - rz);
	    w001 = (1.0 - rx) * (1.0 - ry) * (rz);
	    w010 = (1.0 - rx) * (ry)       * (1.0 - rz);
	    w011 = (1.0 - rx) * (ry)       * (rz);
	    w100 = (rx)       * (1.0 - ry) * (1.0 - rz);
	    w101 = (rx)       * (1.0 - ry) * (rz);
	    w110 = (rx)       * (ry)       * (1.0 - rz);
	    w111 = (rx)       * (ry)       * (rz);

	    // see which corners of cube are valid
	    bool
	      floorXIn = (fx >= 0), ceilXIn = (fx < (int) size.x-1),
	      floorYIn = (fy >= 0), ceilYIn = (fy < (int) size.y-1),
	      floorZIn = (fz >= 0), ceilZIn = (fz < (int) size.z-1);

	    if (floorXIn && floorYIn && floorZIn)
	      {
		defImage(fx, fy, fz)       += w000 * ix;
		count(fx, fy, fz)          += w000;
	      }
	    if (floorXIn && floorYIn && ceilZIn)
	      {
		defImage(fx, fy, fz+1)     += w001 * ix;
		count(fx, fy, fz+1)        += w001;
	      }
	    if (floorXIn && ceilYIn && floorZIn)
	      {
		defImage(fx, fy+1, fz)     += w010 * ix;
		count(fx, fy+1, fz)        += w010;
	      }
	    if (floorXIn && ceilYIn && ceilZIn)
	      {
		defImage(fx, fy+1, fz+1)   += w011 * ix;
		count(fx, fy+1, fz+1)      += w011;
	      }
	    if (ceilXIn && floorYIn && floorZIn)
	      {
		defImage(fx+1, fy, fz)     += w100 * ix;
		count(fx+1, fy, fz)        += w100;
	      }
	    if (ceilXIn && floorYIn && ceilZIn)
	      {
		defImage(fx+1, fy, fz+1)   += w101 * ix;
		count(fx+1, fy, fz+1)      += w101;
	      }
	    if (ceilXIn && ceilYIn && floorZIn)
	      {
		defImage(fx+1, fy+1, fz)   += w110 * ix;
		count(fx+1, fy+1, fz)      += w110;
	      }
	    if (ceilXIn && ceilYIn && ceilZIn)
	      {
		defImage(fx+1, fy+1, fz+1) += w111 * ix;
		count(fx+1, fy+1, fz+1)    += w111;
	      }
	  }
      }
    }
  }    
    
  // normalize counts (NOTE: no rounding for integer types)
  if(normalize){
    unsigned int numElements = defImage.getNumElements();
    for (unsigned int e = 0; e < numElements; ++e)
      {
	defImage(e) = count(e) > 0 ? defImage(e) / count(e) : background;
      }
  }

  // jsp 2010:  I don't think we want to do this anymore, really a hack
//   // Fill holes on interior
//   for (unsigned int z = 1; z < size.z - 1; ++z)
//     for (unsigned int y = 1; y < size.y - 1; ++y)
//       for (unsigned int x = 1; x < size.x - 1; ++x)
// 	if (count(x,y,z) < 0.0001f)
// 	  { // This is a hole, we need to set defImage to average of all it's 6 neighbors
// 	    defImage(x,y,z) = ( defImage(x+1,y,z) + defImage(x-1,y,z)
//                                 + defImage(x,y+1,z) + defImage(x,y-1,z) 
//                                 + defImage(x,y,z+1) + defImage(x,y,z-1) ) / 6.0f;
// 	  }
}

template <class T, class U>
inline
void
HField3DUtils::
forwardApplyPeriodic(const Array3D<T>& image,
                     const Array3D<Vector3D<U> >& hField,
                     Array3D<T>& defImage,
                     bool normalize)
{
  Vector3D<unsigned int> size = image.getSize();
  defImage.fill(0.0f);
  Array3D<float> count(defImage.getSize());
  count.fill(0.0f);

  // trilinear weights
  float w000, w001, w010, w011, w100, w101, w110, w111;

  T ix;
    
  // floor index and residuals
  U hx, hy, hz;
  int fx, fy, fz;
  float rx, ry, rz;
    
  for (unsigned int z = 0; z < size.z; ++z)
    {
    for (unsigned int y = 0; y < size.y; ++y)
      {
      for (unsigned int x = 0; x < size.x; ++x)
        {
        // get intensity value
        ix = image(x,y,z);
	  
        // get h field value---where this intensity should go
        hx = hField(x,y,z).x;
        hy = hField(x,y,z).y;
        hz = hField(x,y,z).z;

        // this is a fast version of the floor function
        fx = static_cast<int>(hx);
        fy = static_cast<int>(hy);
        fz = static_cast<int>(hz);
        // this just fixes our floor function for negative values
        if (hx < 0 && hx != static_cast<int>(hx)) --fx;
        if (hy < 0 && hy != static_cast<int>(hy)) --fy;
        if (hz < 0 && hz != static_cast<int>(hz)) --fz;

        if (hx > -1 && hx < (int) size.x &&  // inside vol with 1 pix border
                            hy > -1 && hy < (int) size.y &&
                                            hz > -1 && hz < (int) size.z)
          {
          // compute trilinear weights
          rx = hx - fx;
          ry = hy - fy;
          rz = hz - fz;
          w000 = (1.0 - rx) * (1.0 - ry) * (1.0 - rz);
          w001 = (1.0 - rx) * (1.0 - ry) * (rz);
          w010 = (1.0 - rx) * (ry)       * (1.0 - rz);
          w011 = (1.0 - rx) * (ry)       * (rz);
          w100 = (rx)       * (1.0 - ry) * (1.0 - rz);
          w101 = (rx)       * (1.0 - ry) * (rz);
          w110 = (rx)       * (ry)       * (1.0 - rz);
          w111 = (rx)       * (ry)       * (rz);

          // No longer need to check if any corners are inside the volume
          // For periodic image they're all inside, but we need to adjust
          // the indices properly
          fx = safe_mod(fx, size.x);
          fy = safe_mod(fy, size.y);
          fz = safe_mod(fz, size.z);
          int fxp = safe_mod(fx+1, size.x);
          int fyp = safe_mod(fy+1, size.y);
          int fzp = safe_mod(fz+1, size.z);

          defImage(fx, fy, fz)       += w000 * ix;
          count(fx, fy, fz)          += w000;

          defImage(fx, fy, fzp)      += w001 * ix;
          count(fx, fy, fzp)         += w001;

          defImage(fx, fyp, fz)      += w010 * ix;
          count(fx, fyp, fz)         += w010;

          defImage(fx, fyp, fzp)     += w011 * ix;
          count(fx, fyp, fzp)        += w011;

          defImage(fxp, fy, fz)     += w100 * ix;
          count(fxp, fy, fz)        += w100;

          defImage(fxp, fy, fzp)   += w101 * ix;
          count(fxp, fy, fzp)      += w101;

          defImage(fxp, fyp, fz)   += w110 * ix;
          count(fxp, fyp, fz)      += w110;

          defImage(fxp, fyp, fzp) += w111 * ix;
          count(fxp, fyp, fzp)    += w111;
          }
        }
      }
    }    
    
  // normalize counts (NOTE: no rounding for integer types)
  if(normalize){
  unsigned int numElements = defImage.getNumElements();
  for (unsigned int e = 0; e < numElements; ++e)
    {
    defImage(e) = count(e) > 0 ? defImage(e) / count(e) : 0.0f;
    }
  }
}

template <class T, class U, 
	  Array3DUtils::ScalarBackgroundStrategyT BackgroundStrategy,
	  Array3DUtils::InterpT InterpMethod>
inline
void
HField3DUtils::
applyWithROI(const Array3D<T>& image,
	     const ROI<int, unsigned int>& roi,
	     const Array3D<Vector3D<U> >& hField,
	     Array3D<T>& defImage,
	     const T& background)
{
  
  for (int z = roi.getStartZ(), i=0; 
       z <= roi.getStopZ(); ++z, ++i)
    {
      for (int y = roi.getStartY(), j=0; 
	   y <= roi.getStopY(); ++y, ++j)
	{
	  for (int x = roi.getStartX(), k=0; 
	       x <= roi.getStopX(); ++x, ++k)
	    {
	      defImage(x, y, z) = 
		Array3DUtils::
		interp<T, BackgroundStrategy, InterpMethod>(image,
							    hField(k, j, i),
							    background);
	    }
	}
    }
}

template <class T>
inline
void
HField3DUtils::
apply(Surface& surface,
      const Array3D<Vector3D<T> >& h)
{
  unsigned int numVertices = surface.numVertices();
  for (unsigned int i = 0; i < numVertices; ++i)
    {
      float hx, hy, hz;
      trilerp(h,
	      (T) surface.vertices[i].x,
	      (T) surface.vertices[i].y,
	      (T) surface.vertices[i].z,
	      hx, hy, hz);
      surface.vertices[i].x = hx;
      surface.vertices[i].y = hy;
      surface.vertices[i].z = hz;
    }
}

template <class T>
inline
void
HField3DUtils::
applyWithROI(Surface& surface,
	     const ROI<int, unsigned int>& roi,
	     const Array3D<Vector3D<T> >& h)
{
  unsigned int numVertices = surface.numVertices();
  for (unsigned int i = 0; i < numVertices; ++i)
    {
      float x = surface.vertices[i].x - roi.getStartX();
      float y = surface.vertices[i].y - roi.getStartY();
      float z = surface.vertices[i].z - roi.getStartZ();
      float Nx,Ny,Nz;
      HField3DUtils::trilerp(h,x,y,z,Nx,Ny,Nz);
      surface.vertices[i].x = Nx + roi.getStartX();
      surface.vertices[i].y = Ny + roi.getStartY();
      surface.vertices[i].z = Nz + roi.getStartZ();
    }
}

template <class T>
inline
void
HField3DUtils::
inverseApply(Surface& surface,
	     const Array3D<Vector3D<T> >& h)
{
  unsigned int numVertices = surface.numVertices();
  for (unsigned int i = 0; i < numVertices; ++i)
    {
      if (i % 100 == 0) std::cerr << i << std::endl;
      float hinvx, hinvy, hinvz;
      hinvx=hinvy=hinvz=0; // make compiler happy
      if( !inverseOfPoint(h, 
			  (float) surface.vertices[i].x,
			  (float) surface.vertices[i].y,
			  (float) surface.vertices[i].z,
			  hinvx, hinvy, hinvz))
	{
	  std::cerr << "Error, inverse not found";
	}
      surface.vertices[i].x = hinvx;
      surface.vertices[i].y = hinvy;
      surface.vertices[i].z = hinvz;
    }
}

template <class T>
inline
void
HField3DUtils::
inverseApply(Surface& surface,
	     const Array3D<Vector3D<T> >& h,
	     const ROI<int, unsigned int>& roi)               
{
  inverseApply(surface, h, 
	       roi.getStartX(), roi.getStartY(), roi.getStartZ());
}
               
template <class T>
inline
void
HField3DUtils::
inverseApply(Surface& surface,
	     const Array3D<Vector3D<T> >& h,
	     int hFieldStartX,
	     int hFieldStartY,
	     int hFieldStartZ)
{
  unsigned int numVertices = surface.numVertices();
  for (unsigned int i = 0; i < numVertices; ++i)
    {
      float x = surface.vertices[i].x - hFieldStartX;
      float y = surface.vertices[i].y - hFieldStartY;
      float z = surface.vertices[i].z - hFieldStartZ;
      float Nx=0.f,Ny=0.f,Nz=0.f;
      if(!inverseOfPoint(h, x, y, z, Nx, Ny, Nz))
	{
	  std::cerr << "Error, inverse not found";
	}
      surface.vertices[i].x = Nx + hFieldStartX;
      surface.vertices[i].y = Ny + hFieldStartY;
      surface.vertices[i].z = Nz + hFieldStartZ;
    }    
}


template <class T>
inline
void
HField3DUtils::
sincUpsample(const Array3D<Vector3D<T> >& inHField, 
	     Array3D<Vector3D<T> >& outHField,
	     unsigned int factor)
{

  Vector3D<unsigned int> oldSize = inHField.getSize();
  Vector3D<unsigned int> newSize = oldSize*factor;

  sincUpsample(inHField, outHField, newSize);
}

template <class T>
inline
void
HField3DUtils::
sincUpsample(const Array3D<Vector3D<T> >& inHField, 
	     Array3D<Vector3D<T> >& outHField,
	     Vector3D<unsigned int> &newSize)
{

  Vector3D<unsigned int> oldSize = inHField.getSize();

  // set up the complex data arrays.  We pad by one pixel to make
  // the hfield dimensions odd.
  Vector3D<unsigned int> oldCSize = oldSize+1;
  Vector3D<unsigned int> newCSize = newSize+1;
  unsigned int cDataOldSize = 2*3*static_cast<unsigned int>(oldCSize.productOfElements());
  float *cDataOld = new float[cDataOldSize];
  unsigned int cDataNewSize = 2*3*static_cast<unsigned int>(newCSize.productOfElements());
  float *cDataNew = new float[cDataNewSize];
  // zero out the complex arrays
  for(unsigned int i = 0; i < cDataOldSize; i++){
    cDataOld[i] = static_cast<float>(0);
  }
  for(unsigned int i = 0; i < cDataNewSize; i++){
    cDataNew[i] = static_cast<float>(0);
  }

  // set up the fftw plans
  fftwf_plan fftwForwardPlan;
  fftwf_plan fftwBackwardPlan;
  fftwf_plan_with_nthreads(2);
  int dims[3];
  dims[0] = oldCSize.x;
  dims[1] = oldCSize.y;
  dims[2] = oldCSize.z;
  int newDims[3];
  newDims[0] = newCSize.x;
  newDims[1] = newCSize.y;
  newDims[2] = newCSize.z;
  // in-place transforms
  fftwForwardPlan = fftwf_plan_many_dft(3, dims, 3, (fftwf_complex *)cDataOld, 0, 3, 1, (fftwf_complex *)(cDataOld),0, 3, 1, -1 , FFTW_ESTIMATE);
  fftwBackwardPlan = fftwf_plan_many_dft(3, newDims, 3, (fftwf_complex *)cDataNew, 0, 3, 1, (fftwf_complex *)cDataNew, 0, 3, 1, +1, FFTW_ESTIMATE);

  if(!fftwForwardPlan){
    std::cerr << "fftw forward plan could not be initialized!" << std::endl;
    return;
  }
  if(!fftwBackwardPlan){
    std::cerr << "fftw backward plan could not be initialized!" << std::endl;
    return;
  }

  // copy the data into the complex array
  for (unsigned int z = 0; z < oldSize.z; ++z) {
    for (unsigned int y = 0; y < oldSize.y; ++y) {
      for (unsigned int x = 0; x < oldSize.x; ++x) {
	unsigned int cIndex = 2*3*((x+1) + ((y+1) + (z+1)*oldCSize.y)*oldCSize.x);
	const Vector3D<T> *dataPtr = &inHField(x,y,z);
	cDataOld[cIndex] = static_cast<float>(dataPtr->x);
	cDataOld[cIndex+2] = static_cast<float>(dataPtr->y);
	cDataOld[cIndex+4] = static_cast<float>(dataPtr->z);
      }
    }
  }

  // execute the fft
  fftwf_execute(fftwForwardPlan);

  // copy the data to the larger array.
  // Data goes in four corners.
  Vector3D<unsigned int> freqSplitIdx;
  for(int i=0;i<3;i++){
    freqSplitIdx[i] = oldCSize[i]/2 + 1;
  }

  Vector3D<unsigned int> newIdx;
  for (unsigned int z = 0; z < oldCSize.z; ++z) {
    if(z < freqSplitIdx.z){
      newIdx.z = z;
    }else{
      newIdx.z = newCSize.z+(z-oldCSize.z);
    }
    for (unsigned int y = 0; y < oldCSize.y; ++y) {
      if(y < freqSplitIdx.y){
	newIdx.y = y;
      }else{
	newIdx.y = newCSize.y+(y-oldCSize.y);
      }
      for (unsigned int x = 0; x < oldCSize.x; ++x) {
	if(x < freqSplitIdx.x){
	  newIdx.x = x;
	}else{
	  newIdx.x = newCSize.x+(x-oldCSize.x);
	}
	unsigned int cIndexOld = 2*3*(x + (y + z*oldCSize.y)*oldCSize.x);
	unsigned int cIndexNew = 2*3*(newIdx.x + (newIdx.y + newIdx.z*newCSize.y)*newCSize.x);
	// copy real and complex values for x, y, and z elements
	for(int i=0;i<6;i++){
	  cDataNew[cIndexNew+i] = cDataOld[cIndexOld+i];
	}
      }
    }
  }
    
  // execute the ifft
  fftwf_execute(fftwBackwardPlan);

  // resize the output hfield
  outHField.resize(newSize);
    
  // copy out the data to the output hfield
  for (unsigned int z = 0; z < newSize.z; ++z) {
    for (unsigned int y = 0; y < newSize.y; ++y) {
      for (unsigned int x = 0; x < newSize.x; ++x) {
	unsigned int cIndex = 2*3*((x+1) + ((y+1) + (z+1)*newCSize.y)*newCSize.x);
	// take the real parts of the data
	outHField(x,y,z).x = static_cast<T>(cDataNew[cIndex]);
	outHField(x,y,z).y = static_cast<T>(cDataNew[cIndex+2]);
	outHField(x,y,z).z = static_cast<T>(cDataNew[cIndex+4]);
      }
    }
  }

  outHField.scale(1.0/static_cast<T>(oldSize.productOfElements()));

}

template<class U>
inline
void 
HField3DUtils::
resampleNew(const Array3D<Vector3D<U> >& inHField, 
	    Array3D<Vector3D<U> >& outHField,
	    const unsigned int outputSizeX,
	    const unsigned int outputSizeY,
	    const unsigned int outputSizeZ,
	    VectorBackgroundStrategy backgroundStrategy,
	    bool rescaleVectors)
{
  Vector3D<unsigned int> inSize = inHField.getSize();
    
  if((outputSizeX == inSize.x) &&
     (outputSizeY == inSize.y) &&
     (outputSizeZ == inSize.z))
    {
      // no need to resample, already the same size
      outHField = inHField;
    }
  else
    {
      outHField.resize(outputSizeX, outputSizeY, outputSizeZ);
	
      // scale factors to convert outputSize to inputSize
      double rX = static_cast<double>(inSize.x) 
	/ static_cast<double>(outputSizeX);
      double rY = static_cast<double>(inSize.y) 
	/ static_cast<double>(outputSizeY);
      double rZ = static_cast<double>(inSize.z) 
	/ static_cast<double>(outputSizeZ);
	
      // sampling origin (for proper centering)
      double oX = (rX-1.0)/2.0;
      double oY = (rY-1.0)/2.0;
      double oZ = (rZ-1.0)/2.0;

      Vector3D<float> hInOfX;
      Vector3D<double> inIndex;
      for (unsigned int z = 0; z < outputSizeZ; ++z){
	inIndex.z = oZ + z * rZ;
	for (unsigned int y = 0; y < outputSizeY; ++y){
	  inIndex.y = oY + y * rY;
	  for (unsigned int x = 0; x < outputSizeX; ++x){
	    inIndex.x = oX + x * rX;

	    // get vector for corresponding position in inHField
	    trilerp(inHField,
		    (float) inIndex.x, (float) inIndex.y, (float) inIndex.z,
		    hInOfX.x, hInOfX.y, hInOfX.z, 
		    backgroundStrategy);

	    outHField(x,y,z) = hInOfX;
	  }
	}
      }
      if(rescaleVectors){
	Vector3D<U> scaleFactor(1.0/rX, 1.0/rY, 1.0/rZ);
	for (unsigned int z = 0; z < outputSizeZ; ++z){
	  for (unsigned int y = 0; y < outputSizeY; ++y){
	    for (unsigned int x = 0; x < outputSizeX; ++x){
	      // rescale vector before setting to outHField
	      outHField(x,y,z) *= scaleFactor;
	    }
	  }
	}
      } // end if recaleVectors
    }
}


template<class U>
inline
void 
HField3DUtils::
resampleNew(const Array3D<Vector3D<U> >& inHField, 
	    Array3D<Vector3D<U> >& outHField, 
	    const Vector3D<unsigned int>& outputSize,
	    VectorBackgroundStrategy backgroundStrategy,
	    bool rescaleVectors)
{
  resampleNew(inHField, outHField, 
	      outputSize.x, outputSize.y, outputSize.z, 
	      backgroundStrategy, rescaleVectors);
}
  
template<class U>
inline
void 
HField3DUtils::
resample(const Array3D<Vector3D<U> >& inHField, 
	 Array3D<Vector3D<U> >& outHField,
	 const unsigned int outputSizeX,
	 const unsigned int outputSizeY,
	 const unsigned int outputSizeZ,
	 VectorBackgroundStrategy backgroundStrategy)
{
  Vector3D<unsigned int> inSize = inHField.getSize();
    
  if((outputSizeX == inSize.x) &&
     (outputSizeY == inSize.y) &&
     (outputSizeZ == inSize.z))
    {
      // no need to resample, already the same size
      outHField = inHField;
    }
  else
    {
      outHField.resize(outputSizeX, outputSizeY, outputSizeZ);
	
      // scale factors to convert outputSize to inputSize
      double rX = static_cast<double>(inSize.x) 
	/ static_cast<double>(outputSizeX);
      double rY = static_cast<double>(inSize.y) 
	/ static_cast<double>(outputSizeY);
      double rZ = static_cast<double>(inSize.z) 
	/ static_cast<double>(outputSizeZ);
	
      Vector3D<float> hInOfX;
      Vector3D<double> inIndex;
      for (unsigned int z = 0; z < outputSizeZ; ++z){
	inIndex.z = z * rZ;
	for (unsigned int y = 0; y < outputSizeY; ++y){
	  inIndex.y = y * rY;
	  for (unsigned int x = 0; x < outputSizeX; ++x){
	    inIndex.x = x * rX;

	    // get vector for corresponding position in inHField
	    trilerp(inHField,
		    (float) inIndex.x, (float) inIndex.y, (float) inIndex.z,
		    hInOfX.x, hInOfX.y, hInOfX.z, 
		    backgroundStrategy);

	    // rescale vector before setting to outHField
	    outHField(x,y,z).set(hInOfX.x / rX,
				 hInOfX.y / rY,
				 hInOfX.z / rZ);
	  }
	}
      }
    }
}

template<class U>
inline
void 
HField3DUtils::
resample(const Array3D<Vector3D<U> >& inHField, 
	 Array3D<Vector3D<U> >& outHField, 
	 const Vector3D<unsigned int>& outputSize,
	 VectorBackgroundStrategy backgroundStrategy)
{
  resample(inHField, outHField, 
	   outputSize.x, outputSize.y, outputSize.z, backgroundStrategy);
}

template <class T>
inline
void
HField3DUtils::
trilerp(const Array3D<Vector3D<T> >& h,
	const T& x, const T& y, const T& z,
	T& hx, T& hy, T& hz,
	VectorBackgroundStrategy backgroundStrategy)
{
  // a (much) faster version of the floor function
  int floorX = static_cast<int>(x);
  int floorY = static_cast<int>(y);
  int floorZ = static_cast<int>(z);
  if (x < 0 && x != static_cast<int>(x)) --floorX;
  if (y < 0 && y != static_cast<int>(y)) --floorY;
  if (z < 0 && z != static_cast<int>(z)) --floorZ;

  // this is not truly ceiling, but floor + 1, which is usually ceiling    
  int ceilX = floorX + 1;
  int ceilY = floorY + 1;
  int ceilZ = floorZ + 1;

  const double t = x - floorX;
  const double u = y - floorY;
  const double v = z - floorZ;
  const double oneMinusT = 1.0 - t;
  const double oneMinusU = 1.0 - u;
  const double oneMinusV = 1.0 - v;

  //
  // ^
  // |  v3   v2       -z->        v4   v5
  // y           --next slice-->      
  // |  v0   v1                   v7   v6
  //
  //      -x->
  //   

  T v0X=0, v0Y=0, v0Z=0;
  T v1X=0, v1Y=0, v1Z=0;
  T v2X=0, v2Y=0, v2Z=0;
  T v3X=0, v3Y=0, v3Z=0;
  T v4X=0, v4Y=0, v4Z=0;
  T v5X=0, v5Y=0, v5Z=0;
  T v6X=0, v6Y=0, v6Z=0;
  T v7X=0, v7Y=0, v7Z=0;

  int sizeX = h.getSizeX();
  int sizeY = h.getSizeY();
  int sizeZ = h.getSizeZ();

  bool inside = true;

  if(backgroundStrategy == BACKGROUND_STRATEGY_WRAP){
    // wrap
  floorX = safe_mod(floorX, sizeX);
  floorY = safe_mod(floorY, sizeY);
  floorZ = safe_mod(floorZ, sizeZ);
  ceilX = safe_mod(ceilX, sizeX);
  ceilY = safe_mod(ceilY, sizeY);
  ceilZ = safe_mod(ceilZ, sizeZ);

    // if(floorX < 0){
    //   floorX = sizeX + floorX;
    //   if(ceilX < 0) ceilX = sizeX + ceilX;
    // }
    // if(ceilX >= sizeX){
    //   ceilX = ceilX % sizeX;
    //   if(floorX >= sizeX) floorX = floorX % sizeX;
    // }
    // if(floorY < 0){
    //   floorY = sizeY + floorY;
    //   if(ceilY < 0) ceilY = sizeY + ceilY;
    // }
    // if(ceilY >= sizeY){
    //   ceilY = ceilY % sizeY;
    //   if(floorY >= sizeY) floorY = floorY % sizeY;
    // }
    // if(floorZ < 0){
    //   floorZ = sizeZ + floorZ;
    //   if(ceilZ < 0) ceilZ = sizeZ + ceilZ;
    // }
    // if(ceilZ >= sizeZ){
    //   ceilZ = ceilZ % sizeZ;
    //   if(floorZ >= sizeZ) floorZ = floorZ % sizeZ;
    // }
  }else if(backgroundStrategy == BACKGROUND_STRATEGY_CLAMP){
    // clamp
    if(floorX < 0){
      floorX = 0;
      if(ceilX < 0) ceilX = 0;
    }
    if(ceilX >= sizeX){
      ceilX = sizeX-1;
      if(floorX >= sizeX) floorX = sizeX-1;
    }
    if(floorY < 0){
      floorY = 0;
      if(ceilY < 0) ceilY = 0;
    }
    if(ceilY >= sizeY){
      ceilY = sizeY-1;
      if(floorY >= sizeY) floorY = sizeY-1;
    }
    if(floorZ < 0){
      floorZ = 0;
      if(ceilZ < 0) ceilZ = 0;
    }
    if(ceilZ >= sizeZ){
      ceilZ = sizeZ-1;
      if(floorZ >= sizeZ) floorZ = sizeZ-1;
    }
  }else{
    inside = (floorX >= 0 && ceilX < sizeX &&
	      floorY >= 0 && ceilY < sizeY &&
	      floorZ >= 0 && ceilZ < sizeZ);
  }

  if (inside)
    {
      //
      // coordinate is inside volume, fill in 
      // eight corners of cube
      //
      v0X = h(floorX, floorY, floorZ).x;
      v0Y = h(floorX, floorY, floorZ).y;
      v0Z = h(floorX, floorY, floorZ).z;

      v1X = h(ceilX, floorY, floorZ).x;
      v1Y = h(ceilX, floorY, floorZ).y;
      v1Z = h(ceilX, floorY, floorZ).z;

      v2X = h(ceilX, ceilY, floorZ).x;
      v2Y = h(ceilX, ceilY, floorZ).y;
      v2Z = h(ceilX, ceilY, floorZ).z;

      v3X = h(floorX, ceilY, floorZ).x;
      v3Y = h(floorX, ceilY, floorZ).y;
      v3Z = h(floorX, ceilY, floorZ).z;

      v4X = h(floorX, ceilY, ceilZ).x;
      v4Y = h(floorX, ceilY, ceilZ).y;
      v4Z = h(floorX, ceilY, ceilZ).z;

      v5X = h(ceilX, ceilY, ceilZ).x;
      v5Y = h(ceilX, ceilY, ceilZ).y;
      v5Z = h(ceilX, ceilY, ceilZ).z;

      v6X = h(ceilX, floorY, ceilZ).x;
      v6Y = h(ceilX, floorY, ceilZ).y;
      v6Z = h(ceilX, floorY, ceilZ).z;

      v7X = h(floorX, floorY, ceilZ).x;
      v7Y = h(floorX, floorY, ceilZ).y;
      v7Z = h(floorX, floorY, ceilZ).z;
    }
  else if (backgroundStrategy == BACKGROUND_STRATEGY_ID)
    {
      //
      // coordinate is not inside volume, return identity
      //
      hx = x; hy = y; hz = z;
      return;
    }
  else if (backgroundStrategy == BACKGROUND_STRATEGY_ZERO)
    {
      hx = 0; hy = 0; hz = 0;
      return;
    }
  else if (backgroundStrategy == BACKGROUND_STRATEGY_PARTIAL_ID ||
	   backgroundStrategy == BACKGROUND_STRATEGY_PARTIAL_ZERO)
    {
      //
      // coordinate is not inside volume; initialize cube
      // corners to identity/zero then set any corners of cube that
      // fall on volume boundary
      //
      if(backgroundStrategy == BACKGROUND_STRATEGY_PARTIAL_ID)
	{
	  //
	  // coordinate is inside volume, fill in 
	  // eight corners of cube
	  //
	  v0X = floorX;
	  v0Y = floorY;
	  v0Z = floorZ;

	  v1X = ceilX;
	  v1Y = floorY;
	  v1Z = floorZ;

	  v2X = ceilX;
	  v2Y = ceilY;
	  v2Z = floorZ;

	  v3X = floorX;
	  v3Y = ceilY;
	  v3Z = floorZ;

	  v4X = floorX;
	  v4Y = ceilY;
	  v4Z = ceilZ;

	  v5X = ceilX;
	  v5Y = ceilY;
	  v5Z = ceilZ;

	  v6X = ceilX;
	  v6Y = floorY;
	  v6Z = ceilZ;

	  v7X = floorX;
	  v7Y = floorY;
	  v7Z = ceilZ;

	  // 	    v0X = x; v0Y = y; v0Z = z;
	  // 	    v1X = x; v1Y = y; v1Z = z;
	  // 	    v2X = x; v2Y = y; v2Z = z;
	  // 	    v3X = x; v3Y = y; v3Z = z;
	  // 	    v4X = x; v4Y = y; v4Z = z;
	  // 	    v5X = x; v5Y = y; v5Z = z;
	  // 	    v6X = x; v6Y = y; v6Z = z;
	  // 	    v7X = x; v7Y = y; v7Z = z;
	}
      else // BACKGROUND_STRATEGY_PARTIAL_ZERO
	{
	  v0X = 0; v0Y = 0; v0Z = 0;
	  v1X = 0; v1Y = 0; v1Z = 0;
	  v2X = 0; v2Y = 0; v2Z = 0;
	  v3X = 0; v3Y = 0; v3Z = 0;
	  v4X = 0; v4Y = 0; v4Z = 0;
	  v5X = 0; v5Y = 0; v5Z = 0;
	  v6X = 0; v6Y = 0; v6Z = 0;
	  v7X = 0; v7Y = 0; v7Z = 0;
	}

      bool floorXIn = floorX >= 0 && floorX < sizeX;
      bool floorYIn = floorY >= 0 && floorY < sizeY;
      bool floorZIn = floorZ >= 0 && floorZ < sizeZ;
	
      bool ceilXIn = ceilX >= 0 && ceilX < sizeX;
      bool ceilYIn = ceilY >= 0 && ceilY < sizeY;
      bool ceilZIn = ceilZ >= 0 && ceilZ < sizeZ;
	
      if (floorXIn && floorYIn && floorZIn)
	{
	  v0X = h(floorX, floorY, floorZ).x;
	  v0Y = h(floorX, floorY, floorZ).y;
	  v0Z = h(floorX, floorY, floorZ).z;	  
	}
      if (ceilXIn && floorYIn && floorZIn)
	{
	  v1X = h(ceilX, floorY, floorZ).x;
	  v1Y = h(ceilX, floorY, floorZ).y;
	  v1Z = h(ceilX, floorY, floorZ).z;
	}
      if (ceilXIn && ceilYIn && floorZIn)
	{
	  v2X = h(ceilX, ceilY, floorZ).x;
	  v2Y = h(ceilX, ceilY, floorZ).y;
	  v2Z = h(ceilX, ceilY, floorZ).z;
	}
      if (floorXIn && ceilYIn && floorZIn)
	{
	  v3X = h(floorX, ceilY, floorZ).x;
	  v3Y = h(floorX, ceilY, floorZ).y;
	  v3Z = h(floorX, ceilY, floorZ).z;
	}
      if (floorXIn && ceilYIn && ceilZIn)
	{
	  v4X = h(floorX, ceilY, ceilZ).x;
	  v4Y = h(floorX, ceilY, ceilZ).y;
	  v4Z = h(floorX, ceilY, ceilZ).z;	  
	}
      if (ceilXIn && ceilYIn && ceilZIn)
	{
	  v5X = h(ceilX, ceilY, ceilZ).x;
	  v5Y = h(ceilX, ceilY, ceilZ).y;
	  v5Z = h(ceilX, ceilY, ceilZ).z;	  
	}
      if (ceilXIn && floorYIn && ceilZIn)
	{
	  v6X = h(ceilX, floorY, ceilZ).x;
	  v6Y = h(ceilX, floorY, ceilZ).y;
	  v6Z = h(ceilX, floorY, ceilZ).z;	  
	}
      if (floorXIn && floorYIn && ceilZIn)
	{
	  v7X = h(floorX, floorY, ceilZ).x;
	  v7Y = h(floorX, floorY, ceilZ).y;
	  v7Z = h(floorX, floorY, ceilZ).z;	  
	}
    }
    
  //
  // do trilinear interpolation
  //
    
  //
  // this is the basic trilerp function...
  //
  //     h = 
  //       v0 * (1 - t) * (1 - u) * (1 - v) +
  //       v1 * t       * (1 - u) * (1 - v) +
  //       v2 * t       * u       * (1 - v) +
  //       v3 * (1 - t) * u       * (1 - v) +
  //       v4 * (1 - t) * u       * v       +
  //       v5 * t       * u       * v       +
  //       v6 * t       * (1 - u) * v       +
  //       v7 * (1 - t) * (1 - u) * v;
  //
  // the following nested version saves 30 multiplies.
  //
    
  hx = 
    oneMinusT * (oneMinusU * (v0X * oneMinusV + v7X * v)  +
		 u         * (v3X * oneMinusV + v4X * v)) +
    t         * (oneMinusU * (v1X * oneMinusV + v6X * v)  +
		 u         * (v2X * oneMinusV + v5X * v));
    
  hy = 
    oneMinusT * (oneMinusU * (v0Y * oneMinusV + v7Y * v)  +
		 u         * (v3Y * oneMinusV + v4Y * v)) +
    t         * (oneMinusU * (v1Y * oneMinusV + v6Y * v)  +
		 u         * (v2Y * oneMinusV + v5Y * v));
    
  hz = 
    oneMinusT * (oneMinusU * (v0Z * oneMinusV + v7Z * v)  +
		 u         * (v3Z * oneMinusV + v4Z * v)) +
    t         * (oneMinusU * (v1Z * oneMinusV + v6Z * v)  +
		 u         * (v2Z * oneMinusV + v5Z * v));
}
  
template <class T, class U>
inline
void
HField3DUtils::
divergence(const Array3D<Vector3D<U> >& hField, Array3D<T>& divergence, Vector3D<double> spacing,
		       bool wrap)
{
  Vector3D<unsigned int> size = hField.getSize();
  unsigned int numElements = hField.getNumElements();

  // resize divergence array if necessary
  if (divergence.getSize() != size)
    {
      divergence.resize(size);
    }

  // build scalar images h1, h2, and h3
  // from the transformation field
  Array3D<U> h1(size);
  Array3D<U> h2(size);
  Array3D<U> h3(size);
  unsigned int i; // stupid vc++
  for (i = 0; i < numElements; ++i)
    {
      h1(i) = hField(i).x;
      h2(i) = hField(i).y;
      h3(i) = hField(i).z;
    }
    
  // compute the gradients of h1, h2, and h3
  Array3D<Vector3D<U> > grad_h1(size);
  Array3D<Vector3D<U> > grad_h2(size);
  Array3D<Vector3D<U> > grad_h3(size);   
  Array3DUtils::computeGradient(h1, grad_h1, spacing, wrap);
  Array3DUtils::computeGradient(h2, grad_h2, spacing, wrap);
  Array3DUtils::computeGradient(h3, grad_h3, spacing, wrap);

  // compute the divergence
  T t1, t2, t3;
  for (i = 0; i < numElements; ++i)
    {
      t1 = static_cast<T>(grad_h1(i).x);
      t2 = static_cast<T>(grad_h2(i).y);
      t3 = static_cast<T>(grad_h3(i).z);

      divergence(i) = t1 + t2 + t3;
    }
}
  

template <class T>
inline
Vector3D<T>  HField3DUtils::sumTest(Array3D<Vector3D<T> >& hField)
{
  Vector3D<double> sum_d(0,0,0);
  Vector3D<unsigned int> size = hField.getSize();
  for (unsigned int z = 0; z < size.z; ++z) {
    for (unsigned int y = 0; y < size.y; ++y) {
      for (unsigned int x = 0; x < size.x; ++x) {
            sum_d.x += hField(x, y, z).x;
            sum_d.y += hField(x, y, z).y;
            sum_d.z += hField(x, y, z).z;
      }
    }
  }
  return Vector3D<T>(sum_d.x,sum_d.y,sum_d.z);
}

template <class T, class U>
inline
void
HField3DUtils::
jacobian(const Array3D<Vector3D<U> >& hField,
	 Array3D<T>& jacobian,
	 Vector3D<double> spacing)
{
  Vector3D<unsigned int> size = hField.getSize();
  unsigned int numElements = hField.getNumElements();

  // resize jacobian array if necessary
  if (jacobian.getSize() != size)
    {
      jacobian.resize(size);
    }

  // build scalar images h1, h2, and h3
  // from the transformation field
  Array3D<U> h1(size);
  Array3D<U> h2(size);
  Array3D<U> h3(size);
  unsigned int i; // stupid vc++
  for (i = 0; i < numElements; ++i)
    {
      h1(i) = hField(i).x;
      h2(i) = hField(i).y;
      h3(i) = hField(i).z;
    }
    
  // compute the gradients of h1, h2, and h3
  Array3D<Vector3D<U> > grad_h1(size);
  Array3D<Vector3D<U> > grad_h2(size);
  Array3D<Vector3D<U> > grad_h3(size);   
  Array3DUtils::computeGradient(h1, grad_h1, spacing);
  Array3DUtils::computeGradient(h2, grad_h2, spacing);
  Array3DUtils::computeGradient(h3, grad_h3, spacing);

  // compute the jacobian
  T t1, t2, t3;
  for (i = 0; i < numElements; ++i)
    {
      t1 = static_cast<T>(grad_h1(i).x * (grad_h2(i).y * grad_h3(i).z -
					  grad_h2(i).z * grad_h3(i).y));
      t2 = static_cast<T>(grad_h1(i).y * (grad_h2(i).x * grad_h3(i).z -
					  grad_h2(i).z * grad_h3(i).x));
      t3 = static_cast<T>(grad_h1(i).z * (grad_h2(i).x * grad_h3(i).y -
					  grad_h2(i).y * grad_h3(i).x));
      jacobian(i) = t1 - t2 + t3;
    }
}

template <class T>
inline
void
HField3DUtils::
pointwiseL2Norm(const Array3D<Vector3D<T> >& hField,
		Array3D<T> &norms)
{
  norms.resize(hField.getSize());
  unsigned int size = hField.getSize().productOfElements();
  for (unsigned int i = 0; i < size; ++i) {    
    norms(i) = 
      sqrt((hField(i).x) * (hField(i).x) +
	   (hField(i).y) * (hField(i).y) +
	   (hField(i).z) * (hField(i).z));
  }
}

template <class T>
inline
void
HField3DUtils::
minMaxDeformationL2Norm(const Array3D<Vector3D<T> >& hField,
			double& min, double& max)
{
  Vector3D<unsigned int> size = hField.getSize();

  //
  // compute min, max
  //
  min = std::numeric_limits<double>::max();
  max = 0;
  for (unsigned int z = 0; z < size.z; ++z) {    
    for (unsigned int y = 0; y < size.y; ++y) {    
      for (unsigned int x = 0; x < size.x; ++x) {    
	double normL2 = 
	  sqrt((hField(x,y,z).x - x) * (hField(x,y,z).x - x) +
	       (hField(x,y,z).y - y) * (hField(x,y,z).y - y) +
	       (hField(x,y,z).z - z) * (hField(x,y,z).z - z));
	if (normL2 < min) min = normL2;
	if (normL2 > max) max = normL2;
      }
    }
  }
}

template <class T>
inline
void
HField3DUtils::
minMaxVelocityL2Norm(const Array3D<Vector3D<T> >& hField,
		     double& min, double& max)
{
  Vector3D<unsigned int> size = hField.getSize();

  //
  // compute min, max
  //
  min = std::numeric_limits<double>::max();
  max = 0;
  for (unsigned int z = 0; z < size.z; ++z) {    
    for (unsigned int y = 0; y < size.y; ++y) {    
      for (unsigned int x = 0; x < size.x; ++x) {    
	double normL2 = 
	  sqrt((hField(x,y,z).x) * (hField(x,y,z).x) +
	       (hField(x,y,z).y) * (hField(x,y,z).y) +
	       (hField(x,y,z).z) * (hField(x,y,z).z));
	if (normL2 < min) min = normL2;
	if (normL2 > max) max = normL2;
      }
    }
  }
}

template <class T>
inline
void
HField3DUtils::
jacobian(const Array3D<Vector3D<T> >& h,
	 const T& x, const T& y, const T& z,
	 double* const J)
{
  // a (much) faster version of the floor function
  int floorX = static_cast<int>(x);
  int floorY = static_cast<int>(y);
  int floorZ = static_cast<int>(z);
  if (x < 0 && x != static_cast<int>(x)) --floorX;
  if (y < 0 && y != static_cast<int>(y)) --floorY;
  if (z < 0 && z != static_cast<int>(z)) --floorZ;
  int ceilX = floorX + 1;
  int ceilY = floorY + 1;
  int ceilZ = floorZ + 1;

  //
  // ^
  // |  J3   J2       -z->        J4   J5
  // y           --next slice-->      
  // |  J0   J1                   J7   J6
  //
  //      -x->
  //   

  double J0[9]; double J1[9]; double J2[9]; double J3[9]; 
  double J4[9]; double J5[9]; double J6[9]; double J7[9];

  jacobianAtGridpoint(h, floorX, floorY, floorZ, J0);
  jacobianAtGridpoint(h, ceilX,  floorY, floorZ, J1);
  jacobianAtGridpoint(h, ceilX,  ceilY,  floorZ, J2);
  jacobianAtGridpoint(h, floorX, ceilY,  floorZ, J3);
  jacobianAtGridpoint(h, floorX, ceilY,  ceilZ,  J4);
  jacobianAtGridpoint(h, ceilX,  ceilY,  ceilZ,  J5);
  jacobianAtGridpoint(h, ceilX,  floorY, ceilZ,  J6);
  jacobianAtGridpoint(h, floorX, floorY, ceilZ,  J7);

  //
  // do trilinear interpolation
  //
  const double t = x - floorX;
  const double u = y - floorY;
  const double v = z - floorZ;
  const double oneMinusT = 1.0 - t;
  const double oneMinusU = 1.0 - u;
  const double oneMinusV = 1.0 - v;

  //
  // this is the basic trilerp function...
  //
  //     h = 
  //       v0 * (1 - t) * (1 - u) * (1 - v) +
  //       v1 * t       * (1 - u) * (1 - v) +
  //       v2 * t       * u       * (1 - v) +
  //       v3 * (1 - t) * u       * (1 - v) +
  //       v4 * (1 - t) * u       * v       +
  //       v5 * t       * u       * v       +
  //       v6 * t       * (1 - u) * v       +
  //       v7 * (1 - t) * (1 - u) * v;
  //
  // the following nested version saves 30 multiplies.
  //

  for (int i = 0; i < 9; ++i)
    {
      J[i] =
	oneMinusT * (oneMinusU * (J0[i] * oneMinusV + J7[i] * v)  +
		     u         * (J3[i] * oneMinusV + J4[i] * v)) +
	t         * (oneMinusU * (J1[i] * oneMinusV + J6[i] * v)  +
		     u         * (J2[i] * oneMinusV + J5[i] * v));
    }
}
  
template <class T>
inline
void
HField3DUtils::
jacobianAtGridpoint(const Array3D<Vector3D<T> >& h,
		    int x, int y, int z,
		    double* const J)
{
  Vector3D<int> size = h.getSize();

  //
  // arbitrarily set Jacobian to identity if outside region
  //
  if (x < 0 || x >= size.x ||
      y < 0 || y >= size.y ||
      z < 0 || z >= size.z)
    {
      J[0] = J[4] = J[8] = 1;
      J[1] = J[2] = J[3] = J[5] = J[6] = J[7] = 0;
      return;
    }

  //
  // do symmetric difference unless on the edge, then do forward
  // difference
  //

  if (x > 0 && x < (size.x - 1))
    {
      // symmetric difference
      J[0] = (h(x+1,y,z).x - h(x-1,y,z).x) / 2.0;
      J[3] = (h(x+1,y,z).y - h(x-1,y,z).y) / 2.0;
      J[6] = (h(x+1,y,z).z - h(x-1,y,z).z) / 2.0;
    }
  else if (x == 0)
    {	  
      // forward difference
      J[0] = (h(x+1,y,z).x - h(x,y,z).x);
      J[3] = (h(x+1,y,z).y - h(x,y,z).y);
      J[6] = (h(x+1,y,z).z - h(x,y,z).z);
    }
  else
    {
      // backwards difference
      J[0] = (h(x,y,z).x - h(x-1,y,z).x);
      J[3] = (h(x,y,z).y - h(x-1,y,z).y);
      J[6] = (h(x,y,z).z - h(x-1,y,z).z);
    }

  if (y > 0 && y < (size.y - 1))
    {
      // symmetric difference
      J[1] = (h(x,y+1,z).x - h(x,y-1,z).x) / 2.0;
      J[4] = (h(x,y+1,z).y - h(x,y-1,z).y) / 2.0;
      J[7] = (h(x,y+1,z).z - h(x,y-1,z).z) / 2.0;
    }
  else if (y == 0)
    {
      // forward difference
      J[1] = (h(x,y+1,z).x - h(x,y,z).x);
      J[4] = (h(x,y+1,z).y - h(x,y,z).y);
      J[7] = (h(x,y+1,z).z - h(x,y,z).z);
    }
  else
    {
      // backwards difference
      J[1] = (h(x,y,z).x - h(x,y-1,z).x);
      J[4] = (h(x,y,z).y - h(x,y-1,z).y);
      J[7] = (h(x,y,z).z - h(x,y-1,z).z);
    }
    
  if (z > 0 && z < (size.z - 1))
    {
      // symmetric difference
      J[2] = (h(x,y,z+1).x - h(x,y,z-1).x) / 2.0;
      J[5] = (h(x,y,z+1).y - h(x,y,z-1).y) / 2.0;
      J[8] = (h(x,y,z+1).z - h(x,y,z-1).z) / 2.0;
    }
  else if (z == 0)
    {
      // forward difference
      J[2] = (h(x,y,z+1).x - h(x,y,z).x);
      J[5] = (h(x,y,z+1).y - h(x,y,z).y);
      J[8] = (h(x,y,z+1).z - h(x,y,z).z);
    }
  else
    {
      // backwards difference
      J[2] = (h(x,y,z).x - h(x,y,z-1).x);
      J[5] = (h(x,y,z).y - h(x,y,z-1).y);
      J[8] = (h(x,y,z).z - h(x,y,z-1).z);
    }
}

template <class T>
inline
void
HField3DUtils::
hessianAtGridpoint(const Array3D<Vector3D<T> >& h,
		   int x, int y, int z,
		   double* const H)
{
  Vector3D<int> size = h.getSize();

  //
  // arbitrarily set hessian to zero if outside region
  //
  if (x < 1 || x >= size.x - 1 ||
      y < 1 || y >= size.y - 1 ||
      z < 1 || z >= size.z - 1)
    {
      for (int i = 0; i < 27; ++i)
	{
	  H[i] = 0;
	}
      return;
    }

  //
  // non mixed partials
  //
  // dx^2
  H[0]  = h(x+1,y,z).x - 2.0 * h(x,y,z).x + h(x-1,y,z).x;
  H[9]  = h(x+1,y,z).y - 2.0 * h(x,y,z).y + h(x-1,y,z).y;
  H[18] = h(x+1,y,z).z - 2.0 * h(x,y,z).z + h(x-1,y,z).z;

  // dy^2
  H[4]  = h(x,y+1,z).x - 2.0 * h(x,y,z).x + h(x,y-1,z).x;
  H[13] = h(x,y+1,z).y - 2.0 * h(x,y,z).y + h(x,y-1,z).y;
  H[22] = h(x,y+1,z).z - 2.0 * h(x,y,z).z + h(x,y-1,z).z;

  // dz^2
  H[8]  = h(x,y,z+1).x - 2.0 * h(x,y,z).x + h(x,y,z-1).x;
  H[17] = h(x,y,z+1).y - 2.0 * h(x,y,z).y + h(x,y,z-1).y;
  H[26] = h(x,y,z+1).z - 2.0 * h(x,y,z).z + h(x,y,z-1).z;
    
  //
  // mixed partials
  //
  // dxdy
  H[1]  = H[3]  =
    + 0                  + h(x,y-1,z).x/2.0 - h(x+1,y-1,z).x/2.0
    + h(x-1,y,z).x/2.0   - h(x,y,z).x       + h(x+1,y,z).x/2.0
    - h(x-1,y+1,z).x/2.0 + h(x,y+1,z).x/2.0 + 0;
  H[10] = H[12] = 
    + 0                  + h(x,y-1,z).y/2.0 - h(x+1,y-1,z).y/2.0
    + h(x-1,y,z).y/2.0   - h(x,y,z).y       + h(x+1,y,z).y/2.0
    - h(x-1,y+1,z).y/2.0 + h(x,y+1,z).y/2.0 + 0;
  H[19] = H[21] = 
    + 0                  + h(x,y-1,z).z/2.0 - h(x+1,y-1,z).z/2.0
    + h(x-1,y,z).z/2.0   - h(x,y,z).z       + h(x+1,y,z).z/2.0
    - h(x-1,y+1,z).z/2.0 + h(x,y+1,z).z/2.0 + 0;

  // dxdz
  H[2]  = H[6]  = 
    + 0                  + h(x,y,z-1).x/2.0 - h(x+1,y,z-1).x/2.0
    + h(x-1,y,z).x/2.0   - h(x,y,z).x       + h(x+1,y,z).x/2.0
    - h(x-1,y,z+1).x/2.0 + h(x,y,z+1).x/2.0 + 0;
  H[11] = H[15] = 
    + 0                  + h(x,y,z-1).y/2.0 - h(x+1,y,z-1).y/2.0
    + h(x-1,y,z).y/2.0   - h(x,y,z).y       + h(x+1,y,z).y/2.0
    - h(x-1,y,z+1).y/2.0 + h(x,y,z+1).y/2.0 + 0;
  H[20] = H[24] = 
    + 0                  + h(x,y,z-1).z/2.0 - h(x+1,y,z-1).z/2.0
    + h(x-1,y,z).z/2.0   - h(x,y,z).z       + h(x+1,y,z).z/2.0
    - h(x-1,y,z+1).z/2.0 + h(x,y,z+1).z/2.0 + 0;

  // dydz
  H[5]  = H[7]  = 
    + 0                  + h(x,y,z-1).x/2.0 - h(x,y+1,z-1).x/2.0
    + h(x,y-1,z).x/2.0   - h(x,y,z).x       + h(x,y+1,z).x/2.0
    - h(x,y-1,z+1).x/2.0 + h(x,y,z+1).x/2.0 + 0;
  H[14] = H[16] = 
    + 0                  + h(x,y,z-1).y/2.0 - h(x,y+1,z-1).y/2.0
    + h(x,y-1,z).y/2.0   - h(x,y,z).y       + h(x,y+1,z).y/2.0
    - h(x,y-1,z+1).y/2.0 + h(x,y,z+1).y/2.0 + 0;
  H[23] = H[25] = 
    + 0                  + h(x,y,z-1).z/2.0 - h(x,y+1,z-1).z/2.0
    + h(x,y-1,z).z/2.0   - h(x,y,z).z       + h(x,y+1,z).z/2.0
    - h(x,y-1,z+1).z/2.0 + h(x,y,z+1).z/2.0 + 0;
}

template <class T>
inline
bool
HField3DUtils::
inverseOfPoint(const Array3D<Vector3D<T> >& h,
	       const T& x, const T& y, const T& z,
	       T& hinvx, T& hinvy, T& hinvz,
	       float thresholdDistance)
{
  // get a coarse estimate
  int estimateX, estimateY, estimateZ;
  // make compiler happy
  estimateX = estimateY = estimateZ = 0;

  float dist = inverseClosestPoint(h, 
				   x, y, z,
				   estimateX, estimateY, estimateZ);

  if (dist > thresholdDistance) {
    hinvx = x;
    hinvy = y;
    hinvz = z;
    return false;
  }

  // refine estimate
  return inverseOfPointRefine(h, 
			      x, y, z,
			      estimateX, estimateY, estimateZ,
			      hinvx, hinvy, hinvz);
}

template <class T>
inline
float 
HField3DUtils::
inverseClosestPoint(const Array3D<Vector3D<T> >& h,
		    const T& x, const T& y, const T& z,
		    int& hinvx, int& hinvy, int& hinvz)
{
  if (h.getNumElements() == 0) return HUGE_VAL;

  Vector3D<T> xVec(x,y,z);
  Vector3D<int> guess(0,0,0);
  double minDistSq = h(0,0,0).distanceSquared(xVec);
    
  // search on a coarse grid
  int gridsize = 3;
  for (unsigned int z = 0; z < h.getSizeZ(); z += gridsize) {
    for (unsigned int y = 0; y < h.getSizeY(); y += gridsize) {
      for (unsigned int x = 0; x < h.getSizeX(); x += gridsize) {
	double currDistSq = h(x,y,z).distanceSquared(xVec);
	if (currDistSq < minDistSq) 
	  {
	    minDistSq = currDistSq;
	    guess.x = x;
	    guess.y = y;
	    guess.z = z;
	  }
      }
    }
  }

  int slop = 10;
  unsigned int xRegionMin = guess.x - slop > 0 ? guess.x - slop : 0;
  unsigned int xRegionMax = 
    guess.x + slop < (int) h.getSizeX() 
    ? guess.x + slop
    : h.getSizeX() - 1;
  unsigned int yRegionMin = guess.y - slop > 0 ? guess.y - slop : 0;
  unsigned int yRegionMax =
    guess.y + slop < (int) h.getSizeY() 
    ? guess.y + slop
    : h.getSizeY() - 1;
  unsigned int zRegionMin = guess.z - slop > 0 ? guess.z - slop : 0;
  unsigned int zRegionMax =
    guess.z + slop < (int) h.getSizeZ() 
    ? guess.z + slop
    : h.getSizeZ() - 1;
    
  // search around best coarse point
  for (unsigned int z = zRegionMin; z <= zRegionMax; ++z) {
    for (unsigned int y = yRegionMin; y <= yRegionMax; ++y) {
      for (unsigned int x = xRegionMin; x <= xRegionMax; ++x) {
	double currDistSq = h(x,y,z).distanceSquared(xVec);
	if (currDistSq < minDistSq) 
	  {
	    minDistSq = currDistSq;
	    guess.x = x;
	    guess.y = y;
	    guess.z = z;
	  }
      }
    }
  }

  hinvx = guess.x;
  hinvy = guess.y;
  hinvz = guess.z;
  return minDistSq;
}

template <class T>
inline
bool
HField3DUtils::
inverseOfPointRefine(const Array3D<Vector3D<T> >& h,
		     const T& x, const T& y, const T& z, 
		     const int& x0, const int& y0, const int& z0,
		     T& hinvx, T& hinvy, T& hinvz)
{
  if (x0 < 0 || x0 >= (int) h.getSizeX() ||
      y0 < 0 || y0 >= (int) h.getSizeY() ||
      z0 < 0 || z0 >= (int) h.getSizeZ())
    {
      return false;
    }

  //
  // approximate the inverse using a first order Taylor exapnsion,
  // according to the following derivation
  // 
  // hInv(x) = x0 + d
  // h(x0 + d) = x
  // h(x0 + d) = h(x0) + Jh|x0 * d + h.o.t.
  // x = h(x0) + Jh|x0 * d + h.o.t.
  // d = JInvh|x0 * (x - h(x0))
  // 
  // hinv(x) = x0 + JInvh|x0 * (x - h(x0))
  //    
  double J[9];
  //double JInv[9];
  jacobianAtGridpoint(h, x0, y0, z0, J);
  Matrix3D<double> JMat(J);
//   Matrix3D<double> JJt = JMat*JMat.Transpose();
   Matrix3D<double> JInvMat;
  
//  bool success = Matrix3D<double>::computeInverse(JJt.a, JInvMat.a);
  bool success = Matrix3D<double>::computeInverse(JMat.a, JInvMat.a);
  //JInvMat = JMat*JInvMat;
  if (!success) 
    {
      // J(h(x)) was not invertable, arbitrarily go with identity
      std::cerr << "At (" << x << "," << y << "," << z 
 		<< "): J(h(x)) is non-invertable." << std::endl;
      hinvx = x0;
      hinvy = y0;
      hinvz = z0;
    }
  else
    {
      
      hinvx = x0 
	+ JInvMat.a[0] * (x - h(x0,y0,z0).x) 
	+ JInvMat.a[1] * (y - h(x0,y0,z0).y) 
	+ JInvMat.a[2] * (z - h(x0,y0,z0).z); 
      hinvy = y0 
	+ JInvMat.a[3] * (x - h(x0,y0,z0).x) 
	+ JInvMat.a[4] * (y - h(x0,y0,z0).y) 
	+ JInvMat.a[5] * (z - h(x0,y0,z0).z); 
      hinvz = z0 
	+ JInvMat.a[6] * (x - h(x0,y0,z0).x) 
	+ JInvMat.a[7] * (y - h(x0,y0,z0).y) 
	+ JInvMat.a[8] * (z - h(x0,y0,z0).z); 
    }    
  return true;
}

inline
void
HField3DUtils::
fillByFunction(Array3D< Vector3D<float> >& h,
	       Vector3D<double> origin,
	       Vector3D<double> spacing,
	       Vector3D<float> (*func)(double, double, double))
{
  for (unsigned int z = 0; z < h.getSizeZ(); ++z) {
    for (unsigned int y = 0; y < h.getSizeY(); ++y) {
      for (unsigned int x = 0; x < h.getSizeX(); ++x) {
	Vector3D<double> point(x, y, z);
	Vector3D<double> pointWC = point * spacing + origin;
	Vector3D<float> offset = func(pointWC.x, pointWC.y, pointWC.z);
	Vector3D<float> hFieldVal = point + offset / spacing;
	h(x, y, z) = hFieldVal;
      }
    }
  }
}  

