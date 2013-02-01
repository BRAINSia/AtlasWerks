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

#ifndef __GREEDY_ITERATOR_CPU__
#define __GREEDY_ITERATOR_CPU__

#include "AtlasWerksTypes.h"
#include "Energy.h"
#include "DeformationIteratorInterface.h"
#include "GreedyIteratorParam.h"
#include "GreedyDeformationData.h"
#include "DiffOper.h"
#include "MultiGaussianKernel.h"
#include "AffineTransform3D.h"

class GreedyIteratorCPU 
  : public DeformationIteratorInterface
{
  
public:

  typedef GreedyDeformationData DeformationDataType;
  typedef GreedyIteratorParam ParamType;
  

  GreedyIteratorCPU(const Vector3D<unsigned int> &size, 
		    const Vector3D<Real> &origin,
		    const Vector3D<Real> &spacing,
		    bool debug=true);
  ~GreedyIteratorCPU();
  
  void SetScaleLevel(const MultiscaleManager &scaleManager,
		     const GreedyIteratorParam &param);
  
  void Iterate(DeformationDataInterface &deformaitonData);

  void UpdateStepSizeNextIteration(){ mUpdateStepSizeNextIter = true; }
  
protected:
  
  void pointwiseMultiplyBy_FFTW_Safe(DiffOper::FFTWVectorField &lhs, 
				     const Array3D<Real> &rhs);

  void updateDeformation(GreedyDeformationData &data);
  
  Real calcMaxDisplacement();
  
  Real mMaxPert;

  bool mUpdateStepSizeNextIter;

  /** Do we calculate energy for debugging? */
  bool mDebug;

  /** norm kernel */
  KernelInterface *mKernel;
  /** Pointer to the internal field of the DiffOper */
  DiffOper::FFTWVectorField *mDiffOpVF;
  /** Scratch Image */
  RealImage *mScratchI;
  /** Scratch Vector Field */
  VectorField *mScratchV;

};

#endif // __GREEDY_ITERATOR_CPU__
