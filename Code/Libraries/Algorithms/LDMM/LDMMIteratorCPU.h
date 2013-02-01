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

#ifndef __LDMM_ITERATOR_CPU__
#define __LDMM_ITERATOR_CPU__

#include "AtlasWerksTypes.h"
#include "DeformationIteratorInterface.h"
#include "KernelInterface.h"
#include "KernelFactory.h"
#include "LDMMDeformationData.h"
#include "LDMMEnergy.h"
#include "LDMMIteratorParam.h"
#include "MultiscaleManager.h"

class LDMMIteratorCPU 
  : public DeformationIteratorInterface
{
  
public:

  typedef LDMMDeformationData DeformationDataType;
  typedef LDMMIteratorParam ParamType;

  //
  // Constructor
  //
  LDMMIteratorCPU(const SizeType &size, 
		  const OriginType &origin,
		  const SpacingType &spacing,
		  unsigned int nTimeSteps,
		  bool debug=true);
  ~LDMMIteratorCPU();

  void SetScaleLevel(const MultiscaleManager &scaleManager,
		     const LDMMIteratorParam &param);
  
  void Iterate(DeformationDataInterface &deformaitonData);
  
  void Iterate(DeformationDataInterface &defData,
	       bool computeAlphasOnly);

  void UpdateStepSizeNextIteration(){ mUpdateStepSizeNextIter = true; }

  /**
   * Compute the jacobian determinant of a series of velocity fields. 
   */
  void ComputeJacDet(std::vector<VectorField*> &v, RealImage *jacDet);
  
  /**
   * Before requesting alphas, Iterate must be called with
   * computeAlphasOnly = true.  Results are only valid until the next
   * call to Iterate with computeAlphasOnly = false.
   */
  RealImage &GetAlpha(unsigned int t);

  /**
   * Reparameterize velocity fields to have constant speed
   */
  void ReParameterize(LDMMDeformationData &defData);

protected:
  
  void IterateNew(DeformationDataInterface &deformationData,
		  bool computeAlphasOnly);

  void IterateOrig(DeformationDataInterface &deformationData,
		   bool computeAlphasOnly);

  void pointwiseMultiplyBy_FFTW_Safe(KernelInterface::KernelInternalVF &lhs, 
				     const Array3D<Real> &rhs);

  void updateVelocity(VectorField &v,
		      Real vFac,
		      const VectorField &u,
		      Real uFac);

  Real calcMaxDisplacement(VectorField &v,
			   const KernelInterface::KernelInternalVF &uField);
//   // Used for adaptive step size calculation
//   Real calcUpdate(const VectorField &curV,
// 		  const KernelInterface::KernelInternalVF &uField,
// 		  VectorField &update);
  
  const LDMMIteratorParam *mParam;

  /** The number of timesteps (intermediate images) */
  const unsigned int mNTimeSteps;
  
  /** weight importance of smooth velocity vs. matched images */
  Real mSigma;

  /** Do we calculate energy for debugging? */
  bool mDebug;

  /** normalized method for choosing step size, takes values between 0 and 1 */
  Real mMaxPert;
  /** Should the step size be updated based on MaxPert on the next iteration? */
  bool mUpdateStepSizeNextIter;

  /** composition of displacement fields */
  VectorField *mHField;

  // ==========================================================
  bool mUseOrigGradientCalc;
  // ==== These are used for adjoint gradient calculation: ====
  /** backwards-deformed images */
  RealImage **mJ0t;
  /** jacobian determinant */
  RealImage *mDiffIm;
  // ==== These are used for original gradient calculation: ====
  RealImage **mJTt;
  RealImage **mJacDet;
  // ==========================================================

  /** Differential operator */
  KernelInterface *mKernel;
  /** Pointer to the kernel internal vector field */
  KernelInterface::KernelInternalVF *mKernelVF;
  /** Scratch Image */
  RealImage *mScratchI;
  /** Scratch Vector Field */
  VectorField *mScratchV;

};

#endif // __LDMM_ITERATOR_CPU__
