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

#ifndef __LDMM_AD_SHOOTING_ITERATOR_CPU__
#define __LDMM_AD_SHOOTING_ITERATOR_CPU__

#include "AtlasWerksTypes.h"
#include "DeformationIteratorInterface.h"
#include "MultiscaleManager.h"
#include "KernelInterface.h"
#include "LDMMShootingDefData.h"
#include "LDMMIteratorParam.h"

class LDMMAdShootingIteratorCPU
  : public DeformationIteratorInterface
{
  
public:

  typedef LDMMIteratorParam ParamType;
  typedef LDMMShootingDefData DeformationDataType;

  LDMMAdShootingIteratorCPU(const SizeType &size, 
			  const OriginType &origin,
			  const SpacingType &spacing,
			  unsigned int nTimeSteps,
			  bool debug=true);
  ~LDMMAdShootingIteratorCPU();
  
  void SetScaleLevel(const MultiscaleManager &scaleManager,
		     const LDMMIteratorParam &param);
  
  void Iterate(DeformationDataInterface &deformaitonData);

  void UpdateStepSizeNextIteration(){ mUpdateStepSizeNextIter = true; }

  void ComputeJacDet(LDMMShootingDefData &data, RealImage &jacDet);

  void finalUpdatePhi0T(LDMMShootingDefData &data);
  void finalUpdatePhiT0(LDMMShootingDefData &data);
  
  /**
   * Reparameterize velocity fields to have constant speed
   */
  void ReParameterize(LDMMShootingDefData &defData){
    throw AtlasWerksException(__FILE__, __LINE__, 
			      "Error, ReParameterize not implemented yet");
  }

protected:
  
  void pointwiseMultiplyBy_FFTW_Safe(KernelInterface::KernelInternalVF &lhs, 
				     const Array3D<Real> &rhs);

  void
  UpdatePhi0T(VectorField &phi, 
	      RealImage &jacDet, 
	      const VectorField &v);


  void backwardAdIntegration(LDMMShootingDefData &data);
  /** The number of timesteps (intermediate images) */
  const unsigned int mNTimeSteps;
  
  /** weight importance of smooth velocity vs. matched images */
  Real mSigma;

  /** Do we calculate energy for debugging? */
  bool mDebug;

  /** Has SetScaleLevel been called yet? */
  bool mInitialized;
  
  /** normalized method for choosing step size, takes values between 0 and 1 */
  Real mMaxPert;
  /** Should the step size be updated based on MaxPert on the next iteration? */
  bool mUpdateStepSizeNextIter;

  /** commented out mVT because we are saving vector fields at all timepoints for backward integration using adjoint equations */
  // v_t

  // VectorField *mVT;
  /** Vector to store V's for each timepoint (not storing alphaT's and IT's) */ 
  std::vector<VectorField*> mVVec;
  RealImage *mIHatT;
  RealImage *mAlphaHatT;
  RealImage *mITildeT;
  RealImage *mAlphaTildeT;
  RealImage *mIT;

  // alpha_t
  RealImage *mAlphaT;

  /** Differential operator */
  KernelInterface *mKernel;
  /** Pointer to the internal field of the Kernel */
  KernelInterface::KernelInternalVF *mKernelVF;

  /** Scratch Images */
  RealImage *mScratchI;
  RealImage *mScratchI2;
  RealImage *mAlpha;
  /** Scratch Vector Field */
  VectorField *mScratchV;  
  VectorField *mScratchV2;
};

#endif // __LDMM_AD_SHOOTING_ITERATOR_CPU__
