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

#ifndef __LDMM_VEL_SHOOTING_ITERATOR_CPU__
#define __LDMM_VEL_SHOOTING_ITERATOR_CPU__

#include "AtlasWerksTypes.h"
#include "DeformationIteratorInterface.h"
#include "MultiscaleManager.h"
#include "KernelInterface.h"
#include "LDMMVelShootingDefData.h"
#include "LDMMIteratorParam.h"

class LDMMVelShootingIteratorCPU
  : public DeformationIteratorInterface
{
  
public:

  typedef LDMMIteratorParam ParamType;
  typedef LDMMVelShootingDefData DeformationDataType;

  LDMMVelShootingIteratorCPU(const SizeType &size, 
			  const OriginType &origin,
			  const SpacingType &spacing,
			  unsigned int nTimeSteps,
			  bool debug=true);
  ~LDMMVelShootingIteratorCPU();
  
  void SetScaleLevel(const MultiscaleManager &scaleManager,
		     const LDMMIteratorParam &param);
  
  void Iterate(DeformationDataInterface &deformaitonData);

  void UpdateStepSizeNextIteration(){ mUpdateStepSizeNextIter = true; }

  void ComputeJacDet(LDMMVelShootingDefData &data, RealImage &jacDet);
  
  /**
   * Reparameterize velocity fields to have constant speed
   */
  void ReParameterize(LDMMVelShootingDefData &defData){
    throw AtlasWerksException(__FILE__, __LINE__, 
			      "Error, ReParameterize not implemented yet");
  }

protected:
  
  void pointwiseMultiplyBy_FFTW_Safe(KernelInterface::KernelInternalVF &lhs, 
				     const Array3D<Real> &rhs);

  void updateVelocity(VectorField &v,
		      const KernelInterface::KernelInternalVF &uField,
		      Real stepSize);

  Real calcMaxDisplacement(VectorField &v,
			   const KernelInterface::KernelInternalVF &uField);
  void
  UpdatePhi0T(VectorField &phi, 
	      RealImage &jacDet, 
	      const VectorField &v);

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

  VectorField *mVT;
  VectorField *mLV0;
  VectorField *mLV0Def;
  // D \phi_{T0}
  VectorField *mDPhiT0_x;
  VectorField *mDPhiT0_y;
  VectorField *mDPhiT0_z;
  RealImage *mJacDet;
  RealImage *mVx;
  RealImage *mVy;
  RealImage *mVz;

  /** Differential operator */
  KernelInterface *mKernel;
  /** Just a pointer to the internal kernel VectorField*/
  KernelInterface::KernelInternalVF *mKernelVF;

  /** Scratch Images */
  RealImage *mScratchI;
  /** Scratch Vector Field */
  VectorField *mScratchV;

};

#endif // __LDMM_VEL_SHOOTING_ITERATOR_CPU__
