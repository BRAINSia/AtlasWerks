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

#ifndef __LDMM_AD_SHOOTING_ITERATOR_GPU__
#define __LDMM_AD_SHOOTING_ITERATOR_GPU__

#ifndef TRADE_MEM_FOR_SPEED
#define TRADE_MEM_FOR_SPEED 1
#endif

#include "AtlasWerksTypes.h"
#include "DeformationIteratorInterface.h"
#include "MultiscaleManager.h"
#include "VectorMath.h"
#include "cudaInterface.h"
#include "cudaReduce.h"
#include "KernelInterfaceGPU.h"
#include "LDMMShootingDefDataGPU.h"
#include "LDMMIteratorParam.h"

class LDMMAdShootingIteratorGPU
  : public DeformationIteratorInterface
{
  
public:

  typedef LDMMIteratorParam ParamType;
  typedef LDMMShootingDefDataGPU DeformationDataType;

  LDMMAdShootingIteratorGPU(const SizeType &size, 
			    const OriginType &origin,
			    const SpacingType &spacing,
			    unsigned int nTimeSteps,
			    bool debug=true);

  ~LDMMAdShootingIteratorGPU();
  
  void SetScaleLevel(const MultiscaleManager &scaleManager,
		     const LDMMIteratorParam &param);
  
  void Iterate(DeformationDataInterface &deformaitonData);

  void UpdateStepSizeNextIteration(){ mUpdateStepSizeNextIter = true; }

  void finalUpdatePhi0T(DeformationDataType &data);
  void finalUpdatePhiT0(DeformationDataType &data);
  
  /**
   * Reparameterize velocity fields to have constant speed
   */
  void ReParameterize(DeformationDataType &defData){
    throw AtlasWerksException(__FILE__, __LINE__, 
			      "Error, ReParameterize not implemented yet");
  }

protected:
  
  void backwardAdIntegration(DeformationDataType &data);
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

  /** Vector to store V's for each timepoint (not storing alphaT's and IT's) */ 
#if TRADE_MEM_FOR_SPEED
  std::vector<cplVector3DArray> mdPhiT0;
  std::vector<cplVector3DArray> mdPhi0T;
  cplVector3DArray mdVT;
#else
  std::vector<cplVector3DArray> mdVVec;
#endif
  float *mdIHatT;
  float *mdAlphaHatT;
  float *mdITildeT;
  float *mdAlphaTildeT;
  float *mdIT;

  // alpha_t
  float *mdAlphaT;

  const LDMMIteratorParam *mParam;

  /** Reduce plan*/
  cplReduce *mdRd; 

  /** Differential operator */
  KernelInterfaceGPU *mdKernel;

  /** Scratch Images */
  float *mdScratchI;
  float *mdScratchI2;
  /** Scratch Vector Field */
  cplVector3DArray mdScratchV;  
  cplVector3DArray mdScratchV2;
};

#endif // __LDMM_AD_SHOOTING_ITERATOR_GPU__
