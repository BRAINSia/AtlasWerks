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

#ifndef __LDMM_SHOOTING_ITERATOR_GPU__
#define __LDMM_SHOOTING_ITERATOR_GPU__

#include "AtlasWerksTypes.h"
#include "VectorMath.h"
#include "cudaInterface.h"
#include "cudaReduce.h"
#include "KernelInterfaceGPU.h"

#include "LDMMShootingDefDataGPU.h"
#include "LDMMIteratorParam.h"
#include "LDMMEnergy.h"
#include "MultiscaleManager.h"

#include "log.h"

class LDMMShootingIteratorGPU
  : public DeformationIteratorInterface
{
  
public:

  typedef LDMMIteratorParam ParamType;
  typedef LDMMShootingDefDataGPU DeformationDataType;

  LDMMShootingIteratorGPU(const SizeType &size, 
			  const OriginType &origin,
			  const SpacingType &spacing,
			  unsigned int nTimeSteps,
			  bool debug=true);
  ~LDMMShootingIteratorGPU();
  
  void SetScaleLevel(const MultiscaleManager &scaleManager,
		     const LDMMIteratorParam &param);
  
  void Iterate(DeformationDataInterface &deformaitonData);

  void UpdateStepSizeNextIteration(){ mUpdateStepSizeNextIter = true; }

  /** Update dPhi0T and dPhiT0 from the alpha0 in the defData */
  void UpdateDeformations(LDMMShootingDefDataGPU &defData);

  /**
   * Reparameterize velocity fields to have constant speed
   */
  void ReParameterize(LDMMShootingDefDataGPU &defData){
    throw AtlasWerksException(__FILE__, __LINE__, 
			      "Error, ReParameterize not implemented yet");
  }
protected:
  
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

  const LDMMIteratorParam *mParam;

  /** Reduce plan*/
  cplReduce *mdRd; 

  /** Differential operator */
  KernelInterfaceGPU *mdKernel;

  // v_t
  cplVector3DArray mdVT;

  // alpha_t
  float *mdAlphaT;

  /** Scratch Images */
  float *mdScratchI;
  float *mdAlpha;
  /** Scratch Vector Field */
  cplVector3DArray mdScratchV;

};

#endif // __LDMM_SHOOTING_ITERATOR_GPU__
