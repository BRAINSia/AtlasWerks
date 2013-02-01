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


#ifndef __LDMM_ITERATOR_GPU_H__
#define __LDMM_ITERATOR_GPU_H__

#include "AtlasWerksTypes.h"
#include "VectorMath.h"
#include "cudaInterface.h"
#include "cudaReduce.h"
#include "DiffOperGPU.h"
#include "KernelInterfaceGPU.h"
#include "LDMMParam.h"
#include "LDMMDeformationDataGPU.h"

class LDMMIteratorGPU
  : public DeformationIteratorInterface
{
public:

  typedef LDMMDeformationDataGPU DeformationDataType;
  typedef LDMMIteratorParam ParamType;

  LDMMIteratorGPU(SizeType &size, 
		  OriginType &origin,
		  SpacingType &spacing,
		  unsigned int &nTimeSteps,
		  bool debug=true);
  
  ~LDMMIteratorGPU();

  void SetScaleLevel(const MultiscaleManager &scaleManager,
		     const LDMMIteratorParam &param);
  
  //   /** Get/set whether to calculate debugging info (energy calculations, etc.) */
  //   void SetDebug(bool debug){mDebug = debug; }
  //   bool GetDebug(){return mDebug; }
  
  //   /** If debugging is enabled, these return the calculated energies
  //     from the previous iteration */
  //   Real GetImageEnergy(){ return mImageEnergy; }
  //   Real GetVectorEnergy(){ return mVectorEnergy; }
  //   Real GetTotalEnergy(){ return mTotalEnergy; }
  //   /** Returns a pointer to an internal array containing the vector 
  //       energy from each timestep.*
  //   const Real* GetVectorStepEnergy(){ return mVectorStepEnergy; }

  void Iterate(DeformationDataInterface &defData);

  void Iterate(DeformationDataInterface &defData,
	       bool computeAlphasOnly);

  void UpdateStepSizeNextIteration(){ mUpdateStepSizeNextIter = true; }

  /**
   * Compute the jacobian determinant of a series of velocity fields. 
   */
  void ComputeJacDet(std::vector<cplVector3DArray> &dV, float *dJacDet);
  /**
   * Compute a forward-deformed image given a series of velocity
   * feilds and initial image.  dH will contain the composed hField
   * upon return.
   */
  void ComputeForwardImage(std::vector<cplVector3DArray> &dV, 
			   float *dI0, cplVector3DArray &dH, 
			   float *dIDef);

  /**
   * Before requesting alphas, Iterate must be called with
   * computeAlphasOnly = true.  Results are only valid until the next
   * call to Iterate with computeAlphasOnly = false.
   */
  float *GetAlpha(unsigned int t);

  /**
   * Just access to internal data structures
   */
  cplVector3DArray& dH(){ return mdH; }

  /**
   * Just access to internal data structures
   */
  cplVector3DArray& dScratchV(){ return mdScratchV; }
  
  /**
   * Reparameterize velocity fields to have constant speed
   */
  void ReParameterize(LDMMDeformationDataGPU &defData);

protected:

  void IterateNew(DeformationDataInterface &deformationData,
		  bool computeAlphasOnly);

  void IterateOrig(DeformationDataInterface &deformationData,
		   bool computeAlphasOnly);

  void jacobianDetHField(float* dJ, cplVector3DArray& dH,
			 int sizeX, int sizeY, int sizeZ,
			 float spX, float spY, float spZ);

  /** The number of timesteps (intermediate images) */
  const unsigned int mNTimeSteps;
  /** Automatically adjust the step size? */
  bool mUseAdaptiveStepSize;
  /** If using adaptive step size, this will adjust the step size in
      relation to the minimum image spacing */
  Real mMaxPert;
  /** weight importance of smooth velocity vs. matched images */
  Real mSigma;

  const LDMMIteratorParam *mParam;

  /** Do we calculate energy for debugging? */
  bool mDebug;

  /** Should the step size be updated based on MaxPert on the next iteration? */
  bool mUpdateStepSizeNextIter;

  // ==========================================================
  bool mUseOrigGradientCalc;
  // ==== These are used for adjoint gradient calculation: ====
  /** backwards-deformed images */
  float **mdJ0t;
  /** jacobian determinant */
  float *mdDiffIm;
  // ==== These are used for original gradient calculation: ====
  float **mdJTt;
  float **mdJacDet;
  // ==========================================================
  
  /** Reduce plan*/
  cplReduce *mRd; 
  /** u field (v = Lu)*/
  cplVector3DArray mdU;
    

  /** FFT Solver */
  KernelInterfaceGPU *mKernel;

  /** Used for jacobian calculation */
  cplVector3DArray mdH;
  /** Scratch Image */
  float *mdScratchI;
  /** Scratch Vector Field */
  cplVector3DArray mdScratchV;

  // only used for computing jacobian determinant, should remove.
  cplVector3DArray mdGX;
  cplVector3DArray mdGY;
  cplVector3DArray mdGZ;
  
};

#endif
