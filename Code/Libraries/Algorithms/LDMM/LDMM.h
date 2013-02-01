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

#ifndef __LDMM_H__
#define __LDMM_H__

#include <pthread.h>

#include "AtlasWerksTypes.h"

#include "ApplicationUtils.h"
#include "Array3D.h"
#include "Array3DUtils.h"
#include "DiffOper.h"
#include "HField3DIO.h"
#include "HField3DUtils.h"
#include "Image.h"
#include "ImageUtils.h"
#include "MultiscaleManager.h"

#include "CmdLineParser.h"

class LDMMIteratorOldParam : public CompoundParam {
public:
  LDMMIteratorOldParam(const std::string& name = "LDMMIterator", 
		       const std::string& desc = "Settings for LDMM iteration", 
		       ParamLevel level = PARAM_COMMON)
    : CompoundParam(name, desc, level)
  {
    this->AddChild(DiffOperParam("DiffOper"));

    this->
      AddChild(ValueParam<Real>("Sigma",
				"Controls tradeoff between image matching and velocity field smoothness",
				PARAM_COMMON,
				5.0));
    this->
      AddChild(ValueParam<Real>("StepSize",
				"Gradient descent step size, or maximum step size when using adaptive step size",
				PARAM_COMMON,
				0.005));
    this->
      AddChild(ValueParam<bool>("UseAdaptiveStepSize",
				"Use an adaptive step size, where each step is scaled to be maxPert*minSpacing",
				PARAM_COMMON,
				true));
    this->
      AddChild(ValueParam<Real>("MaxPert",
				"when using adaptive step size, step will be scaled to maxPert*minSpacing",
				PARAM_COMMON,
				0.1));
  }
  
  ParamAccessorMacro(DiffOperParam, DiffOper)
  ValueParamAccessorMacro(Real, Sigma)
  ValueParamAccessorMacro(Real, StepSize)
  ValueParamAccessorMacro(bool, UseAdaptiveStepSize)
  ValueParamAccessorMacro(Real, MaxPert)

  CopyFunctionMacro(LDMMIteratorOldParam)

};

class LDMMScaleLevelParam : public ScaleLevelSettingsParam {
public:
  LDMMScaleLevelParam(const std::string& name = "LDMMScaleLevel", 
		      const std::string& desc = "Settings for single-scale LDMM registration", 
		      ParamLevel level = PARAM_COMMON)
    : ScaleLevelSettingsParam(name, desc, level)
  {
    this->AddChild(LDMMIteratorOldParam("LDMMIterator"));
    this->
      AddChild(ValueParam<unsigned int>("NIterations",
					"Number of iterations",
					PARAM_COMMON,
					50));
    // debugging output
    this->
      AddChild(ValueParam<unsigned int>("OutputEveryNIterations",
					"If non-zero, write out intermediate data every N iteratioins",
					PARAM_COMMON,
					0));
  }
  
  ParamAccessorMacro(LDMMIteratorOldParam, LDMMIterator)
  ValueParamAccessorMacro(unsigned int, NIterations)
  ValueParamAccessorMacro(unsigned int, OutputEveryNIterations);

  CopyFunctionMacro(LDMMScaleLevelParam)
  
};

class LDMMOldParam : public MultiscaleParamBase<LDMMScaleLevelParam> {

public:
  LDMMOldParam(const std::string& name = "LDMM", 
	       const std::string& desc = "Settings for LDMM registration/atlas building", 
	       ParamLevel level = PARAM_COMMON)
    : MultiscaleParamBase<LDMMScaleLevelParam>(LDMMScaleLevelParam("LDMMScaleLevel"), name, desc, level)
  {
    this->
      AddChild(ValueParam<unsigned int>("NTimeSteps",
					"Number of timesteps (and therefore intermediate vector fields) to use",
					PARAM_COMMON,
					5));
    
    // If these aren't set, they inherit from the parameter file params of the same name
    this->AddChild(ValueParam<std::string>("OutputPrefix", "filename prefix to use", PARAM_COMMON, ""));  
    this->AddChild(ValueParam<std::string>("OutputSuffix", "filename extension to use (determines format)", PARAM_COMMON, ""));
    
    // Debugging params...
    this->AddChild(ValueParam<bool>("WriteMean",
				    "Write intermediate mean",
				    PARAM_COMMON,
				    false));

    this->AddChild(ValueParam<bool>("WriteAlphas", "Compute and save alpha images?", PARAM_COMMON, false));
  }

  ValueParamAccessorMacro(unsigned int, NTimeSteps)
  ValueParamAccessorMacro(bool, WriteMean)
  ValueParamAccessorMacro(bool, WriteAlphas)
  ValueParamAccessorMacro(std::string, OutputPrefix)
  ValueParamAccessorMacro(std::string, OutputSuffix)
  
  CopyFunctionMacro(LDMMOldParam)

};

class LDMMIterator {
public:
  LDMMIterator(const Vector3D<unsigned int> &size, 
	       const Vector3D<Real> &origin,
	       const Vector3D<Real> &spacing,
	       const unsigned int &nTimeSteps,
	       const LDMMIteratorOldParam &param,
	       bool debug=false);
  ~LDMMIterator();

  /** Get/set sigma, which controls tradeoff between smooth velocity
      field and more accurate image matching*/
  void SetSigma(Real sigma){ mSigma = sigma; }
  Real GetSigma(){ return mSigma; }
  /**
     Get/set the use of adaptive step size.  True by default.
   */
  void SetUseAdaptiveStepSize(bool b){ mUseAdaptiveStepSize = b; }
  bool GetUseAdaptiveStepSize(){ return mUseAdaptiveStepSize; }
  /** If using adaptive step size, the step size will be at most 
      MaxPert * (minimum voxel spacing) */
  void SetMaxPert(Real maxPert){ mMaxPert = maxPert; }
  Real GetMaxPert(){ return mMaxPert; }
  /** If not using adaptive step size, this step size will be used.
    Otherwise StepSize sets a maximum step size to use.
   */
  void SetStepSize(Real stepSize){ mStepSize = stepSize; }
  Real GetStepSize(){ return mStepSize; }
  /** Get/set whether to calculate debugging info (energy calculations, etc.) */
  void SetDebug(bool debug){mDebug = debug; }
  bool GetDebug(){return mDebug; }
  /** If debugging is enabled, these return the calculated energies
    from the previous iteration */
  Real GetImageEnergy(){ return mImageEnergy; }
  Real GetVectorEnergy(){ return mVectorEnergy; }
  Real GetTotalEnergy(){ return mTotalEnergy; }
  /** Returns a pointer to an internal array containing the vector 
      energy from each timestep.*/
  const Real* GetVectorStepEnergy(){ return mVectorStepEnergy; }

  void SetImages(const RealImage *initial, 
		 const RealImage *final);

  void Iterate(VectorField** v, VectorField *hField);

  Real ShootingIterateShoot(const RealImage *alpha0, const RealImage *I0, 
			    VectorField *vt,  RealImage *alphat, 
			    VectorField *hField, RealImage *Dphit, RealImage *J0t);
  void ShootingIterate(RealImage *alpha0, RealImage *alpha0inv, 
		       VectorField *hField, VectorField *hFieldInv,
		       RealImage *I0AtT, RealImage *ITAt0);

  /**
   * Compute the deformation from I0 to IT -- that is, the deformation
   * such that IT(hField) ~= I0
   */
  void ComputeForwardDef(const VectorField* const* v, VectorField *hField);
  /**
   * Compute the deformation from IT to I0 -- that is, the deformation
   * such that I0(hField) ~= IT
   */
  void ComputeReverseDef(const VectorField* const* v, VectorField *hField);
  
  /**
   * Compute the jacobian determinant of the set of velocity fields v
   */
  void ComputeJacDet(const VectorField* const* v, RealImage *jacDet);

  void SaveAlphaImages(RealImage **alpha);

  void SaveUFields(VectorField **uField);
  
  /**
   * get a copy of the DiffOper internal field.  Must be
   * correctly sized.
   */
  void GetUField(VectorField &vf);
  
  /**
   * Get the jacobian determinant as calculated in the previous call
   * to Iterate()
   */
  const RealImage* GetJacDet(){ return mJacDet[0]; }

protected:

  /** The number of timesteps (intermediate images) */
  const unsigned int mNTimeSteps;
  /** Automatically adjust the step size? */
  bool mUseAdaptiveStepSize;
  /** If using adaptive step size, this will adjust the step size in
    relation to the minimum image spacing */
  Real mMaxPert;
  /** The velocity field update step size */
  Real mStepSize;
  /** weight importance of smooth velocity vs. matched images */
  Real mSigma;
  /** The image size */
  Vector3D<unsigned int> mImSize;
  /** The image origin */
  Vector3D<Real> mImOrigin;
  /** The image spacing */
  Vector3D<Real> mImSpacing;
  /** Do we calculate energy for debugging? */
  bool mDebug;
  /** debug info, only calculated if mDebug is true */
  Real mVectorEnergy;
  Real mImageEnergy;
  Real mTotalEnergy;
  Real *mVectorStepEnergy;
  /** The initial image (at time 0)*/
  const RealImage *mI0;
  /** The final image (at time T) */
  const RealImage *mIT;
  /** backwards-deformed images */
  RealImage **mJTt;
  /** jacobian determinant */
  RealImage **mJacDet;
  /** Differential operator */
  DiffOper *mOp;
  /** Pointer to the internal field of the DiffOper */
  DiffOper::FFTWVectorField *mDiffOpVF;
  /** Scratch Image */
  RealImage *mScratchI;
  /** Scratch Vector Field */
  VectorField *mScratchV;
  /** Holds the alpha images if we're saving them */
  RealImage **mAlpha;
  /** Holds the UField images if we're saving them */
  VectorField **mUField;

  static
  void
  pointwiseMultiplyBy_FFTW_Safe(DiffOper::FFTWVectorField &lhs, 
				const Array3D<Real> &rhs);

  /** Calculate the update field (placed in update) and return the
    step size to use */
  Real 
  calcUpdate(const VectorField &curV,
	     const DiffOper::FFTWVectorField &uField,
	     VectorField &update);
  
  /** Update the velocity field using a fixed step size
   */
  void 
  updateVelocity(VectorField &v,
		 const DiffOper::FFTWVectorField &uField);
  
};

class LDMM {

  class ThreadInfo{
  public:
    unsigned int threadIndex;
    unsigned int imageIndex;
    VectorField **vFields;
    VectorField *hField;
    RealImage *I0;
    RealImage *finalImage;
    LDMMIterator *iterator;
  };

public:  

  /**
   * Create a transformation between image1 and image2
   */
  static
  void 
  LDMMRegistration(const RealImage *image1,
		   const RealImage *image2,
		   const unsigned int numTimeSteps,
		   const LDMMScaleLevelParam &params,
		   std::vector<RealImage*> &morphImages,
		   std::vector<VectorField*> &defFields);

  static  
  void 
  LDMMMultiscaleRegistration(const RealImage *image1,
			     const RealImage *image2,
			     const LDMMOldParam &param,
			     std::vector<RealImage*> &morphImages,
			     std::vector<VectorField*> &defFields);
  
  static
  void 
  LDMMMultiscaleAtlas(std::vector<const RealImage *> images,
		      const LDMMOldParam & param,
		      std::vector<RealImage*> &finalMorphImages,
		      std::vector<VectorField*> &finalDefFields);

  static
  void 
  LDMMMultiscaleAtlas(std::vector<const RealImage *> images,
		      std::vector<Real> &weights,
		      const LDMMOldParam & param,
		      std::vector<RealImage*> &finalMorphImages,
		      std::vector<VectorField*> &finalDefFields);

  static
  void 
  LDMMMultiscaleMultithreadedAtlas(std::vector<const RealImage *> images,
				   const LDMMOldParam & param,
				   unsigned int nThreads,
				   RealImage *MeanImage = NULL,
				   std::vector<RealImage*> *finalMorphImages = NULL,
				   std::vector<VectorField*> *finalDefFields = NULL,
				   std::vector<std::vector<VectorField*> > *finalVecFields = NULL);

  static
  void 
  LDMMMultiscaleMultithreadedAtlas(std::vector<const RealImage *> images,
				   std::vector<Real> &weights,
				   const LDMMOldParam & param,
				   unsigned int nThreads,
				   RealImage *MeanImage = NULL,
				   std::vector<RealImage*> *finalMorphImages = NULL,
				   std::vector<VectorField*> *finalDefFields = NULL,
				   std::vector<std::vector<VectorField*> > *finalVecFields = NULL);

  /**
   * Start from alpha0 and I0, and 'shoot' this image through a
   * deformation formed by the geodesic defined by the initial velocity
   * field.  params.nTimeSteps is used to determine the number of
   * timesteps the deformation is propogated over.
   */
  static
  void 
  GeodesicShooting(const RealImage *I0,
		   const RealImage *alpha0,
		   const unsigned int nTimeSteps,
		   const DiffOperParam &diffOpParam,
		   std::vector<RealImage*> &finalMorphImages,
		   std::vector<VectorField*> &finalVecFields,
		   const CompoundParam *debugOptions=NULL);

  /**
   * Start from an initial image and velocity field, and 'shoot' this
   * image through a deformation formed by the geodesic defined by the
   * initial velocity field.  params.nTimeSteps is used to determine the
   * number of timesteps the deformation is propogated over.  Requires
   * approximating alpha0 from v0.
   */
  static
  void 
  GeodesicShooting(const RealImage *I0,
		   const VectorField *v0,
		   const unsigned int nTimeSteps,
		   const DiffOperParam &diffOpParam,
		   std::vector<RealImage*> &finalMorphImages,
		   std::vector<VectorField*> &finalVecFields,
		   const CompoundParam *debugOptions=NULL);

  static
  void 
  LDMMShootingRegistration(const RealImage *image1,
			   const RealImage *image2,
			   const unsigned int nTimeSteps,
			   const LDMMScaleLevelParam &params,
			   std::vector<RealImage*> &morphImages,
			   std::vector<VectorField*> &defFields);

  static
  void 
  LDMMShootingRegistration2(const RealImage *image1,
			    const RealImage *image2,
			    const unsigned int nTimeSteps,
			    const LDMMScaleLevelParam &params,
			    std::vector<RealImage*> &morphImages,
			    std::vector<VectorField*> &defFields);

  /**
   * Compute \f$alpha\f$ fields from LDMM equations given the final velocity
   * fields
   */ 
  static
  void
  computeAlphaFields(std::vector<VectorField *> &v,
		     const RealImage *I0,
		     const RealImage *IT,
		     const Real &sigma,
		     std::vector<RealImage *> &momentum);

  /**
   * A bit of a hack to compute \f$alpha_t\f$ given \f$a_t = Lv_t\f$
   * and \f$\nabla J^0_t\f$
   */
  static
  Real
  computeAlphaFromA(const VectorField *at,
		    const VectorField *gJ0t,
		    RealImage *alphat=NULL,
		    RealImage *angleDiff=NULL, 
		    RealImage *mask=NULL);

protected:

  static
  void*
  LDMMThreadedUpdateVelocities(void* arg);

  static
  void 
  ComputeWeightedMean(const std::vector<RealImage*> &images, 
		      RealImage *mean, 
		      const std::vector<Real> &weights);

};

#endif// __LDMM_H__
