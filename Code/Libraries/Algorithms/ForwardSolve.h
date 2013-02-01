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

#ifndef __FORWARD_SOLVE_H__
#define __FORWARD_SOLVE_H__

#include "AtlasWerksTypes.h"
#include "HField3DUtils.h"
#include "CompoundParam.h"
#include "ValueParam.h"

class ForwardSolveParam : public CompoundParam {
public:
  ForwardSolveParam(const std::string& name = "ForwardSolve", 
		    const std::string& desc = "Settings for iteratively solving I1(h(x)) = I0 for I1 given I0 and h(x)", 
		    ParamLevel level = PARAM_COMMON)
    : CompoundParam(name, desc, level)
  {
    this->
      AddChild(ValueParam<unsigned int>("NIterations",
					"Number of iterations to run",
					PARAM_COMMON,
					20));
    this->
      AddChild(ValueParam<Real>("StepSize",
				"Stepsize to use during iteration",
				PARAM_COMMON,
				1.0));
    this->
      AddChild(ValueParam<Real>("GaussianSigma",
				"standard deviation of the gaussian used for regularization",
				PARAM_COMMON,
				1.0));
    this->
      AddChild(ValueParam<unsigned int>("GaussianKernelSize",
					"kernel size of the gaussian used for regularization",
					PARAM_COMMON,
					3));
    this->
      AddChild(ValueParam<bool>("GaussianRegularize",
				"Regularize via gaussian smoothing?",
				PARAM_COMMON,
				false));
    this->
      AddChild(ValueParam<bool>("PrintError",
				"Print out the error at each iteration",
				PARAM_COMMON,
				false));
    this->
      AddChild(ValueParam<bool>("InitializeWithTInv",
				"Initialize with I0(x-v(x))",
				PARAM_COMMON,
				false));
    this->
      AddChild(ValueParam<std::string>("IterationOutputFileFormat",
				       "If non-null, an image will be written out at each iteration "
				       "using this printf-style format string.",
				       PARAM_COMMON,
				       ""));

  }
  
  ValueParamAccessorMacro(unsigned int, NIterations)
  ValueParamAccessorMacro(Real, StepSize)
  ValueParamAccessorMacro(Real, GaussianSigma);
  ValueParamAccessorMacro(unsigned int, GaussianKernelSize)
  ValueParamAccessorMacro(bool, GaussianRegularize)
  ValueParamAccessorMacro(bool, PrintError)
  ValueParamAccessorMacro(bool, InitializeWithTInv)
  ValueParamAccessorMacro(std::string, IterationOutputFileFormat)

  CopyFunctionMacro(ForwardSolveParam)

};

class ForwardSolve {
  
public:
  
  ForwardSolve(const Vector3D<unsigned int> &size, 
	       const Vector3D<double> &origin,
	       const Vector3D<double> &spacing);

  ForwardSolve(const Vector3D<unsigned int> &size, 
	       const Vector3D<double> &origin,
	       const Vector3D<double> &spacing,
	       const ForwardSolveParam &param);
  
  ~ForwardSolve();

  void SetParams(const ForwardSolveParam &param);
  ForwardSolveParam GetParams();

  void SetPrintError(bool print){ mPrintError = print; }
  bool GetPrintError(){ return mPrintError; }

  void SetNIterations(unsigned int niters){ mNIterations=niters; }
  unsigned int GetNIterations(){ return mNIterations; }
  void SetStepSize(Real step){ mStepSize = step; }
  Real GetStepSize(){ return mStepSize; }

  /**
   * Initialize output with I0(x-v) instead of zero image?
   */
  void SetInitializeWithTInv(bool useTInv){ mInitializeWithTInv = useTInv; }
  bool GetInitializeWithTInv(){ return mInitializeWithTInv; }

  /**
   * Regularize via gaussian smoothing?
   */
  void SetGaussianRegularize(bool reg){ mGaussianRegularize = reg; }
  bool GetGaussianRegularize(){ return mGaussianRegularize; }
  /** Gaussian Parameters */
  void SetGaussianSigma(Real sigma){ mGaussianSigma = sigma; }
  Real GetGaussianSigma(){ return mGaussianSigma; }
  void SetGaussianKernelSize(unsigned int kernelSize){ mGaussianKernelSize = kernelSize; }
  unsigned int GetGaussianKernelSize(){ return mGaussianKernelSize; }

  /**
   * If non-null, an image will be written out at each iteration using
   * this printf-style format string.  One %d-style format expected
   * (filled with iteration #).
   */
  void SetIterationFileFormat(const std::string &iterFormat){ mIterFormat = iterFormat; }
  std::string GetIterationFileFormat(){ return mIterFormat; }

  // Solve directly for pushed-forward image
  void
  Solve(const RealImage &I0,
	const VectorField &v,
	RealImage &I1,
	unsigned int maxIter);

  // Solve directly for pushed-forward image
  void
  Solve(const RealImage &I0,
	const VectorField &v,
	RealImage &I1);

  // static version of Solve, allocates an instance of ForwardSolve
  static
  void
  RunForwardSolve(const RealImage &I0,
		  const VectorField &v,
		  RealImage &I1,
		  unsigned int maxIters);

protected:

  unsigned int mNIterations;
  Real mStepSize;
  bool mPrintError;
  bool mInitializeWithTInv;
  bool mGaussianRegularize;
  Real mGaussianSigma;
  unsigned int mGaussianKernelSize;
  std::string mIterFormat;

  Vector3D<unsigned int> mSize;
  Vector3D<double> mOrigin;
  Vector3D<double> mSpacing;

  RealImage *mTAdjI0;
  VectorField *mHField;
  RealImage *mScratchI1;
  RealImage *mScratchI2;
  
};


#endif // __FORWARD_SOLVE_H__
