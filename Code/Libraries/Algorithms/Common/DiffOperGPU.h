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

#ifndef __DIFF_OPER_GPU_H__
#define __DIFF_OPER_GPU_H__

#include <vector>

#include "DiffOperParam.h"
#include "KernelInterfaceGPU.h"
#include <cudaVector3DArray.h>
#include <cudaComplex.h>
#include <VectorMath.h>
#include <cudaInterface.h>
#include <cudaFFTSolver.h>

class DiffOperGPU :
  public KernelInterfaceGPU 
{
  
public:  
  
  // ===== Constructors =====
  
  DiffOperGPU();
  ~DiffOperGPU();
  
  // ===== Public Members =====

  void SetSize(const SizeType &logicalSize, 
	       const SpacingType &spacing,
	       const DiffOperParam &params);

  void SetSize(const SizeType &logicalSize, 
	       const SpacingType &spacing,
	       const KernelParam &params);
  
  /** Set all parameters via DiffOperParam */
  void SetParams(const DiffOperParam &param);
  /** Get all parameters in a DiffOperParam */
  DiffOperParam GetParams();
  /** Set the \f$\alpha\f$ parameter.  Controls fluid viscosity. */
  void SetAlpha(Real alpha);
  /** Get the \f$\alpha\f$ parameter.  Controls fluid viscosity. */
  Real GetAlpha();
  /** Set the \f$\beta\f$ parameter.  Controls fluid viscosity. */
  void SetBeta(Real beta);
  /** Get the \f$\beta\f$ parameter.  Controls fluid viscosity. */
  Real GetBeta();
  /** Set the \f$\gamma\f$ parameter.  Usually << 1, maintains invertability. */
  void SetGamma(Real gamma);
  /** Get the \f$\gamma\f$ parameter.  Usually << 1, maintains invertability. */
  Real GetGamma();
  /** Set the power of L.  One by default */
  void SetLPow(Real p);
  /** Get the power of L */
  Real GetLPow();
  /** Set whether to perform precomputation to gain speed at the expense of memory */
  void SetUseEigenLUT(bool b);
  /** Get whether precomputation is performed to gain speed at the expense of memory */
  bool GetUseEigenLUT();

  /**
   * If SetDivergenceFree is set to true, incompressibility of the
   * fluid transformation will be enforced by projecting each
   * deformation step to the 'nearest' divergence-free deformation
   * step.
   */
  void SetDivergenceFree(bool df);
  /**
   * See SetDivergenceFree()
   */
  bool GetDivergenceFree();
  
  /**
   * f = Lv
   * 
   * v field is overwritten in this operation (holds f).
   */
  void ApplyOperator(float* dVx,
		     float* dVy,
		     float* dVz);
  
  /**
   * f = Lv
   * 
   * v field is overwritten in this operation (holds f).
   */
  void ApplyOperator(cplVector3DArray& dV);
  
  /**
   * v = Kf
   * 
   * f field is overwritten in this operation (holds v).
   */
  void ApplyInverseOperator(float* dFx,
			    float* dFy,
			    float* dFz);

  /**
   * v = Kf
   * 
   * f field is overwritten in this operation (holds v).
   */
  void ApplyInverseOperator(cplVector3DArray& dF);

protected:

  // ===== Member Data =====
  
  SizeType mSize;
  SpacingType mSpacing;

  bool mInitialized;

  //
  // parameters for calculating L
  //
  Real mAlpha;
  Real mBeta;
  Real mGamma;

  FFTSolverPlan3D mFFTSolver;

};


#endif //__DIFF_OPER_GPU_H__
