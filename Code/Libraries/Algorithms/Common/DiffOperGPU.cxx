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


#include "DiffOperGPU.h"

// ################ DiffOperGPU ################ //

// ======== Public Members ======== //

DiffOperGPU::
DiffOperGPU()
  : mInitialized(false)
{
}

DiffOperGPU::
~DiffOperGPU()
{
}

void
DiffOperGPU::
SetSize(const SizeType &size,
        const SpacingType &spacing,
        const DiffOperParam &param)
{
  mSize = size;
  mSpacing = spacing;

  mFFTSolver.setSize(mSize, mSpacing);
  
  this->SetParams(param);
  
  mInitialized=true;
}

void
DiffOperGPU::
SetSize(const SizeType &size,
        const SpacingType &spacing,
        const KernelParam &param)
{
  const DiffOperParam *diffOpParam = param.AsDiffOperParam();
  if(diffOpParam == NULL){
    throw AtlasWerksException(__FILE__,__LINE__,"Error, kernel param is not "
			      "DiffOperParam");
  }
  this->SetSize(size, spacing, *diffOpParam);
}

void 
DiffOperGPU::
SetParams(const DiffOperParam &param){
  // check for unimplemented parameters
  // set values from param
  this->SetAlpha(param.Alpha());
  this->SetBeta(param.Beta());
  this->SetGamma(param.Gamma());
  this->SetLPow(param.LPow());
  this->SetUseEigenLUT(param.UseEigenLUT());
  this->SetDivergenceFree(param.DivergenceFree());

  mFFTSolver.setParams(mAlpha, mBeta, mGamma);

}

DiffOperParam 
DiffOperGPU::
GetParams(){
  DiffOperParam param;
  param.Alpha() = this->GetAlpha();
  param.Beta() = this->GetBeta();
  param.Gamma() = this->GetGamma();
  param.LPow() = this->GetLPow();
  param.UseEigenLUT() = this->GetUseEigenLUT();
  param.DivergenceFree() = this->GetDivergenceFree();
  return param;
}

void DiffOperGPU::
SetAlpha(Real alpha){
  if(alpha != this->mAlpha){
    this->mAlpha = alpha;
  }
}

Real DiffOperGPU::
GetAlpha(){
  return this->mAlpha;
}

void DiffOperGPU::
SetBeta(Real beta){
  if(beta != this->mBeta){
    this->mBeta = beta;
  }
}

Real DiffOperGPU::
GetBeta(){
  return this->mBeta;
}

void DiffOperGPU::
SetGamma(Real gamma){
  if(gamma != this->mGamma){
    this->mGamma = gamma;
  }
}

Real DiffOperGPU::
GetGamma(){
  return this->mGamma;
}

void DiffOperGPU::
SetDivergenceFree(bool df){
  if(df){
    throw AtlasWerksException(__FILE__,__LINE__,"Error, divergence free projection not implemented on the GPU!");
  }
}

bool DiffOperGPU::
GetDivergenceFree(){
  return false;
}

void DiffOperGPU::
SetLPow(Real p){
  if(p != 1.0){
    throw AtlasWerksException(__FILE__,__LINE__,"Error, powers of L not supported on the GPU!");
  }
}

Real DiffOperGPU::
GetLPow(){
  return 1.0f;
}

void DiffOperGPU::
SetUseEigenLUT(bool b){
  if(b){
    throw AtlasWerksException(__FILE__,__LINE__,"Error, eigen LUT not supported on the GPU!");
  }
}

bool DiffOperGPU::
GetUseEigenLUT(){
  return false;
}

void 
DiffOperGPU::
ApplyInverseOperator(float* dFx,
		     float* dFy,
		     float* dFz)
{
  if(!mInitialized){
    throw AtlasWerksException(__FILE__,__LINE__,"Error, DiffOperGPU not initialized, please call SetScale first!");
  }
  mFFTSolver.apply(dFx, dFy, dFz, true);
}

void 
DiffOperGPU::
ApplyInverseOperator(cplVector3DArray& dF)
{
  if(!mInitialized){
    throw AtlasWerksException(__FILE__,__LINE__,"Error, DiffOperGPU not initialized, please call SetScale first!");
  }
  mFFTSolver.applyInverseOperator(dF);
}

void 
DiffOperGPU::
ApplyOperator(float* dVx,
	      float* dVy,
	      float* dVz)
{
  if(!mInitialized){
    throw AtlasWerksException(__FILE__,__LINE__,"Error, DiffOperGPU not initialized, please call SetScale first!");
  }
  mFFTSolver.apply(dVx, dVy, dVz, false);
}

void 
DiffOperGPU::
ApplyOperator(cplVector3DArray& dV)
{
  if(!mInitialized){
    throw AtlasWerksException(__FILE__,__LINE__,"Error, DiffOperGPU not initialized, please call SetScale first!");
  }
  mFFTSolver.applyOperator(dV);
}
