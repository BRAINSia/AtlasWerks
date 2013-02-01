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

#include "LDMMGeodesicShootingCPU.h"

LDMMGeodesicShootingCPU::
LDMMGeodesicShootingCPU(SizeType imSize, 
			SpacingType imSpacing, 
			DiffOperParam &diffOpParam)
  : LDMMGeodesicShooting(imSize, imSpacing),
    mDiffOp(imSize, imSpacing, diffOpParam),
    mPhi0T(imSize),
    mPhiT0(imSize),
    mVT(imSize),
    mScratchV(imSize),
    mAlphaT(imSize),
    mIT(imSize)
{
  
}

LDMMGeodesicShootingCPU::
~LDMMGeodesicShootingCPU()
{
}

void 
LDMMGeodesicShootingCPU::
ShootImage(const RealImage &I0,
	   const RealImage &alpha0,
	   unsigned int nTimeSteps)
{

  // Initialize values

  mNTimeSteps = nTimeSteps;

  mAlphaT = alpha0;

  if(mSaveAlphas){
    RealImage *alpha = new RealImage(mAlphaT);
    mAlphaVec.push_back(alpha);
  }

  mIT = I0;

  if(mSaveImages){
    RealImage *im = new RealImage(mIT);
    mImVec.push_back(im);
  }

  HField3DUtils::setToIdentity(mPhi0T);
  HField3DUtils::setToIdentity(mPhiT0);
  
  LDMMEnergy energy(mNTimeSteps, 0.f);

  // compute v0 = K(alpha0*gI0)
  Array3DUtils::computeGradient(mIT,mScratchV,mImSpacing,false);
  mScratchV.pointwiseMultiplyBy(mAlphaT);
  mDiffOp.CopyIn(mScratchV);
  mDiffOp.ApplyInverseOperator();
  mDiffOp.CopyOut(mVT);

  if(mSaveVecs){
    VectorField *v = new VectorField(mVT);
    mVVec.push_back(v);
  }

  Real vecEnergy;

  LOGNODE(logDEBUG) <<  "Finished initialization";

  for(unsigned int t = 1; t < (unsigned int)mNTimeSteps; t++){

    LOGNODE(logDEBUG) <<  "Running timestep " << t;
    // update vector energy, this step energy = <Lv_t,v_t>
    mDiffOp.CopyIn(mVT);
    mDiffOp.ApplyOperator();
    mDiffOp.CopyOut(mScratchV);
    vecEnergy = HField3DUtils::l2DotProd(mScratchV, mVT, mImSpacing);
    energy.SetVecStepEnergy(vecEnergy);

    /* #### Compute New Alpha #### */

    // update mPhi0T
    HField3DUtils::composeVH(mVT, mPhi0T, mScratchV, mImSpacing,
			     HField3DUtils::BACKGROUND_STRATEGY_PARTIAL_ZERO);
    mPhi0T = mScratchV;
 
    // create alpha_t
    HField3DUtils::forwardApply(alpha0, mPhi0T, mAlphaT, 0.0f, false);

    if(mSaveAlphas){
      RealImage *alpha = new RealImage(mAlphaT);
      mAlphaVec.push_back(alpha);
    }

    /* #### Compute New I #### */

    // update the deformation (hfield) for shooting
    HField3DUtils::composeHVInv(mPhiT0, mVT, mScratchV, mImSpacing,
 				HField3DUtils::BACKGROUND_STRATEGY_PARTIAL_ID);
    // HField3DUtils::composeHVInvIterative(mPhiT0, mPhi0T, *mVT, *mScratchV, N_REFINEMENT_ITERS, mImSpacing,
    // 	 HField3DUtils::BACKGROUND_STRATEGY_PARTIAL_ID, true);
    mPhiT0 = mScratchV;
    
    // create deformed image
    HField3DUtils::apply(I0, mPhiT0, mIT);

    if(mSaveImages){
      RealImage *im = new RealImage(mIT);
      mImVec.push_back(im);
    }

    /* #### Compute New V #### */

    // compute next vt = K(alphat*gJ0t)
    Array3DUtils::computeGradient(mIT, mScratchV, mImSpacing, false);
    mScratchV.pointwiseMultiplyBy(mAlphaT);

    mDiffOp.CopyIn(mScratchV);
    mDiffOp.ApplyInverseOperator();
    mDiffOp.CopyOut(mVT);

    if(mSaveVecs){
      VectorField *v = new VectorField(mVT);
      mVVec.push_back(v);
    }

  }

  // final update

  // get the last step energy
  mDiffOp.CopyIn(mVT);
  mDiffOp.ApplyOperator();
  mDiffOp.CopyOut(mScratchV);
  vecEnergy = HField3DUtils::l2DotProd(mScratchV, mVT, mImSpacing);
  energy.SetVecStepEnergy(vecEnergy);
  
  // final update to the deformation fields
  HField3DUtils::composeHVInv(mPhiT0, mVT, mScratchV, mImSpacing,
			      HField3DUtils::BACKGROUND_STRATEGY_PARTIAL_ID);
  mPhiT0 = mScratchV;

  HField3DUtils::composeVH(mVT, mPhi0T, mScratchV, mImSpacing,
			   HField3DUtils::BACKGROUND_STRATEGY_PARTIAL_ZERO);
  mPhi0T = mScratchV;

  // compute final image
  HField3DUtils::apply(I0, mPhiT0, mIT);

  if(mSaveImages){
    RealImage *im = new RealImage(mIT);
    mImVec.push_back(im);
  }

  LOGNODE(logDEBUG) <<  "Shooting finished";
}
  
void 
LDMMGeodesicShootingCPU::
ShootImage(const RealImage &I0,
	   const VectorField &v0,
	   unsigned int nTimeSteps)
{
  mNTimeSteps = nTimeSteps;
  OriginType imOrigin = I0.getOrigin();
  unsigned int numElements = v0.getNumElements();

  HField3DUtils::setToIdentity(mPhiT0);
  HField3DUtils::setToIdentity(mPhi0T);
  mVT = v0;
  
  LDMMEnergy energy(mNTimeSteps, 0.f);
  
  // init some objects
  RealImage jacDet(mImSize, imOrigin, mImSpacing);
  RealImage v1(mImSize, imOrigin, mImSpacing);
  RealImage v2(mImSize, imOrigin, mImSpacing);
  RealImage v3(mImSize, imOrigin, mImSpacing);
  RealImage scratchI(mImSize, imOrigin, mImSpacing);

  VectorField grad_h1(mImSize);
  VectorField grad_h2(mImSize);
  VectorField grad_h3(mImSize);   

  // set to identity
  grad_h1.fill(Vector3D<Real>(1.0, 0.0, 0.0));
  grad_h2.fill(Vector3D<Real>(0.0, 1.0, 0.0));
  grad_h3.fill(Vector3D<Real>(0.0, 0.0, 1.0));

  VectorField Lv0(mImSize);
  VectorField Lv0Def(mImSize);

  mDiffOp.CopyIn(v0);
  mDiffOp.ApplyOperator();
  mDiffOp.CopyOut(Lv0);

  for(unsigned int t = 0; t < (unsigned int)mNTimeSteps; t++){

    LOGNODE(logDEBUG) <<  "Running timestep " << t;

    if(mSaveVecs){
      VectorField *v = new VectorField(mVT);
      mVVec.push_back(v);
    }

    if(mSaveImages){
      RealImage *im = new RealImage(mImSize, imOrigin, mImSpacing);
      HField3DUtils::apply(I0, mPhiT0, *im);
      mImVec.push_back(im);
    }
  
    // update the deformation (hfield) for shooting
    HField3DUtils::composeHVInv(mPhiT0, mVT, mScratchV, mImSpacing,
 				HField3DUtils::BACKGROUND_STRATEGY_PARTIAL_ID);
    mPhiT0 = mScratchV;

    // phi0T isn't necessary for computation, but calculate it anyway
    HField3DUtils::composeVH(mVT, mPhi0T, mScratchV, mImSpacing,
			     HField3DUtils::BACKGROUND_STRATEGY_PARTIAL_ZERO);
    mPhi0T = mScratchV;

    // stop if this is the last timestep
    if(t == static_cast<unsigned int>(mNTimeSteps - 1)){
      break;
    }

    // compute Lv0(mPhiT0)
    HField3DUtils::compose(Lv0, mPhiT0, Lv0Def, 
			   HField3DUtils::BACKGROUND_STRATEGY_PARTIAL_ZERO);

    for (unsigned int i = 0; i < numElements; ++i)
      {
	v1(i) = mVT(i).x;
	v2(i) = mVT(i).y;
	v3(i) = mVT(i).z;
      }
    
    // compute the gradients of v1, v2, and v3 (form rows of
    // jacobian), and update jacobian
    HField3DUtils::apply(v1, mPhiT0, scratchI);
    Array3DUtils::computeGradient(scratchI, mScratchV, mImSpacing);
    grad_h1.pointwiseAdd(mScratchV);
    HField3DUtils::apply(v2, mPhiT0, scratchI);
    Array3DUtils::computeGradient(scratchI, mScratchV, mImSpacing);
    grad_h2.pointwiseAdd(mScratchV);
    HField3DUtils::apply(v3, mPhiT0, scratchI);
    Array3DUtils::computeGradient(scratchI, mScratchV, mImSpacing);
    grad_h3.pointwiseAdd(mScratchV);

    // compute compute DphiT0^T * ( Lv0(phiT0) ) and |Dphi|

    Real t1, t2, t3;
    for (unsigned int i = 0; i < numElements; ++i)
      {
	// DphiT0^T * Lv0Def

	mScratchV(i).x = 
	  grad_h1(i).x*Lv0Def(i).x + 
	  grad_h2(i).x*Lv0Def(i).y +
	  grad_h3(i).x*Lv0Def(i).z;

	mScratchV(i).y = 
	  grad_h1(i).y*Lv0Def(i).x + 
	  grad_h2(i).y*Lv0Def(i).y +
	  grad_h3(i).y*Lv0Def(i).z;

	mScratchV(i).z = 
	  grad_h1(i).z*Lv0Def(i).x + 
	  grad_h2(i).z*Lv0Def(i).y +
	  grad_h3(i).z*Lv0Def(i).z;

	// jacobian determinant calculation

	t1 = grad_h1(i).x * (grad_h2(i).y * grad_h3(i).z -
			     grad_h2(i).z * grad_h3(i).y);
	t2 = grad_h1(i).y * (grad_h2(i).x * grad_h3(i).z -
			     grad_h2(i).z * grad_h3(i).x);
	t3 = grad_h1(i).z * (grad_h2(i).x * grad_h3(i).y -
			     grad_h2(i).y * grad_h3(i).x);
	jacDet(i) = t1 - t2 + t3;

      }

    // multiply by jac. det.
    mScratchV.pointwiseMultiplyBy(jacDet);
    
    // create the new V
    mDiffOp.CopyIn(mScratchV);
    mDiffOp.ApplyInverseOperator();
    mDiffOp.CopyOut(mVT);

  }

  if(mSaveImages){
    RealImage *im = new RealImage(mImSize, imOrigin, mImSpacing);
    HField3DUtils::apply(I0, mPhiT0, mIT);
    *im = mIT;
    mImVec.push_back(im);
  }

  
}
