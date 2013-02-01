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


#include "LDMMShootingIteratorCPU.h"
#include "KernelFactory.h"

//
// ################ LDMMShootingIteratorCPU Implementation ################ //
//

LDMMShootingIteratorCPU::
LDMMShootingIteratorCPU(const SizeType &size, 
			const OriginType &origin,
			const SpacingType &spacing,
			unsigned int nTimeSteps,
			bool debug)
  : DeformationIteratorInterface(size, origin, spacing),
    mNTimeSteps(nTimeSteps),
    mSigma(0.f),
    mDebug(debug),
    mInitialized(false),
    mUpdateStepSizeNextIter(false),
    mVT(NULL),
    mAlphaT(NULL),
    mKernel(NULL),
    mKernelVF(NULL),
    mScratchI(NULL),
    mAlpha(NULL),
    mScratchV(NULL)
{
}

LDMMShootingIteratorCPU::
~LDMMShootingIteratorCPU()
{
  delete mVT;
  delete mAlphaT;

  delete mScratchI;
  delete mScratchV;

  delete mKernel;
}

void 
LDMMShootingIteratorCPU::
SetScaleLevel(const MultiscaleManager &scaleManager,
	      const LDMMIteratorParam &param)
{
  ScaleLevelDataInfo::SetScaleLevel(scaleManager);

  mMaxPert = param.MaxPert();
  mSigma = param.Sigma();

  if(!mInitialized){
    mVT = new VectorField(mCurSize);
    mAlphaT = new RealImage(mCurSize, mImOrigin, mCurSpacing);
    
    mScratchI = new RealImage(mCurSize, mImOrigin, mCurSpacing);
    mScratchV = new VectorField(mCurSize);
    
    mInitialized = true;
  }else{
    mVT->resize(mCurSize);
    mAlphaT->resize(mCurSize);
    mAlphaT->setSpacing(mCurSpacing);
    
    mScratchI->resize(mCurSize);
    mScratchI->setSpacing(mCurSpacing);
    mScratchV->resize(mCurSize);
  }

  // create DiffOper
  if(mKernel) delete mKernel;
  mKernel = KernelFactory::NewKernel(param.Kernel(), mCurSize, mCurSpacing);
  mKernelVF = mKernel->GetInternalFFTWVectorField();
    
}

void 
LDMMShootingIteratorCPU::
pointwiseMultiplyBy_FFTW_Safe(KernelInterface::KernelInternalVF &lhs, 
			      const Array3D<Real> &rhs)
{
  Vector3D<unsigned int> size = rhs.getSize();
  for (unsigned int z = 0; z < size.z; ++z) {
    for (unsigned int y = 0; y < size.y; ++y) {
      for (unsigned int x = 0; x < size.x; ++x) {
	lhs(x,y,z) *= rhs(x,y,z);
      }
    }
  }
}

void 
LDMMShootingIteratorCPU::
ComputeJacDet(LDMMShootingDefData &data, RealImage &jacDet)
{
  if(data.I0().getSize() != mCurSize){
    throw AtlasWerksException(__FILE__,__LINE__,"Error, jacobian determinant "
			   "calculation must be performed on images of the "
			   "same size as the iterator");
  }

  // splat 'ones' image by PhiT0 to create jacobian determinant of Phi0T
  mScratchI->fill(1.0);
  HField3DUtils::forwardApply(*mScratchI, data.PhiT0(), jacDet, 0.0f, false);
}

/**  
 * Update Phi0T and JacDetPhi0T.  This is no longer used.
 */  
void
LDMMShootingIteratorCPU::
UpdatePhi0T(VectorField &phi, 
	    RealImage &jacDet, 
	    const VectorField &v)
{
  unsigned int numElements = phi.getNumElements();

  // build scalar images h1, h2, and h3
  // from the transformation field
  RealImage h1(mCurSize);
  RealImage h2(mCurSize);
  RealImage h3(mCurSize);
  
  unsigned int i; // stupid vc++
  for (i = 0; i < numElements; ++i)
    {
      h1(i) = v(i).x;
      h2(i) = v(i).y;
      h3(i) = v(i).z;
    }

  // compute the gradients of h1, h2, and h3 (form rows of jacobian)
  VectorField grad_h1(mCurSize);
  VectorField grad_h2(mCurSize);
  VectorField grad_h3(mCurSize);   
  Array3DUtils::computeGradient(h1, grad_h1, mCurSpacing);
  Array3DUtils::computeGradient(h2, grad_h2, mCurSpacing);
  Array3DUtils::computeGradient(h3, grad_h3, mCurSpacing);

  // add identity to the matrix
  grad_h1.add(Vector3D<Real>(1.0, 0.0, 0.0));
  grad_h2.add(Vector3D<Real>(0.0, 1.0, 0.0));
  grad_h3.add(Vector3D<Real>(0.0, 0.0, 1.0));
  
  // compute the determinant
  Real t1, t2, t3;
  for (i = 0; i < numElements; ++i)
    {
      t1 = grad_h1(i).x * (grad_h2(i).y * grad_h3(i).z -
			   grad_h2(i).z * grad_h3(i).y);
      t2 = grad_h1(i).y * (grad_h2(i).x * grad_h3(i).z -
			   grad_h2(i).z * grad_h3(i).x);
      t3 = grad_h1(i).z * (grad_h2(i).x * grad_h3(i).y -
			   grad_h2(i).y * grad_h3(i).x);
      (*mScratchI)(i) = t1 - t2 + t3;
    }

  // deform the determinant (store in h1)
  HField3DUtils::apply(*mScratchI, phi, h1, 1.0f);
  
  // update the deformed jac det (stored in h1)
  jacDet.pointwiseMultiplyBy(h1);

  // update phi
  HField3DUtils::composeVH(v, phi, *mScratchV, mCurSpacing,
			   HField3DUtils::BACKGROUND_STRATEGY_PARTIAL_ZERO);
  phi = *mScratchV;
}

#define N_REFINEMENT_ITERS 0

void 
LDMMShootingIteratorCPU::
Iterate(DeformationDataInterface &deformationData)
{
  if(!mInitialized){
    throw AtlasWerksException(__FILE__,__LINE__,"Error, iterator not initialized!");
  }

  LDMMShootingDefData &data = dynamic_cast<LDMMShootingDefData&>(deformationData);
  VectorField &phi0T = data.Phi0T();
  VectorField &phiT0 = data.PhiT0();
  
  LDMMEnergy energy(mNTimeSteps, mSigma);

  // Initialize values
  HField3DUtils::setToIdentity(phi0T);
  HField3DUtils::setToIdentity(phiT0);
  *mAlphaT = data.Alpha0();
  
  // compute v0 = K(alpha0*gI0)
  Array3DUtils::computeGradient(data.I0(),*mScratchV,mCurSpacing,false);
  mScratchV->pointwiseMultiplyBy(*mAlphaT);
  mKernel->CopyIn(*mScratchV);
  mKernel->ApplyInverseOperator();
  mKernel->CopyOut(*mVT);

  Real vecEnergy;

  for(unsigned int t = 1; t < mNTimeSteps; t++){

    // update vector energy, this step energy = <Lv_t,v_t>
    mKernel->CopyIn(*mVT);
    mKernel->ApplyOperator();
    mKernel->CopyOut(*mScratchV);
    vecEnergy = HField3DUtils::l2DotProd(*mScratchV, *mVT, mCurSpacing);
    energy.SetVecStepEnergy(vecEnergy);

    /* #### Compute New Alpha #### */

    // update phi0T
    HField3DUtils::composeVH(*mVT, phi0T, *mScratchV, mCurSpacing,
			     HField3DUtils::BACKGROUND_STRATEGY_PARTIAL_ZERO);
    phi0T = *mScratchV;
 
    // create alpha_t
    HField3DUtils::forwardApply(data.Alpha0(), phi0T, *mAlphaT, 0.0f, false);

    /* #### Compute New I #### */

    // update the deformation (hfield) for shooting
    HField3DUtils::composeHVInv(phiT0, *mVT, *mScratchV, mCurSpacing,
 				HField3DUtils::BACKGROUND_STRATEGY_PARTIAL_ID);
    // HField3DUtils::composeHVInvIterative(phiT0, phi0T, *mVT, *mScratchV, N_REFINEMENT_ITERS, mCurSpacing,
    // 	 HField3DUtils::BACKGROUND_STRATEGY_PARTIAL_ID, true);
    phiT0 = *mScratchV;
    
    // create deformed image
    HField3DUtils::apply(data.I0(), phiT0, *mScratchI);

    /* #### Compute New V #### */

    // compute next vt = K(alphat*gJ0t)
    Array3DUtils::computeGradient(*mScratchI,*mScratchV,mCurSpacing,false);
    mScratchV->pointwiseMultiplyBy(*mAlphaT);
    mKernel->CopyIn(*mScratchV);
    mKernel->ApplyInverseOperator();
    mKernel->CopyOut(*mVT);
  }

  // final update

  // get the last step energy
  mKernel->CopyIn(*mVT);
  mKernel->ApplyOperator();
  mKernel->CopyOut(*mScratchV);
  vecEnergy = HField3DUtils::l2DotProd(*mScratchV, *mVT, mCurSpacing);
  energy.SetVecStepEnergy(vecEnergy);
  
  // final update to the deformation fields
  HField3DUtils::composeHVInv(phiT0, *mVT, *mScratchV, mCurSpacing,
			      HField3DUtils::BACKGROUND_STRATEGY_PARTIAL_ID);
  phiT0 = *mScratchV;

  HField3DUtils::composeVH(*mVT, phi0T, *mScratchV, mCurSpacing,
			   HField3DUtils::BACKGROUND_STRATEGY_PARTIAL_ZERO);
  phi0T = *mScratchV;

  // compute image difference term
  HField3DUtils::apply(data.I0(), phiT0, *mScratchI);

  mScratchI->pointwiseSubtract(data.I1());

  // splat image difference term to time 0
  HField3DUtils::forwardApply(*mScratchI, phiT0, *mAlphaT, 0.0f, false);

  // alpha0 = alpha0 - mStepSize * ( alpha0 - lambda * imDiff)
  //        = vFac*alpha0 + uFac*imDiff
  Real vFac = 1.0-data.StepSize();
  Real uFac = data.StepSize()/(mSigma*mSigma);

  Real *alpha0Data = data.Alpha0().getDataPointer();
  Real *uData = mAlphaT->getDataPointer();

  unsigned int nVox = mCurSize.productOfElements();
  for (unsigned int vIdx = 0; vIdx < nVox; vIdx++) {
    alpha0Data[vIdx] = 
      vFac * alpha0Data[vIdx] + 
      uFac * uData[vIdx];
  }

  // compute image energy  = ||JT0 - I1||^2
  // create the final deformed image
  HField3DUtils::apply(data.I0(), phiT0, *mScratchI);
  mScratchI->pointwiseSubtract(data.I1());
  energy.SetImageEnergy(ImageUtils::l2NormSqr(*mScratchI));
  data.AddEnergy(energy);
  
}

