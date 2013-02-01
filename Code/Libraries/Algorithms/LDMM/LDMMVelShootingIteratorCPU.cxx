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


#include "LDMMVelShootingIteratorCPU.h"
#include "KernelFactory.h"

//
// ################ LDMMVelShootingIteratorCPU Implementation ################ //
//

LDMMVelShootingIteratorCPU::
LDMMVelShootingIteratorCPU(const SizeType &size, 
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
    mLV0(NULL),
    mLV0Def(NULL),
    mDPhiT0_x(NULL),
    mDPhiT0_y(NULL),
    mDPhiT0_z(NULL),
    mJacDet(NULL),
    mVx(NULL),
    mVy(NULL),
    mVz(NULL),
    mKernel(NULL),
    mScratchI(NULL),
    mScratchV(NULL)
{
}

LDMMVelShootingIteratorCPU::~LDMMVelShootingIteratorCPU(){
  delete mVT;
  delete mLV0;
  delete mLV0Def;
  delete mDPhiT0_x;
  delete mDPhiT0_y;
  delete mDPhiT0_z;
  delete mJacDet;
  delete mVx;
  delete mVy;
  delete mVz;

  delete mScratchI;
  delete mScratchV;

  delete mKernel;
}

void 
LDMMVelShootingIteratorCPU::
SetScaleLevel(const MultiscaleManager &scaleManager,
	      const LDMMIteratorParam &param)
{
  ScaleLevelDataInfo::SetScaleLevel(scaleManager);

  mMaxPert = param.MaxPert();
  mSigma = param.Sigma();

  if(!mInitialized){
    mVT = new VectorField(mCurSize);
    mLV0 = new VectorField(mCurSize);
    mLV0Def = new VectorField(mCurSize);
    mDPhiT0_x = new VectorField(mCurSize);
    mDPhiT0_y = new VectorField(mCurSize);
    mDPhiT0_z = new VectorField(mCurSize);
    mJacDet = new RealImage(mCurSize, mImOrigin, mCurSpacing);
    mVx = new RealImage(mCurSize, mImOrigin, mCurSpacing);
    mVy = new RealImage(mCurSize, mImOrigin, mCurSpacing);
    mVz = new RealImage(mCurSize, mImOrigin, mCurSpacing);
    
    mScratchI = new RealImage(mCurSize, mImOrigin, mCurSpacing);
    mScratchV = new VectorField(mCurSize);
    
    mInitialized = true;
  }else{
    mVT->resize(mCurSize);
    mLV0->resize(mCurSize);
    mLV0Def->resize(mCurSize);
    mDPhiT0_x->resize(mCurSize);
    mDPhiT0_y->resize(mCurSize);
    mDPhiT0_z->resize(mCurSize);
    mJacDet->resize(mCurSize);
    mJacDet->setSpacing(mCurSpacing);
    mVx->resize(mCurSize);
    mVx->setSpacing(mCurSpacing);
    mVy->resize(mCurSize);
    mVy->setSpacing(mCurSpacing);
    mVz->resize(mCurSize);
    mVz->setSpacing(mCurSpacing);
    
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
LDMMVelShootingIteratorCPU::
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
LDMMVelShootingIteratorCPU::
updateVelocity(VectorField &v,
	       const KernelInterface::KernelInternalVF &uField,
	       Real stepSize)
{
  Vector3D<unsigned int> size = v.getSize();
  Real vFac = 1.0-2.0*stepSize;
  Real uFac = (2*stepSize)/(mSigma*mSigma);
  for (unsigned int z = 0; z < size.z; ++z) {
    for (unsigned int y = 0; y < size.y; ++y) {
      for (unsigned int x = 0; x < size.x; ++x) {
	v(x,y,z) = vFac*v(x,y,z) + uFac*uField(x,y,z);
      }
    }
  }
}

Real
LDMMVelShootingIteratorCPU::
calcMaxDisplacement(VectorField &v,
		    const KernelInterface::KernelInternalVF &uField)
{
  Vector3D<unsigned int> size = v.getSize();
  Real vFac = 2.0;
  Real uFac = -2.0/(mSigma*mSigma);
  Real maxLen = 0.f;
  Vector3D<Real> vec;
  for (unsigned int z = 0; z < size.z; ++z) {
    for (unsigned int y = 0; y < size.y; ++y) {
      for (unsigned int x = 0; x < size.x; ++x) {
	vec = vFac*v(x,y,z) + uFac*uField(x,y,z);
	Real len = vec.length();
	if(len > maxLen) maxLen = len;
      }
    }
  }
  return maxLen;
}

void 
LDMMVelShootingIteratorCPU::
ComputeJacDet(LDMMVelShootingDefData &data, RealImage &jacDet)
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

void 
LDMMVelShootingIteratorCPU::
Iterate(DeformationDataInterface &deformationData)
{
  if(!mInitialized){
    throw AtlasWerksException(__FILE__,__LINE__,"Error, iterator not initialized!");
  }
  
  LDMMVelShootingDefData &data = dynamic_cast<LDMMVelShootingDefData&>(deformationData);
  VectorField &phi0T = data.Phi0T();
  VectorField &phiT0 = data.PhiT0();

  unsigned int numElements = data.V0().getNumElements();
  
  LDMMEnergy energy(mNTimeSteps, mSigma);
  
  // Initialize values
  HField3DUtils::setToIdentity(phi0T);
  HField3DUtils::setToIdentity(phiT0);
  *mVT = data.V0();
  
  Real vecEnergy;

  // set to identity
  mDPhiT0_x->fill(Vector3D<Real>(1.0, 0.0, 0.0));
  mDPhiT0_y->fill(Vector3D<Real>(0.0, 1.0, 0.0));
  mDPhiT0_z->fill(Vector3D<Real>(0.0, 0.0, 1.0));

  // compute Lv0
  mKernel->CopyIn(data.V0());
  mKernel->ApplyOperator();
  mKernel->CopyOut(*mLV0);

  for(unsigned int t = 0; t < (unsigned int)mNTimeSteps; t++){
    
    // update vector energy, this step energy = <Lv_t,v_t>
    mKernel->CopyIn(*mVT);
    mKernel->ApplyOperator();
    mKernel->CopyOut(*mScratchV);
    vecEnergy = HField3DUtils::l2DotProd(*mScratchV, *mVT, mCurSpacing);
    energy.SetVecStepEnergy(vecEnergy);

    // update the deformation (hfield) for shooting
    HField3DUtils::composeHVInv(phiT0, *mVT, *mScratchV, mCurSpacing,
 				HField3DUtils::BACKGROUND_STRATEGY_PARTIAL_ID);
    phiT0 = *mScratchV;

    // phi0T isn't necessary for computation, but calculate it anyway
    HField3DUtils::composeVH(*mVT, phi0T, *mScratchV, mCurSpacing,
			     HField3DUtils::BACKGROUND_STRATEGY_PARTIAL_ZERO);
    phi0T = *mScratchV;

    // stop if this is the last timestep
    if(t == static_cast<unsigned int>(mNTimeSteps - 1)){
      break;
    }

    for (unsigned int i = 0; i < numElements; ++i)
      {
	(*mVx)(i) = (*mVT)(i).x;
	(*mVy)(i) = (*mVT)(i).y;
	(*mVz)(i) = (*mVT)(i).z;
      }
    
    // compute the gradients of vx, vy, and vz (form rows of
    // jacobian), and update jacobian
    HField3DUtils::apply(*mVx, phiT0, *mScratchI);
    Array3DUtils::computeGradient(*mScratchI, *mScratchV, mCurSpacing);
    mDPhiT0_x->pointwiseAdd(*mScratchV);
    HField3DUtils::apply(*mVy, phiT0, *mScratchI);
    Array3DUtils::computeGradient(*mScratchI, *mScratchV, mCurSpacing);
    mDPhiT0_y->pointwiseAdd(*mScratchV);
    HField3DUtils::apply(*mVz, phiT0, *mScratchI);
    Array3DUtils::computeGradient(*mScratchI, *mScratchV, mCurSpacing);
    mDPhiT0_z->pointwiseAdd(*mScratchV);
    
    // compute Lv0(phiT0)
    HField3DUtils::compose(*mLV0, phiT0, *mLV0Def, 
			   HField3DUtils::BACKGROUND_STRATEGY_PARTIAL_ZERO);

    // compute compute DphiT0^T * ( Lv0(phiT0) ) and |Dphi|

    Real t1, t2, t3;
    for (unsigned int i = 0; i < numElements; ++i)
      {
	// DphiT0^T * LV0Def

	(*mScratchV)(i).x = 
	  (*mDPhiT0_x)(i).x*(*mLV0Def)(i).x + 
	  (*mDPhiT0_y)(i).x*(*mLV0Def)(i).y +
	  (*mDPhiT0_z)(i).x*(*mLV0Def)(i).z;

	(*mScratchV)(i).y = 
	  (*mDPhiT0_x)(i).y*(*mLV0Def)(i).x + 
	  (*mDPhiT0_y)(i).y*(*mLV0Def)(i).y +
	  (*mDPhiT0_z)(i).y*(*mLV0Def)(i).z;

	(*mScratchV)(i).z = 
	  (*mDPhiT0_x)(i).z*(*mLV0Def)(i).x + 
	  (*mDPhiT0_y)(i).z*(*mLV0Def)(i).y +
	  (*mDPhiT0_z)(i).z*(*mLV0Def)(i).z;

	// jacobian determinant calculation (store in mScratchI)
	
	t1 = (*mDPhiT0_x)(i).x * ((*mDPhiT0_y)(i).y * (*mDPhiT0_z)(i).z -
				(*mDPhiT0_y)(i).z * (*mDPhiT0_z)(i).y);
	t2 = (*mDPhiT0_x)(i).y * ((*mDPhiT0_y)(i).x * (*mDPhiT0_z)(i).z -
				(*mDPhiT0_y)(i).z * (*mDPhiT0_z)(i).x);
	t3 = (*mDPhiT0_x)(i).z * ((*mDPhiT0_y)(i).x * (*mDPhiT0_z)(i).y -
				(*mDPhiT0_y)(i).y * (*mDPhiT0_z)(i).x);
	(*mScratchI)(i) = t1 - t2 + t3;

      }

    // multiply by jac. det.
    mScratchV->pointwiseMultiplyBy(*mScratchI);
    
    // create the new V
    mKernel->CopyIn(*mScratchV);
    mKernel->ApplyInverseOperator();
    mKernel->CopyOut(*mVT);
  }

  // compute difference image, store in mScratchI
  HField3DUtils::apply(data.I0(), phiT0, *mScratchI);
  mScratchI->pointwiseSubtract(data.I1());

  // compute image energy  = ||JT0 - I1||^2
  energy.SetImageEnergy(ImageUtils::l2NormSqr(*mScratchI));
  data.AddEnergy(energy);
  
  // splat the difference image from dest to source, store in mVx
  HField3DUtils::forwardApply(*mScratchI, phiT0, *mVx, 0.0f, false);
 
  // create the gradient of I0, store in DiffOper vector field
  Array3DUtils::computeGradient(data.I0(), *mKernelVF, mCurSpacing, false);
  
  // scale the gradient
  pointwiseMultiplyBy_FFTW_Safe(*mKernelVF, *mVx);

  // apply K
  mKernel->ApplyInverseOperator();
  
  if(mUpdateStepSizeNextIter){
    Real maxDisplacement = calcMaxDisplacement(data.V0(),*mKernelVF);
    data.StepSize(mMaxPert / maxDisplacement);
    mUpdateStepSizeNextIter = false;
  }else{
    updateVelocity(data.V0(),*mKernelVF,data.StepSize());
  } // end mUpdateStepSizeNextIter

}

