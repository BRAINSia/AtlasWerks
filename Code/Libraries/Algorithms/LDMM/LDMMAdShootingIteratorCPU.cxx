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


#include "LDMMAdShootingIteratorCPU.h"
#include "KernelFactory.h"

//
// ################ LDMMAdShootingIteratorCPU Implementation ################ //
//

LDMMAdShootingIteratorCPU::
LDMMAdShootingIteratorCPU(const SizeType &size, 
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
    mIHatT(NULL),
    mAlphaHatT(NULL),
    mITildeT(NULL),
    mAlphaTildeT(NULL),
    mIT(NULL),
    mAlphaT(NULL),
    mKernel(NULL),
    mKernelVF(NULL),
    mScratchI(NULL),
    mScratchI2(NULL),
    mAlpha(NULL),
    mScratchV(NULL),
    mScratchV2(NULL)
{
}

LDMMAdShootingIteratorCPU::~LDMMAdShootingIteratorCPU(){
  //delete mVT;
  for (uint i=0; i<mNTimeSteps; i++){
    delete mVVec[i];
  }
  mVVec.clear();  
  delete mIHatT;
  delete mAlphaHatT;
  delete mITildeT;
  delete mAlphaTildeT;
  delete mScratchI2;
  delete mScratchV2;
  delete mIT;
  delete mAlphaT;
  delete mScratchI;
  delete mScratchV;
  delete mKernel;
}

void 
LDMMAdShootingIteratorCPU::
SetScaleLevel(const MultiscaleManager &scaleManager,
	      const LDMMIteratorParam &param)
{

  ScaleLevelDataInfo::SetScaleLevel(scaleManager);

  mMaxPert = param.MaxPert();
  mSigma = param.Sigma();

  if(!mInitialized){
    //mVT = new VectorField(mCurSize);
    mAlphaT = new RealImage(mCurSize, mImOrigin, mCurSpacing);
    
    /** Initialize std:: vector to store V's for each timepoint */ 
    mVVec = std::vector<VectorField*>(mNTimeSteps);
    for (uint i=0; i<mNTimeSteps; i++){
      mVVec[i] = new VectorField(mCurSize);
    }

    mIHatT = new RealImage(mCurSize, mImOrigin, mCurSpacing);
    mAlphaHatT = new RealImage(mCurSize, mImOrigin, mCurSpacing);
    mITildeT = new RealImage(mCurSize, mImOrigin, mCurSpacing);
    mAlphaTildeT = new RealImage(mCurSize, mImOrigin, mCurSpacing);
    mIT = new RealImage(mCurSize, mImOrigin, mCurSpacing);
    mScratchI = new RealImage(mCurSize, mImOrigin, mCurSpacing);
    mScratchV = new VectorField(mCurSize);
    mScratchI2 = new RealImage(mCurSize, mImOrigin, mCurSpacing);
    mScratchV2 = new VectorField(mCurSize);
  
    mInitialized = true;
  }else{
    //mVT->resize(mCurSize);
    for (uint i=0; i<mNTimeSteps; i++){
      mVVec[i]->resize(mCurSize);
    }
    mIHatT->resize(mCurSize);
    mIHatT->setSpacing(mCurSpacing);

    mAlphaHatT->resize(mCurSize);
    mAlphaHatT->setSpacing(mCurSpacing);

    mITildeT->resize(mCurSize);
    mITildeT->setSpacing(mCurSpacing);

    mAlphaTildeT->resize(mCurSize);
    mAlphaTildeT->setSpacing(mCurSpacing);

    mScratchI2->resize(mCurSize);
    mScratchI2->setSpacing(mCurSpacing);

    mScratchV2->resize(mCurSize);

    mIT->resize(mCurSize);    
    mIT->setSpacing(mCurSpacing);    

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
LDMMAdShootingIteratorCPU::
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
LDMMAdShootingIteratorCPU::
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
LDMMAdShootingIteratorCPU::
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
LDMMAdShootingIteratorCPU::
Iterate(DeformationDataInterface &deformationData)
{
  if(!mInitialized){
    throw AtlasWerksException(__FILE__,__LINE__,"Error, iterator not initialized!");
  }

  /** ####ADJOINT ALGORITHM PART 1: SHOOTING ALGORITHM####  */

  LDMMShootingDefData &data = dynamic_cast<LDMMShootingDefData&>(deformationData);
  VectorField &phi0T = data.Phi0T();
  VectorField &phiT0 = data.PhiT0();
  
  LDMMEnergy energy(mNTimeSteps, mSigma);

  // Initialize values
  HField3DUtils::setToIdentity(phi0T);
  HField3DUtils::setToIdentity(phiT0);

  *mAlphaT = data.Alpha0();
  
  // compute v0 = -K(alpha0*gI0)
  Array3DUtils::computeGradient(data.I0(),*mScratchV,mCurSpacing,false);
  mScratchV->pointwiseMultiplyBy(*mAlphaT);
  mKernel->CopyIn(*mScratchV);
  mKernel->ApplyInverseOperator();
  mKernel->CopyOut(*(mVVec[0]));
  mVVec[0]->scale(-1.0f);

  Real vecEnergy;

  for(unsigned int t = 1; t < mNTimeSteps; t++){

    /* #### Compute New Alpha #### */

    // update phi0T
    HField3DUtils::composeVH(*(mVVec[t-1]), phi0T, *mScratchV, mCurSpacing,
			     HField3DUtils::BACKGROUND_STRATEGY_PARTIAL_ZERO);
    phi0T = *mScratchV;
 
    // create alpha_t
    HField3DUtils::forwardApply(data.Alpha0(), phi0T, *mAlphaT, 0.0f, false);

    /* #### Compute New I #### */

    // update the deformation (hfield) for shooting
    HField3DUtils::composeHVInv(phiT0, *(mVVec[t-1]), *mScratchV, mCurSpacing,
 				HField3DUtils::BACKGROUND_STRATEGY_PARTIAL_ID);
    phiT0 = *mScratchV;
    
    // create deformed image
    HField3DUtils::apply(data.I0(), phiT0, *mScratchI);

    /* #### Compute New V #### */

    // compute next vt = -K(alphat*gJ0t)
    Array3DUtils::computeGradient(*mScratchI,*mScratchV,mCurSpacing,false);
    mScratchV->pointwiseMultiplyBy(*mAlphaT);
    mKernel->CopyIn(*mScratchV);
    mKernel->ApplyInverseOperator();
    mKernel->CopyOut(*(mVVec[t]));
    mVVec[t]->scale(-1.0f);
  }

  // final update  
  // final update to the deformation fields
  HField3DUtils::composeHVInv(phiT0, *(mVVec[mNTimeSteps-1]), *mScratchV, mCurSpacing,
			      HField3DUtils::BACKGROUND_STRATEGY_PARTIAL_ID);
  phiT0 = *mScratchV;

  HField3DUtils::composeVH(*(mVVec[mNTimeSteps-1]), phi0T, *mScratchV, mCurSpacing,
			   HField3DUtils::BACKGROUND_STRATEGY_PARTIAL_ZERO);
  phi0T = *mScratchV;


  // get the vector energy term: Note that this is just <Lv0,v0> and not integrated over time
  mKernel->CopyIn(*(mVVec[0]));
  mKernel->ApplyOperator();
  mKernel->CopyOut(*mScratchV);
  vecEnergy = HField3DUtils::l2DotProd(*mScratchV, *(mVVec[0]), mCurSpacing);
  energy.SetVecStepEnergy(vecEnergy);

  // compute image energy  = ||JT0 - I1||^2
  // create the final deformed image
  HField3DUtils::apply(data.I0(), phiT0, *mScratchI);
  mScratchI->pointwiseSubtract(data.I1());
  
  energy.SetImageEnergy(ImageUtils::l2NormSqr(*mScratchI));
  data.AddEnergy(energy);

  /** ####ADJOINT ALGORITHM PART 2: GRADIENT COMPUTATION BY SOLVING ADJOINT EQUATIONS VIA BACKWARD INTEGRATION####  */  
  backwardAdIntegration(data);

}

void 
LDMMAdShootingIteratorCPU::backwardAdIntegration(LDMMShootingDefData &data){

  VectorField &phi0T = data.Phi0T();
  VectorField &phiT0 = data.PhiT0();

  *mAlphaT = data.Alpha0();

  /** Initial conditions: compute \tilde{alpha}(1) and \tilde{I}(1) */  
  // \tilde{alpha}(1) is zero
  mAlphaTildeT->fill(0.0f);

  // \tilde{I}(1), phiT0 contains phi(1,0)
  HField3DUtils::apply(data.I0(), phiT0, *mScratchI);
  *mIHatT = data.I1();
  mIHatT->pointwiseSubtract(*mScratchI);
  mIHatT->scale( ((1.0f)/(mSigma*mSigma)) ); 

  // splat image difference term to time 0 : \tilde{I(1)}=Jac(Phi0T)IhatT o PhiOT
  HField3DUtils::forwardApply(*mIHatT, phiT0, *mITildeT, 0.0f, false);


  for (int i=mNTimeSteps-1; i>= 0; i--){    

    if(i!=(static_cast<int>(mNTimeSteps)-1)){  //1st iteration is separated because we can use Phi0T, PhiT0 computed earlier in shooting
      HField3DUtils::setToIdentity(phi0T);
      HField3DUtils::setToIdentity(phiT0);
      //Get \phi_{t_{i+1},0} and \phi_{0,t_{i+1}}
      for(int j=1; j<=(i+1); j++){

	
	HField3DUtils::composeVH(*(mVVec[j-1]), phi0T, *mScratchV, mCurSpacing,
				 HField3DUtils::BACKGROUND_STRATEGY_PARTIAL_ZERO);
	phi0T = *mScratchV;

	HField3DUtils::composeHVInv(phiT0, *(mVVec[j-1]), *mScratchV, mCurSpacing,
				    HField3DUtils::BACKGROUND_STRATEGY_PARTIAL_ID);
	phiT0 = *mScratchV;	

      }
    }
    // create deformed image, I_{t+1}
    HField3DUtils::apply(data.I0(), phiT0, *mIT);
    
    // create deformed alpha0, alpha_{t+1}
    HField3DUtils::forwardApply(data.Alpha0(), phi0T, *mAlphaT, 0.0f, false);

    //Compute \hat{alpha}(t_{i+1})
    HField3DUtils::apply(*mAlphaTildeT, phiT0,*mAlphaHatT); // \tilde{Alpha}_{t+1} from previous iteration/initialization

    //Compute \hat{I}(t_{i+1}): splat \tilde{I}_{t_{i+1}} to time t_{i+1}
    HField3DUtils::forwardApply(*mITildeT, phi0T, *mIHatT, 0.0f, false);    // \tilde{I}_{t+1} from previous iteration/initialization

    //Compute \hat{v}(t_{i+1})
    Array3DUtils::computeGradient(*mAlphaHatT, *mScratchV, mCurSpacing, false);
    Array3DUtils::computeGradient(*mIT, *mScratchV2, mCurSpacing, false); //mIT has I_{t+1}

    mScratchV->pointwiseMultiplyBy(*mAlphaT);//    mScratchV has \nabla \hat{alpha}(t_{i+1})
    mScratchV2->pointwiseMultiplyBy(*mIHatT); //    mScratchV2 has \nabla I(t_{i+1})

    mScratchV->pointwiseSubtract(*mScratchV2);

    mKernel->CopyIn(*mScratchV);
    mKernel->ApplyInverseOperator();
    mKernel->CopyOut(*(mScratchV));

    //Result of pointwise innerprod \nabla I(t_{i+1}) \dot \hat{v}(t_{i+1}) will be stored in mScratchI
    Array3DUtils::computeGradient(*mIT, *mScratchV2, mCurSpacing, false); //Computing    \nabla I(t_{i+1}) again because mScratchV2 was overwritten earlier

    HField3DUtils::pointwiseL2DotProd(*mScratchV2,*mScratchV,*mScratchI); 

    //Compute \tilde{P}(t_i)
    HField3DUtils::apply(*mScratchI, phi0T, *mScratchI2);    
    mAlphaTildeT->pointwiseSubtract(*mScratchI2);

    //Result of divergence computation \nabla \dot (\alpha_{t_{i+1}} \hat{v}(t_{i+1})) will be strored in mScratchI
    mScratchV->pointwiseMultiplyBy(*mAlphaT); 
    HField3DUtils::divergence(*mScratchV,*mScratchI,mCurSpacing, false);
    HField3DUtils::forwardApply(*mScratchI, phiT0, *mScratchI2,0.0f,false); //splat
    //Compute new \tilde{I}(t_i)
    mITildeT->pointwiseAdd(*mScratchI2);

  }  // By the end of above loop we have \tilde{alpha}(0) in mAlphaTildeT

  // Compute gradient of energy functional 
  Array3DUtils::computeGradient(data.I0(), *mScratchV, mCurSpacing, false);
  HField3DUtils::pointwiseL2DotProd(*mScratchV,*(mVVec[0]),*mScratchI);

  mScratchI->scale((-1.0f));
  mScratchI->pointwiseSubtract(*mAlphaTildeT); 

  mScratchI->scale(data.StepSize());  
  RealImage &Alpha0 = data.Alpha0(); 

  // Updated initial momenta for this gradient descent iteration
  Alpha0.pointwiseSubtract(*mScratchI);
}

void 
LDMMAdShootingIteratorCPU::finalUpdatePhi0T(LDMMShootingDefData &data){

  // Update final phi0T
  VectorField &phi0T = data.Phi0T();
  HField3DUtils::setToIdentity(phi0T);
  for(uint j=1; j<=mNTimeSteps; j++){	
    HField3DUtils::composeVH(*(mVVec[j-1]), phi0T, *mScratchV, mCurSpacing,
			     HField3DUtils::BACKGROUND_STRATEGY_PARTIAL_ZERO);
    phi0T = *mScratchV;  
  }
}

void 
LDMMAdShootingIteratorCPU::finalUpdatePhiT0(LDMMShootingDefData &data){
  // Update final phiT0
  VectorField &phiT0 = data.PhiT0();
  HField3DUtils::setToIdentity(phiT0);
  for(uint j=1; j<=mNTimeSteps; j++){	   
    HField3DUtils::composeHVInv(phiT0, *(mVVec[j-1]), *mScratchV, mCurSpacing,
				HField3DUtils::BACKGROUND_STRATEGY_PARTIAL_ID);
    phiT0 = *mScratchV;	    
  }
}
