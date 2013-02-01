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


#include "LDMMIteratorCPU.h"

//
// ################ LDMMIteratorCPU Implementation ################ //
//

LDMMIteratorCPU::
LDMMIteratorCPU(const SizeType &size, 
		const OriginType &origin,
		const SpacingType &spacing,
		unsigned int nTimeSteps,
		bool debug)
  : DeformationIteratorInterface(size, origin, spacing),
    mNTimeSteps(nTimeSteps),
    mSigma(0.f),
    mDebug(debug),
    mUpdateStepSizeNextIter(false),
    mHField(NULL),
    mUseOrigGradientCalc(true),
    mJ0t(NULL),
    mDiffIm(NULL),
    mJTt(NULL),
    mJacDet(NULL),
    mKernel(NULL),
    mKernelVF(NULL),
    mScratchI(NULL),
    mScratchV(NULL)
{
}

LDMMIteratorCPU::~LDMMIteratorCPU(){
  if(mUseOrigGradientCalc){
    for(unsigned int i=0;i<mNTimeSteps;i++){
      delete mJTt[i];
      delete mJacDet[i];
    }
    delete [] mJTt;
    delete [] mJacDet;
  }else{
    for(unsigned int i=0;i<mNTimeSteps;i++){
      delete mJ0t[i];
    }
    delete [] mJ0t;
    delete mDiffIm;
  }

  delete mScratchI;
  delete mScratchV;

  if(mKernel) delete mKernel;
}

void 
LDMMIteratorCPU::
SetScaleLevel(const MultiscaleManager &scaleManager,
	      const LDMMIteratorParam &param)
{
  ScaleLevelDataInfo::SetScaleLevel(scaleManager);

  mParam = &param;
  mMaxPert = mParam->MaxPert();
  mSigma = mParam->Sigma();
  
  if(mKernel) delete mKernel;
  
  mKernel = KernelFactory::NewKernel(param.Kernel(), mCurSize, mCurSpacing);
  mKernelVF = mKernel->GetInternalFFTWVectorField();

  // allocate memory
  if(mUseOrigGradientCalc){
    // allocate / resize mJTt
    if(mJTt == NULL){
      mJTt = new RealImage*[mNTimeSteps];
      for(unsigned int i=0;i<mNTimeSteps;i++){
	mJTt[i] = NULL;
      }
    }
    for(unsigned int i=0;i<mNTimeSteps;i++){
      if(mJTt[i]){
	mJTt[i]->resize(mCurSize);
	mJTt[i]->setSpacing(mCurSpacing);
      }else{
	mJTt[i] = new RealImage(mCurSize, mImOrigin, mCurSpacing);
      }
    }
    // allocate / resize mJacDet
    if(mJacDet == NULL){
      mJacDet = new RealImage*[mNTimeSteps];
      for(unsigned int i=0;i<mNTimeSteps;i++){
	mJacDet[i] = NULL;
      }
    }
    for(unsigned int i=0;i<mNTimeSteps;i++){
      if(mJacDet[i]){
	mJacDet[i]->resize(mCurSize);
	mJacDet[i]->setSpacing(mCurSpacing);
      }else{
	mJacDet[i] = new RealImage(mCurSize, mImOrigin, mCurSpacing);
      }
    }
  }else{
    if(mJ0t == NULL){
      mJ0t = new RealImage*[mNTimeSteps];
      for(unsigned int i=0;i<mNTimeSteps;i++){
	mJ0t[i] = NULL;
      }
    }
    for(unsigned int i=0;i<mNTimeSteps;i++){
      if(mJ0t[i]){
	mJ0t[i]->resize(mCurSize);
	mJ0t[i]->setSpacing(mCurSpacing);
      }else{
	mJ0t[i] = new RealImage(mCurSize, mImOrigin, mCurSpacing);
      }
    }
    if(mDiffIm){
      mDiffIm->resize(mCurSize);
      mDiffIm->setSpacing(mCurSpacing);
    }else{
      mDiffIm = new RealImage(mCurSize, mImOrigin, mCurSpacing);
    }
  }
  
  if(mScratchI){
    mScratchI->resize(mCurSize);
    mScratchI->setSpacing(mCurSpacing);
  }else{
    mScratchI = new RealImage(mCurSize, mImOrigin, mCurSpacing);
  }
  
  if(mHField){
    mHField->resize(mCurSize);
  }else{
    mHField = new VectorField(mCurSize);
  }

  if(mScratchV){
    mScratchV->resize(mCurSize);
  }else{
    mScratchV = new VectorField(mCurSize);
  }
  

}

void 
LDMMIteratorCPU::
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
LDMMIteratorCPU::
updateVelocity(VectorField &v,
	       Real vFac,
	       const VectorField &u,
	       Real uFac)
{
  Vector3D<unsigned int> size = v.getSize();
  for (unsigned int z = 0; z < size.z; ++z) {
    for (unsigned int y = 0; y < size.y; ++y) {
      for (unsigned int x = 0; x < size.x; ++x) {
	v(x,y,z) = vFac*v(x,y,z) + uFac*u(x,y,z);
      }
    }
  }
}

Real
LDMMIteratorCPU::
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

RealImage&
LDMMIteratorCPU::
GetAlpha(unsigned int t)
{
  if(mUseOrigGradientCalc){
    return *mJTt[t];
  }else{
    return *mJ0t[t];
  }
}

template <class T>
inline
Vector3D<T>  sumTest(Array3D<Vector3D<T> >& hField)
{
  Vector3D<double> sum_d(0,0,0);
  Vector3D<unsigned int> size = hField.getSize();
  for (unsigned int z = 0; z < size.z; ++z) {
    for (unsigned int y = 0; y < size.y; ++y) {
      for (unsigned int x = 0; x < size.x; ++x) {
	sum_d.x += hField(x, y, z).x;
	sum_d.y += hField(x, y, z).y;
	sum_d.z += hField(x, y, z).z;
      }
    }
  }
  return Vector3D<T>(sum_d.x,sum_d.y,sum_d.z);
}

template Vector3D<float>  sumTest(Array3D<Vector3D<float> >& hField);

inline float  sumTest(Array3D<float>& hField){
  double sum = 0.0;
  Vector3D<unsigned int> size = hField.getSize();
  for (unsigned int z = 0; z < size.z; ++z)
    for (unsigned int y = 0; y < size.y; ++y)
      for (unsigned int x = 0; x < size.x; ++x)
	sum += hField(x,y,z);
  return sum;
}

template <class T, class U>
inline
void jacobian(const Array3D<Vector3D<U> >& hField,
              Array3D<T>& jacobian,
              Vector3D<double> spacing)
{
  Vector3D<unsigned int> size = hField.getSize();
  unsigned int numElements = hField.getNumElements();

  // resize jacobian array if necessary
  if (jacobian.getSize() != size)
    {
      jacobian.resize(size);
    }

  // build scalar images h1, h2, and h3
  // from the transformation field
  Array3D<U> h1(size);
  Array3D<U> h2(size);
  Array3D<U> h3(size);
  unsigned int i; // stupid vc++
  for (i = 0; i < numElements; ++i)
    {
      h1(i) = hField(i).x;
      h2(i) = hField(i).y;
      h3(i) = hField(i).z;
    }
    
  // compute the gradients of h1, h2, and h3
  Array3D<Vector3D<U> > grad_h1(size);
  Array3D<Vector3D<U> > grad_h2(size);
  Array3D<Vector3D<U> > grad_h3(size);   
  Array3DUtils::computeGradient(h1, grad_h1, spacing);
  Array3DUtils::computeGradient(h2, grad_h2, spacing);
  Array3DUtils::computeGradient(h3, grad_h3, spacing);

  std::cerr << "Check step CPU " << sumTest(grad_h1) << sumTest(grad_h2) << sumTest(grad_h3) << std::endl; 
  // compute the jacobian
  T t1, t2, t3;
  for (i = 0; i < numElements; ++i)
    {
      t1 = static_cast<T>(grad_h1(i).x * (grad_h2(i).y * grad_h3(i).z -
					  grad_h2(i).z * grad_h3(i).y));
      t2 = static_cast<T>(grad_h1(i).y * (grad_h2(i).x * grad_h3(i).z -
					  grad_h2(i).z * grad_h3(i).x));
      t3 = static_cast<T>(grad_h1(i).z * (grad_h2(i).x * grad_h3(i).y -
					  grad_h2(i).y * grad_h3(i).x));
      jacobian(i) = t1 - t2 + t3;
    }
  std::cerr << "Check jacobian CPU " << sumTest(jacobian) << std::endl; 
}

void 
LDMMIteratorCPU::
ComputeJacDet(std::vector<VectorField*> &v, RealImage *jacDet)
{
  for(int i = mNTimeSteps-1; i >= 0; i--){

    // compute determinant of jacobian of current deformation:
    // |D(h_{t-1})| = (|D(h_t)|(x+v_t))*|D(x+v_t)|

    // get identity in world coords
    HField3DUtils::setToIdentity(*mScratchV);
    mScratchV->scale(mCurSpacing);
    // and add velocity in world coords
    mScratchV->pointwiseAdd(*v[i]);
    
    // mScratchI = |D(x+v(x))|
    HField3DUtils::jacobian(*mScratchV,*mScratchI,mCurSpacing);

    // get a scratch image
    RealImage &tmp = this->GetAlpha(0);

    /////HField3DUtils::jacobian(*mScratchV,*mScratchI,Vector3D<Real>(1.0,1.0,1.0));
    if(i == (int)mNTimeSteps-1){
      *jacDet = *mScratchI;
    }else{
      // deform current det. of jac. (just using tmp as scratch)
      HField3DUtils::applyU(*jacDet, *v[i], tmp, mCurSpacing);
      *jacDet = tmp;
      // scale by new deformation jacobian
      jacDet->pointwiseMultiplyBy(*mScratchI);
    }

  }
}

// void
// LDMMIteratorCPU::
// ComputeJacDet(std::vector<VectorField*> &v, RealImage *jacDet)
// {
//   jacDet->resize(mCurSize);
//   HField3DUtils::setToIdentity(*mHField);
//   for(unsigned int t = 0; t < mNTimeSteps; t++){
//     HField3DUtils::composeHVInv(*mHField, *v[t], *mScratchV, mCurSpacing,
// 				HField3DUtils::BACKGROUND_STRATEGY_PARTIAL_ID);
//     *mHField = *mScratchV;
//   }
//   mScratchI->fill(1.0);
//   HField3DUtils::forwardApply(*mScratchI, *mHField, *jacDet, 0.f, false);
// }

void 
LDMMIteratorCPU::
Iterate(DeformationDataInterface &defData)
{
  this->Iterate(defData, false);
}

void 
LDMMIteratorCPU::
Iterate(DeformationDataInterface &deformationData,
	bool computeAlphasOnly)
{
  if(mUseOrigGradientCalc){
    this->IterateOrig(deformationData, false);
  }else{
    this->IterateNew(deformationData, false);
  }
  
}

void 
LDMMIteratorCPU::
IterateOrig(DeformationDataInterface &deformationData,
	    bool computeAlphasOnly)
{
  
  LDMMDeformationData &data = dynamic_cast<LDMMDeformationData&>(deformationData);
  
  LDMMEnergy energy(mNTimeSteps, mSigma);
  if(mParam->VerboseEnergy()){ energy.Verbose(true); }

  // when calculating step size, this holds max displacement across all velocity fields
  Real totalMaxDisplacement = 0.f;

  // compute mdJacDet[t] and mdJT[t] for all t
  HField3DUtils::setToIdentity(*mHField);
  for (int t = mNTimeSteps-1; t>=0; --t){
    // 1. Compute JTt
    // Compute the current deformation h(t) = h_(t+1)(x + vt);
    // cudaHField3DUtils::composeHV(mdScratchV, mdH, data.dV(t),
    // 				 mCurSize, mCurSpacing, BACKGROUND_STRATEGY_PARTIAL_ID);
    HField3DUtils::composeHV(*mHField, data.v(t), *mScratchV, mCurSpacing,
			     HField3DUtils::BACKGROUND_STRATEGY_PARTIAL_ID);
    // swap(mdH, mdScratchV);
    *mHField = *mScratchV;
	
    // deform the image base on the current deformation JTt= I(h(t))
    //cudaHField3DUtils::apply(mdJTt[t], data.dI1(), mdH, mCurSize);
    HField3DUtils::apply(data.I1(), *mHField, *mJTt[t]);
	
    // 2. Compute DPhiT
    // Compute determinent of jacobian of current deformation 
    // D(h_{t-1}) = ( D(h_t) |(x+ v_t)) * |D (x + v_t)|
    // Compute x + v_t

    // get identity in world coords
    HField3DUtils::setToIdentity(*mScratchV);
    mScratchV->scale(mCurSpacing);
    // and add velocity in world coords
    mScratchV->pointwiseAdd(data.v(t));
    
	
    // Compute |D(x + v_t)|
    HField3DUtils::jacobian(*mScratchV, *mScratchI, mCurSpacing);

    if (t == (int)mNTimeSteps - 1){
      *mJacDet[t] = *mScratchI;
    }else {
      //D(h_t) |(x+v_t)
      HField3DUtils::applyU(*mJacDet[t+1], data.v(t), *mJacDet[t], mCurSpacing);
      mJacDet[t]->pointwiseMultiplyBy(*mScratchI);
    }

  }

  // update each velocity field from source to dest
  HField3DUtils::setToIdentity(*mHField);
  for (unsigned int t=0; t< mNTimeSteps; ++t){
    // 3. Compute J0Tt
    // Compute the defomed image

    if (t==0){
      *mScratchI = data.I0();
    }
    else {
      HField3DUtils::apply(data.I0(), *mHField, *mScratchI);
    }
      
    // Compute the gradient of deformed image
    Array3DUtils::computeGradient(*mScratchI, *mScratchV, mCurSpacing, false);

    mScratchI->pointwiseSubtract(*mJTt[t]);
    mScratchI->pointwiseMultiplyBy(*mJacDet[t]);

    if(computeAlphasOnly){
      // scale alphas appropriately
      mScratchI->scale((float)1.0/(mSigma*mSigma));
      // store alphas in mdJTt array
      *mJTt[t] = *mScratchI;
    }else{
	
      mScratchV->pointwiseMultiplyBy(*mScratchI);

      mKernel->CopyIn(*mScratchV);
      mKernel->ApplyInverseOperator();
      mKernel->CopyOut(*mScratchV);
	
      if(mUpdateStepSizeNextIter){
	// update velocity fields based on step size
	
	this->updateVelocity(*mScratchV, -2.0 / (mSigma * mSigma), data.v(t), 2);
	double unused=0;
	double maxDisplacement = 0.f;
	HField3DUtils::minMaxVelocityL2Norm(*mScratchV,unused,maxDisplacement);

	if(maxDisplacement != maxDisplacement){
	  throw AtlasWerksException(__FILE__, __LINE__, 
				    "Error, max displacement is NaN");
	}
	if(maxDisplacement > totalMaxDisplacement){
	  totalMaxDisplacement = maxDisplacement;
	}
      }else {
	// update velocity fields based on step size
	this->updateVelocity(data.v(t), 
			     1.f - 2 * data.StepSize(), 
			     *mScratchV, 
			     2*data.StepSize()/(mSigma*mSigma));
      }
    } // end if compute only alphas
      
    HField3DUtils::composeHVInv(*mHField, data.v(t), *mScratchV, mCurSpacing,
				HField3DUtils::BACKGROUND_STRATEGY_PARTIAL_ID);
    *mHField = *mScratchV;

    if(mDebug){
      // compute vector energy = <Lv,v>
      mKernel->CopyIn(data.v(t));
      mKernel->ApplyOperator();
      mKernel->CopyOut(*mScratchV);
      Real vecEnergy = HField3DUtils::l2DotProd(*mScratchV, data.v(t), mCurSpacing);
      //energy.SetVecStepEnergy(vecEnergy);
#if USE_LV
      Real estVecEnergy = 
	l2DotProd(data.dLV(t), data.dV(t), mCurSpacing);
      std::cerr << "Ref " << vecEnergy << " Est " << estVecEnergy << " Diff " << estVecEnergy - vecEnergy << std::endl;
      energy.SetVecStepEnergy(estVecEnergy);
#else
      energy.SetVecStepEnergy(vecEnergy);
#endif
    }
    
  }

  if(mUpdateStepSizeNextIter){
    data.StepSize(mMaxPert / totalMaxDisplacement);
    LOGNODETHREAD(logDEBUG) << "Step size is " << data.StepSize();
    mUpdateStepSizeNextIter = false;
  }else{
    if(mDebug){
      HField3DUtils::apply(data.I0(), *mHField, *mScratchI);
      mScratchI->pointwiseSubtract(data.I1());
      energy.SetImageEnergy(ImageUtils::l2NormSqr(*mScratchI));
      // test for NaN
      if(energy.ImageEnergy() != energy.ImageEnergy()){
	throw AtlasWerksException(__FILE__, __LINE__, "Error, NaN encountered");
      }
      data.AddEnergy(energy);
    }
  }
}

void 
LDMMIteratorCPU::
IterateNew(DeformationDataInterface &deformationData,
	   bool computeAlphasOnly)
{
  LDMMDeformationData &data = 
    dynamic_cast<LDMMDeformationData&>(deformationData);
  
  // this will only be filled if mDebug is true
  LDMMEnergy energy(mNTimeSteps, mSigma);
  if(mParam->VerboseEnergy()){ energy.Verbose(true); }

  Real totalMaxDisplacement = 0.f;

  // Compute deformed images from source to dest
  HField3DUtils::setToIdentity(*mHField);
  for(unsigned int i = 0; i < mNTimeSteps; i++){
    // update the deformation (hfield) from dest to this timepoint
    HField3DUtils::composeHVInv(*mHField, data.v(i), *mScratchV, mCurSpacing,
				HField3DUtils::BACKGROUND_STRATEGY_PARTIAL_ID);
    *mHField = *mScratchV;
    // create the deformed image from source to this timepoint
    HField3DUtils::apply(data.I0(), *mHField, *mJ0t[i]);
  }
  
  // compute difference image
  *mDiffIm = *mJ0t[mNTimeSteps-1];
  mDiffIm->pointwiseSubtract(data.I1());

  // update each velocity field from dest to source
  HField3DUtils::setToIdentity(*mHField);
  for(int t = mNTimeSteps-1; t >= 0; t--){

    // update the deformation (hfield) from dest to this timepoint
    HField3DUtils::composeVHInv(data.v(t), *mHField, *mScratchV, mCurSpacing,
				HField3DUtils::BACKGROUND_STRATEGY_PARTIAL_ZERO);
    *mHField = *mScratchV;

    // splat the difference image from dest to this timepoint
    HField3DUtils::forwardApply(*mDiffIm, *mHField, *mScratchI, 0.0f, false);

    // mScratchI now holds the alpha field for this timestep (used for
    // geodesic shooting) -- this could be of interest to us
    if(computeAlphasOnly){
      if(data.ComputeAlphas() || 
	 (data.ComputeAlpha0() && t == 0))
	{
	  data.Alpha(t) = *mScratchI;
	  data.Alpha(t).scale(1.0/(mSigma*mSigma));
	}
    }else{
      // compute gradient of forward-deformed image, store in internal vector field
      // of differential operator
      if(t > 0){
	Array3DUtils::computeGradient(*mJ0t[t-1],*mKernelVF,mCurSpacing,false);
      }else{
	Array3DUtils::computeGradient(data.I0(),*mKernelVF,mCurSpacing,false);
      }
      
      // create the body force by scaling the gradient
      pointwiseMultiplyBy_FFTW_Safe(*mKernelVF, *mScratchI);
      
      mKernel->ApplyInverseOperator();
      
      if(mUpdateStepSizeNextIter){
	Real maxDisplacement = calcMaxDisplacement(data.v(t),*mKernelVF);
	if(maxDisplacement > totalMaxDisplacement) 
	  totalMaxDisplacement = maxDisplacement;
      }else{
	this->updateVelocity(data.v(t), 
			     1.f - 2 * data.StepSize(), 
			     *mKernelVF, 
			     2*data.StepSize()/(mSigma*mSigma));
	// compute vector energy, <Lv,v>
	if(mDebug){
	  // save the current field
	  mKernel->CopyIn(data.v(t));
	  mKernel->ApplyOperator();
	  mKernel->CopyOut(*mScratchV);
	  Real vecEnergy = HField3DUtils::l2DotProd(*mScratchV, data.v(t), mCurSpacing);
	  energy.SetVecStepEnergy(vecEnergy);
	}
      } // end mUpdateStepSizeNextIter
    } // end computeAlphasOnly
  } // end loop over timesteps

  if(mUpdateStepSizeNextIter){
    data.StepSize(mMaxPert / totalMaxDisplacement);
    LOGNODETHREAD(logDEBUG) << "Step size is " << data.StepSize();
    mUpdateStepSizeNextIter = false;
  }else{
    if(mDebug){
      energy.SetImageEnergy(ImageUtils::l2NormSqr(*mDiffIm));
      // test for NaN
      if(energy.ImageEnergy() != energy.ImageEnergy()){
	throw AtlasWerksException(__FILE__, __LINE__, "Error, NaN encountered");
      }
      
      data.AddEnergy(energy);
    }
  }

}

void 
LDMMIteratorCPU::
ReParameterize(LDMMDeformationData &defData)
{
  std::vector<VectorField*> vVec;
  std::vector<Real> lVec;

  // last energy
  const Energy &e = defData.LastEnergy();
  const LDMMEnergy &oldEnergy = dynamic_cast<const LDMMEnergy&>(e);
  // will hold new energy after reparameterization
  LDMMEnergy energy(mNTimeSteps, mSigma);
  if(mParam->VerboseEnergy()){ energy.Verbose(true); }
  
  // find length of path
  Real l = 0.f;
  lVec.push_back(l);
  for(unsigned int t=0;t<mNTimeSteps;++t){
    l += sqrt(oldEnergy.VecStepEnergy(t));
    lVec.push_back(l);
  }

  LOGNODETHREAD(logDEBUG2) << "Geodesic Length: " << l 
			   << ", average length^2: " 
			   << (l/mNTimeSteps)*(l/mNTimeSteps)
			   << std::endl;

  // compute reparameterized v's
  for(unsigned int t=0;t<mNTimeSteps;++t){
    Real time = (l*t)/mNTimeSteps;
    int base = 0;
    while(base+1 < (int)lVec.size() && (Real)lVec[base+1] < time) ++base;
    time = ((Real)base) + (time-lVec[base])/(lVec[base+1]-lVec[base]);

    LOGNODETHREAD(logDEBUG2) << "Timestep " << t 
			     << " constant-speed reparameterized time is " 
			     << time << std::endl; 
    
    VectorField *v = new VectorField();
    vVec.push_back(v);
    defData.InterpV(*v, time);
    mKernel->CopyIn(*v);
    mKernel->ApplyOperator();
    mKernel->CopyOut(*mScratchV);
    Real scaleFac = HField3DUtils::l2DotProd(*mScratchV, *v, mCurSpacing);
    scaleFac = l/(mNTimeSteps*sqrt(scaleFac));
    v->scale(scaleFac);

    // compute vector energy
    mKernel->CopyIn(*v);
    mKernel->ApplyOperator();
    mKernel->CopyOut(*mScratchV);
    Real vecEnergy = HField3DUtils::l2DotProd(*mScratchV, *v, mCurSpacing);
    energy.SetVecStepEnergy(vecEnergy);
  }
  
  // assign vector fields and clean up memory
  for(unsigned int t=0;t<mNTimeSteps;++t){
    defData.v(t) = *vVec[t];
    delete vVec[t];
  }

  // get a scratch image
  RealImage &tmp = this->GetAlpha(0);
  // calculate new image energy
  defData.GetI0At1(tmp);
  tmp.pointwiseSubtract(defData.I1());
  energy.SetImageEnergy(ImageUtils::l2NormSqr(tmp));
  defData.GetEnergyHistory().AddEvent(ReparameterizeEvent(energy));

}

