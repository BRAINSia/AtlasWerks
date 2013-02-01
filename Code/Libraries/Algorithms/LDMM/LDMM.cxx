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

#include "LDMM.h"
#include "stdio.h"
#include "ApplicationUtils.h"
#include "log.h"

LDMMIterator::
LDMMIterator(const Vector3D<unsigned int> &size, 
	     const Vector3D<Real> &origin,
	     const Vector3D<Real> &spacing,
	     const unsigned int &nTimeSteps,
	     const LDMMIteratorOldParam &param,
	     bool debug)
  :
  mNTimeSteps(nTimeSteps),
  mUseAdaptiveStepSize(param.UseAdaptiveStepSize()),
  mMaxPert(param.MaxPert()),
  mStepSize(param.StepSize()),
  mSigma(param.Sigma()),
  mImSize(size),
  mImOrigin(origin),
  mImSpacing(spacing),
  mDebug(debug),
  mVectorEnergy(0.0),
  mImageEnergy(0.0),
  mTotalEnergy(0.0),
  mVectorStepEnergy(NULL),
  mAlpha(NULL),
  mUField(NULL)
{
  // allocate memory
  mJTt = new RealImage*[mNTimeSteps];
  mJacDet = new RealImage*[mNTimeSteps];
  for(unsigned int i=0;i<mNTimeSteps;i++){
    mJTt[i] = new RealImage(mImSize, mImOrigin, mImSpacing);
    mJacDet[i] = new RealImage(mImSize, mImOrigin, mImSpacing);
  }
  mScratchI = new RealImage(mImSize, mImOrigin, mImSpacing);
  mScratchV = new VectorField(mImSize);
  
  // create DiffOper
  mOp = new DiffOper(mImSize, mImSpacing, param.DiffOper());
  mDiffOpVF = mOp->GetInternalFFTWVectorField();
}

LDMMIterator::~LDMMIterator(){
  for(unsigned int i=0;i<mNTimeSteps;i++){
    delete mJTt[i];
    delete mJacDet[i];
  }
  delete [] mJTt;
  delete [] mJacDet;

  delete mScratchI;
  delete mScratchV;

  delete mOp;
}

void
LDMMIterator::
SetImages(const RealImage *initial, const RealImage *final)
{
  this->mI0 = initial;
  this->mIT = final;
}

void 
LDMMIterator::
SaveAlphaImages(RealImage **alpha)
{
  mAlpha = alpha;
}

void 
LDMMIterator::
SaveUFields(VectorField **uField)
{
  mUField = uField;
}

void 
LDMMIterator::
pointwiseMultiplyBy_FFTW_Safe(DiffOper::FFTWVectorField &lhs, 
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

Real 
LDMMIterator::
calcUpdate(const VectorField &curV,
	   const DiffOper::FFTWVectorField &uField,
	   VectorField &update)
{
  Vector3D<unsigned int> size = curV.getSize();
  Real uFac = 2.0/(mSigma*mSigma);
  Real maxLen = 0.0;
  for (unsigned int z = 0; z < size.z; ++z) {
    for (unsigned int y = 0; y < size.y; ++y) {
      for (unsigned int x = 0; x < size.x; ++x) {
	update(x,y,z) = 2.0*curV(x,y,z) - uFac*uField(x,y,z);
	Real len = update(x,y,z).normL2();
	maxLen = len > maxLen ? len : maxLen;
      }
    }
  }

  Real step = mMaxPert*mImSpacing.minElement()/maxLen;
  //std::cout << "maxLen: " << maxLen << ", CalcStep : " << step;
  if(step > mStepSize){
    step = mStepSize;
  }
  //std::cout << ", FinalStep : " << step << std::endl;
  return step;
}

void 
LDMMIterator::
updateVelocity(VectorField &v,
	       const DiffOper::FFTWVectorField &uField)
{
  Vector3D<unsigned int> size = v.getSize();
  Real vFac = 1.0-2.0*mStepSize;
  Real uFac = (2*mStepSize)/(mSigma*mSigma);
  for (unsigned int z = 0; z < size.z; ++z) {
    for (unsigned int y = 0; y < size.y; ++y) {
      for (unsigned int x = 0; x < size.x; ++x) {
	v(x,y,z) = vFac*v(x,y,z) + uFac*uField(x,y,z);
      }
    }
  }
}

void LDMMIterator::
GetUField(VectorField &vf)
{
  mOp->CopyOut(vf);
}

void 
LDMMIterator::
Iterate(VectorField **v, VectorField *hField)
{
  
  // reset energy accumulation (for debugging only)
  mVectorEnergy = 0.0;
  if(mDebug){
    if(!mVectorStepEnergy){
      mVectorStepEnergy = new Real[mNTimeSteps];
    }
    for(unsigned int i=0;i<mNTimeSteps;i++){
      mVectorStepEnergy[i] = 0.0;
    }
  }

  // Compute deformed images from dest to source
  HField3DUtils::setToIdentity(*hField);
  for(int i = mNTimeSteps-1; i >= 0; i--){

    // update the deformation (hfield) from dest to this timepoint
    HField3DUtils::composeHV(*hField, *v[i], *mScratchV, mImSpacing,
			     HField3DUtils::BACKGROUND_STRATEGY_PARTIAL_ID);
    
    *hField = *mScratchV;

    // create the deformed image from dest to this timepoint
    HField3DUtils::apply(*mIT, *hField, *mJTt[i]);

    // compute determinant of jacobian of current deformation:
    // |D(h_{t-1})| = (|D(h_t)|(x+v_t))*|D(x+v_t)|

    // get identity in world coords
    HField3DUtils::setToIdentity(*mScratchV);
    mScratchV->scale(mImSpacing);
    // and add velocity in world coords
    mScratchV->pointwiseAdd(*v[i]);
    // mScratchI = |D(x+v(x))|
    HField3DUtils::jacobian(*mScratchV,*mScratchI,mImSpacing);
    /////HField3DUtils::jacobian(*mScratchV,*mScratchI,Vector3D<Real>(1.0,1.0,1.0));
    if(i == (int)mNTimeSteps-1){
      *mJacDet[i] = *mScratchI;
    }else{
      // deform current det. of jac.
      HField3DUtils::applyU(*mJacDet[i+1], *v[i], *mJacDet[i], mImSpacing);
      // scale by new deformation jacobian
      mJacDet[i]->pointwiseMultiplyBy(*mScratchI);
    }
  }

  // update each velocity field from source to dest
  HField3DUtils::setToIdentity(*hField);
  for(unsigned int i = 0; i < mNTimeSteps; i++){

    // create the deformed image from source to this timepoint
    HField3DUtils::apply(*mI0, *hField, *mScratchI);

    // compute gradient of deformed image, store in internal vector field
    // of differential operator
    Array3DUtils::computeGradient(*mScratchI,*mDiffOpVF,mImSpacing,false);
    /////Array3DUtils::computeGradient(*mScratchI,*mDiffOpVF,Vector3D<Real>(1.0,1.0,1.0),false);

    // subtract the reverse-deformed image
    mScratchI->pointwiseSubtract(*mJTt[i]);

    // multiply by the jacobian determinant
    mScratchI->pointwiseMultiplyBy(*mJacDet[i]);

    // mScratchI now holds the alpha field for this timestep (used for
    // geodesic shooting) -- this could be of interest to us
    if(mAlpha){
      *mAlpha[i] = *mScratchI;
    }

    // create the body force by scaling the gradient
    pointwiseMultiplyBy_FFTW_Safe(*mDiffOpVF, *mScratchI);

    mOp->ApplyInverseOperator();
    
    if(mUField){
      mOp->CopyOut(*mUField[i]);
    }

    // calculates the update:
    // v_{k+1} = v_k - stepsize*(2v_k - (2/sigma^2)*u_k)
    if(mUseAdaptiveStepSize){
      Real step = calcUpdate(*v[i],*mDiffOpVF,*mScratchV);
      mScratchV->scale(step);
      v[i]->pointwiseSubtract(*mScratchV);
    }else{
      updateVelocity(*v[i],*mDiffOpVF);
    }

    // update the deformation from source to this timepoint
    HField3DUtils::composeHVInv(*hField, *v[i], *mScratchV, mImSpacing,
				HField3DUtils::BACKGROUND_STRATEGY_PARTIAL_ID);
    *hField = *mScratchV;

    // compute vector energy, <Lv,v>
    if(mDebug){
      // save the current field
      mOp->CopyIn(*v[i]);
      mOp->ApplyOperator();
      mOp->CopyOut(*mScratchV);
      mVectorStepEnergy[i] = (1.0/mNTimeSteps)*HField3DUtils::l2DotProd(*mScratchV, *v[i], mImSpacing);
      mVectorEnergy += mVectorStepEnergy[i];
    }

  }

  if(mDebug){
    // create the deformed image
    HField3DUtils::apply(*mI0, *hField, *mScratchI);
    mScratchI->pointwiseSubtract(*mIT);
    mImageEnergy = (1.0/(mSigma*mSigma))*ImageUtils::l2NormSqr(*mScratchI);
    mTotalEnergy = mVectorEnergy+mImageEnergy;
  }

}

void updateJacDet(RealImage &jacDet, 
		  const VectorField &v, 
		  const Vector3D<Real> &spacing, 
		  RealImage &scratchI, 
		  VectorField &scratchV)
{
  scratchV = v;
  scratchV.scale(-1.0);
  // deform last timestep's jacobian
  HField3DUtils::applyU(jacDet, scratchV, scratchI, spacing);
  jacDet = scratchI;
  // compute x+v in world space
  HField3DUtils::setToIdentity(scratchV);
  scratchV.scale(spacing);
  scratchV.pointwiseSubtract(v);
  // compute jacobian determinant of x-v
  HField3DUtils::jacobian(scratchV,scratchI,spacing);
  // scale deformed jacobian by this update
  jacDet.pointwiseMultiplyBy(scratchI);
}

// inputs are alpha0 and I0, outputs are J0t, hField and Dphit, and vt
// and alphat are scratch, returns vector energy
Real
LDMMIterator::
ShootingIterateShoot(const RealImage *alpha0, const RealImage *I0, 
		     VectorField *vt,  RealImage *alphat, 
		     VectorField *hField, RealImage *Dphit, RealImage *J0t)
{
  Real vecEnergy = 0.f;

  // compute initial velocity from alpha0s
  Array3DUtils::computeGradient(*mI0,*vt,mImSpacing,false);
  vt->pointwiseMultiplyBy(*alpha0);

  mOp->CopyIn(*vt);
  mOp->ApplyInverseOperator();
  mOp->CopyOut(*vt);

  // initial image
  *J0t = *I0;

  // set up initial values
  HField3DUtils::setToIdentity(*hField);
  *alphat = *alpha0;
  Dphit->fill(1);

  // compute vector energy
  for(unsigned int t=1;t<mNTimeSteps;t++){

    // compute vec energy
    mOp->CopyIn(*vt);
    mOp->ApplyOperator();
    mOp->CopyOut(*mScratchV);
    vecEnergy += HField3DUtils::l2DotProd(*mScratchV, *vt, mImSpacing);

    // update hField
    HField3DUtils::composeHVInv(*hField, *vt, *mScratchV, mImSpacing,
				HField3DUtils::BACKGROUND_STRATEGY_PARTIAL_ID);
    *hField = *mScratchV;

    // compute alpha_t
    updateJacDet(*Dphit, *vt, mImSpacing, *mScratchI, *mScratchV);
    
    // deform alpha0
    HField3DUtils::apply(*alpha0, *hField, *alphat);
    
    // scale by jac. det.
    alphat->pointwiseMultiplyBy(*Dphit);

    // compute deformed image
    HField3DUtils::apply(*I0, *hField, *J0t);

    // compute a_t
    Array3DUtils::computeGradient(*J0t,*vt,mImSpacing,false);
    vt->pointwiseMultiplyBy(*alphat);

    // compute v_t
    mOp->CopyIn(*vt);
    mOp->ApplyInverseOperator();
    mOp->CopyOut(*vt);
  }

  // update hField
  HField3DUtils::composeHVInv(*hField, *vt, *mScratchV, mImSpacing,
			      HField3DUtils::BACKGROUND_STRATEGY_PARTIAL_ID);
  *hField = *mScratchV;

  // update jac det
  updateJacDet(*Dphit, *vt, mImSpacing, *mScratchI, *mScratchV);

  // compute vec energy
  mOp->CopyIn(*vt);
  mOp->ApplyOperator();
  mOp->CopyOut(*mScratchV);
  vecEnergy += HField3DUtils::l2DotProd(*mScratchV, *vt, mImSpacing);

  return vecEnergy;
}

void 
LDMMIterator::
ShootingIterate(RealImage *alpha0, RealImage *alpha0inv, 
		VectorField *hField, VectorField *hFieldInv,
		RealImage *I0AtT, RealImage *ITAt0)
{
  
  // shoot both alphas
  VectorField *vt = new VectorField(mImSize);
  RealImage *alphat = new RealImage(mImSize,mImOrigin,mImSpacing);

  Real vecEnergyFor, vecEnergyRev;
  Real imEnergyFor, imEnergyRev;
  
  vecEnergyFor = ShootingIterateShoot(alpha0, mI0, 
				      vt,  alphat,
				      hField, mJacDet[0], I0AtT);

  vecEnergyRev = ShootingIterateShoot(alpha0inv, mIT, 
				      vt,  alphat,
				      hFieldInv, mJacDet[1], ITAt0);

  // update alpha0
  *mScratchI = *ITAt0;
  mScratchI->pointwiseSubtract(*mI0);
  mScratchI->scale(-1.0/(mSigma*mSigma));
  imEnergyFor = ImageUtils::l2NormSqr(*mScratchI);
  mScratchI->pointwiseMultiplyBy(*mJacDet[1]);
  *alphat = *alpha0;
  alphat->pointwiseSubtract(*mScratchI);
  alphat->scale(mStepSize);
  alpha0->pointwiseSubtract(*alphat);

  *mScratchI = *I0AtT;
  mScratchI->pointwiseSubtract(*mIT);
  mScratchI->scale(-1.0/(mSigma*mSigma));
  imEnergyRev = ImageUtils::l2NormSqr(*mScratchI);
  mScratchI->pointwiseMultiplyBy(*mJacDet[0]);
  *alphat = *alpha0inv;
  alphat->pointwiseSubtract(*mScratchI);
  alphat->scale(mStepSize);
  alpha0inv->pointwiseSubtract(*alphat);

  std::cout << "Forward energy: " << vecEnergyFor + imEnergyFor << " = " 
	    << imEnergyFor << " (image) + " 
	    << vecEnergyFor << " (vec)" << std::endl;
  std::cout << "Reverse energy: " << vecEnergyRev + imEnergyRev << " = " 
	    << imEnergyRev << " (image) + " 
	    << vecEnergyRev << " (vec)" << std::endl;

  delete vt;
  delete alphat;
  
}

void 
LDMMIterator::
ComputeForwardDef(const VectorField* const* v, VectorField *hField)
{
  HField3DUtils::setToIdentity(*hField);
  for(int i = mNTimeSteps-1; i >= 0; i--){
    HField3DUtils::composeHV(*hField, *v[i], *mScratchV, mImSpacing,
			     HField3DUtils::BACKGROUND_STRATEGY_PARTIAL_ID);
    *hField = *mScratchV;
  }
}

void 
LDMMIterator::
ComputeReverseDef(const VectorField* const* v, VectorField *hField)
{
  HField3DUtils::setToIdentity(*hField);
  for(unsigned int i = 0; i < mNTimeSteps; i++){
    HField3DUtils::composeHVInv(*hField, *v[i], *mScratchV, mImSpacing,
		 HField3DUtils::BACKGROUND_STRATEGY_PARTIAL_ID);
    *hField = *mScratchV;
  }
}

void
LDMMIterator::
ComputeJacDet(const VectorField* const* v, RealImage *jacDet)
{
  jacDet->fill(1.0);
  for(int t = mNTimeSteps-1; t >= 0; t--){
    // compute determinant of jacobian of current deformation:
    // |D(h_{t-1})| = (|D(h_t)|(x+v_t))*|D(x+v_t)|
    
    // get identity in world coords
    HField3DUtils::setToIdentity(*mScratchV, mImSpacing);
    // and add velocity in world coords
    mScratchV->pointwiseAdd(*v[t]);
    // mScratchI = |D(x+v(x))|
    HField3DUtils::jacobian(*mScratchV,*mScratchI,mImSpacing);
    // deform current det. of jac.
    HField3DUtils::applyU(*jacDet, *v[t], *mJacDet[0], mImSpacing, 1.0f);
    *jacDet = *mJacDet[0];
    // scale by new deformation jacobian
    jacDet->pointwiseMultiplyBy(*mScratchI);
  }
}

void 
LDMM::
ComputeWeightedMean(const std::vector<RealImage*> &images, 
		    RealImage *mean, 
		    const std::vector<Real> &weights)
{
  // Initialize average as linear image average
  if(mean->getSize() != images[0]->getSize()){
    std::cerr << "Error, mean image must be same size as images to be averaged" << std::endl;
    return;
  }
  if(images.size() != weights.size()){
    std::cerr << "Error, number of wieghts must be the same as number of images" << std::endl;
    return;
  }
  mean->fill(0.0);
  for(unsigned int i = 0; i < weights.size(); i++)
    {
      Real *averageData = mean->getDataPointer();
      Real *imageData = images[i]->getDataPointer();
      unsigned int size = images[i]->getNumElements();
      for(unsigned int j = 0; j < size; j++)
	averageData[j] += imageData[j] * weights[i];
    }


} 

void 
LDMM::
LDMMMultiscaleMultithreadedAtlas(std::vector<const RealImage *> images,
				 const LDMMOldParam & params,
				 unsigned int nThreads,
				 RealImage *finalMeanImage,
				 std::vector<RealImage*> *finalMorphImages,
				 std::vector<VectorField*> *finalDefFields,
				 std::vector<std::vector<VectorField*> > *finalVecFields)
{
  std::vector<Real> equalWeights;
  Real val = 1.0/static_cast<Real>(images.size());
  equalWeights.insert(equalWeights.end(),images.size(),val);
  LDMMMultiscaleMultithreadedAtlas(images, equalWeights, params, nThreads, 
				   finalMeanImage, finalMorphImages, finalDefFields, finalVecFields);
}

void 
LDMM::
LDMMMultiscaleMultithreadedAtlas(std::vector<const RealImage *> images,
				 std::vector<Real> &weights,
				 const LDMMOldParam & params,
				 unsigned int nThreads,
				 RealImage *finalMeanImage,
				 std::vector<RealImage*> *finalMorphImages,
				 std::vector<VectorField*> *finalDefFields,
				 std::vector<std::vector<VectorField*> > *finalVecFields)
{

  unsigned int numLevels = params.GetNumberOfScaleLevels();
  unsigned int nTimeSteps = params.NTimeSteps();
  unsigned int numImages = images.size();
  
  Vector3D<unsigned int> size = images[0]->getSize();
  Vector3D<Real> origin = images[0]->getOrigin();
  Vector3D<Real> spacing = images[0]->getSpacing();

  MultiscaleManager scaleManager(size, spacing, origin, params);

  // initial images at current scale
  std::vector<RealImage *> I0(numImages,NULL); 
  for(unsigned int i=0;i<numImages;i++){
    I0[i] = scaleManager.GenerateBaseLevelImage(images[i]);
  }
  
  // mean image  
  RealImage *IHat = scaleManager.GenerateBaseLevelImage();
  
  // Vector fields
  std::vector<VectorField**> vFields(numImages);
  for(unsigned int imIdx=0;imIdx<numImages;imIdx++){
    vFields[imIdx] = new VectorField*[nTimeSteps];
    for(unsigned int tIdx=0; tIdx<nTimeSteps;tIdx++){
      vFields[imIdx][tIdx] = scaleManager.GenerateBaseLevelVectorField();
    }
  }
  
  if(finalDefFields && finalDefFields->size() > 0){
    std::cerr << "Error: non-empty initial deformation field vector?" << std::endl;
    return;
  }
  
  for(unsigned int scaleLevel = 0; scaleLevel < numLevels; scaleLevel++)
    {
      
      std::cout << "==== Scale level " << scaleLevel 
 		<< " ====" 
 		<< std::endl;

      Vector3D<unsigned int> curSize = scaleManager.CurScaleSize();
      Vector3D<Real> curSpacing = scaleManager.CurScaleSpacing();

      const LDMMScaleLevelParam &scaleLevelParam = params.GetScaleLevel(scaleLevel);

      // allocate IHat, hFields, deformed images
      std::vector<VectorField*> hFields(numImages);
      std::vector<RealImage *> finalImages(numImages);
      for(unsigned int i=0;i<numImages;i++){
 	finalImages[i] = new RealImage(curSize, origin, curSpacing);
 	*finalImages[i] = *I0[i];
 	hFields[i] = new VectorField(curSize);
      }
    
      // compute initial mean (just intensity average)
      ComputeWeightedMean(finalImages, IHat, weights);
      // TEST
      Real testVal = Array3DUtils::sumOfSquaredElements(*IHat);
      std::cout << "Mean image L2 norm: " << testVal << std::endl;
      ApplicationUtils::SaveImageITK("InitialAverageImage.mha", *IHat);
      // END TEST
    
      if(nThreads == 0){
	nThreads = numImages;
	std::cout << "Using " << nThreads << " threads for " << numImages << " images" << std::endl;
      }

      if(nThreads > numImages){
	std::cerr << "The number of threads must be <= number of images, changing from " 
		  << nThreads << " threads to " << numImages << " threads." 
		  << std::endl;
	nThreads = numImages;
      }

      // initialize the threads
      std::vector<pthread_t> threads(nThreads);

      // initialize the iterators
      std::vector<ThreadInfo> threadInfoVec(nThreads);

      for(unsigned int threadIdx=0; threadIdx < nThreads; threadIdx++){
	LDMMIterator *it = 
	  new LDMMIterator(curSize,
			   origin,
			   curSpacing,
			   nTimeSteps,
			   scaleLevelParam.LDMMIterator(),
			   true);
	threadInfoVec[threadIdx].threadIndex = threadIdx;
	threadInfoVec[threadIdx].iterator = it;
      }
      for(unsigned int iter=0;iter<(unsigned int)scaleLevelParam.NIterations();iter++){

	unsigned int imIdx = 0;
	Real totalEnergy = 0.0;
	Real imageEnergy = 0.0;
	Real vecEnergy = 0.0;

	// continue to create groups of threads until all images are processed
 	while(imIdx<numImages){
	  // create threads that will update images
	  unsigned int nUsedThreads = 0;
 	  for (unsigned int threadIdx = 0; threadIdx < nThreads; threadIdx++)
 	    {
	      threadInfoVec[threadIdx].imageIndex  = imIdx;
	      threadInfoVec[threadIdx].vFields = vFields[imIdx];
	      threadInfoVec[threadIdx].hField = hFields[imIdx];
	      threadInfoVec[threadIdx].I0 = I0[imIdx];
	      threadInfoVec[threadIdx].finalImage = finalImages[imIdx];
	      threadInfoVec[threadIdx].iterator->SetImages(IHat,I0[imIdx]);
	      
 	      int rv = pthread_create(&threads[threadIdx], NULL,
				      &LDMMThreadedUpdateVelocities, 
 				      &threadInfoVec[threadIdx]);
 	      if (rv != 0)
 		{
 		  throw std::runtime_error("Error creating thread.");
 		}
	      imIdx++;
	      nUsedThreads++;
	      if(imIdx >= numImages) break;
 	    }
	  
	  // join threads
	  for (unsigned int threadIdx = 0; threadIdx < nUsedThreads; threadIdx++)
	    {
	      int rv = pthread_join(threads[threadIdx], NULL);
	      if (rv != 0)
		{
		  throw std::runtime_error("Error joining thread.");
		}
	      // get the energy from the iterator
	      totalEnergy += threadInfoVec[threadIdx].iterator->
		GetTotalEnergy();
	      imageEnergy += threadInfoVec[threadIdx].iterator->
		GetImageEnergy();
	      vecEnergy += threadInfoVec[threadIdx].iterator->
		GetVectorEnergy();
	    }
 	}  // end iterate over each image
	
	// update average
	ComputeWeightedMean(finalImages, IHat, weights);
	// TEST
	testVal = Array3DUtils::sumOfSquaredElements(*IHat);
	std::cout << "Mean image L2 norm: " << testVal << std::endl;
	// END TEST
	
	// print out energy
	std::cout << "Scale " << scaleLevel << " Iter " << iter << " energy = " << totalEnergy 
		  << " = " << imageEnergy  << " (image) + " 
		  << vecEnergy  << " (vec)" << std::endl;

	// DEBUGGING
	if(iter > 0 &&
	   scaleLevelParam.OutputEveryNIterations() != 0 && 
	   iter%scaleLevelParam.OutputEveryNIterations() == 0)
	  {
	    static char curIterFName[256];
	    static char lastIterFName[256];
	    const char curIterFormat[] = "Image%03dTimestep%02dVelocity_Current.mha";
	    const char lastIterFormat[] = "Image%03dTimestep%02dVelocity_Last.mha";
	    std::cout << "Writing intermediate vector fields (iteration " << iter << ")..." << std::endl;
	    std::cout << "   Moving current fields to old fields..." << std::endl;
	    for(unsigned int imIdx=0;imIdx<numImages;imIdx++){
	      for(unsigned int t=0;t<nTimeSteps;t++){
		sprintf(curIterFName, curIterFormat, imIdx, t);
		sprintf(lastIterFName, lastIterFormat, imIdx, t);
		if(ApplicationUtils::FileExists(curIterFName)){
		  if(rename(curIterFName, lastIterFName) != 0){
		    std::cerr << "Unable to rename " << curIterFName << " to " 
			      << lastIterFName << " at iteration " << iter << std::endl;
		  }
		}
	      }
	    }
	    std::cout << "   writing new fields..." << std::endl;
	    for(unsigned int imIdx=0;imIdx<numImages;imIdx++){
	      for(unsigned int t=0;t<nTimeSteps;t++){
		sprintf(curIterFName, curIterFormat, imIdx, t);
		ApplicationUtils::SaveHFieldITK(curIterFName, *vFields[imIdx][t]);
		// HACK FOR NIKHIL
		if(iter == 50){
		  sprintf(curIterFName, "Image%03dTimestep%02dVelocity_Iter50.mha", imIdx, t);
		  ApplicationUtils::SaveHFieldITK(curIterFName, *vFields[imIdx][t]);
		}
	      }
	    }
	    std::cout << "Done writing intermediate images (iteration " << iter << ")" << std::endl;
	  }

      }// end iteration loop
      
      // if this is the final scale level, copy out results
      if(scaleLevel == numLevels-1){
	std::cout << "Performing final cleanup..." << std::endl;
	if(finalMeanImage){
	  std::cout << "copying mean image..." << std::endl;
	  *finalMeanImage = *IHat;
	}
	if(finalMorphImages){
	  std::cout << "returning deformed images..." << std::endl;
	  for(unsigned int imIdx=0;imIdx<numImages;imIdx++){
	    finalMorphImages->push_back(finalImages[imIdx]);
	  }
	}else{
	  std::cout << "deleting deformed images..." << std::endl;
	  for(unsigned int imIdx=0;imIdx<numImages;imIdx++){
	    delete finalImages[imIdx];
	  }
	}
	if(finalDefFields){
	  std::cout << "returning deformation fields..." << std::endl;
	  for(unsigned int imIdx=0;imIdx<numImages;imIdx++){
	    finalDefFields->push_back(hFields[imIdx]);
	  }
	}else{
	  std::cout << "deleting deformation fields..." << std::endl;
	  for(unsigned int imIdx=0;imIdx<numImages;imIdx++){
	    delete hFields[imIdx];
	  }
	}
	if(finalVecFields){
	  std::cout << "returning vector fields..." << std::endl;
	  for(unsigned int imIdx=0;imIdx<numImages;imIdx++){
	    std::vector<VectorField*> imVelocities;
	    for(unsigned int t=0;t<nTimeSteps;t++){
	      if(!scaleManager.Detach(vFields[imIdx][t])){
		std::cerr << "Error detaching vector field for image " << imIdx 
			  << " timestep " << t << std::endl;
	      }
	      imVelocities.push_back(vFields[imIdx][t]);
	    }
	    finalVecFields->push_back(imVelocities);
	  }
	}
      }else{
	// not the final scale level, do all the up/down sampling
	std::cout << "Generating images at next scale level..." << std::endl;
	scaleManager.NextScaleLevel();

	std::cout << "Cleaning up scale level data structures" << std::endl;
	// clean up -- these aren't automatically deleted on the final scale level,
	// as we may be returning them
	for(unsigned int imIdx=0;imIdx<numImages;imIdx++){
	  delete finalImages[imIdx];
	  delete hFields[imIdx];
	}
      }

      // clean up (performed at the end of all scale levels)
      std::cout << "Cleaning up scale level iterators" << std::endl;
      for(unsigned int threadIdx=0; threadIdx < nThreads; threadIdx++){
	delete threadInfoVec[threadIdx].iterator;
      }

    } // end iterate over scale levels

  std::cout << "Atlas generation completed succesfully" << std::endl;

}

void*
LDMM::LDMMThreadedUpdateVelocities(void* arg)
{
  ThreadInfo *threadInfo = static_cast<ThreadInfo*>(arg);

  threadInfo->iterator->
    Iterate(threadInfo->vFields, 
	    threadInfo->hField);

  threadInfo->iterator->
    ComputeForwardDef(threadInfo->vFields, 
		      threadInfo->hField);

  HField3DUtils::apply(*threadInfo->I0, 
 		       *threadInfo->hField, 
		       *threadInfo->finalImage);
  return NULL;
}

void 
LDMM::LDMMMultiscaleAtlas(std::vector<const RealImage *> images,
			  const LDMMOldParam & params,
			  std::vector<RealImage*> &finalMorphImages,
			  std::vector<VectorField*> &finalDefFields)
{
  std::vector<Real> equalWeights;
  Real val = 1.0/static_cast<Real>(images.size());
  equalWeights.insert(equalWeights.end(),images.size(),val);
  LDMMMultiscaleAtlas(images, equalWeights, params, finalMorphImages, finalDefFields);
}

void 
LDMM::LDMMMultiscaleAtlas(std::vector<const RealImage *> images,
			  std::vector<Real> &weights,
			  const LDMMOldParam & params,
			  std::vector<RealImage*> &finalMorphImages,
			  std::vector<VectorField*> &finalDefFields)
{

  unsigned int numLevels = params.GetNumberOfScaleLevels();
  unsigned int nTimeSteps = params.NTimeSteps();
  unsigned int numImages = images.size();
  
  Vector3D<unsigned int> size = images[0]->getSize();
  Vector3D<Real> origin = images[0]->getOrigin();
  Vector3D<Real> spacing = images[0]->getSpacing();
  
  MultiscaleManager scaleManager(size, spacing, origin, params);

  // initial images at current scale
  std::vector<RealImage *> I0(numImages,NULL); 
  for(unsigned int i=0;i<numImages;i++){
    I0[i] = scaleManager.GenerateBaseLevelImage(images[i]);
  }

  // mean image  
  RealImage *IHat = scaleManager.GenerateBaseLevelImage();

  // deformation fields -- vFields[image][timestep]
  std::vector<VectorField**> vFields(numImages);
  for(unsigned int imIdx=0;imIdx<numImages;imIdx++){
    vFields[imIdx] = new VectorField*[nTimeSteps];
    for(unsigned int tIdx=0; tIdx<nTimeSteps;tIdx++){
      vFields[imIdx][tIdx] = scaleManager.GenerateBaseLevelVectorField();
    }
  }
  
  if(finalDefFields.size() > 0){
    std::cerr << "Error: non-empty initial deformation field vector?" << std::endl;
    return;
  }
  
  for(unsigned int scaleLevel = 0; scaleLevel < numLevels; scaleLevel++)
    {
      
      std::cout << "==== Scale level " << scaleLevel 
 		<< " ====" 
 		<< std::endl;

      const LDMMScaleLevelParam &scaleLevelParam = params.GetScaleLevel(scaleLevel);
      
      Vector3D<unsigned int> curSize = scaleManager.CurScaleSize();
      Vector3D<Real> curSpacing = scaleManager.CurScaleSpacing();

      // allocate IHat, hFields, deformed images
      IHat = new RealImage(curSize, origin, curSpacing);
      std::vector<VectorField*> hFields(numImages);
      std::vector<RealImage*> finalImages(numImages);
      for(unsigned int i=0;i<numImages;i++){
	finalImages[i] = new RealImage(curSize, origin, curSpacing);
	*finalImages[i] = *I0[i];
	hFields[i] = new VectorField(curSize);
      }
      
      // compute initial mean (just intensity average)
      ComputeWeightedMean(finalImages, IHat, weights);
      
      // initialize the iterator
      LDMMIterator iterator(curSize,
			    origin,
			    curSpacing,
			    nTimeSteps,
			    scaleLevelParam.LDMMIterator(),
			    true);
      
      for(unsigned int iter=0;iter<(unsigned int)scaleLevelParam.NIterations();iter++){
	Real totalEnergy = 0.0;
	Real imageEnergy = 0.0;
	Real vecEnergy = 0.0;
	for(unsigned int imIdx=0;imIdx<numImages;imIdx++){
	  iterator.SetImages(IHat,I0[imIdx]);
	  iterator.Iterate(vFields[imIdx], hFields[imIdx]);

	  iterator.
	    ComputeForwardDef(vFields[imIdx], 
			      hFields[imIdx]);

	  HField3DUtils::apply(*I0[imIdx], 
			       *hFields[imIdx], 
			       *finalImages[imIdx]);

	  // get the energy from the iterator
	  totalEnergy += iterator.GetTotalEnergy();
	  imageEnergy += iterator.GetImageEnergy();
	  vecEnergy += iterator.GetVectorEnergy();

	}  // end iterate over each image
	
	// update average
	ComputeWeightedMean(finalImages, IHat, weights);

	// print out energy
	std::cout << "Step size: " 
		  << iterator.GetStepSize()
		  << std::endl;
	LOGNODE(logINFO) 
	  << "Scale " << scaleLevel << " Iter " << iter << " energy = " << totalEnergy 
	  << " = " << imageEnergy  << " (image) + " 
	  << vecEnergy  << " (vec)" << std::endl;

      }// end iteration loop

      if(scaleLevel == numLevels-1){
 	for(unsigned int imIdx=0;imIdx<numImages;imIdx++){
 	  RealImage *tmpIm = new RealImage(*finalImages[imIdx]);
 	  finalMorphImages.push_back(tmpIm);
 	  VectorField *tmpVF = new VectorField(*hFields[imIdx]);
 	  finalDefFields.push_back(tmpVF);
 	}
      }else{
	// otherwise, do all the up/down sampling
	std::cout << "Generating images at next scale level..." << std::endl;
	scaleManager.NextScaleLevel();
      }
      // clean up
      for(unsigned int imIdx=0;imIdx<numImages;imIdx++){
	delete finalImages[imIdx];
	delete hFields[imIdx];
      }

    } // end iterate over scale levels
  
}

void 
LDMM::LDMMMultiscaleRegistration(const RealImage *image1,
				 const RealImage *image2,
				 const LDMMOldParam &params,
				 std::vector<RealImage*> &morphImages,
				 std::vector<VectorField*> &vFields)
{

  unsigned int numLevels = params.GetNumberOfScaleLevels();
  unsigned int nTimeSteps = params.NTimeSteps();

  Vector3D<unsigned int> size = image1->getSize();
  Vector3D<Real> origin = image1->getOrigin();
  Vector3D<Real> spacing = image1->getSpacing();

  MultiscaleManager scaleManager(size, spacing, origin, params);
  
  RealImage *I0 = scaleManager.GenerateBaseLevelImage(image1);
  RealImage *IT = scaleManager.GenerateBaseLevelImage(image2); // final image

  char fname[256];

  if(vFields.size() > 0){
    std::cerr << "Error: non-empty initial deformation field vector?" << std::endl;
    return;
  }

  // Vector fields
  std::vector<VectorField*> v;
  for(unsigned int tIdx=0; tIdx<nTimeSteps;tIdx++){
    v.push_back(scaleManager.GenerateBaseLevelVectorField());
  }

  for(unsigned int scaleLevel = 0; scaleLevel < numLevels; scaleLevel++)
    {
      std::cout << "Scale " << scaleLevel << "Image size = " << I0->getSize() << std::endl;
      
      Vector3D<unsigned int> curSize = scaleManager.CurScaleSize();
      
      // run the registration for this scale level
      LDMMRegistration(I0, IT, params.NTimeSteps(), 
		       params.GetScaleLevel(scaleLevel),
		       morphImages, v);

      std::cout << "Saving scale level final def" << std::endl;
      sprintf(fname, "FinalDefImg_ScaleLevel%02d.mha", scaleLevel);
      ApplicationUtils::SaveImageITK(fname, *morphImages[nTimeSteps]);

      if(scaleLevel != numLevels-1){
	// if this is not the final iteration,
	// generate images/fields at next scale level
	scaleManager.NextScaleLevel();
	// and delete old images
	for(unsigned int tIdx=0;tIdx <= nTimeSteps;tIdx++){
	  delete morphImages[tIdx];
	}
	morphImages.clear();
      }else{
	// if this is the final iteration, copy final fields
	for(unsigned int tIdx=0;tIdx < nTimeSteps;tIdx++){
	  vFields.push_back(new VectorField(*v[tIdx]));
	}
      }

    } // end iterate over scales

}

void 
LDMM::LDMMRegistration(const RealImage *image1,
		       const RealImage *image2,
		       const unsigned int nTimeSteps,
		       const LDMMScaleLevelParam &params,
		       std::vector<RealImage*> &morphImages,
		       std::vector<VectorField*> &defFields)
{

  Vector3D<unsigned int> curSize = image1->getSize();
  unsigned int size = 0;
  Vector3D<Real> origin = image1->getOrigin();

  // update size and spacing for downsampling
  size = curSize.productOfElements();
  
  Vector3D<Real> spacing = image1->getSpacing();

  // create vector fields, or use ones passed in to defFields
  VectorField **v = new VectorField*[nTimeSteps];
  if(defFields.size() == 0){
    for(unsigned int i = 0; i < nTimeSteps; i++){
      v[i] = new VectorField(curSize);
      v[i]->fill(Vector3D<Real>(0,0,0));
    }
  }else if(defFields.size() == nTimeSteps){
    for(unsigned int i = 0; i < nTimeSteps; i++){
      v[i] = defFields[i];
    }
  }else{
    std::cerr << "Error, input velocity field vector not empty and not equal to number of timesteps! " 
	      << "nTimeSteps = " << nTimeSteps
	      << ", defFeilds.size() = " << defFields.size()
	      << std::endl;
    return;
  }
  VectorField *hField = new VectorField(curSize);
  RealImage *im = new RealImage(curSize, origin, spacing);

  LDMMIterator iterator(curSize,
			origin,
			spacing,
			nTimeSteps,
			params.LDMMIterator(),
			true);

  iterator.SetImages(image1,image2);
  
  // ### TEST ONLY
  RealImage **alpha = new RealImage*[nTimeSteps];
  VectorField **ufields = new VectorField*[nTimeSteps];
  for(unsigned int i = 0; i < nTimeSteps; i++){
    alpha[i] = new RealImage(curSize);
    ufields[i] = new VectorField(curSize);
  }
  iterator.SaveAlphaImages(alpha);
  iterator.SaveUFields(ufields);
  // ### END TEST

  for(unsigned int iter = 0; iter < (unsigned int)params.NIterations(); iter++)
    {

      std::cout << "Iteration " << iter << std::endl;

      iterator.Iterate(v, hField);
      
       std::cout << "Iter " << iter << " energy = " << iterator.GetTotalEnergy() 
 		<< " = " << iterator.GetImageEnergy()  << " (image) + " 
 		<< iterator.GetVectorEnergy()  << " (vec, ";
       for(unsigned int i=0;i<nTimeSteps;i++){
 	std::cout << iterator.GetVectorStepEnergy()[i];
 	if(i<nTimeSteps-1)
 	  std::cout << " + ";
       }
       std::cout << ")"
 		<< std::endl;

       if(params.OutputEveryNIterations() > 0 &&
	  (iter+1) % params.OutputEveryNIterations() == 0)
	 {
	   for(unsigned int t = 0; t < nTimeSteps; t++){
	     char fname[256];
	     sprintf(fname, "IntermediateVelocity%02d_Iter%04d.mha", t, iter);
	     ApplicationUtils::SaveHFieldITK(fname, *v[t]);
	   }
	 }
    }
  
  // create morph images to return
  VectorField scratchV(curSize);
  *im = *image1;
  morphImages.push_back(im);
  HField3DUtils::setToIdentity(*hField);
  for(unsigned int i = 0; i < nTimeSteps; i++){
    if(i<nTimeSteps){
      if(defFields.size() < i+1){
	defFields.push_back(v[i]);
      }else{
	*defFields[i] = *v[i];
      }
    }
    im = new RealImage(curSize, origin, spacing);
    // update the deformation from source to this timepoint
    HField3DUtils::composeHVInv(*hField, *v[i], scratchV, spacing,
				HField3DUtils::BACKGROUND_STRATEGY_PARTIAL_ID);
    *hField = scratchV;
    HField3DUtils::apply(*image1, *hField, *im);
    morphImages.push_back(im);
  }

  // ### TEST ONLY
  char buff[1024];
  for(unsigned int i = 0; i < nTimeSteps; i++){
    sprintf(buff, "LDMMAlpha%02d.mha", i);
    ApplicationUtils::SaveImageITK(buff, *alpha[i]);
    sprintf(buff, "LDMMUField%02d.mha", i);
    ApplicationUtils::SaveHFieldITK(buff, *ufields[i]);
  }
  // ### END TEST

}

/**  
 * Update Phi0T and JacDetPhi0T
 */  
void
UpdatePhi0T(VectorField &phi, 
	    RealImage &jacDet, 
	    const VectorField &v,
	    SpacingType spacing)
{
  SizeType size = jacDet.getSize();
  unsigned int numElements = phi.getNumElements();

  // build scalar images h1, h2, and h3
  // from the transformation field
  RealImage h1(size);
  RealImage h2(size);
  RealImage h3(size);
  
  unsigned int i; // stupid vc++
  for (i = 0; i < numElements; ++i)
    {
      h1(i) = v(i).x;
      h2(i) = v(i).y;
      h3(i) = v(i).z;
    }

  // compute the gradients of h1, h2, and h3 (form rows of jacobian)
  VectorField grad_h1(size);
  VectorField grad_h2(size);
  VectorField grad_h3(size);   
  Array3DUtils::computeGradient(h1, grad_h1, spacing);
  Array3DUtils::computeGradient(h2, grad_h2, spacing);
  Array3DUtils::computeGradient(h3, grad_h3, spacing);

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
      h1(i) = t1 - t2 + t3;
    }

  // deform h1 (the determinant) and store in h2
  HField3DUtils::apply(h1, phi, h2, 1.0f);
  
  // update the deformed jac det (stored in h1)
  jacDet.pointwiseMultiplyBy(h2);

  // update phi = phi + v(phi)
  HField3DUtils::composeVH(v, phi, grad_h1, spacing,
			   HField3DUtils::BACKGROUND_STRATEGY_PARTIAL_ZERO);
  phi = grad_h1;
}

/**
 * Just a test of concept, optimize only alpha0 and shoot forward from
 * this image. 
 */ 
void 
LDMM::LDMMShootingRegistration(const RealImage *I0,
			       const RealImage *IT,
			       const unsigned int nTimeSteps,
			       const LDMMScaleLevelParam &params,
			       std::vector<RealImage*> &morphImages,
			       std::vector<VectorField*> &defFields)
{
  Vector3D<unsigned int> size = I0->getSize();
  Vector3D<Real> origin = I0->getOrigin();
  Vector3D<Real> spacing = I0->getSpacing();
  
  // scratch memory
  VectorField hField(size);
  RealImage jacDet(size,origin,spacing);
  VectorField scratchV(size);
  RealImage scratchI(size, origin, spacing);
  RealImage scratchI2(size, origin, spacing);
  
  // create vector fields, or use ones passed in to defFields
  VectorField **v = new VectorField*[nTimeSteps];
  if(defFields.size() == 0){
    for(unsigned int i = 0; i < nTimeSteps; i++){
      v[i] = new VectorField(size);
      v[i]->fill(Vector3D<Real>(0,0,0));
    }
  }else if(defFields.size() == nTimeSteps){
    for(unsigned int i = 0; i < nTimeSteps; i++){
      v[i] = defFields[i];
    }
    defFields.clear();
  }else{
    std::cerr << "Error, input velocity field vector not empty and not equal to number of timesteps! " 
	      << "nTimeSteps = " << nTimeSteps
	      << ", defFeilds.size() = " << defFields.size()
	      << std::endl;
    return;
  }

  LDMMIterator iterator(size,
			origin,
			spacing,
			nTimeSteps,
			params.LDMMIterator(),
			true);

  iterator.SetImages(I0,IT);
  
  // we'll need alpha for shooting
  RealImage **alpha = new RealImage*[nTimeSteps];
  for(unsigned int i = 0; i < nTimeSteps; i++){
    alpha[i] = new RealImage(size);
  }
  iterator.SaveAlphaImages(alpha);

  // create arrays to hold shooting results
  std::vector<RealImage*> shootingImages;
  std::vector<VectorField*> shootingVelocities;

  Real sigma = params.LDMMIterator().Sigma();
  Real step = params.LDMMIterator().StepSize();
  Vector3D<Real> imSize = I0->getSize();
  RealImage *alpha0 = new RealImage(imSize);
  alpha0->fill(0.0);

  for(unsigned int iter = 0; iter < (unsigned int)params.NIterations(); iter++)
    {
      
      std::cout << "Iteration " << iter << std::endl;

      std::cout << "Running shooting" << std::endl;
      
      // delete old images/velocities
      for(unsigned int i=0;i<shootingImages.size();i++){
	delete shootingImages[i];
      }
      shootingImages.clear();
      for(unsigned int i=0;i<shootingVelocities.size();i++){
	delete shootingVelocities[i];
      }
      shootingVelocities.clear();
      
      GeodesicShooting(I0,
		       alpha0,
		       nTimeSteps,
		       params.LDMMIterator().DiffOper(),
		       shootingImages,
		       shootingVelocities);

      // copy the shooting velocities into LDMM velocity fields
      for(unsigned int i=0;i<nTimeSteps;i++){
	scratchV = *shootingVelocities[i];
	//scratchV.scale(-1.0);
	*v[i] = scratchV;
      }
      
      // update alpha0

      bool updateFromLDMM = false;
      
      if(updateFromLDMM){
	iterator.ComputeForwardDef(v, &hField);
	iterator.ComputeJacDet(v, &jacDet);
      }else{
	HField3DUtils::setToIdentity(hField);
	jacDet.fill(1.0);
	for(unsigned int t=0;t<nTimeSteps;t++){
	  UpdatePhi0T(hField,jacDet,*v[t],spacing);
	}
      }
	
      // TEST
      if(iter == 50){
	std::string fname;
	//
	// compute using backward method
	//
	iterator.ComputeForwardDef(v, &hField);
	iterator.ComputeJacDet(v, &jacDet);
	// save results
	fname = StringUtils::strPrintf("JacDetBackwardMethodIter%03d.mha",iter);
	ApplicationUtils::SaveImageITK(fname.c_str(), jacDet);
	fname = StringUtils::strPrintf("HFieldBackwardMethodIter%03d.mha",iter);
	ApplicationUtils::SaveHFieldITK(fname.c_str(), hField);
	HField3DUtils::jacobian(hField, scratchI, spacing);
	fname = StringUtils::strPrintf("JacDetSingleStepBackwardIter%03d.mha",iter);
	ApplicationUtils::SaveImageITK(fname.c_str(), scratchI);
	//
	// compute using forward method
	//
	HField3DUtils::setToIdentity(hField);
	jacDet.fill(1.0);
	for(unsigned int t=0;t<nTimeSteps;t++){
	  UpdatePhi0T(hField,jacDet,*v[t],spacing);
	  fname = StringUtils::strPrintf("VFieldIter%03dStep%02d.mha",iter,t);
	  ApplicationUtils::SaveHFieldITK(fname.c_str(), *v[t]);
	  fname = StringUtils::strPrintf("JacDetForwardMethodIter%03dStep%02d.mha",iter,t);
	  ApplicationUtils::SaveImageITK(fname.c_str(), jacDet);
	}
	fname = StringUtils::strPrintf("JacDetForwardMethodIter%03d.mha",iter);
	ApplicationUtils::SaveImageITK(fname.c_str(), jacDet);
	fname = StringUtils::strPrintf("HFieldForwardMethodIter%03d.mha",iter);
	ApplicationUtils::SaveHFieldITK(fname.c_str(), hField);
	HField3DUtils::jacobian(hField, scratchI, spacing);
	fname = StringUtils::strPrintf("JacDetSingleStepForwardIter%03d.mha",iter);
	ApplicationUtils::SaveImageITK(fname.c_str(), scratchI);
      }
      // END TEST				     
      
      // create new alpha
      HField3DUtils::apply(*IT, hField, scratchI);
      scratchI.scale(-1.0);
      scratchI.pointwiseAdd(*I0);
      jacDet.scale(1.0/(sigma*sigma));
      scratchI.pointwiseMultiplyBy(jacDet);

      scratchI.scale(-1.0);
      scratchI.pointwiseAdd(*alpha0);
      scratchI.scale(step);
      alpha0->pointwiseSubtract(scratchI);

      // Do this just to compute energy

      std::cout << "Running LDMM step" << std::endl;
      // run LDMM iteration step to get updated alpha value
      iterator.Iterate(v, &hField);
      
      // print energy
      std::cout << "Iter " << iter << " energy = " << iterator.GetTotalEnergy() 
 		<< " = " << iterator.GetImageEnergy()  << " (image) + " 
 		<< iterator.GetVectorEnergy()  << " (vec, ";
      for(unsigned int t=0;t<nTimeSteps;t++){
 	std::cout << iterator.GetVectorStepEnergy()[t];
 	if(t<nTimeSteps-1)
 	  std::cout << " + ";
      }
      std::cout << ")"
 		<< std::endl;
      
    }
  
  // copy out final fields
  for(unsigned int i=0;i<nTimeSteps;i++){
    morphImages.push_back(shootingImages[i]);
    defFields.push_back(shootingVelocities[i]);
  }
  morphImages.push_back(shootingImages[nTimeSteps]);
  
  
}

/**
 * Just a test of concept, optimize only alpha0 and shoot forward from
 * this image.  Doesn't work so far.
 */ 
void 
LDMM::LDMMShootingRegistration2(const RealImage *I0,
				const RealImage *IT,
				const unsigned int nTimeSteps,
				const LDMMScaleLevelParam &params,
				std::vector<RealImage*> &morphImages,
				std::vector<VectorField*> &defFields)
{
  Vector3D<unsigned int> size = I0->getSize();
  Vector3D<Real> origin = I0->getOrigin();
  Vector3D<Real> spacing = I0->getSpacing();
  
  // scratch memory
  VectorField hField(size);
  VectorField hFieldInv(size);
  RealImage alpha0(size, origin, spacing);
  RealImage alpha0inv(size, origin, spacing);
  RealImage ITAt0(size, origin, spacing);
  RealImage I0AtT(size, origin, spacing);

  alpha0.fill(0.0);
  alpha0inv.fill(0.0);

  LDMMIterator iterator(size,
			origin,
			spacing,
			nTimeSteps,
			params.LDMMIterator(),
			true);

  iterator.SetImages(I0,IT);
  
  char fname[1024];
  for(unsigned int iter = 0; iter < (unsigned int)params.NIterations(); iter++)
    {
      
      std::cout << "Iteration " << iter << std::endl;
      iterator.ShootingIterate(&alpha0, &alpha0inv, 
			       &hField, &hFieldInv,
			       &I0AtT, &ITAt0);
      sprintf(fname, "I0AtTIter%02d.mha", iter);
      ApplicationUtils::SaveImageITK(fname, I0AtT);
      sprintf(fname, "ITAt0Iter%02d.mha", iter);
      ApplicationUtils::SaveImageITK(fname, ITAt0);
    }

}

void
LDMM::
computeAlphaFields(std::vector<VectorField *> &v,
		   const RealImage *I0,
		   const RealImage *IT,
		   const Real &sigma,
		   std::vector<RealImage *> &momentum)
{
  
  Vector3D<unsigned int> size = I0->getSize();
  Vector3D<Real> origin(0.0, 0.0, 0.0);
  Vector3D<Real> spacing = I0->getSpacing();

  VectorField hField(size);
  VectorField scratchV(size);
  RealImage scratchI(size, origin, spacing);
  RealImage J0t(size,origin,spacing);

  int nTimeSteps = v.size();

  std::vector<RealImage *>JTt(v.size());
  std::vector<RealImage *>jacDet(v.size());
  for(size_t i=0;i<JTt.size();i++){
    JTt[i] = new RealImage(size,origin,spacing);
    jacDet[i] = new RealImage(size,origin,spacing);
  }

  // Compute deformed images from dest to source
  HField3DUtils::setToIdentity(hField);
  for(int i = nTimeSteps-1; i >= 0; i--){

    // update the deformation (hfield) from dest to this timepoint
    HField3DUtils::composeHV(hField, *v[i], scratchV, spacing,
			     HField3DUtils::BACKGROUND_STRATEGY_PARTIAL_ID);
    
    hField = scratchV;

    // create the deformed image from dest to this timepoint
    HField3DUtils::apply(*IT, hField, *JTt[i]);

    // TEST
    char fname[1024];
    sprintf(fname, "RevDef%02d.mha", i);
    ApplicationUtils::SaveImageITK(fname, *JTt[i]);
    // END TEST

    // compute determinant of jacobian of current deformation:
    // |D(h_{t-1})| = (|D(h_t)|(x+v_t))*|D(x+v_t)|
    HField3DUtils::setToIdentity(scratchV);
    scratchV.scale(spacing);
    scratchV.pointwiseAdd(*v[i]);
    // mScratchI = |D(x+v(x))|
    HField3DUtils::jacobian(scratchV,scratchI,spacing);
    if(i == (int)nTimeSteps-1){
      *jacDet[i] = scratchI;
    }else{
      // deform current det. of jac.
      HField3DUtils::applyU(*jacDet[i+1], *v[i], *jacDet[i], spacing);
      // scale by new deformation jacobian
      jacDet[i]->pointwiseMultiplyBy(scratchI);
    }
  }

  // update each velocity field from source to dest
  HField3DUtils::setToIdentity(hField);
  for(unsigned int i = 0; i < (unsigned int)nTimeSteps; i++){

    // create the deformed image from source to this timepoint
    HField3DUtils::apply(*I0, hField, scratchI);

    // TEST
    char fname[1024];
    sprintf(fname, "ForDef%02d.mha", i);
    ApplicationUtils::SaveImageITK(fname, scratchI);
    // END TEST

    // subtract the reverse-deformed image
    scratchI.pointwiseSubtract(*JTt[i]);

    // multiply by the jacobian determinant
    scratchI.pointwiseMultiplyBy(*jacDet[i]);

    scratchI.scale(1.0/(sigma*sigma));

    if(momentum.size() <= i){
      RealImage *alpha = new RealImage(scratchI);
      momentum.push_back(alpha);
    }else{
      *momentum[i] = scratchI;
    }

    // update the deformation from source to this timepoint
    HField3DUtils::composeHVInv(hField, *v[i], scratchV, spacing,
				HField3DUtils::BACKGROUND_STRATEGY_PARTIAL_ID);
    hField = scratchV;

  }

  for(size_t i=0;i<JTt.size();i++){
    delete JTt[i];
    delete jacDet[i];
  }

}

Real
LDMM::
computeAlphaFromA(const VectorField *at,
		  const VectorField *gJ0t,
		  RealImage *alphat,
		  RealImage *angleDiff, 
		  RealImage *mask)
{

  Vector3D<unsigned int> size = at->getSize();

  RealImage atNorm(size);
  RealImage gJ0tNorm(size);
  HField3DUtils::pointwiseL2Norm(*gJ0t,gJ0tNorm);
  HField3DUtils::pointwiseL2Norm(*at,atNorm);

  Real momentum = 0.0;
  if(mask) mask->fill(0);
  unsigned int dataSize = at->getSize().productOfElements();
  for (unsigned int i = 0; i < dataSize; ++i)
    {
      Real gJ0tData = gJ0tNorm(i);
      Real val = atNorm(i);

      if(gJ0tData > 0.0001){
	val /= gJ0tData;
      }

      Vector3D<Real> atVec = (*at)(i);
      Vector3D<Real> gJ0tVec = (*gJ0t)(i);
      Real normProd = (atVec.length()*gJ0tVec.length());
      Real angle = 0.0;
      if(normProd != 0.0){
	angle = atVec.dot(gJ0tVec)/normProd;
 	if(std::fabs(angle) > 1.0){
 	  angle = angle/std::fabs(angle);
 	}
	angle = std::acos(angle);
	if(angle != angle){
	  std::cerr << "Error, angle is NaN: " << angle << std::endl;
	}

	if(angle > M_PI/2.0){
	  val *= -1;
	}

	if(angle > M_PI/2.0)
	  angle -= M_PI;
      }else{
	if(mask) (*mask)(i) = 1.0;
      }
      
      if(angleDiff) (*angleDiff)(i) = angle;

      if(alphat) (*alphat)(i) = val;

      momentum += val;
    }
  return momentum;
}

void 
LDMM::GeodesicShooting(const RealImage *I0,
		       const RealImage *alpha0,
		       const unsigned int nTimeSteps,
		       const DiffOperParam &diffOpParam,
		       std::vector<RealImage*> &finalMorphImages,
		       std::vector<VectorField*> &finalVecFields,
		       const CompoundParam *debugOptions)
{

  Vector3D<unsigned int> size = I0->getSize();
  Vector3D<Real> origin = I0->getOrigin();
  Vector3D<Real> spacing = I0->getSpacing();
  
  // deformation fields -- vFields[image][timestep]
  char fname[256];
  
  VectorField *hField = new VectorField(size);
  RealImage *alphat = new RealImage(size, origin, spacing);
  RealImage *Dphit = new RealImage(size, origin, spacing);
  VectorField *at = new VectorField(size);
  VectorField *scratchV = new VectorField(size);
  RealImage *scratchI = new RealImage(size, origin, spacing);

  // create DiffOper
  DiffOper op(size, spacing, diffOpParam);

  // compute initial velocity from alpha0
  Array3DUtils::computeGradient(*I0,*at,spacing,false);
  at->pointwiseMultiplyBy(*alpha0);

  RealImage *J0t = NULL;
  VectorField *vt = NULL;

  J0t = new RealImage(size,origin,spacing);
  vt = new VectorField(size);

  op.CopyIn(*at);
  op.ApplyInverseOperator();
  op.CopyOut(*vt);

  // initial image
  *J0t = *I0;

  // add initial image and vector field to result vectors
  finalMorphImages.push_back(J0t);
  finalVecFields.push_back(vt);

  // set up initial values
  HField3DUtils::setToIdentity(*hField);
  *alphat = *alpha0;
  Dphit->fill(1);
  for(unsigned int t=1;t<nTimeSteps;t++){

    // update hField
    HField3DUtils::composeHVInv(*hField, *finalVecFields[t-1], *scratchV, spacing,
				HField3DUtils::BACKGROUND_STRATEGY_PARTIAL_ID);
    *hField = *scratchV;

    // compute alpha_t

    // ### Jacobian-scaled deformation version ###
    {
      
      updateJacDet(*Dphit, *finalVecFields[t-1], spacing, *scratchI, *scratchV);
      
      // deform alpha0
      HField3DUtils::apply(*alpha0, *hField, *alphat);
      
      // scale by jac. det.
      alphat->pointwiseMultiplyBy(*Dphit);
    }

     // ### Forward-Splat Version ###
//       {

//         VectorField splatHField(size);
//         HField3DUtils::setToIdentity(splatHField);
//         for(int i=t-1;i>=0;i--){
//   	HField3DUtils::composeHV(splatHField, *finalVecFields[t-1], *scratchV, spacing,
//   				 HField3DUtils::BACKGROUND_STRATEGY_PARTIAL_ID);
//   	splatHField = *scratchV;
//         }
//         HField3DUtils::forwardApply(*alpha0, splatHField, *alphat, (Real)0.0, false);
//       }
    

    // TEST
    sprintf(fname, "alpha%02d.mha", t);
    ApplicationUtils::SaveImageITK(fname, *alphat);
    // END TEST

    // compute deformed image
    J0t = new RealImage(size, origin, spacing);
    HField3DUtils::apply(*I0, *hField, *J0t);

    // compute a_t
    Array3DUtils::computeGradient(*J0t,*at,spacing,false);
    at->pointwiseMultiplyBy(*alphat);

    // ### TEST ###
    Real momentum = Array3DUtils::mass(*alphat);

    std::cout << "timestep " << t << " momentum: " << momentum << std::endl;
    // ### END TEST ###

    // compute v_t
    vt = new VectorField(size);
    op.CopyIn(*at);
    op.ApplyInverseOperator();
    op.CopyOut(*vt);

    finalMorphImages.push_back(J0t);
    finalVecFields.push_back(vt);
  }

  if(debugOptions){
    
    // Have to compute the jacobian update for the last velocity still....
    updateJacDet(*Dphit, *finalVecFields.back(), spacing, *scratchI, *scratchV);
    
    std::string prefix = GetChildVal<std::string>(debugOptions, "OutputPrefix");
    std::string suffix = GetChildVal<std::string>(debugOptions, "OutputSuffix");
    if(GetChildVal<bool>(debugOptions, "WriteJacDet")){
      std::string jacDetName = prefix + "JacDet." + suffix;
      ApplicationUtils::SaveImageITK(jacDetName.c_str(), *Dphit);
    }

  }

  // create the final deformation
  // update hField
  HField3DUtils::composeHVInv(*hField, *finalVecFields[nTimeSteps-1], *scratchV, spacing,
			     HField3DUtils::BACKGROUND_STRATEGY_PARTIAL_ID);
  *hField = *scratchV;
  J0t = new RealImage(size, origin, spacing);
  HField3DUtils::apply(*I0, *hField, *J0t);
  finalMorphImages.push_back(J0t);


  // clean up
  delete hField;
  delete alphat;
  delete Dphit;
  delete at;
  delete scratchV;

}

void 
LDMM::GeodesicShooting(const RealImage *I0,
		       const VectorField *v0,
		       const unsigned int nTimeSteps,
		       const DiffOperParam &diffOpParam,
		       std::vector<RealImage*> &finalMorphImages,
		       std::vector<VectorField*> &finalVecFields,
		       const CompoundParam *debugOptions)
{
  
  Vector3D<unsigned int> size = I0->getSize();
  Vector3D<Real> origin = I0->getOrigin();
  Vector3D<Real> spacing = I0->getSpacing();
  
  // deformation fields -- vFields[image][timestep]
  char fname[256];
  
  // create DiffOper
  DiffOper op(size, spacing, diffOpParam);

  VectorField *a0 = new VectorField(size);
  RealImage *alpha0 = new RealImage(size, origin, spacing);
  VectorField *gradI0 = new VectorField(size);
  RealImage *gradI0Norm = new RealImage(size);
  RealImage *a0Norm = new RealImage(size, origin, spacing);

  Array3DUtils::computeGradient(*I0,*gradI0,spacing,false);

  HField3DUtils::pointwiseL2Norm(*gradI0,*gradI0Norm);

  op.CopyIn(*v0);
  op.ApplyOperator();
  op.CopyOut(*a0);

  HField3DUtils::pointwiseL2Norm(*a0,*a0Norm);

  VectorField *hField = new VectorField(size);
  RealImage *alphat = new RealImage(size, origin, spacing);
  RealImage *Dphit = new RealImage(size, origin, spacing);
  VectorField *at = new VectorField(size);
  VectorField *scratchV = new VectorField(size);
  RealImage *scratchI = new RealImage(size, origin, spacing);
  RealImage *scratchI2 = new RealImage(size, origin, spacing);
  
  // create alpha0
  Real computedMomentum = 
    computeAlphaFromA(a0,
		      gradI0,
		      alpha0,
		      scratchI, // will hold angle diff
		      alphat); // will hold mask

  // ### TEST ###

  std::cout << "Initial computed momentum: " << computedMomentum << std::endl;

  // write out initial alpha and gradI
  ApplicationUtils::SaveImageITK("alpha00.mha", *alpha0);
  ApplicationUtils::SaveImageITK("gradI0Norm.mha", *gradI0Norm);
  ApplicationUtils::SaveImageITK("alphaMask.mha", *alphat);
  ApplicationUtils::SaveImageITK("angleDiff.mha", *scratchI);
  
  // compute v0 from alpha0, see how far off we are
  *scratchV = *gradI0;
  scratchV->pointwiseMultiplyBy(*alpha0);
  op.CopyIn(*scratchV);
  op.ApplyInverseOperator();
  op.CopyOut(*scratchV);
  // save norm of computed field
  HField3DUtils::pointwiseL2Norm(*scratchV,*scratchI);
  ApplicationUtils::SaveImageITK("v0ComputedNorm.mha",*scratchI);
  
  // test against original v0
  scratchV->pointwiseSubtract(*v0);
  // save norm error
  HField3DUtils::pointwiseL2Norm(*scratchV,*scratchI);
  ApplicationUtils::SaveImageITK("v0ComputedFromAlpha0Err.mha",*scratchI);
  // save original v0 norm too, for comparison
  HField3DUtils::pointwiseL2Norm(*v0,*scratchI);
  ApplicationUtils::SaveImageITK("v0Norm.mha",*scratchI);
  
  // ### END TEST ###

  RealImage *J0t = NULL;
  VectorField *vt = NULL;
  
  J0t = new RealImage(size, origin, spacing);
  *J0t = *I0;
  vt = new VectorField(size);
  *vt = *v0;
  

  // set up initial values

  finalMorphImages.push_back(J0t);
  finalVecFields.push_back(vt);

  HField3DUtils::setToIdentity(*hField);
  *alphat = *alpha0;
  Dphit->fill(1);
  for(unsigned int t=1;t<nTimeSteps;t++){
    // update hField
    HField3DUtils::composeHVInv(*hField, *finalVecFields[t-1], *scratchV, spacing,
				HField3DUtils::BACKGROUND_STRATEGY_PARTIAL_ID);
    *hField = *scratchV;

    // compute alpha_t

    // scratchV = x+v(x)
    *scratchV = *finalVecFields[t-1];
    scratchV->scale(-1.0);
    // deform the old jacobian by the new velocity
    HField3DUtils::applyU(*Dphit,*scratchV,*scratchI,spacing);
    *Dphit = *scratchI;
    // compute the determinant of jacobian of the velocity
    // deformation
    HField3DUtils::setToIdentity(*scratchV);
    scratchV->scale(spacing);
    scratchV->pointwiseSubtract(*finalVecFields[t-1]);
    HField3DUtils::jacobian(*scratchV,*scratchI,spacing);
    // compute new jacobian determinant
    Dphit->pointwiseMultiplyBy(*scratchI);

    // deform alpha0
    HField3DUtils::apply(*alpha0, *hField, *alphat);

    // scale by jac. det.
    alphat->pointwiseMultiplyBy(*Dphit);
     
//      // This was to scale by abs(Dphit)
//      *scratchI = *Dphit;
//      // take absolute value
//      unsigned int dataSize = size.productOfElements();
//      for (unsigned int i = 0; i < dataSize; ++i){
//        (*scratchI)(i) = std::fabs((*Dphit)(i));
//      }
//      alphat->pointwiseMultiplyBy(*scratchI);

    // TEST
    sprintf(fname, "alpha%02d.mha", t);
    ApplicationUtils::SaveImageITK(fname, *alphat);
    // END TEST

    // compute deformed image
    J0t = new RealImage(size, origin, spacing);
    HField3DUtils::apply(*I0, *hField, *J0t);

    // compute a_t
    Array3DUtils::computeGradient(*J0t,*at,spacing,false);
    at->pointwiseMultiplyBy(*alphat);

    // ### TEST ###
    Real momentum = Array3DUtils::mass(*alphat);

    // compute alpha_t
    Array3DUtils::computeGradient(*J0t,*gradI0,spacing,false);
    computedMomentum =   
      computeAlphaFromA(at,
			gradI0,
			alphat,
			scratchI,
			scratchI2);

    std::cout << "timestep " << t << " momentum: " << momentum << "/" << computedMomentum << std::endl;
    sprintf(fname, "alpha%02dComputed.mha", t);
    ApplicationUtils::SaveImageITK(fname, *alphat);
    sprintf(fname, "angleDiff%02d.mha", t);
    ApplicationUtils::SaveImageITK(fname, *scratchI);
    sprintf(fname, "alphaMask%02d.mha", t);
    ApplicationUtils::SaveImageITK(fname, *scratchI2);

    // ### END TEST ###

    // compute v_t
    vt = new VectorField(size);
    op.CopyIn(*at);
    op.ApplyInverseOperator();
    op.CopyOut(*vt);

    finalMorphImages.push_back(J0t);
    finalVecFields.push_back(vt);
  }

  // create the final deformation
  // update hField
  HField3DUtils::composeHVInv(*hField, *finalVecFields[nTimeSteps-1], *scratchV, spacing,
			     HField3DUtils::BACKGROUND_STRATEGY_PARTIAL_ID);
  *hField = *scratchV;
  J0t = new RealImage(size, origin, spacing);
  HField3DUtils::apply(*I0, *hField, *J0t);
  finalMorphImages.push_back(J0t);
  
  if(debugOptions){
    std::string prefix = GetChildVal<std::string>(debugOptions, "OutputPrefix");
    std::string suffix = GetChildVal<std::string>(debugOptions, "OutputSuffix");
    if(GetChildVal<bool>(debugOptions, "WriteJacDet")){
      std::string jacDetName = prefix + "JacDet." + suffix;
      ApplicationUtils::SaveImageITK(jacDetName.c_str(), *Dphit);
    }
  }

  // clean up
  delete a0;
  delete alpha0;
  delete gradI0;
  delete gradI0Norm;
  delete a0Norm;
  delete hField;
  delete alphat;
  delete Dphit;
  delete at;
  delete scratchV;

}

// void 
// LDMM::LDMMBuildAtlas(std::vector<const RealImage*>images,
// 		     const ParamMap &params,
// 		     RealImage &meanImage,
// 		     std::vector<VectorField*> &defFields)
// {
//}
