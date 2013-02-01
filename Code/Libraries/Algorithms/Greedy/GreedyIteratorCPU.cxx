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


#include "GreedyIteratorCPU.h"
#include "KernelFactory.h"

GreedyIteratorCPU::
GreedyIteratorCPU(const Vector3D<unsigned int> &size, 
		  const Vector3D<Real> &origin,
		  const Vector3D<Real> &spacing,
		  bool debug)
  : DeformationIteratorInterface(size, origin, spacing),
    mUpdateStepSizeNextIter(false),
    mDebug(debug),
    mKernel(NULL),
    mDiffOpVF(NULL)
{
  // allocate memory
  mScratchI = new RealImage();
  mScratchV = new VectorField();
}

GreedyIteratorCPU::~GreedyIteratorCPU(){
  delete mScratchI;
  delete mScratchV;

  if(mKernel){
    delete mKernel;
  }
}

void 
GreedyIteratorCPU::
SetScaleLevel(const MultiscaleManager &scaleManager,
	      const GreedyIteratorParam &param)
{
  ScaleLevelDataInfo::SetScaleLevel(scaleManager);

  mMaxPert = param.MaxPert();

  // allocate memory
  mScratchI->resize(mCurSize);
  mScratchI->setOrigin(mImOrigin);
  mScratchI->setSpacing(mCurSpacing);
  mScratchV->resize(mCurSize);
  
  // create DiffOper
  if(mKernel) delete mKernel;
  mKernel = KernelFactory::NewKernel(param.Kernel(), mCurSize, mCurSpacing);
  mDiffOpVF = mKernel->GetInternalFFTWVectorField();

}

Real
GreedyIteratorCPU::
calcMaxDisplacement()
{
  Real max = 0.0f;
  for (unsigned int z = 0; z < mCurSize.z; ++z) {
    for (unsigned int y = 0; y < mCurSize.y; ++y) {
      for (unsigned int x = 0; x < mCurSize.x; ++x) {
	Real l = mDiffOpVF->get(x,y,z).normL2();
	if(l > max) max = l;
      }
    }
  }
  return max;
}

void 
GreedyIteratorCPU::
updateDeformation(GreedyDeformationData &data)
{
  mDiffOpVF->scale(data.StepSize());
  HField3DUtils::composeHV(data.Def1To0(), *mDiffOpVF, *mScratchV, mCurSpacing,
			   HField3DUtils::BACKGROUND_STRATEGY_PARTIAL_ID);
  data.Def1To0() = *mScratchV;
  if(data.ComputeInverseHField()){
    
    HField3DUtils::composeVHInv(*mDiffOpVF, data.Def0To1(), *mScratchV, mCurSpacing,
				HField3DUtils::BACKGROUND_STRATEGY_PARTIAL_ZERO);
    data.Def0To1() = *mScratchV;
  }
}

void
GreedyIteratorCPU::
Iterate(DeformationDataInterface &deformationData)
{
  GreedyDeformationData &data = dynamic_cast<GreedyDeformationData&>(deformationData);
  
  // this will only be filled if mDebug is true
  Energy energy;

  // create the deformed image 
  HField3DUtils::apply(data.I0(), data.Def1To0(), *mScratchI);

  // compute gradient of deformed image, store in internal vector field
  // of differential operator
  Array3DUtils::computeGradient(*mScratchI,*mDiffOpVF,mCurSpacing,false);
  // subtract the final image from the deformed image
  mScratchI->pointwiseSubtract(data.I1());

  mScratchI->scale(-1.0);

  // scale the gradient by the scaled image difference
  mKernel->pointwiseMultiplyBy_FFTW_Safe(*mScratchI);
  
  // compute energy, actually a step behind
  energy.SetEnergy(ImageUtils::l2NormSqr(*mScratchI));
  data.AddEnergy(energy);

  // apply K
  mKernel->ApplyInverseOperator();

  // mDiffOpVF now holds the update vector field, we can now calculate
  // new step size based on MaxPert if needed
  if(mUpdateStepSizeNextIter){
    Real maxDisplacement = this->calcMaxDisplacement();
    LOGNODE(logDEBUG) << "Max Displacement is " << maxDisplacement;
    data.StepSize(mMaxPert / maxDisplacement);
    mUpdateStepSizeNextIter = false;
  }
  
  this->updateDeformation(data);

}
