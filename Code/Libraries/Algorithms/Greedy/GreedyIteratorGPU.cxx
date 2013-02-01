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


#include "GreedyIteratorGPU.h"

#include "KernelFactory.h"
#include <cudaHField3DUtils.h>
#include <cudaImage3D.h>
#include <cudaUpsample3D.h>
#include <cutil_comfunc.h>
#include "log.h"

GreedyIteratorGPU::
GreedyIteratorGPU(SizeType &size, 
		OriginType &origin,
		SpacingType &spacing,
		bool debug)
  : DeformationIteratorInterface(size, origin, spacing),
    mUpdateStepSizeNextIter(false),
    mDebug(debug)
{
  mRd     = new cplReduce();
  mKernel = NULL;

  int memSize = mNVox * sizeof(float);
  
  // allocate scratch memory
  allocateDeviceArray((void**)&mdScratchI, memSize);
  allocateDeviceVector3DArray(mdScratchV, mNVox);
  allocateDeviceVector3DArray(mdU, mNVox);

  checkCUDAError("Memory allocation");
}

GreedyIteratorGPU::
~GreedyIteratorGPU()
{
  delete mRd;
  if(mKernel)
    delete mKernel;
  
  cudaSafeDelete(mdScratchI);
  freeDeviceVector3DArray(mdScratchV);
  freeDeviceVector3DArray(mdU);
}

void 
GreedyIteratorGPU::
SetScaleLevel(const MultiscaleManager &scaleManager,
	      const GreedyIteratorParam &param)
{
  ScaleLevelDataInfo::SetScaleLevel(scaleManager);

  mParam = &param;
  mMaxPert = mParam->MaxPert();

  if(!mKernel) mKernel = KernelFactory::NewGPUKernel(param.Kernel());
  
  mKernel->SetSize(mCurSize, mCurSpacing, mParam->Kernel());
}

void 
GreedyIteratorGPU::
updateDeformation(GreedyDeformationDataGPU &data)
{
  // scale by step size
  cplVector3DOpers::MulC_I(mdU, data.StepSize(), mCurVox);
  cudaHField3DUtils::composeHV(mdScratchV, data.dDef1To0(), mdU, 
			       mCurSize, mCurSpacing, 
			       BACKGROUND_STRATEGY_PARTIAL_ID);
  copyArrayDeviceToDevice(data.dDef1To0(), mdScratchV, mCurVox);
  
  if(data.ComputeInverseHField()){
    cudaHField3DUtils::composeVHInv(mdScratchV, mdU, data.dDef0To1(), 
				    mCurSize, mCurSpacing,
				    BACKGROUND_STRATEGY_PARTIAL_ZERO);
    copyArrayDeviceToDevice(data.dDef0To1(), mdScratchV, mCurVox);
  }
}

void 
GreedyIteratorGPU::
Iterate(DeformationDataInterface &deformationData)
{

  GreedyDeformationDataGPU &data = dynamic_cast<GreedyDeformationDataGPU&>(deformationData);

  // this will only be filled if mDebug is true
  Energy energy;

  cudaHField3DUtils::apply(mdScratchI, data.dI0(), data.dDef1To0(), mCurSize);

  // Compute the gradient of deformed image
  cplComputeGradient(mdScratchV, mdScratchI,
		      mCurSize.x, mCurSize.y, mCurSize.z,
		      mCurSpacing.x, mCurSpacing.y, mCurSpacing.z);

  // compute I1 - IDef
  cplVectorOpers::SubMulC_I(mdScratchI, data.dI1(), -1.0f, mCurVox);

  // scale the gradient by the image difference
  cplVector3DOpers::Mul_I(mdScratchV, mdScratchI, mCurVox);

  if(mDebug){
    float imEnergy = mCurSpacing.productOfElements()*mRd->Sum2(mdScratchI, mCurVox);
    energy.SetEnergy(imEnergy);
    data.AddEnergy(energy);
  }

  copyArrayDeviceToDevice(mdU, mdScratchV);

  mKernel->ApplyInverseOperator(mdU);

  if(mUpdateStepSizeNextIter){
      cplVector3DOpers::Magnitude(mdScratchI, mdU, mCurVox);
      Real maxDisplacement = mRd->Max(mdScratchI, mCurVox);
      LOGNODE(logDEBUG) << "Max Displacement is " << maxDisplacement;
      data.StepSize(mMaxPert / maxDisplacement);
      mUpdateStepSizeNextIter = false;
  }
  
  this->updateDeformation(data);

}
