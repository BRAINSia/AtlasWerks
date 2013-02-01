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


#include "LDMMShootingIteratorGPU.h"
#include "KernelFactory.h"
#include <cudaSplat.h>
#include <cudaImage3D.h>

//
// ################ LDMMShootingIteratorGPU Implementation ################ //
//

LDMMShootingIteratorGPU::
LDMMShootingIteratorGPU(const SizeType &size, 
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
    mdKernel(NULL),
    mdAlphaT(NULL),
    mdScratchI(NULL),
    mdAlpha(NULL)
{

  int memSize = mNVox * sizeof(float);

  allocateDeviceArray((void**)&mdAlphaT, memSize);
  allocateDeviceArray((void**)&mdScratchI, memSize);
  allocateDeviceArray((void**)&mdAlpha, memSize);

  allocateDeviceVector3DArray(mdVT, mNVox);
  allocateDeviceVector3DArray(mdScratchV, mNVox);

  // create DiffOper
  mdRd = new cplReduce();
}

LDMMShootingIteratorGPU::~LDMMShootingIteratorGPU(){
  cudaSafeDelete(mdAlphaT);
  cudaSafeDelete(mdScratchI);
  cudaSafeDelete(mdAlpha);

  freeDeviceVector3DArray(mdVT);
  freeDeviceVector3DArray(mdScratchV);

  delete mdKernel;
}

void 
LDMMShootingIteratorGPU::
SetScaleLevel(const MultiscaleManager &scaleManager,
	      const LDMMIteratorParam &param)
{
  ScaleLevelDataInfo::SetScaleLevel(scaleManager);

  mParam = &param;
  
  if(param.UseAdaptiveStepSize()){
    throw AtlasWerksException(__FILE__,__LINE__,"Error, GPU version of LDMMShooting does not support adaptive step size");
  }

  mMaxPert = mParam->MaxPert();
  mSigma = mParam->Sigma();

  LOGNODETHREAD(logDEBUG1) <<  "Resizing diffOper with size " << mCurSize << " and spacing " << mCurSpacing;
  // resize DiffOper
  if(!mdKernel) mdKernel = KernelFactory::NewGPUKernel(mParam->Kernel());
  mdKernel->SetSize(mCurSize, mCurSpacing, mParam->Kernel());

  CUDAUtilities::CheckCUDAError(__FILE__,__LINE__);

  mInitialized = true;

}

void 
LDMMShootingIteratorGPU::
Iterate(DeformationDataInterface &deformationData)
{

  if(!mInitialized){
    throw AtlasWerksException(__FILE__,__LINE__,"Error, iterator not initialized!");
  }

  float voxelVol = mCurSpacing.productOfElements();

  LDMMShootingDefDataGPU &data = dynamic_cast<LDMMShootingDefDataGPU&>(deformationData);
  cplVector3DArray &phi0T = data.dPhi0T();
  cplVector3DArray &phiT0 = data.dPhiT0();
  
  LDMMEnergy energy(mNTimeSteps, mSigma);

  // Initialize values
  cudaHField3DUtils::setToIdentity(phi0T, mCurSize);
  cudaHField3DUtils::setToIdentity(phiT0, mCurSize);
  copyArrayDeviceToDevice(mdAlphaT, data.dAlpha0(), mCurVox);
  
  //TEST
//   {
//     float phiSum = CUDAUtilities::DeviceVectorSum(phiT0, mCurVox);
//     LOGNODETHREAD(logDEBUG2) << "phiT0 sum is " << phiSum << " after setToIdentity";
//   }
  // END TEST

  // compute v0 = K(alpha0*gI0)
  cplComputeGradient(mdScratchV, data.dI0(), mCurSize, mCurSpacing);
  cplVector3DOpers::Mul(mdVT, mdScratchV, mdAlphaT, mCurVox);

  //TEST
//   {
//     float phiSum = CUDAUtilities::DeviceVectorSum(phiT0, mCurVox);
//     LOGNODETHREAD(logDEBUG2) << "phiT0 sum is " << phiSum << " before inverse op";
//   }
  // END TEST

  mdKernel->ApplyInverseOperator(mdVT);

  //TEST
//   {
//     float phiSum = CUDAUtilities::DeviceVectorSum(phiT0, mCurVox);
//     LOGNODETHREAD(logDEBUG2) << "phiT0 sum is " << phiSum << " after inverse op";
//   }
  // END TEST
  
  for(unsigned int t = 1; t < mNTimeSteps; t++){

    // update vector energy, this step energy = <Lv_t,v_t>
    copyArrayDeviceToDevice(mdScratchV, mdVT);
    mdKernel->ApplyOperator(mdScratchV);

    //TEST
//     {
//       float phiSum = CUDAUtilities::DeviceVectorSum(phiT0, mCurVox);
//       LOGNODETHREAD(logDEBUG2) << "phiT0 sum is " << phiSum << " after applyOper";
//     }
    // END TEST

    Real vecEnergy = 0.f;
    vecEnergy += voxelVol*mdRd->Dot(mdScratchV.x, mdVT.x, mCurVox);
    vecEnergy += voxelVol*mdRd->Dot(mdScratchV.y, mdVT.y, mCurVox);
    vecEnergy += voxelVol*mdRd->Dot(mdScratchV.z, mdVT.z, mCurVox);

    //TEST
//     {
//       float phiSum = CUDAUtilities::DeviceVectorSum(phiT0, mCurVox);
//       LOGNODETHREAD(logDEBUG2) << "phiT0 sum is " << phiSum << " before SetVecStepEnergy";
//     }
    // END TEST

    energy.SetVecStepEnergy(vecEnergy);

    //TEST
//     {
//       float phiSum = CUDAUtilities::DeviceVectorSum(phiT0, mCurVox);
//       LOGNODETHREAD(logDEBUG2) << "phiT0 sum is " << phiSum << " after energy computation";
//     }
    // END TEST

    /* #### Compute New Alpha #### */

    // update phi0T
    cudaHField3DUtils::composeVH(mdScratchV, mdVT, phi0T, mCurSize, mCurSpacing,
				 BACKGROUND_STRATEGY_PARTIAL_ZERO);
    copyArrayDeviceToDevice(phi0T, mdScratchV);

    cplSplat3DH(mdAlphaT, data.dAlpha0(), phi0T, mCurSize);
    // cplSplatingHFieldAtomicSigned(mdAlphaT, data.dAlpha0(), 
    //                               phi0T.x, phi0T.y, phi0T.z,
    //                               mCurSize.x, mCurSize.y, mCurSize.z);
    
    
    /* #### Compute New I #### */

    //TEST
//     {
//       float phiSum = CUDAUtilities::DeviceVectorSum(phiT0, mCurVox);
//       LOGNODETHREAD(logDEBUG2) << "phiT0 sum is " << phiSum << " during Iterate() step " << t;
//     }
    // END TEST

    // update the deformation (hfield) for shooting
    cudaHField3DUtils::composeHVInv(mdScratchV, phiT0, mdVT, mCurSize, mCurSpacing,
				    BACKGROUND_STRATEGY_PARTIAL_ID);
    copyArrayDeviceToDevice(phiT0, mdScratchV);
    
    // create deformed image
    cudaHField3DUtils::apply(mdScratchI, data.dI0(), phiT0, mCurSize);

    /* #### Compute New V #### */

    // compute next vt = K(alphat*gJ0t)
    cplComputeGradient(mdScratchV, mdScratchI,
			mCurSize.x, mCurSize.y, mCurSize.z,
			mCurSpacing.x, mCurSpacing.y, mCurSpacing.z);
    cplVector3DOpers::Mul(mdVT, mdScratchV, mdAlphaT, mCurVox);
    mdKernel->ApplyInverseOperator(mdVT);
  }

  // final update

  // get the last step energy
  copyArrayDeviceToDevice(mdScratchV, mdVT);  
  mdKernel->ApplyOperator(mdScratchV);
  Real vecEnergy = 0.f;
  vecEnergy += voxelVol*mdRd->Dot(mdScratchV.x, mdVT.x, mCurVox);
  vecEnergy += voxelVol*mdRd->Dot(mdScratchV.y, mdVT.y, mCurVox);
  vecEnergy += voxelVol*mdRd->Dot(mdScratchV.z, mdVT.z, mCurVox);
  energy.SetVecStepEnergy(vecEnergy);
  
  // final update to the inverse deformation field, used for computing
  // image energy term
  cudaHField3DUtils::composeHVInv(mdScratchV, phiT0, mdVT, mCurSize, mCurSpacing,
				  BACKGROUND_STRATEGY_PARTIAL_ID);
  copyArrayDeviceToDevice(phiT0, mdScratchV);

  cudaHField3DUtils::composeVH(mdScratchV, mdVT, phi0T, mCurSize, mCurSpacing,
			       BACKGROUND_STRATEGY_PARTIAL_ZERO);
  copyArrayDeviceToDevice(phi0T, mdScratchV);
  
  // update alpha0 -= mStepSize*( alpha0 - newAlpha0)
  cudaHField3DUtils::apply(mdScratchI, data.dI0(), phiT0, mCurSize);
  
  cplVectorOpers::Sub_I(mdScratchI, data.dI1(), mCurVox);

  cplSplat3DH(mdAlphaT, mdScratchI, phiT0, mCurSize);

  // cplSplatingHFieldAtomicSigned(mdAlphaT, mdScratchI, 
  //                               phiT0.x, phiT0.y, phiT0.z,
  //                               mCurSize.x, mCurSize.y, mCurSize.z);

  Real vFac = 1.0-data.StepSize();
  Real uFac = data.StepSize()/(mSigma*mSigma);

  cplVectorOpers::MulC_Add_MulC_I(data.dAlpha0(), vFac, mdAlphaT, uFac, mCurVox);

  // compute image energy  = ||JT0 - I1||^2
  // create the final deformed image
  cudaHField3DUtils::apply(mdScratchI, data.dI0(), phiT0, mCurSize);

  cplVectorOpers::Sub_I(mdScratchI, data.dI1(), mCurVox);
  //compute difference from target image
  Real imEnergy = voxelVol*mdRd->Sum2(mdScratchI, mCurVox);
  energy.SetImageEnergy(imEnergy);
  data.AddEnergy(energy);
}

void 
LDMMShootingIteratorGPU::
UpdateDeformations(LDMMShootingDefDataGPU &defData)
{

  cplVector3DArray &phi0T = defData.dPhi0T();
  cplVector3DArray &phiT0 = defData.dPhiT0();
  
  cudaHField3DUtils::setToIdentity(phi0T, mCurSize);
  cudaHField3DUtils::setToIdentity(phiT0, mCurSize);
  copyArrayDeviceToDevice(mdAlphaT, defData.dAlpha0(), mCurVox);
  
  // compute v0 = K(alpha0*gI0)
  cplComputeGradient(mdScratchV, defData.dI0(),
		      mCurSize.x, mCurSize.y, mCurSize.z,
		      mCurSpacing.x, mCurSpacing.y, mCurSpacing.z);
  cplVector3DOpers::Mul(mdVT, mdScratchV, mdAlphaT, mCurVox);

  mdKernel->ApplyInverseOperator(mdVT);
  
  for(unsigned int t = 1; t < mNTimeSteps; t++){

    /* #### Compute New Alpha #### */

    // update phi0T
    cudaHField3DUtils::composeVH(mdScratchV, mdVT, phi0T, mCurSize, mCurSpacing,
				 BACKGROUND_STRATEGY_PARTIAL_ZERO);
    copyArrayDeviceToDevice(phi0T, mdScratchV);

    // splat the difference image from dest to this timepoint
    cplSplat3DH(mdAlphaT, defData.dAlpha0(), phi0T, mCurSize);
    
    /* #### Compute New I #### */

    // update the deformation (hfield) for shooting
    cudaHField3DUtils::composeHVInv(mdScratchV, phiT0, mdVT, mCurSize, mCurSpacing,
				    BACKGROUND_STRATEGY_PARTIAL_ID);
    copyArrayDeviceToDevice(phiT0, mdScratchV);
    
    // create deformed image
    cudaHField3DUtils::apply(mdScratchI, defData.dI0(), phiT0, mCurSize);

    /* #### Compute New V #### */

    // compute next vt = K(alphat*gJ0t)
    cplComputeGradient(mdScratchV, mdScratchI,
		       mCurSize.x, mCurSize.y, mCurSize.z,
		       mCurSpacing.x, mCurSpacing.y, mCurSpacing.z);
    cplVector3DOpers::Mul(mdVT, mdScratchV, mdAlphaT, mCurVox);
    mdKernel->ApplyInverseOperator(mdVT);
  }

  // final update to the inverse deformation field, used for computing
  // image energy term
  cudaHField3DUtils::composeHVInv(mdScratchV, phiT0, mdVT, mCurSize, mCurSpacing,
				  BACKGROUND_STRATEGY_PARTIAL_ID);
  copyArrayDeviceToDevice(phiT0, mdScratchV);

}

