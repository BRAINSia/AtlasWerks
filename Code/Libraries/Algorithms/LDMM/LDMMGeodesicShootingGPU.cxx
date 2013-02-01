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

#include "LDMMGeodesicShootingGPU.h"

#include <cudaSplat.h>
#include <cudaImage3D.h>
#include <cudaHField3DUtils.h>
#include <CUDAUtilities.h>

LDMMGeodesicShootingGPU::
LDMMGeodesicShootingGPU(SizeType imSize, 
			SpacingType imSpacing, 
			DiffOperParam &diffOpParam)
  : LDMMGeodesicShooting(imSize, imSpacing),
    mNVox(imSize.productOfElements()),
    mdDiffOp(),
    mdRd(),
    mIT(imSize),
    mPhi0T(imSize),
    mPhiT0(imSize),
    mdAlphaT(NULL),
    mdIT(NULL),
    mdAlpha0(NULL),
    mdI0(NULL),
    mdScratchI(NULL)
{
  
  int memSize = mNVox * sizeof(float);

  allocateDeviceVector3DArray(mdPhi0T, mNVox);
  allocateDeviceVector3DArray(mdPhiT0, mNVox);
  allocateDeviceVector3DArray(mdVT, mNVox);
  allocateDeviceVector3DArray(mdScratchV, mNVox);
  allocateDeviceArray((void**)&mdAlphaT, memSize);
  allocateDeviceArray((void**)&mdIT, memSize);
  allocateDeviceArray((void**)&mdI0, memSize);
  allocateDeviceArray((void**)&mdAlpha0, memSize);
  allocateDeviceArray((void**)&mdScratchI, memSize);

  mdDiffOp.SetSize(mImSize, mImSpacing, diffOpParam);

  CUDAUtilities::CheckCUDAError(__FILE__,__LINE__);
}

LDMMGeodesicShootingGPU::
~LDMMGeodesicShootingGPU()
{
  freeDeviceVector3DArray(mdPhi0T);
  freeDeviceVector3DArray(mdPhiT0);
  freeDeviceVector3DArray(mdVT);
  freeDeviceVector3DArray(mdScratchV);

  freeDeviceArray(mdAlphaT);
  freeDeviceArray(mdIT);
  freeDeviceArray(mdAlpha0);
  freeDeviceArray(mdI0);
  freeDeviceArray(mdScratchI);

  CUDAUtilities::CheckCUDAError(__FILE__,__LINE__);
}

void 
LDMMGeodesicShootingGPU::
ShootImage(const RealImage &I0,
	   const RealImage &alpha0,
	   unsigned int nTimeSteps)
{

  // Initialize values

  OriginType imOrigin = I0.getOrigin();
  float voxelVol = mImSpacing.productOfElements();

  mNTimeSteps = nTimeSteps;

  copyArrayToDevice(mdAlpha0, alpha0.getDataPointer(), mNVox);
  copyArrayDeviceToDevice(mdAlphaT, mdAlpha0, mNVox);

  if(mSaveAlphas){
    RealImage *alpha = new RealImage(mImSize, imOrigin, mImSpacing);
    copyArrayFromDevice(alpha->getDataPointer(), mdAlphaT, mNVox);
    mAlphaVec.push_back(alpha);
  }

  copyArrayToDevice(mdI0, I0.getDataPointer(), mNVox);
  copyArrayDeviceToDevice(mdIT, mdI0, mNVox);

  if(mSaveImages){
    RealImage *im = new RealImage(mImSize, imOrigin, mImSpacing);
    copyArrayFromDevice(im->getDataPointer(), mdIT, mNVox);
    mImVec.push_back(im);
  }

  cudaHField3DUtils::setToIdentity(mdPhi0T, mImSize);
  cudaHField3DUtils::setToIdentity(mdPhiT0, mImSize);
  
  LDMMEnergy energy(mNTimeSteps, 0.f);

  // compute v0 = K(alpha0*gI0)
  cplComputeGradient(mdScratchV, mdIT, mImSize, mImSpacing);
  cplVector3DOpers::Mul(mdVT, mdScratchV, mdAlphaT, mNVox);

  mdDiffOp.ApplyInverseOperator(mdVT);
  
  CUDAUtilities::CheckCUDAError(__FILE__,__LINE__);

  if(mSaveVecs){
    VectorField *v = new VectorField(mImSize);
    CUDAUtilities::CopyVectorFieldFromDevice(mdVT, *v, true);
    mVVec.push_back(v);
  }

  LOGNODE(logDEBUG) <<  "Finished initialization";

  for(unsigned int t = 1; t < (unsigned int)mNTimeSteps; t++){

    LOGNODE(logDEBUG) <<  "Running timestep " << t;
    // update vector energy, this step energy = <Lv_t,v_t>
    copyArrayDeviceToDevice(mdScratchV, mdVT);
    mdDiffOp.ApplyOperator(mdScratchV);

    CUDAUtilities::CheckCUDAError(__FILE__,__LINE__);

    Real vecEnergy = 0.f;
    vecEnergy += (voxelVol/mNTimeSteps)*mdRd.Dot(mdScratchV.x, mdVT.x, mNVox);
    vecEnergy += (voxelVol/mNTimeSteps)*mdRd.Dot(mdScratchV.y, mdVT.y, mNVox);
    vecEnergy += (voxelVol/mNTimeSteps)*mdRd.Dot(mdScratchV.z, mdVT.z, mNVox);
    energy.SetVecStepEnergy(vecEnergy);

    CUDAUtilities::CheckCUDAError(__FILE__,__LINE__);
    /* #### Compute New Alpha #### */

    // update mdPhi0T
    cudaHField3DUtils::composeVH(mdScratchV, mdVT, mdPhi0T, mImSize, mImSpacing,
				 BACKGROUND_STRATEGY_PARTIAL_ZERO);
    copyArrayDeviceToDevice(mdPhi0T, mdScratchV);

    cplSplat3DH(mdAlphaT, mdAlpha0, mdPhi0T, mImSize);

    if(mSaveAlphas){
      RealImage *alpha = new RealImage(mImSize, imOrigin, mImSpacing);
      copyArrayFromDevice(alpha->getDataPointer(), mdAlphaT, mNVox);
      mAlphaVec.push_back(alpha);
    }
    CUDAUtilities::CheckCUDAError(__FILE__,__LINE__);

    /* #### Compute New I #### */

    // update the deformation (hfield) for shooting
    cudaHField3DUtils::composeHVInv(mdScratchV, mdPhiT0, mdVT, mImSize, mImSpacing,
				    BACKGROUND_STRATEGY_PARTIAL_ID);
    copyArrayDeviceToDevice(mdPhiT0, mdScratchV);
    
    // create deformed image
    cudaHField3DUtils::apply(mdScratchI, mdI0, mdPhiT0, mImSize);

    if(mSaveImages){
      RealImage *im = new RealImage(mImSize, imOrigin, mImSpacing);
      copyArrayFromDevice(im->getDataPointer(), mdIT, mNVox);
      mImVec.push_back(im);
    }
    CUDAUtilities::CheckCUDAError(__FILE__,__LINE__);

    /* #### Compute New V #### */

    // compute next vt = K(alphat*gJ0t)
    cplComputeGradient(mdScratchV, mdScratchI,
			mImSize.x, mImSize.y, mImSize.z,
			mImSpacing.x, mImSpacing.y, mImSpacing.z);
    CUDAUtilities::CheckCUDAError(__FILE__,__LINE__);
    cplVector3DOpers::Mul(mdVT, mdScratchV, mdAlphaT, mNVox);

    CUDAUtilities::CheckCUDAError(__FILE__,__LINE__);
    mdDiffOp.ApplyInverseOperator(mdVT);
    CUDAUtilities::CheckCUDAError(__FILE__,__LINE__);

    if(mSaveVecs){
      VectorField *v = new VectorField(mImSize);
      CUDAUtilities::CopyVectorFieldFromDevice(mdVT, *v, true);
      mVVec.push_back(v);
    }

  }

  // final update

  // get the last step energy
  copyArrayDeviceToDevice(mdScratchV, mdVT);  
  mdDiffOp.ApplyOperator(mdScratchV);
  Real vecEnergy = 0.f;
  vecEnergy += (voxelVol/mNTimeSteps)*mdRd.Dot(mdScratchV.x, mdVT.x, mNVox);
  vecEnergy += (voxelVol/mNTimeSteps)*mdRd.Dot(mdScratchV.y, mdVT.y, mNVox);
  vecEnergy += (voxelVol/mNTimeSteps)*mdRd.Dot(mdScratchV.z, mdVT.z, mNVox);
  energy.SetVecStepEnergy(vecEnergy);
  
  // final update to the inverse deformation field, used for computing
  // image energy term
  cudaHField3DUtils::composeHVInv(mdScratchV, mdPhiT0, mdVT, mImSize, mImSpacing,
				  BACKGROUND_STRATEGY_PARTIAL_ID);
  copyArrayDeviceToDevice(mdPhiT0, mdScratchV);

  cudaHField3DUtils::composeVH(mdScratchV, mdVT, mdPhi0T, mImSize, mImSpacing,
			       BACKGROUND_STRATEGY_PARTIAL_ZERO);
  copyArrayDeviceToDevice(mdPhi0T, mdScratchV);
  
  // compute final image
  cudaHField3DUtils::apply(mdIT, mdI0, mdPhiT0, mImSize);

  if(mSaveImages){
    RealImage *im = new RealImage(mImSize, imOrigin, mImSpacing);
    copyArrayFromDevice(im->getDataPointer(), mdIT, mNVox);
    mImVec.push_back(im);
  }

  // copy results back to host
  copyArrayFromDevice(mIT.getDataPointer(), mdIT, mNVox);
  mIT.setOrigin(imOrigin);
  mIT.setSpacing(mImSpacing);
  CUDAUtilities::CopyVectorFieldFromDevice(mdPhi0T, mPhi0T, true);
  CUDAUtilities::CopyVectorFieldFromDevice(mdPhiT0, mPhiT0, true);

  CUDAUtilities::CheckCUDAError(__FILE__,__LINE__);

  LOGNODE(logDEBUG) <<  "Shooting finished";
}
  
void 
LDMMGeodesicShootingGPU::
ShootImage(const RealImage &I0,
	   const VectorField &v0,
	   unsigned int nTimeSteps)
{
  throw AtlasWerksException(__FILE__,__LINE__,"Velocity geodesic shooting not supported on GPU yet");
}
  


