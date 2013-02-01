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


#include "LDMMAdShootingIteratorGPU.h"
#include "KernelFactory.h"
#include <cudaSplat.h>
#include <cudaImage3D.h>

//
// ################ LDMMAdShootingIteratorGPU Implementation ################ //
//

LDMMAdShootingIteratorGPU::
LDMMAdShootingIteratorGPU(const SizeType &size, 
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
#if TRADE_MEM_FOR_SPEED
    mdPhiT0(nTimeSteps+1),
    mdPhi0T(nTimeSteps+1),
#else
    mdVVec(nTimeSteps),
#endif
    mdIHatT(NULL),
    mdAlphaHatT(NULL),
    mdITildeT(NULL),
    mdAlphaTildeT(NULL),
    mdIT(NULL),
    mdAlphaT(NULL),
    mdKernel(NULL),
    mdScratchI(NULL),
    mdScratchI2(NULL)
{
  int memSize = mNVox * sizeof(float);

  allocateDeviceArray((void**)&mdIHatT, memSize);
  allocateDeviceArray((void**)&mdAlphaHatT, memSize);
  allocateDeviceArray((void**)&mdITildeT, memSize);
  allocateDeviceArray((void**)&mdAlphaTildeT, memSize);
  allocateDeviceArray((void**)&mdScratchI2, memSize);
  allocateDeviceArray((void**)&mdIT, memSize);
  allocateDeviceArray((void**)&mdAlphaT, memSize);
  allocateDeviceArray((void**)&mdScratchI, memSize);

  allocateDeviceVector3DArray(mdScratchV, mNVox);
  allocateDeviceVector3DArray(mdScratchV2, mNVox);
#if TRADE_MEM_FOR_SPEED
  allocateDeviceVector3DArray(mdVT, mNVox);
#endif

  for (uint i=0; i<mNTimeSteps; i++){
#if TRADE_MEM_FOR_SPEED
    allocateDeviceVector3DArray(mdPhiT0[i], mNVox);
    allocateDeviceVector3DArray(mdPhi0T[i], mNVox);
#else
    allocateDeviceVector3DArray(mdVVec[i], mNVox);
#endif
  }

  mdRd = new cplReduce();

  CUDAUtilities::CheckCUDAError(__FILE__,__LINE__);

}

LDMMAdShootingIteratorGPU::~LDMMAdShootingIteratorGPU(){

  cudaSafeDelete(mdIHatT);
  cudaSafeDelete(mdAlphaHatT);
  cudaSafeDelete(mdAlphaTildeT);
  cudaSafeDelete(mdScratchI2);
  cudaSafeDelete(mdIT);
  cudaSafeDelete(mdAlphaT);
  cudaSafeDelete(mdScratchI);

  freeDeviceVector3DArray(mdScratchV);
  freeDeviceVector3DArray(mdScratchV2);

  for (uint i=0; i<mNTimeSteps; i++){
#if TRADE_MEM_FOR_SPEED
    freeDeviceVector3DArray(mdPhiT0[i]);
    freeDeviceVector3DArray(mdPhi0T[i]);
#else
    freeDeviceVector3DArray(mdVVec[i]);
#endif
  }

  delete mdKernel;
  delete mdRd;

}

void 
LDMMAdShootingIteratorGPU::
SetScaleLevel(const MultiscaleManager &scaleManager,
	      const LDMMIteratorParam &param)
{
  ScaleLevelDataInfo::SetScaleLevel(scaleManager);

  mParam = &param;

  if(param.UseAdaptiveStepSize()){
    throw AtlasWerksException(__FILE__,__LINE__,"Error, GPU version of LDMMShooting does not support adaptive step size");
  }
  
  mMaxPert = param.MaxPert();
  mSigma = param.Sigma();
  
  if(!mdKernel) mdKernel = KernelFactory::NewGPUKernel(mParam->Kernel());
  mdKernel->SetSize(mCurSize, mCurSpacing, mParam->Kernel());

  CUDAUtilities::CheckCUDAError(__FILE__,__LINE__);

  mInitialized = true;
    
}

void 
LDMMAdShootingIteratorGPU::
Iterate(DeformationDataInterface &deformationData)
{
  if(!mInitialized){
    throw AtlasWerksException(__FILE__,__LINE__,"Error, iterator not initialized!");
  }
  
  CUDAUtilities::CheckCUDAError(__FILE__,__LINE__);

  float voxelVol = mCurSpacing.productOfElements();

  /** ####ADJOINT ALGORITHM PART 1: SHOOTING ALGORITHM####  */

  DeformationDataType &data = dynamic_cast<DeformationDataType&>(deformationData);

#if TRADE_MEM_FOR_SPEED
  cplVector3DArray *phi0T = &mdPhi0T[0];
  cplVector3DArray *phiT0 = &mdPhiT0[0];
  cplVector3DArray *vt = &mdVT;
#else
  cplVector3DArray *phi0T = &data.dPhi0T();
  cplVector3DArray *phiT0 = &data.dPhiT0();
  cplVector3DArray *vt = &mdVVec[0];
#endif

  LDMMEnergy energy(mNTimeSteps, mSigma);

  // Initialize values
  cudaHField3DUtils::setToIdentity(*phi0T, mCurSize);
  cudaHField3DUtils::setToIdentity(*phiT0, mCurSize);

  copyArrayDeviceToDevice(mdAlphaT, data.dAlpha0(), mCurVox);
  
  // compute v0 = -K(alpha0*gI0)
  cplComputeGradient(mdScratchV, data.dI0(), mCurSize, mCurSpacing);
  cplVector3DOpers::MulMulC(*vt, mdScratchV, mdAlphaT, -1.f, mCurVox);

  // Lv_0
  copyArrayDeviceToDevice(mdScratchV, *vt);

  // calc v_0
  mdKernel->ApplyInverseOperator(*vt);

  // calc <Lv,v>
  Real vecEnergy = 0.f;
  vecEnergy += voxelVol*mdRd->Dot(mdScratchV.x, vt->x, mCurVox);
  vecEnergy += voxelVol*mdRd->Dot(mdScratchV.y, vt->y, mCurVox);
  vecEnergy += voxelVol*mdRd->Dot(mdScratchV.z, vt->z, mCurVox);
  energy.SetVecStepEnergy(vecEnergy);

  for(unsigned int t = 1; t < mNTimeSteps; t++){

    /* #### Compute New Alpha #### */

    // update phi0T and phiT0
    cudaHField3DUtils::composeVH(mdScratchV, *vt, *phi0T, mCurSize, mCurSpacing,
				 BACKGROUND_STRATEGY_PARTIAL_ZERO);
    cudaHField3DUtils::composeHVInv(mdScratchV2, *phiT0, *vt, 
				    mCurSize, mCurSpacing,
				    BACKGROUND_STRATEGY_PARTIAL_ID);

#if TRADE_MEM_FOR_SPEED
    phiT0 = &mdPhiT0[t];
    phi0T = &mdPhi0T[t];
#else
    vt = &mdVVec[t];
#endif

    copyArrayDeviceToDevice(*phi0T, mdScratchV);
    copyArrayDeviceToDevice(*phiT0, mdScratchV2);
 
    // create alpha_t
    cplSplat3DH(mdAlphaT, data.dAlpha0(), *phi0T, mCurSize);

    /* #### Compute New I #### */

    
    // create deformed image
    cudaHField3DUtils::apply(mdScratchI, data.dI0(), *phiT0, mCurSize);

    /* #### Compute New V #### */

    // compute next vt = -K(alphat*gJ0t)
    cplComputeGradient(*vt, mdScratchI, mCurSize, mCurSpacing);

    cplVector3DOpers::MulMulC_I(*vt, mdAlphaT, -1.0, mCurVox);
    mdKernel->ApplyInverseOperator(*vt);
  }

  // final update to the deformation fields
  cudaHField3DUtils::composeHVInv(mdScratchV, *phiT0, *vt, 
				  mCurSize, mCurSpacing,
				  BACKGROUND_STRATEGY_PARTIAL_ID);
  cudaHField3DUtils::composeVH(mdScratchV2, *vt, *phi0T, 
			       mCurSize, mCurSpacing,
			       BACKGROUND_STRATEGY_PARTIAL_ZERO);
  
#if TRADE_MEM_FOR_SPEED
  phiT0 = &data.dPhiT0();
  phi0T = &data.dPhi0T();
#endif
  
  copyArrayDeviceToDevice(*phiT0, mdScratchV);
  copyArrayDeviceToDevice(*phi0T, mdScratchV2);


  // compute image energy  = ||JT0 - I1||^2
  // create the final deformed image
  cudaHField3DUtils::apply(mdScratchI, data.dI0(), *phiT0, mCurSize);
  cplVectorOpers::Sub_I(mdScratchI, data.dI1(), mCurVox);

  Real imEnergy = voxelVol*mdRd->Sum2(mdScratchI, mCurVox);
  energy.SetImageEnergy(imEnergy);
  data.AddEnergy(energy);

  CUDAUtilities::CheckCUDAError(__FILE__,__LINE__);

  /** ####ADJOINT ALGORITHM PART 2: GRADIENT COMPUTATION BY SOLVING ADJOINT EQUATIONS VIA BACKWARD INTEGRATION####  */  
  backwardAdIntegration(data);

}

void 
LDMMAdShootingIteratorGPU::backwardAdIntegration(DeformationDataType &data){

  cplVector3DArray *phi0T = &data.dPhi0T();
  cplVector3DArray *phiT0 = &data.dPhiT0();

  copyArrayDeviceToDevice(mdAlphaT, data.dAlpha0(), mCurVox);

  /** Initial conditions: compute \tilde{alpha}(1) and \tilde{I}(1) */  
  // \tilde{alpha}(1) is zero
  cplVectorOpers::SetMem(mdAlphaTildeT, 0.f, mCurVox);

  // \tilde{I}(1), phiT0 contains phi(1,0)
  cudaHField3DUtils::apply(mdScratchI, data.dI0(), *phiT0, mCurSize);

  copyArrayDeviceToDevice(mdIHatT, data.dI1(), mCurVox);
  
  CUDAUtilities::CheckCUDAError(__FILE__,__LINE__);

  cplVectorOpers::SubMulC_I(mdIHatT, mdScratchI, ((1.0f)/(mSigma*mSigma)), mCurVox);

  // splat image difference term to time 0 : \tilde{I(1)}=Jac(Phi0T)IhatT o PhiOT
  cplSplat3DH(mdITildeT, mdIHatT, *phiT0, mCurSize);

  CUDAUtilities::CheckCUDAError(__FILE__,__LINE__);

  for (int i=mNTimeSteps-1; i>= 0; i--){    

    if(i!=(static_cast<int>(mNTimeSteps)-1)){  //1st iteration is separated because we can use Phi0T, PhiT0 computed earlier in shooting
#if TRADE_MEM_FOR_SPEED
      phiT0 = &mdPhiT0[i+1];
      phi0T = &mdPhi0T[i+1];
#else
      cudaHField3DUtils::setToIdentity(*phiT0, mCurSize);
      cudaHField3DUtils::setToIdentity(*phi0T, mCurSize);
      //Get \phi_{t_{i+1},0} and \phi_{0,t_{i+1}}
      CUDAUtilities::CheckCUDAError(__FILE__,__LINE__);
      for(int j=1; j<=(i+1); j++){

	cudaHField3DUtils::composeVH(mdScratchV, mdVVec[j-1], *phi0T, 
				     mCurSize, mCurSpacing,
				     BACKGROUND_STRATEGY_PARTIAL_ZERO);
	copyArrayDeviceToDevice(*phi0T, mdScratchV);
	
	CUDAUtilities::CheckCUDAError(__FILE__,__LINE__);

	cudaHField3DUtils::composeHVInv(mdScratchV, *phiT0, mdVVec[j-1], 
					mCurSize, mCurSpacing,
					BACKGROUND_STRATEGY_PARTIAL_ID);
	copyArrayDeviceToDevice(*phiT0, mdScratchV);

	CUDAUtilities::CheckCUDAError(__FILE__,__LINE__);
      }
#endif
    }
    CUDAUtilities::CheckCUDAError(__FILE__,__LINE__);

    // create deformed image, I_{t+1}
    cudaHField3DUtils::apply(mdIT, data.dI0(), *phiT0, mCurSize);

    CUDAUtilities::CheckCUDAError(__FILE__,__LINE__);

    // create deformed alpha0, alpha_{t+1}
    cplSplat3DH(mdAlphaT, data.dAlpha0(), *phi0T, mCurSize);

    //Compute \hat{alpha}(t_{i+1})
    cudaHField3DUtils::apply(mdAlphaHatT, mdAlphaTildeT, *phiT0, mCurSize); // \tilde{Alpha}_{t+1} from previous iteration/initialization

    //Compute \hat{I}(t_{i+1}): splat \tilde{I}_{t_{i+1}} to time t_{i+1}
    cplSplat3DH(mdIHatT, mdITildeT, *phi0T, mCurSize); // \tilde{I}_{t+1} from previous iteration/initialization

    //Compute \hat{v}(t_{i+1})
    cplComputeGradient(mdScratchV, mdAlphaHatT, mCurSize, mCurSpacing);
    cplComputeGradient(mdScratchV2, mdIT, mCurSize, mCurSpacing); //mIT has I_{t+1}
    CUDAUtilities::CheckCUDAError(__FILE__,__LINE__);

    cplVector3DOpers::Mul_I(mdScratchV, mdAlphaT, mCurVox); // mScratchV has \nabla \hat{alpha}(t_{i+1})
    cplVector3DOpers::Mul_I(mdScratchV2, mdIHatT, mCurVox); //    mScratchV2 has \nabla I(t_{i+1})

    cplVector3DOpers::Sub_I(mdScratchV, mdScratchV2, mCurVox);

    CUDAUtilities::CheckCUDAError(__FILE__,__LINE__);
    mdKernel->ApplyInverseOperator(mdScratchV);
    CUDAUtilities::CheckCUDAError(__FILE__,__LINE__);

    //Result of pointwise innerprod \nabla I(t_{i+1}) \dot \hat{v}(t_{i+1}) will be stored in mScratchI
    cplComputeGradient(mdScratchV2, mdIT, mCurSize, mCurSpacing); //Computing    \nabla I(t_{i+1}) again because mScratchV2 was overwritten earlier

    cplVector3DOpers::DotProd(mdScratchI, mdScratchV2, mdScratchV, mCurVox);

    //Compute \tilde{P}(t_i)
    cudaHField3DUtils::apply(mdScratchI2, mdScratchI, *phi0T, mCurSize);
    cplVectorOpers::Sub_I(mdAlphaTildeT, mdScratchI2, mCurVox);

    //Result of divergence computation \nabla \dot (\alpha_{t_{i+1}} \hat{v}(t_{i+1})) will be strored in mScratchI
    cplVector3DOpers::Mul_I(mdScratchV, mdAlphaT, mCurVox);

    cudaHField3DUtils::divergence(mdScratchI, mdScratchV,
				  mdScratchV2,
				  mCurSize, mCurSpacing);
    cplSplat3DH(mdScratchI2, mdScratchI, *phiT0, mCurSize); 

    //Compute new \tilde{I}(t_i)
    cplVectorOpers::Add_I(mdITildeT, mdScratchI2, mCurVox);

  }  // By the end of above loop we have \tilde{alpha}(0) in mAlphaTildeT

  // Compute gradient of energy functional 
  cplComputeGradient(mdScratchV, data.dI0(), mCurSize, mCurSpacing);
#if TRADE_MEM_FOR_SPEED
  cplVector3DArray *vt = &mdVT;
  // compute v0 = -K(alpha0*gI0)
  cplComputeGradient(*vt, data.dI0(), mCurSize, mCurSpacing);
  cplVector3DOpers::MulMulC_I(*vt, data.dAlpha0(), -1.f, mCurVox);
  mdKernel->ApplyInverseOperator(*vt);
#else
  cplVector3DArray *vt = &mdVVec[0];
#endif

  cplVector3DOpers::DotProd(mdScratchI, mdScratchV, *vt, mCurVox);

  cplVectorOpers::MulCSub_I(mdScratchI, -1.f, mdAlphaTildeT, mCurVox);

  // Updated initial momenta for this gradient descent iteration
  cplVectorOpers::EpsUpdate(data.dAlpha0(), mdScratchI, data.StepSize(), mCurVox);

}

void 
LDMMAdShootingIteratorGPU::finalUpdatePhi0T(DeformationDataType &data){

#if TRADE_MEM_FOR_SPEED
  // nothing to do, phi0T still holds final deformation
#else
  // Update final phi0T
  cplVector3DArray &phi0T = data.dPhi0T();
  cudaHField3DUtils::setToIdentity(phi0T, mCurSize);
  for(uint j=1; j<=mNTimeSteps; j++){	
    cudaHField3DUtils::composeVH(mdScratchV, mdVVec[j-1], phi0T, 
				 mCurSize, mCurSpacing,
				 BACKGROUND_STRATEGY_PARTIAL_ZERO);
    copyArrayDeviceToDevice(phi0T, mdScratchV);
  }
#endif
}

void 
LDMMAdShootingIteratorGPU::finalUpdatePhiT0(DeformationDataType &data){
#if TRADE_MEM_FOR_SPEED
  // nothing to do, phiT0 still holds final deformation
#else
  // Update final phiT0
  cplVector3DArray &phiT0 = data.dPhiT0();
  cudaHField3DUtils::setToIdentity(phiT0, mCurSize);
  for(uint j=1; j<=mNTimeSteps; j++){	   
    cudaHField3DUtils::composeHVInv(mdScratchV, phiT0, mdVVec[j-1], 
				    mCurSize, mCurSpacing,
				    BACKGROUND_STRATEGY_PARTIAL_ID);
    copyArrayDeviceToDevice(phiT0, mdScratchV);
  }
#endif
}
