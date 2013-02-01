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


#include "LDMMIteratorGPU.h"
#include "KernelFactory.h"
#include <cudaHField3DUtils.h>
#include <cudaImage3D.h>
#include <cudaUpsample3D.h>
#include <cudaSplat.h>
#include <FileIO.h>
#include <cutil_comfunc.h>
#include "log.h"
#include "CUDAUtilities.h"

#include <cutil_inline.h>
LDMMIteratorGPU::
LDMMIteratorGPU(SizeType &size, 
                OriginType &origin,
                SpacingType &spacing,
                unsigned int &nTimeSteps,
                bool debug)
  : DeformationIteratorInterface(size, origin, spacing),
    mNTimeSteps(nTimeSteps),
    mDebug(debug),
    mUpdateStepSizeNextIter(false),
    mUseOrigGradientCalc(true),
    mdJ0t(NULL),
    mdDiffIm(NULL),
    mdJTt(NULL),
    mdJacDet(NULL),
    mRd(NULL),
    mKernel(NULL),
    mdScratchI(NULL)
{
    mRd        = new cplReduce();
  
    int memSize = mNVox * sizeof(float);
    
    if(mUseOrigGradientCalc){
      // allocate forward-deformed images
      mdJTt    = new float* [mNTimeSteps];
      mdJacDet    = new float* [mNTimeSteps];
      for (int i=0; i< (int)mNTimeSteps; ++i){
        allocateDeviceArray((void**)&mdJTt[i], memSize);
        allocateDeviceArray((void**)&mdJacDet[i], memSize);
      }
    }else{
      // allocate forward-deformed images
      mdJ0t    = new float* [mNTimeSteps];
      for (int i=0; i< (int)mNTimeSteps; ++i){
        allocateDeviceArray((void**)&mdJ0t[i], memSize);
      }
      allocateDeviceArray((void**)&mdDiffIm, memSize);
    }
  
  
    // allocate scratch memory
    allocateDeviceArray((void**)&mdScratchI, memSize);
    allocateDeviceVector3DArray(mdScratchV, mNVox);
  
    allocateDeviceVector3DArray(mdU, mNVox);
    
    allocateDeviceVector3DArray(mdH, mNVox);
  
    allocateDeviceVector3DArray(mdGX, mNVox);
    allocateDeviceVector3DArray(mdGY, mNVox);
    allocateDeviceVector3DArray(mdGZ, mNVox);
  
    CUDAUtilities::CheckCUDAError(__FILE__,__LINE__);
  
}


LDMMIteratorGPU::
~LDMMIteratorGPU()
{
    delete mRd;
    delete mKernel;
  
    if(mUseOrigGradientCalc){
      for (int i=0; i< (int)mNTimeSteps; ++i){
        cudaSafeDelete(mdJTt[i]);
        cudaSafeDelete(mdJacDet[i]);
      }
    }else{
      for (int i=0; i< (int)mNTimeSteps; ++i){
        cudaSafeDelete(mdJ0t[i]);
      }
      cudaSafeDelete(mdDiffIm);
    }
    
    cudaSafeDelete(mdScratchI);
    freeDeviceVector3DArray(mdScratchV);
    freeDeviceVector3DArray(mdU);
    
    freeDeviceVector3DArray(mdH);

    freeDeviceVector3DArray(mdGX);
    freeDeviceVector3DArray(mdGY);
    freeDeviceVector3DArray(mdGZ);
}

void 
LDMMIteratorGPU::
SetScaleLevel(const MultiscaleManager &scaleManager,
	      const LDMMIteratorParam &param)
{
  ScaleLevelDataInfo::SetScaleLevel(scaleManager);
  
  mParam = &param;
  
  mUseAdaptiveStepSize = mParam->UseAdaptiveStepSize();
  mMaxPert = mParam->MaxPert();
  mSigma = mParam->Sigma();
  
  if(!mKernel) mKernel = KernelFactory::NewGPUKernel(mParam->Kernel());
  mKernel->SetSize(mCurSize, mCurSpacing, mParam->Kernel());
  
  if(mUseAdaptiveStepSize){
    LOGNODETHREAD(logDEBUG1) << "Using adaptive step size";
  }else{
    LOGNODETHREAD(logDEBUG1) <<  "Not using adaptive step size";
  }
  
  LOGNODETHREAD(logDEBUG1) << StringUtils::strPrintf("%f sigma", mSigma);
}

void 
LDMMIteratorGPU::
Iterate(DeformationDataInterface &deformationData)
{
    this->Iterate(deformationData, false);
}

void 
LDMMIteratorGPU::
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
LDMMIteratorGPU::
IterateNew(DeformationDataInterface &deformationData,
	   bool computeAlphasOnly)
{

    LDMMDeformationDataGPU &data = dynamic_cast<LDMMDeformationDataGPU&>(deformationData);

    LDMMEnergy energy(mNTimeSteps, mSigma);
    if(mParam->VerboseEnergy()){ energy.Verbose(true); }

    Real voxelVol = mCurSpacing.productOfElements();

    // when calculating step size, this holds max displacement across all velocity fields
    Real totalMaxDisplacement = 0.f;

    // Compute deformed images from source to dest
    cudaHField3DUtils::setToIdentity(mdH, mCurSize);
    for(unsigned int t = 0; t < mNTimeSteps; t++){
        cudaHField3DUtils::composeHVInv(mdScratchV, mdH, data.dV(t), mCurSize, mCurSpacing, 
                                        BACKGROUND_STRATEGY_PARTIAL_ID);
        swap(mdH, mdScratchV);

        cudaHField3DUtils::apply(mdJ0t[t], data.dI0(), mdH, mCurSize);
    }
  
    // compute difference image
    cplVectorOpers::Sub(mdDiffIm, mdJ0t[mNTimeSteps-1], data.dI1(), mCurVox);
  
    // update each velocity field from dest to source
    cudaHField3DUtils::setToIdentity(mdH, mCurSize);
    for(int t = mNTimeSteps-1; t >= 0; t--){
        cudaHField3DUtils::composeVHInv(mdScratchV, data.dV(t), mdH, mCurSize, mCurSpacing, 
                                        BACKGROUND_STRATEGY_PARTIAL_ZERO);
        swap(mdH, mdScratchV);

        // splat the difference image from dest to this timepoint
        cplSplat3DH(mdScratchI, mdDiffIm, mdH, mCurSize);

        if(computeAlphasOnly){
            // scale alphas appropriately, store in mdJ0t variable
            cplVectorOpers::MulC(mdJ0t[t], mdScratchI, ((float)1.0/(mSigma*mSigma)), mCurVox);
            // copy alphas to host (or only alpha0, if that's all we want)
            if(data.ComputeAlphas() || 
               (data.ComputeAlpha0() && t == 0))
            {
                copyArrayFromDevice(data.Alpha(t).getDataPointer(), mdJ0t[t], mCurVox);
            }
        }else{ // continue computing update...

            // compute gradient of forward-deformed image
            if(t > 0){
                cplComputeGradient(mdScratchV, mdJ0t[t-1],
                                   mCurSize.x, mCurSize.y, mCurSize.z,
                                   mCurSpacing.x, mCurSpacing.y, mCurSpacing.z);
            }else{
                cplComputeGradient(mdScratchV, data.dI0(),
                                   mCurSize.x, mCurSize.y, mCurSize.z,
                                   mCurSpacing.x, mCurSpacing.y, mCurSpacing.z);
            }
	    
            // create force by scaling gradient
            cplVector3DOpers::Mul_I(mdScratchV, mdScratchI, mCurVox); 

            // solve for the update direction
            copyArrayDeviceToDevice(mdU, mdScratchV);

#if USE_LV
                // update LV base on the step size
            cplVector3DOpers::MulC_Add_MulC_I(data.dLV(t), 1.f - 2 * data.StepSize(), mdScratchV, 2  * data.StepSize() / (mSigma * mSigma), mCurVox);
#endif

            mKernel->ApplyInverseOperator(mdU);

            if(mUpdateStepSizeNextIter){
                // update velocity fields based on step size
                cplVector3DOpers::MulC_Add_MulC_I(mdU, -2.0 / (mSigma * mSigma), data.dV(t), 2, mCurVox);
                cplVector3DOpers::Magnitude(mdScratchI, mdU, mCurVox);
                Real maxDisplacement = mRd->Max(mdScratchI, mCurVox);
		if(maxDisplacement != maxDisplacement){
		  throw AtlasWerksException(__FILE__, __LINE__, 
					    "Error, max displacement is NaN");
		}
                if(maxDisplacement > totalMaxDisplacement){
                    totalMaxDisplacement = maxDisplacement;
                }
            }else{

                // update velocity fields based on step size
                cplVector3DOpers::MulC_Add_MulC_I(data.dV(t), 1.f - 2 * data.StepSize(), mdU, 2  * data.StepSize() / (mSigma * mSigma), mCurVox);
                
                if(mDebug){
                    // compute vector energy = <Lv,v>
                    copyArrayDeviceToDevice(mdScratchV, data.dV(t), mCurVox);
                    mKernel->ApplyOperator(mdScratchV);
                    Real vecEnergy = 0.f;
                    vecEnergy += voxelVol*mRd->Dot(mdScratchV.x, data.dV(t).x, mCurVox);
                    vecEnergy += voxelVol*mRd->Dot(mdScratchV.y, data.dV(t).y, mCurVox);
                    vecEnergy += voxelVol*mRd->Dot(mdScratchV.z, data.dV(t).z, mCurVox);
                    //energy.SetVecStepEnergy(vecEnergy);
#if USE_LV
                    Real estVecEnergy = 0.f;
                    estVecEnergy += voxelVol*mRd->Dot(data.dLV(t).x, data.dV(t).x, mCurVox);
                    estVecEnergy += voxelVol*mRd->Dot(data.dLV(t).y, data.dV(t).y, mCurVox);
                    estVecEnergy += voxelVol*mRd->Dot(data.dLV(t).x, data.dV(t).z, mCurVox);
                    std::cerr << "Ref " << vecEnergy << " Est " << estVecEnergy << " Diff " << estVecEnergy - vecEnergy << std::endl;
                    energy.SetVecStepEnergy(estVecEnergy);
#else
                    energy.SetVecStepEnergy(vecEnergy);
#endif
                }

            } // end mUpdateStepSizeNextIter
        } // end computeAlphasOnly
    } // end iterate over timesteps

    if(mUpdateStepSizeNextIter){
      data.StepSize(mMaxPert / totalMaxDisplacement);
      LOGNODETHREAD(logDEBUG) << "Step size is " << data.StepSize();
      mUpdateStepSizeNextIter = false;
    }else{
        // calculate image energy
        if(mDebug){
            energy.SetImageEnergy(voxelVol*mRd->Sum2(mdDiffIm, mCurVox));
            // test for NaN
            if(energy.ImageEnergy() != energy.ImageEnergy()){
                throw AtlasWerksException(__FILE__, __LINE__, "Error, NaN encountered");
            }
      
            data.AddEnergy(energy);
        }
    }
  
    CUDAUtilities::CheckCUDAError(__FILE__,__LINE__);
}

void 
LDMMIteratorGPU::
IterateOrig(DeformationDataInterface &deformationData,
	    bool computeAlphasOnly)
{
  
  LDMMDeformationDataGPU &data = dynamic_cast<LDMMDeformationDataGPU&>(deformationData);
  
  LDMMEnergy energy(mNTimeSteps, mSigma);
  if(mParam->VerboseEnergy()){ energy.Verbose(true); }

  Real voxelVol = mCurSpacing.productOfElements();

  // when calculating step size, this holds max displacement across all velocity fields
  Real totalMaxDisplacement = 0.f;

  // compute mdJacDet[t] and mdJT[t] for all t
  cudaHField3DUtils::setToIdentity(mdH, mCurSize);
  for (int t = mNTimeSteps-1; t>=0; --t){
    // 1. Compute JTt
    // Compute the current deformation h(t) = h_(t+1)(x + vt);
    cudaHField3DUtils::composeHV(mdScratchV, mdH, data.dV(t),
				 mCurSize, mCurSpacing, BACKGROUND_STRATEGY_PARTIAL_ID);
    swap(mdH, mdScratchV);
	
    // deform the image base on the current deformation JTt= I(h(t))
    cudaHField3DUtils::apply(mdJTt[t], data.dI1(), mdH, mCurSize);
	
    // 2. Compute DPhiT
    // Compute determinent of jacobian of current deformation 
    // D(h_{t-1}) = ( D(h_t) |(x+ v_t)) * |D (x + v_t)|
    // Compute x + v_t
    cudaHField3DUtils::velocityToH_US(mdScratchV, data.dV(t), 
				      mCurSize, mCurSpacing);
	
    // Compute |D(x + v_t)|
    cudaHField3DUtils::jacobianDetHField(mdScratchI, mdScratchV,
					 mdGX, mdGY, mdGZ,
					 mCurSize, mCurSpacing);
	
    if (t == (int)mNTimeSteps - 1){
      copyArrayDeviceToDevice(mdJacDet[t], mdScratchI, mCurVox);
    }else {
      //D(h_t) |(x+v_t)
      cudaHField3DUtils::applyU(mdJacDet[t], mdJacDet[t+1], data.dV(t),
				mCurSize, mCurSpacing);
      cplVectorOpers::Mul_I(mdJacDet[t], mdScratchI, mCurVox); 
    }

  }
    
  // update each velocity field from source to dest
  cudaHField3DUtils::setToIdentity(mdH, mCurSize);
  for (unsigned int t=0; t< mNTimeSteps; ++t){
    // 3. Compute J0Tt
    // Compute the defomed image
    if (t==0){
      copyArrayDeviceToDevice(mdScratchI, data.dI0(), mCurVox);
    }
    else {
      cudaHField3DUtils::apply(mdScratchI, data.dI0(), mdH,
			       mCurSize);
    }
      
    // Compute the gradient of deformed image
    cplComputeGradient(mdScratchV, mdScratchI,
		       mCurSize.x, mCurSize.y, mCurSize.z,
		       mCurSpacing.x, mCurSpacing.y, mCurSpacing.z);
      
    cplVectorOpers::SubMul_I(mdScratchI, mdJTt[t], mdJacDet[t], mCurVox);

    if(computeAlphasOnly){
      // scale alphas appropriately
      cplVectorOpers::MulC_I(mdScratchI, ((float)1.0/(mSigma*mSigma)), mCurVox);
      // store alphas in mdJTt array
      copyArrayDeviceToDevice(mdJTt[t], mdScratchI, mCurVox);
    }else{
	
      // finish updating velocity fields
      cplVector3DOpers::Mul_I(mdScratchV, mdScratchI, mCurVox);

      copyArrayDeviceToDevice(mdU, mdScratchV);
      mKernel->ApplyInverseOperator(mdU);
	
      if(mUpdateStepSizeNextIter){
	// update velocity fields based on step size
	cplVector3DOpers::MulC_Add_MulC_I(mdU, -2.0 / (mSigma * mSigma), data.dV(t), 2, mCurVox);
	cplVector3DOpers::Magnitude(mdScratchI, mdU, mCurVox);
	Real maxDisplacement = mRd->Max(mdScratchI, mCurVox);
	if(maxDisplacement != maxDisplacement){
	  throw AtlasWerksException(__FILE__, __LINE__, 
				    "Error, max displacement is NaN");
	}
	if(maxDisplacement > totalMaxDisplacement){
	  totalMaxDisplacement = maxDisplacement;
	}
      }else {
	// update velocity fields based on step size
	cplVector3DOpers::MulC_Add_MulC_I(data.dV(t), 
					  1.f - 2 * data.StepSize(), 
					  mdU, 
					  2*data.StepSize()/(mSigma*mSigma), 
					  mCurVox);
      }
    } // end if compute only alphas
      
    cudaHField3DUtils::composeHVInv(mdScratchV, mdH, data.dV(t),
				    mCurSize, mCurSpacing, 
				    BACKGROUND_STRATEGY_PARTIAL_ID);
    swap(mdScratchV, mdH);

    if(mDebug){
      // compute vector energy = <Lv,v>
      copyArrayDeviceToDevice(mdScratchV, data.dV(t), mCurVox);
      mKernel->ApplyOperator(mdScratchV);
      Real vecEnergy = 0.f;
      vecEnergy += voxelVol*mRd->Dot(mdScratchV.x, data.dV(t).x, mCurVox);
      vecEnergy += voxelVol*mRd->Dot(mdScratchV.y, data.dV(t).y, mCurVox);
      vecEnergy += voxelVol*mRd->Dot(mdScratchV.z, data.dV(t).z, mCurVox);
      //energy.SetVecStepEnergy(vecEnergy);
#if USE_LV
      Real estVecEnergy = 0.f;
      estVecEnergy += voxelVol*mRd->Dot(data.dLV(t).x, data.dV(t).x, mCurVox);
      estVecEnergy += voxelVol*mRd->Dot(data.dLV(t).y, data.dV(t).y, mCurVox);
      estVecEnergy += voxelVol*mRd->Dot(data.dLV(t).x, data.dV(t).z, mCurVox);
      std::cerr << "Ref " << vecEnergy << " Est " << estVecEnergy << " Diff " << estVecEnergy - vecEnergy << std::endl;
      energy.SetVecStepEnergy(estVecEnergy);
#else
      energy.SetVecStepEnergy(vecEnergy);
#endif
    }
    
  } // end iterate over timesteps

  if(mUpdateStepSizeNextIter){
    data.StepSize(mMaxPert / totalMaxDisplacement);
    LOGNODETHREAD(logDEBUG) << "Step size is " << data.StepSize();
    mUpdateStepSizeNextIter = false;
  }else{
    if(mDebug){
      cudaHField3DUtils::apply(mdScratchI, data.dI0(), mdH,
			       mCurSize);
      cplVectorOpers::Sub_I(mdScratchI, data.dI1(), mCurVox);
      energy.SetImageEnergy(voxelVol*mRd->Sum2(mdScratchI, mCurVox));
      // test for NaN
      if(energy.ImageEnergy() != energy.ImageEnergy()){
	throw AtlasWerksException(__FILE__, __LINE__, "Error, NaN encountered");
      }
      data.AddEnergy(energy);
    }
  }
}

void
LDMMIteratorGPU::
ComputeJacDet(std::vector<cplVector3DArray> &dV, float *dJacDet)
{
    float *scratchI = mdDiffIm;
    if(mUseOrigGradientCalc){
      scratchI = mdJacDet[0];
    }

    for (int t = mNTimeSteps-1; t>=0; --t){

        // 2. Compute DPhiT
        // Compute determinent of jacobian of current deformation 
        // D(h_{t-1}) = ( D(h_t) |(x+ v_t)) * |D (x + v_t)|
        // Compute x + v_t
        cudaHField3DUtils::velocityToH_US(mdScratchV, dV[t], mCurSize, mCurSpacing);
    
        // Compute |D(x + v_t)|
        cudaHField3DUtils::jacobianDetHField(mdScratchI, mdScratchV,
                                             mdGX, mdGY, mdGZ,
                                             mCurSize, mCurSpacing);
	
        if (t == (int)mNTimeSteps - 1){
            copyArrayDeviceToDevice(dJacDet, mdScratchI, mCurVox);
        }else {
            cudaHField3DUtils::applyU(scratchI, dJacDet, dV[t],
                                      mCurSize, mCurSpacing);
            cplVectorOpers::Mul(dJacDet, scratchI, mdScratchI, mCurVox);
        }
    }
}

// void
// LDMMIteratorGPU::
// ComputeJacDet(std::vector<cplvVector3DArray> &dV, float *dJacDet)
// {
//   assert(dV.size() == mNTimeSteps);
//   // mdScratchV will be PhiT0
//   cudaHField3DUtils::setToIdentity(mdH, mCurSize);
//   for (unsigned int t = 0; t < mNTimeSteps; t++){
//     cudaHField3DUtils::composeHVInv(mdScratchV, mdH, dV[t], mCurSize, mCurSpacing,
// 				    BACKGROUND_STRATEGY_PARTIAL_ID);
//     copyArrayDeviceToDevice(mdH, mdScratchV, mCurVox);
//   }
//   // splat 'ones' image by PhiT0 to create jacobian determinant of Phi0T
//   cudaSetMem(mdScratchI, 1.f, mCurVox);
//   cplvSplatingHFieldAtomicSigned(dJacDet, mdScratchI, 
// 				 mdH.x, mdH.y, mdH.z,
// 				 mCurSize.x, mCurSize.y, mCurSize.z);
//}

void 
LDMMIteratorGPU::
ComputeForwardImage(std::vector<cplVector3DArray> &dV, float *dI0, cplVector3DArray &dH, float *dIDef)
{
    cudaHField3DUtils::setToIdentity(dH, mCurSize);
    for (unsigned int t = 0 ;t < mNTimeSteps; t++){
        cudaHField3DUtils::composeHVInv(mdScratchV, dH, dV[t],
                                        mCurSize, mCurSpacing, 
                                        BACKGROUND_STRATEGY_PARTIAL_ID);
    
        copyArrayDeviceToDevice(dH, mdScratchV, mCurVox);
    }
    cudaHField3DUtils::apply(dIDef, dI0, dH, mCurSize);
}

float*
LDMMIteratorGPU::
GetAlpha(unsigned int t)
{
  if(mUseOrigGradientCalc){
    return mdJTt[t];
  }else{
    return mdJ0t[t];
  }
}

void 
LDMMIteratorGPU::
ReParameterize(LDMMDeformationDataGPU &defData)
{
  std::vector<cplVector3DArray> vVec;
  std::vector<Real> lVec;

  Real voxelVol = mCurSpacing.productOfElements();

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
    while(base+1 < (int)lVec.size() && lVec[base+1] < time) ++base;
    time = ((Real)base) + (time-lVec[base])/(lVec[base+1]-lVec[base]);

    LOGNODETHREAD(logDEBUG2) << "Timestep " << t 
			     << " constant-speed reparameterized time is " 
			     << time << std::endl; 

    cplVector3DArray dV;
    allocateDeviceVector3DArray(dV, mNVox);
    vVec.push_back(dV);
    defData.InterpV(dV, time);

    copyArrayDeviceToDevice(mdScratchV, dV, mCurVox);
    mKernel->ApplyOperator(mdScratchV);

    Real scaleFac = 0.f;
    scaleFac += voxelVol*mRd->Dot(mdScratchV.x, dV.x, mCurVox);
    scaleFac += voxelVol*mRd->Dot(mdScratchV.y, dV.y, mCurVox);
    scaleFac += voxelVol*mRd->Dot(mdScratchV.z, dV.z, mCurVox);
    scaleFac = l/(mNTimeSteps*sqrt(scaleFac));
    cplVector3DOpers::MulC_I(dV, scaleFac, mCurVox);
    
    copyArrayDeviceToDevice(mdScratchV, dV, mCurVox);
    mKernel->ApplyOperator(mdScratchV);

    Real vecEnergy = 0.f;
    vecEnergy += voxelVol*mRd->Dot(mdScratchV.x, dV.x, mCurVox);
    vecEnergy += voxelVol*mRd->Dot(mdScratchV.y, dV.y, mCurVox);
    vecEnergy += voxelVol*mRd->Dot(mdScratchV.z, dV.z, mCurVox);

    energy.SetVecStepEnergy(vecEnergy);
  }

  // assign vector fields and clean up memory
  for(unsigned int t=0;t<mNTimeSteps;++t){
    copyArrayDeviceToDevice(defData.dV(t), vVec[t], mCurVox);
    freeDeviceVector3DArray(vVec[t]);
  }

  // calculate new image energy
  RealImage diffIm;
  defData.GetI0At1(diffIm);
  diffIm.pointwiseSubtract(defData.I1());
  diffIm.setSpacing(mCurSpacing);
  energy.SetImageEnergy(ImageUtils::l2NormSqr(diffIm));
  defData.GetEnergyHistory().AddEvent(ReparameterizeEvent(energy));

}

