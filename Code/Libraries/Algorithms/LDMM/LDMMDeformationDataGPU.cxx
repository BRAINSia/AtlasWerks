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


#include "LDMMDeformationDataGPU.h"

LDMMDeformationDataGPU::
LDMMDeformationDataGPU(const RealImage *I0, 
		       const RealImage *I1,
		       const LDMMParam &param)
  : LDMMDeformationData(I0, I1, param),
    mDeviceDataInitialized(false),
    mdV(mNTimeSteps)
#if USE_LV
  ,mdLV(mNTimeSteps)
#endif
{
  mNVox = mImSize.productOfElements();
  mCurVox = 0;
}

LDMMDeformationDataGPU::
~LDMMDeformationDataGPU()
{
}

void 
LDMMDeformationDataGPU::
InitializeDeviceData()
{
  
  LOGNODETHREAD(logDEBUG) << "Initializing device data";
  
  unsigned int memSize = mNVox*sizeof(Real);
  
  allocateDeviceArray((void**)&mdI0Orig, memSize);
  allocateDeviceArray((void**)&mdI1Orig, memSize);
  
  LOGNODETHREAD(logDEBUG) << "Copying original images to device (mNVox: " << mNVox << ")";

  copyArrayToDevice<float>(mdI0Orig, mI0Orig->getDataPointer(), mNVox);
  copyArrayToDevice<float>(mdI1Orig, mI1Orig->getDataPointer(), mNVox);

  // TEST
  // LOGNODETHREAD(logDEBUG) << "Host I1Orig squared sum: " << Array3DUtils::sumOfSquaredElements(*mI1Orig);
  // std::string fname = StringUtils::strPrintf("%sInitDeviceDataImage.mha", this->GetName().c_str());
  // CUDAUtilities::SaveDeviceImage(fname.c_str(), mdI1Orig, mImSize, mImOrigin, mImSpacing);
  // fname = StringUtils::strPrintf("%sInitDeviceDataHostImage.mha", this->GetName().c_str());
  // ApplicationUtils::SaveImageITK(fname.c_str(), *mI1Orig);
  // END TEST

  mdI0Ptr = mdI0Orig;
  mdI1Ptr = mdI1Orig;

  allocateDeviceArray((void**)&mdI0Scaled, memSize);
  allocateDeviceArray((void**)&mdI1Scaled, memSize);

  for(int t=0;t<(int)mNTimeSteps;t++){
      allocateDeviceVector3DArray(mdV[t], mNVox);
#if USE_LV
      allocateDeviceVector3DArray(mdLV[t], mNVox);
#endif
  }

  mDeviceDataInitialized = true;

  checkCUDAError("LDMMDeformationDataGPU::InitDeviceData");
  printCUDAMemoryUsage();
  LOGNODETHREAD(logDEBUG) << "Finished Initializing device data";

}

void 
LDMMDeformationDataGPU::
FreeDeviceData()
{

  cudaSafeDelete(mdI0Orig);
  cudaSafeDelete(mdI0Scaled);

  for(unsigned int t=0;t<mNTimeSteps;t++){
    freeDeviceVector3DArray(mdV[t]);
#if USE_LV
    freeDeviceVector3DArray(mdLV[t]);
#endif
  }

  mDeviceDataInitialized = false;

}

void 
LDMMDeformationDataGPU::
InitializeWarp()
{
  this->InitializeDeviceData();
}

void 
LDMMDeformationDataGPU::
FinalizeWarp()
{
  
  for(unsigned int t=0; t<mNTimeSteps;t++){
      CUDAUtilities::CopyVectorFieldFromDevice(mdV[t], *mV[t], true);
#if USE_LV
      CUDAUtilities::CopyVectorFieldFromDevice(mdLV[t], *mLV[t], true);
#endif
  }
  this->FreeDeviceData();
}

void 
LDMMDeformationDataGPU::
SetScaleLevel(const MultiscaleManager &scaleManager)
{
  
  LOGNODETHREAD(logDEBUG) << "Deformation data setting scale level";

  if(!scaleManager.InitialScaleLevel()){
    // copy host data to host
    for(unsigned int t=0; t<mNTimeSteps;t++){
        CUDAUtilities::CopyVectorFieldFromDevice(mdV[t], *mV[t], true);
#if USE_LV
        CUDAUtilities::CopyVectorFieldFromDevice(mdLV[t], *mLV[t], true);
#endif
    }
  }

  // do up/downsampling from superclass
  LDMMDeformationData::SetScaleLevel(scaleManager);

  mCurVox = mCurSize.productOfElements();
  
  // copy host data back to device
  for(unsigned int t=0; t<mNTimeSteps;t++){
    CUDAUtilities::CopyVectorFieldToDevice(*mV[t], mdV[t], true);
#if USE_LV
    CUDAUtilities::CopyVectorFieldToDevice(*mLV[t], mdLV[t], true);
#endif
  }

  // set pointers correctly
  if(scaleManager.FinalScaleLevel()){
    // Use original image
    mdI0Ptr = mdI0Orig;
    mdI1Ptr = mdI1Orig;
  }else{
    if(mScaleI0){
      mdI0Ptr = mdI0Scaled;
    }
    if(mScaleI1){
      mdI1Ptr = mdI1Scaled;
    }
    // copy images to device
    copyArrayToDevice<float>(mdI0Ptr, mI0Ptr->getDataPointer(), mCurVox);
    copyArrayToDevice<float>(mdI1Ptr, mI1Ptr->getDataPointer(), mCurVox);
  }

}

  // TEST
void
LDMMDeformationDataGPU::
GetI0(RealImage &im)
{
  im.resize(mCurSize);
  im.setOrigin(mImOrigin);
  im.setSpacing(mCurSpacing);
  copyArrayFromDevice(im.getDataPointer(), mdI0Ptr, mCurVox);
  if(mdI0Ptr == mdI0Orig){
    LOGNODETHREAD(logDEBUG) << "GetI0: I0 is currently original data";
  }else if(mdI0Ptr == mdI0Scaled){
    LOGNODETHREAD(logDEBUG) << "GetI0: I0 is currently scaled data";
  }else{
    LOGNODETHREAD(logDEBUG) << "GetI0: I0 is currently external pointer";
  }
}
void
LDMMDeformationDataGPU::
GetI1(RealImage &im)
{
  im.resize(mCurSize);
  im.setOrigin(mImOrigin);
  im.setSpacing(mCurSpacing);
  copyArrayFromDevice(im.getDataPointer(), mdI1Ptr, mCurVox);
  if(mdI1Ptr == mdI1Orig){
    LOGNODETHREAD(logDEBUG) << "GetI1: I1 is currently original data";
  }else if(mdI1Ptr == mdI1Scaled){
    LOGNODETHREAD(logDEBUG) << "GetI1: I1 is currently scaled data";
  }else{
    LOGNODETHREAD(logDEBUG) << "GetI1: I1 is currently external pointer";
  }
}
  // END TEST
void 
LDMMDeformationDataGPU::
GetDef0ToT(VectorField &h, unsigned int tIdx)
{
  if(mDeviceDataInitialized){

    cplVector3DArray dH, dScratchV;
    allocateDeviceVector3DArray(dH, mCurVox);
    allocateDeviceVector3DArray(dScratchV, mCurVox);
    
    GetDef0ToT(dH, dScratchV, tIdx);
    
    h.resize(mCurSize);
    CUDAUtilities::CopyVectorFieldFromDevice(dH, h, true);
    
    freeDeviceVector3DArray(dH);
    freeDeviceVector3DArray(dScratchV);

  }else{
    LDMMDeformationData::GetDef0ToT(h, tIdx);
  }

}

void 
LDMMDeformationDataGPU::
GetDefTTo0(VectorField &h, unsigned int tIdx)
{
  if(mDeviceDataInitialized){

    cplVector3DArray dH, dScratchV;
    allocateDeviceVector3DArray(dH, mCurVox);
    allocateDeviceVector3DArray(dScratchV, mCurVox);
    
    GetDefTTo0(dH, dScratchV, tIdx);
    
    h.resize(mCurSize);
    CUDAUtilities::CopyVectorFieldFromDevice(dH, h, true);
    
    freeDeviceVector3DArray(dH);
    freeDeviceVector3DArray(dScratchV);

  }else{
    LDMMDeformationData::GetDefTTo0(h, tIdx);
  }
  
}

void 
LDMMDeformationDataGPU::
GetDef1ToT(VectorField &h, unsigned int tIdx)
{
  if(mDeviceDataInitialized){
    
    cplVector3DArray dH, dScratchV;
    allocateDeviceVector3DArray(dH, mCurVox);
    allocateDeviceVector3DArray(dScratchV, mCurVox);
    
    GetDef1ToT(dH, dScratchV, tIdx);
    
    h.resize(mCurSize);
    CUDAUtilities::CopyVectorFieldFromDevice(dH, h, true);
    
    freeDeviceVector3DArray(dH);
    freeDeviceVector3DArray(dScratchV);
    
  }else{
    LDMMDeformationData::GetDef1ToT(h, tIdx);
  }
  
}

void 
LDMMDeformationDataGPU::
GetDefTTo1(VectorField &h, unsigned int tIdx)
{
  if(mDeviceDataInitialized){

    cplVector3DArray dH, dScratchV;
    allocateDeviceVector3DArray(dH, mCurVox);
    allocateDeviceVector3DArray(dScratchV, mCurVox);
    
    GetDefTTo1(dH, dScratchV, tIdx);
    
    h.resize(mCurSize);
    CUDAUtilities::CopyVectorFieldFromDevice(dH, h, true);
    
    freeDeviceVector3DArray(dH);
    freeDeviceVector3DArray(dScratchV);

  }else{
    LDMMDeformationData::GetDefTTo1(h, tIdx);
  }

}

void 
LDMMDeformationDataGPU::
GetI0AtT(RealImage& iDef, unsigned int tIdx)
{
  if(mDeviceDataInitialized){

    cplVector3DArray dH, dScratchV;
    allocateDeviceVector3DArray(dH, mCurVox);
    allocateDeviceVector3DArray(dScratchV, mCurVox);
    
    GetDefTTo0(dH, dScratchV, tIdx);
    
    float *dIDef;
    allocateDeviceArray((void**)&dIDef, mCurVox*sizeof(Real));

    cudaHField3DUtils::apply(dIDef, mdI0Ptr, dH, mCurSize);

    iDef.resize(mCurSize);
    copyArrayFromDevice<float>(iDef.getDataPointer(), dIDef, mCurVox);
    

    cudaSafeDelete(dIDef);
    freeDeviceVector3DArray(dH);
    freeDeviceVector3DArray(dScratchV);

  }else{
    LDMMDeformationData::GetI0AtT(iDef, tIdx);
  }
}

void 
LDMMDeformationDataGPU::
GetI1AtT(RealImage& iDef, unsigned int tIdx)
{
  if(mDeviceDataInitialized){
    
    cplVector3DArray dH, dScratchV;
    allocateDeviceVector3DArray(dH, mCurVox);
    allocateDeviceVector3DArray(dScratchV, mCurVox);
    
    GetDefTTo1(dH, dScratchV, tIdx);
    
    float *dIDef;
    allocateDeviceArray((void**)&dIDef, mCurVox*sizeof(Real));
    cudaHField3DUtils::apply(dIDef, mdI1Ptr, dH, mCurSize);
    
    iDef.resize(mCurSize);
    copyArrayFromDevice<float>(iDef.getDataPointer(), dIDef, mCurVox);
    
    cudaSafeDelete(dIDef);
    freeDeviceVector3DArray(dH);
    freeDeviceVector3DArray(dScratchV);

  }else{
    LDMMDeformationData::GetI1AtT(iDef, tIdx);
  }
}

void 
LDMMDeformationDataGPU::
GetDef0ToT(cplVector3DArray &dH, cplVector3DArray &dScratchV, unsigned int tIdx)
{
  cudaHField3DUtils::setToIdentity(dH, mCurSize);
  for(unsigned int t = 0; t < tIdx; t++){
    cudaHField3DUtils::composeVH(dScratchV, mdV[t], dH, mCurSize, mCurSpacing, 
				 BACKGROUND_STRATEGY_PARTIAL_ZERO);
    swap(dH, dScratchV);
  }
}

void 
LDMMDeformationDataGPU::
GetDefTTo0(cplVector3DArray &dH, cplVector3DArray &dScratchV, unsigned int tIdx)
{
  cudaHField3DUtils::setToIdentity(dH, mCurSize);
  for(unsigned int t = 0; t < mNTimeSteps; t++){
    cudaHField3DUtils::composeHVInv(dScratchV, dH, mdV[t], mCurSize, mCurSpacing, 
				    BACKGROUND_STRATEGY_PARTIAL_ID);
    swap(dH, dScratchV);
  }
}

void 
LDMMDeformationDataGPU::
GetDef1ToT(cplVector3DArray &dH, cplVector3DArray &dScratchV, unsigned int tIdx)
{
  cudaHField3DUtils::setToIdentity(dH, mCurSize);
  for(int t = mNTimeSteps-1; t >= (int)tIdx; t--){
    cudaHField3DUtils::composeVHInv(dScratchV, mdV[t], dH, mCurSize, mCurSpacing, 
				    BACKGROUND_STRATEGY_PARTIAL_ZERO);
    swap(dH, dScratchV);
  }
}

void 
LDMMDeformationDataGPU::
GetDefTTo1(cplVector3DArray &dH, cplVector3DArray &dScratchV, unsigned int tIdx)
{
  cudaHField3DUtils::setToIdentity(dH, mCurSize);
  for(int t = mNTimeSteps-1; t >= (int)tIdx; t--){
    cudaHField3DUtils::composeHV(dScratchV, dH, mdV[t], mCurSize, mCurSpacing, 
				 BACKGROUND_STRATEGY_PARTIAL_ID);

    swap(dH, dScratchV);
  }
}

void
LDMMDeformationDataGPU::
InterpV(cplVector3DArray &dV, Real tIdx)
{
  // previous integer time
  int tp = static_cast<int>(tIdx);
  Real frac = tIdx - static_cast<Real>(tp);
  if(tp < 0){
    LOGNODETHREAD(logWARNING) << "Time out of range for interp " 
			      << tp << ", from time " 
			      << tIdx
			      << std::endl;
    tp = 0;
    frac = 0.f;
  }

  // no vector field at final timepoint, so if between T-1 and T,
  // assume it stays the same as T-1
  if(tp >= (int)mNTimeSteps-1){
    if(tp >= (int)mNTimeSteps){
      LOGNODETHREAD(logWARNING) << "Time out of range for interp " 
				<< tp << ", from time " 
				<< tIdx
				<< std::endl;
    }
    tp = mNTimeSteps-1;
    frac = 0.f;
  }


  if(frac > 0.f){
    cplVector3DOpers::MulC_Add_MulC(dV, 
				    mdV[tp], 1.0-frac, 
				    mdV[tp+1], frac, mNVox);
  }else{
    copyArrayDeviceToDevice(dV, mdV[tp], mNVox);
  }

}
