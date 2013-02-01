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


#include "LDMMShootingDefDataGPU.h"

LDMMShootingDefDataGPU::
LDMMShootingDefDataGPU(const RealImage *I0, 
		       const RealImage *I1,
		       const LDMMParam &param)
  : LDMMShootingDefData(I0, I1, param),
    mDeviceDataInitialized(false),
    mdI0Orig(NULL),
    mdI1Orig(NULL),
    mdI0Scaled(NULL),
    mdI1Scaled(NULL),
    mdI0Ptr(NULL),
    mdI1Ptr(NULL),
    mdAlpha0(NULL)
{
  mNVox = mImSize.productOfElements();
  mCurVox = 0;
}

LDMMShootingDefDataGPU::
~LDMMShootingDefDataGPU()
{
}

void 
LDMMShootingDefDataGPU::
InitializeDeviceData()
{
  
  LOGNODETHREAD(logDEBUG) << "Initializing device data";
  
  unsigned int memSize = mNVox*sizeof(Real);
  
  allocateDeviceArray((void**)&mdI0Orig, memSize);
  allocateDeviceArray((void**)&mdI1Orig, memSize);
  
  copyArrayToDevice<float>(mdI0Orig, mI0Orig->getDataPointer(), mNVox);
  copyArrayToDevice<float>(mdI1Orig, mI1Orig->getDataPointer(), mNVox);

  mdI0Ptr = mdI0Orig;
  mdI1Ptr = mdI1Orig;

  allocateDeviceArray((void**)&mdI0Scaled, memSize);
  allocateDeviceArray((void**)&mdI1Scaled, memSize);
  
  allocateDeviceVector3DArray(mdPhi0T, mNVox);
  allocateDeviceVector3DArray(mdPhiT0, mNVox);
  allocateDeviceArray((void**)&mdAlpha0, memSize);
  cplVectorOpers::SetMem(mdAlpha0, 0.f, mNVox);
  
  mDeviceDataInitialized = true;

  checkCUDAError("LDMMShootingDefDataGPU::InitDeviceData");
  printCUDAMemoryUsage();
  LOGNODETHREAD(logDEBUG) << "Finished Initializing device data";

}

void 
LDMMShootingDefDataGPU::
FreeDeviceData()
{

  LOGNODETHREAD(logDEBUG) << "Freeing memory from device data";

  cudaSafeDelete(mdI0Orig);
  cudaSafeDelete(mdI0Scaled);
  freeDeviceVector3DArray(mdPhi0T);
  freeDeviceVector3DArray(mdPhiT0);
  cudaSafeDelete(mdAlpha0);

  mDeviceDataInitialized = false;

}

void 
LDMMShootingDefDataGPU::
InitializeWarp()
{
  this->InitializeDeviceData();
}

void 
LDMMShootingDefDataGPU::
FinalizeWarp()
{
  CUDAUtilities::CopyVectorFieldFromDevice(mdPhiT0, *mPhiT0, true);
  CUDAUtilities::CopyVectorFieldFromDevice(mdPhi0T, *mPhi0T, true);
  copyArrayFromDevice<float>(mAlpha0->getDataPointer(), mdAlpha0, mCurVox);
  this->FreeDeviceData();
}

void 
LDMMShootingDefDataGPU::
SetScaleLevel(const MultiscaleManager &scaleManager)
{
  
  if(!scaleManager.InitialScaleLevel()){
    // copy device data back to host
    copyArrayFromDevice<float>(mAlpha0->getDataPointer(), mdAlpha0, mCurVox);
  }

  // do up/downsampling from superclass
  LDMMShootingDefData::SetScaleLevel(scaleManager);

  mCurVox = mCurSize.productOfElements();
  
  // copy host data back to device
  copyArrayToDevice<float>(mdAlpha0, mAlpha0->getDataPointer(), mCurVox);
  CUDAUtilities::CopyVectorFieldToDevice(*mPhi0T, mdPhi0T, true);
  CUDAUtilities::CopyVectorFieldToDevice(*mPhiT0, mdPhiT0, true);

  // TEST
  float alphaSum = CUDAUtilities::DeviceImageSum(mdAlpha0, mCurVox);
  LOGNODETHREAD(logDEBUG2) << "alpha0 sum is " << alphaSum << " after after SetScaleLevel()";
  // END TEST

  // NOTE: while alpha0 is updated correctly, phi0T and phiT0 have not been!
  
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
  
VectorField& 
LDMMShootingDefDataGPU::
PhiT0()
{
  if(mDeviceDataInitialized){
    CUDAUtilities::CopyVectorFieldFromDevice(mdPhiT0, *mPhiT0, true);
  }
  return *mPhiT0;
}

VectorField& 
LDMMShootingDefDataGPU::
Phi0T()
{
  if(mDeviceDataInitialized){
    CUDAUtilities::CopyVectorFieldFromDevice(mdPhi0T, *mPhi0T, true);
  }
  return *mPhi0T;
}

RealImage&
LDMMShootingDefDataGPU::
Alpha0()
{
  if(mDeviceDataInitialized){
    copyArrayFromDevice<float>(mAlpha0->getDataPointer(), mdAlpha0, mCurVox);
  }
  return *mAlpha0;
}

void 
LDMMShootingDefDataGPU::
GetI0At1(RealImage& iDef)
{
  if(mDeviceDataInitialized){
    CUDAUtilities::CopyVectorFieldFromDevice(mdPhiT0, *mPhiT0, true);
  }
  LDMMShootingDefData::GetI0At1(iDef);
}

void 
LDMMShootingDefDataGPU::
GetI1At0(RealImage& iDef)
{
  if(mDeviceDataInitialized){
    CUDAUtilities::CopyVectorFieldFromDevice(mdPhi0T, *mPhi0T, true);
  }
  LDMMShootingDefData::GetI1At0(iDef);
}


