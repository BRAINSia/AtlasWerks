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


#include "GreedyDeformationDataGPU.h"

GreedyDeformationDataGPU::
GreedyDeformationDataGPU(const RealImage *I0, 
			 const RealImage *I1,
			 const GreedyParam &param)
  : GreedyDeformationData(I0, I1, param),
    mDeviceDataInitialized(false)
{
}
  
GreedyDeformationDataGPU::
  ~GreedyDeformationDataGPU()
{
}
  
void 
GreedyDeformationDataGPU::
InitializeDeviceData()
{

  LOGNODETHREAD(logDEBUG) << "Initializing device data";

  unsigned int nVox = mI0Orig->getSize().productOfElements();
  unsigned int memSize = nVox*sizeof(Real);

  allocateDeviceArray((void**)&mdI0Orig, memSize);
  allocateDeviceArray((void**)&mdI1Orig, memSize);

  copyArrayToDevice<float>(mdI0Orig, mI0Orig->getDataPointer(), nVox);
  copyArrayToDevice<float>(mdI1Orig, mI1Orig->getDataPointer(), nVox);

  allocateDeviceArray((void**)&mdI0Scaled, memSize);
  allocateDeviceArray((void**)&mdI1Scaled, memSize);
  
  allocateDeviceVector3DArray(mdDef1To0, nVox);
  allocateDeviceVector3DArray(mdDef0To1, nVox);

  mDeviceDataInitialized = true;

  checkCUDAError("GreedyDeformationDataGPU::InitDeviceData");
  printCUDAMemoryUsage();
  LOGNODETHREAD(logDEBUG) << "Finished Initializing device data";

}

void 
GreedyDeformationDataGPU::
FreeDeviceData()
{

  cudaSafeDelete(mdI0Orig);
  cudaSafeDelete(mdI0Scaled);
  freeDeviceVector3DArray(mdDef1To0);
  freeDeviceVector3DArray(mdDef0To1);

  mDeviceDataInitialized = false;

}
void 
GreedyDeformationDataGPU::
InitializeWarp()
{
  this->InitializeDeviceData();
}
void 
GreedyDeformationDataGPU::
FinalizeWarp()
{
  CUDAUtilities::CopyVectorFieldFromDevice(mdDef1To0, *mDef1To0, true);
  if(mComputeInverseHField){
    CUDAUtilities::CopyVectorFieldFromDevice(mdDef0To1, *mDef0To1, true);
  }
  this->FreeDeviceData();
}

void 
GreedyDeformationDataGPU::
SetScaleLevel(const MultiscaleManager &scaleManager)
{

  //MultiscaleManagerGPU &scaleManagerGPU = dynamic_cast<MultiscaleManagerGPU&>(scaleManager);
  
  if(!scaleManager.InitialScaleLevel()){
    // copy device hfields back to host
    CUDAUtilities::CopyVectorFieldFromDevice(mdDef1To0, *mDef1To0, true);
    if(mComputeInverseHField){
      CUDAUtilities::CopyVectorFieldFromDevice(mdDef0To1, *mDef0To1, true);
    }
  }

  // do up/downsampling from superclass
  GreedyDeformationData::SetScaleLevel(scaleManager);
  
  unsigned int nVox = mCurSize.productOfElements();
  
  // copy host hfields back to device
  CUDAUtilities::CopyVectorFieldToDevice(*mDef1To0, mdDef1To0, true);
  if(mComputeInverseHField){
    CUDAUtilities::CopyVectorFieldToDevice(*mDef0To1, mdDef0To1, true);
  }

  // set pointers correctly
  if(scaleManager.FinalScaleLevel()){
    // Use original image
    mdI0Ptr = mdI0Orig;
    mdI1Ptr = mdI1Orig;
  }else{
    mdI0Ptr = mdI0Scaled;
    mdI1Ptr = mdI1Scaled;
    // copy images to device
    copyArrayToDevice<float>(mdI0Ptr, mI0Ptr->getDataPointer(), nVox);
    copyArrayToDevice<float>(mdI1Ptr, mI1Ptr->getDataPointer(), nVox);
  }
  
}
  
VectorField& 
GreedyDeformationDataGPU::
Def1To0()
{
  if(mDeviceDataInitialized){
    CUDAUtilities::CopyVectorFieldFromDevice(mdDef1To0, *mDef1To0, true);
  }
  return *mDef1To0;
}

VectorField&
GreedyDeformationDataGPU::
Def0To1()
{
  if(mDeviceDataInitialized && mComputeInverseHField){
    CUDAUtilities::CopyVectorFieldFromDevice(mdDef0To1, *mDef0To1, true);
  }
  return *mDef0To1;
}

void 
GreedyDeformationDataGPU::
GetI0At1(RealImage& iDef)
{
  if(mDeviceDataInitialized){
    CUDAUtilities::CopyVectorFieldFromDevice(mdDef1To0, *mDef1To0, true);
  }
  GreedyDeformationData::GetI0At1(iDef);
}

void 
GreedyDeformationDataGPU::
GetI1At0(RealImage& iDef)
{
  if(mDeviceDataInitialized && mComputeInverseHField){
    CUDAUtilities::CopyVectorFieldFromDevice(mdDef0To1, *mDef0To1, true);
  }
  GreedyDeformationData::GetI1At0(iDef);
}

