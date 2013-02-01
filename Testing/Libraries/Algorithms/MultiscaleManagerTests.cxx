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


#include "TestUtils.h"

#include "ImageUtils.h"
#include "ApplicationUtils.h"
#include "MultiscaleManager.h"

#ifdef CUDA_ENABLED
#include "cudaInterface.h"
#include "DiffOperGPU.h"
#include "CUDAUtilities.h"
#include "MultiscaleManagerGPU.h"
#endif // CUDA_ENABLED

int runTest1(){
  

  Vector3D<Real> spacingVec(1.0, 1.0, 1.0);
  Vector3D<unsigned int> sizeVec(65,65,65);
  Vector3D<Real> origin(0,0,0);

  // Create the test data
  RealImage testImOrig(sizeVec, // size
		       origin, // origin
		       spacingVec); // spacing
  TestUtils::GenBullseye(testImOrig, .8,.6,.2);

  MultiscaleManager manager(sizeVec, spacingVec, origin);
  manager.GenerateScaleLevels(3);

  RealImage *downsampledIm = manager.GenerateBaseLevelImage(&testImOrig);
  RealImage *upsampledIm = manager.GenerateBaseLevelImage();
  *upsampledIm = *downsampledIm;
  VectorField *upsampledField = manager.GenerateBaseLevelVectorField();
  TestUtils::GenDilation(*upsampledField, 1.0);

  char fname[1024];
  for(unsigned int i=0;i<manager.NumberOfScaleLevels();i++){
    sprintf(fname,"Scale%02dUpsampledIm.mha",i);
    ApplicationUtils::SaveImageITK(fname,*upsampledIm);
    sprintf(fname,"Scale%02dDownsampledIm.mha",i);
    ApplicationUtils::SaveImageITK(fname,*downsampledIm);
    sprintf(fname,"Scale%02dupsampledField.mha",i);
    ApplicationUtils::SaveHFieldITK(fname,*upsampledField);
    if(i < manager.NumberOfScaleLevels()-1){
      manager.NextScaleLevel();
    }
  }

  return TEST_PASS;
  
}

#ifdef CUDA_ENABLED

// Comparison of results of up/downsampling from CPU/GPU versions

int runTest2(){
  // set up GPU
  cudaSetDevice(0);
  
  for(Real spacing = 1.0; spacing <= 2.0; spacing += 1.0){
    std::cout << "Testing spacing " << spacing << std::endl;

    // create the test image
    SizeType sizeVec(32,32,32);
    OriginType originVec(0,0,0);
    SpacingType spacingVec(spacing, spacing, spacing);
    unsigned int nVox = sizeVec.productOfElements();
    unsigned int memSize = nVox * sizeof(float);
    
    // create images
    RealImage I(sizeVec, // size
		originVec,
		spacingVec); // spacing
    TestUtils::GenBullseye(I, .8,.6,.2);
    float *dI;
    allocateDeviceArray((void**)&dI, memSize);
    copyArrayToDevice(dI, I.getDataPointer(), nVox);
    
    // create the scale managers
    MultiscaleManager scaleManagerCPU(sizeVec, spacingVec, originVec);
    scaleManagerCPU.AddScaleLevel(1);
    scaleManagerCPU.AddScaleLevel(2);

    // create the scale managers
    MultiscaleManagerGPU scaleManagerGPU(sizeVec, spacingVec, originVec);
    scaleManagerGPU.AddScaleLevel(1); // scale level 1
    scaleManagerGPU.AddScaleLevel(2); // scale level 0

    // test image downsample
    SizeType curSize = scaleManagerCPU.CurScaleSize();
    SpacingType curSpacing = scaleManagerCPU.CurScaleSpacing();
    unsigned int curVox = curSize.productOfElements();
    RealImage IDownCPU(curSize, originVec, curSpacing);
    RealImage IDownGPU(curSize, originVec, curSpacing);

    float *dIDown, *dIUp;
    allocateDeviceArray((void**)&dIDown, memSize);
    allocateDeviceArray((void**)&dIUp, memSize);
    
    scaleManagerCPU.DownsampleToLevel(I, scaleManagerCPU.CurScaleLevel(), IDownCPU);
    scaleManagerGPU.DownsampleToLevel(&I, scaleManagerGPU.CurScaleLevel(), dIDown);
    
    copyArrayFromDevice(IDownGPU.getDataPointer(), dIDown, curVox);

    if(!TestUtils::Test(IDownGPU, IDownCPU, 0.01, "Downsample_CPU_GPU.mha")){
      std::cout << "Error, GPU/CPU downsample results failed" << std::endl;
      return TEST_FAIL;
    }else{
      std::cout << "Passed Downsample GPU/CPU test" << std::endl;
    }

    // test image upsample
    RealImage IUpCPU(sizeVec, originVec, spacingVec);
    RealImage IUpGPU(sizeVec, originVec, spacingVec);

    IUpCPU = IDownCPU;
    scaleManagerCPU.UpsampleToLevel(IUpCPU, 1);

    copyArrayDeviceToDevice(dIUp, dIDown, curVox);
    scaleManagerGPU.UpsampleToLevel(dIUp, 0, 1);
    copyArrayFromDevice(IUpGPU.getDataPointer(), dIUp, nVox);

    if(!TestUtils::Test(IUpGPU, IUpCPU, 0.01, "Upsample_CPU_GPU.mha")){
      std::cout << "Error, GPU/CPU upsample results failed" << std::endl;
      return TEST_FAIL;
    }else{
      std::cout << "Passed Upsample GPU/CPU test" << std::endl;
    }

    // test hfield upsample
    VectorField vf(curSize);
    Array3DUtils::computeGradient(IDownCPU, vf, curSpacing, true);
    cplVector3DArray dVf;
    allocateDeviceVector3DArray(dVf, sizeVec.productOfElements());
    CUDAUtilities::CopyVectorFieldToDevice(vf, dVf, true);
    
    VectorField vfUpCPU(curSize);
    vfUpCPU = vf;
    VectorField vfUpGPU(curSize);
    scaleManagerCPU.UpsampleToLevel(vfUpCPU, 1);
    scaleManagerGPU.UpsampleToLevel(dVf, 0, 1);
    CUDAUtilities::CopyVectorFieldFromDevice(dVf, vfUpGPU, true);

    if(!TestUtils::Test(vfUpGPU, vfUpCPU, 0.01, "Upsample_CPU_GPU_VF.mha")){
      std::cout << "Error, GPU/CPU upsample vector field results failed" << std::endl;
      return TEST_FAIL;
    }else{
      std::cout << "Passed Upsample GPU/CPU vector field test" << std::endl;
    }

  }

  return TEST_PASS;
  
}
#endif // CUDA_ENABLED

int main(int argc, char *argv[]){
  if(runTest1() != TEST_PASS){
    return -1;
  }
#ifdef CUDA_ENABLED
  if(runTest2() != TEST_PASS){
    return -1;
  }
#endif // CUDA_ENABLED
  return 0;
}
