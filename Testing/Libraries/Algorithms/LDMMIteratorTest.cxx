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



#include "LDMMIteratorCPU.h"
#include "LDMMIteratorGPU.h"
#include "TestUtils.h"
#include "CUDAUtilities.h"
#include "cudaHField3DUtils.h"

// test a single iteration between CPU and GPU

int runTests1(){

  // set up GPU
  cudaSetDevice(0);
  
  for(Real spacing = 1.0; spacing <= 2.0; spacing += 1.0){
    
    std::cout << "Testing spacing " << spacing << std::endl;
    
    SizeType sizeVec(32,32,32);
    OriginType originVec(0,0,0);
    SpacingType spacingVec(spacing, spacing, spacing);
    
    // create images
    RealImage I0(sizeVec, // size
		 originVec,
		 spacingVec); // spacing
    TestUtils::GenBullseye(I0, .8,.6,.2);
    RealImage I1(sizeVec, // size
		 originVec, // origin
		 spacingVec); // spacing
    TestUtils::GenBullseye(I1, .7,.5,.3);


    // create iterators
    unsigned int nTimeSteps=5;
    LDMMIteratorParam param;
    param.Kernel().SetParam("DiffOper");
    DiffOperParam *doParam = param.Kernel().AsDiffOperParam();
    doParam->UseEigenLUT()=false;
    param.UseAdaptiveStepSize()=false;
    
    MultiscaleManager scaleManager(sizeVec, spacingVec, originVec);
    scaleManager.AddScaleLevel(1);

    LDMMIteratorCPU cpuIter(sizeVec, originVec, spacingVec, nTimeSteps, true);
    cpuIter.SetScaleLevel(scaleManager, param);
    LDMMIteratorGPU gpuIter(sizeVec, originVec, spacingVec, nTimeSteps, true);
    gpuIter.SetScaleLevel(scaleManager, param);
    

    LDMMParam ldmmParam;
    ldmmParam.NTimeSteps() = nTimeSteps;
    ldmmParam.ScaleLevel().AddParsedParam(LDMMScaleLevelParam());
    // iterate CPU version
    std::cout << "Creating CPU iterator" << std::endl;
    LDMMDeformationData cpuDefData(&I0, &I1, ldmmParam);

    cpuDefData.SetScaleLevel(scaleManager);
    cpuDefData.StepSize(param.StepSize());
    
    // iterate GPU version
    checkCUDAError("LDMMIteratorTest :: before GPU setup");
    std::cout << "Creating GPU iterator" << std::endl;
    LDMMDeformationDataGPU gpuDefData(&I0, &I1, ldmmParam);
    gpuDefData.InitializeWarp();
    gpuDefData.SetScaleLevel(scaleManager);
    gpuDefData.StepSize(param.StepSize());

    unsigned int nVox = sizeVec.productOfElements();
    unsigned int memSize = nVox * sizeof(float);

    cplVector3DArray dH;
    allocateDeviceVector3DArray(dH, nVox);
    cplVector3DArray dScratchV;
    allocateDeviceVector3DArray(dScratchV, nVox);
    cplVector3DOpers::SetMem(dH, 0, nVox);
    printCUDAMemoryUsage();

    checkCUDAError("LDMMIteratorTest :: before iterate()");

    for(int i = 0; i < 20; i++){
      std::cout << "CPU iteration " << i << std::endl;
      cpuIter.Iterate(cpuDefData);
      std::cout << "GPU iteration " << i << std::endl;
      gpuIter.Iterate(gpuDefData);
    }
    checkCUDAError("LDMMIteratorTest :: after iterate()");

    // get the CPU deformed image
    std::cout << "Getting CPU deformed image" << std::endl;
    RealImage cpuDefIm(sizeVec, originVec, spacingVec);
    cpuDefData.GetI0At1(cpuDefIm);

    // get the GPU deformed image
    std::cout << "Getting GPU deformed image" << std::endl;
    RealImage gpuDefIm(sizeVec, originVec, spacingVec);
    gpuDefData.GetI0At1(gpuDefIm);

    if(!TestUtils::Test(gpuDefIm, cpuDefIm, 0.001, "LDMM_CPU_GPU_Iterator.mha")){
      std::cout << "Error, GPU/CPU results failed" << std::endl;
      return TEST_FAIL;
    }else{
      std::cout << "Passed LDMM GPU/CPU Iterator test" << std::endl;
    }

    // Test jacobian determinant calculation
//     for(unsigned int t=0;t<nTimeSteps;t++){
//       TestUtils::GenWavy(h, 0.25, 4.0);
//       cpuDefData.v(t) = h;
//       CUDAUtilities::CopyVectorFieldToDevice(h, dV[t], true);
//     }

    float *dIDef;
    allocateDeviceArray((void**)&dIDef, memSize);

    cpuIter.ComputeJacDet(cpuDefData.v(), &cpuDefIm);
    gpuIter.ComputeJacDet(gpuDefData.dV(), dIDef);
    copyArrayFromDevice(gpuDefIm.getDataPointer(), dIDef, nVox);
    if(!TestUtils::Test(gpuDefIm, cpuDefIm, 0.001, "LDMM_CPU_GPU_JacDet.mha")){
      std::cout << "Error, GPU/CPU Jac. Det. results failed" << std::endl;
      return TEST_FAIL;
    }else{
      std::cout << "Passed LDMM GPU/CPU Jac. Det. test" << std::endl;
    }
    
    // Test reverse image computation
    cpuDefData.GetI1At0(cpuDefIm);
    gpuDefData.GetI1At0(gpuDefIm);

    if(!TestUtils::Test(gpuDefIm, cpuDefIm, 0.001, "LDMM_CPU_GPU_RevDef.mha")){
      std::cout << "Error, LDMM GPU/CPU reverse def. test failed" << std::endl;
      return TEST_FAIL;
    }else{
      std::cout << "Passed LDMM GPU/CPU reverse def. test" << std::endl;
    }
    
  } // end loop over spacing

  return TEST_PASS;
    
}

int main(int argc, char *argv[]){
  if(runTests1() != TEST_PASS){
    return -1;
  }
}
