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

#include "cudaInterface.h"
#include "CUDAUtilities.h"

#include "cudaHField3DUtils.h"
#include "cudaSplat.h"
//#include "cudaComposition.h"
#include "HField3DUtils.h"
#include "ImageUtils.h"
#include <cudaComposition.h>

// test definition of adjoint (forwardApply/splatting) for cpu/gpu
int runTests1(){
  
  for(Real spacing = 1.0; spacing <= 2.0; spacing += 1.0){
    
    std::cout << "Testing spacing " << spacing << std::endl;
    
    Vector3D<Real> spacingVec(spacing, spacing, spacing);
    Vector3D<unsigned int> sizeVec(32,32,32);
    Real result;

    // create images
    RealImage I1(sizeVec, // size
		 Vector3D<Real>(0,0,0), // origin
		 spacingVec); // spacing
    TestUtils::GenBullseye(I1, .8,.6,.2);
    RealImage I2(sizeVec, // size
		 Vector3D<Real>(0,0,0), // origin
		 spacingVec); // spacing
    TestUtils::GenBullseye(I2, .7,.5,.3);
    // create some negative values
    I2.add(-0.5f);
    RealImage defIm(sizeVec, // size
		    Vector3D<Real>(0,0,0), // origin
		    spacingVec); // spacing
    RealImage scratchI(sizeVec, // size
		       Vector3D<Real>(0,0,0), // origin
		       spacingVec); // spacing
    
    // create hField
    VectorField h(sizeVec);
    TestUtils::GenWavy(h, 1.0, 4.0);
    HField3DUtils::velocityToH(h, spacingVec);
    
    // compute <T_h(I1),I2>
    HField3DUtils::apply(I1,h,defIm);
    Real dotProd = ImageUtils::l2DotProd(defIm, I2);

    // compute <I1, T_h^T(I2)>
    HField3DUtils::forwardApply(I2, h, defIm, 0.0f, false);
    result = ImageUtils::l2DotProd(I1, defIm);
    if(fabs(result-dotProd) > 0.1){
      std::cerr << "Error, adjoint test failed: " << result << " vs. " << dotProd << std::endl;
      return TEST_FAIL;
    }
    std::cout << "Passed CPU adjoint test: " << dotProd << " vs. " << result << std::endl;
    
    // GPU tests

    //HField3DUtils::hToVelocity(h, spacingVec);
    
    unsigned int nVox = sizeVec.productOfElements();
    unsigned int memSize = nVox*sizeof(Real);
    cplVector3DArray dH;
    allocateDeviceVector3DArray(dH, nVox);
    CUDAUtilities::CopyVectorFieldToDevice(h, dH);

    float *dI, *dDefIm;
    allocateDeviceArray((void**)&dI, memSize);
    allocateDeviceArray((void**)&dDefIm, memSize);

    copyArrayToDevice(dI, I2.getDataPointer(), nVox);

    cplSplat3DH(dDefIm, dI, dH, sizeVec);
    // cplSplatingHFieldAtomicSigned(dDefIm, dI, dH.x, dH.y, dH.z, 
    //                               sizeVec.x, sizeVec.y, sizeVec.z);
    
    copyArrayFromDevice(scratchI.getDataPointer(), dDefIm, nVox);
    if(!TestUtils::Test(scratchI, defIm, 0.1, "CPU_GPU_Splatting.mha")){
      std::cout << "Error, GPU/CPU results failed" << std::endl;
      return TEST_FAIL;
    }else{
      std::cout << "Passed GPU/CPU test" << std::endl;
    }
    
    
  } // end loop over spacing
  
  return TEST_PASS;
  
}


// test composition functions
int runTests2(){
  for(Real spacing = 1.0; spacing <= 2.0; spacing += 1.0){
    std::cout << "Testing spacing " << spacing << std::endl;
    
    Vector3D<Real> spacingVec(spacing, spacing, spacing);
    Vector3D<unsigned int> sizeVec(32,32,32);
    unsigned int nVox = sizeVec.productOfElements();

    // create images
    RealImage I1(sizeVec, // size
		 Vector3D<Real>(0,0,0), // origin
		 spacingVec); // spacing
    TestUtils::GenBullseye(I1, .8,.6,.2);

    // create hField h
    VectorField h(sizeVec);
    TestUtils::GenWavy(h, 1.0, 4.0);
    HField3DUtils::velocityToH(h, spacingVec);

    // create vField from gradient
    VectorField v(sizeVec);
    Array3DUtils::computeGradient(I1, v, spacingVec, true);

    //
    // Compare composeHV with cudaBackwardMap
    //

    VectorField cpuResult(sizeVec);
    HField3DUtils::composeHV(h, v, cpuResult);

    cplVector3DArray dH, dV, dResult;
    allocateDeviceVector3DArray(dH, nVox);
    allocateDeviceVector3DArray(dV, nVox);
    allocateDeviceVector3DArray(dResult, nVox);
    CUDAUtilities::CopyVectorFieldToDevice(h, dH);
    CUDAUtilities::CopyVectorFieldToDevice(v, dV);

    cplBackwardMapping(dResult, dH, dV, sizeVec, 1.0, BACKGROUND_STRATEGY_ID);
    
    VectorField gpuResult(sizeVec);
    CUDAUtilities::CopyVectorFieldFromDevice(dResult, gpuResult);
    
    if(!TestUtils::Test(cpuResult, gpuResult, 0.1, "ComposeHV.mha")){
      std::cout << "ComposeHV vs. cudaBackwardMap Failed!" << std::endl;
      return TEST_FAIL;
    }else{
      std::cout << "Passed ComposeHV vs. cudaBackwardMap test" << std::endl;
    }

  }
  
  return TEST_PASS;
    
}

// test divergence cpu vs. gpu
int runTests3(){
  for(Real spacing = 1.0; spacing <= 2.0; spacing += 1.0){
    std::cout << "Testing spacing " << spacing << std::endl;
    
    Vector3D<Real> spacingVec(spacing, spacing, spacing);
    Vector3D<unsigned int> sizeVec(32,32,32);
    unsigned int nVox = sizeVec.productOfElements();
    unsigned int memSize = nVox*sizeof(Real);

    // create vector field v
    VectorField v(sizeVec);
    TestUtils::GenWavy(v, 1.0, 4.0);

    // copy v to device
    cplVector3DArray dV, dScratchV;
    allocateDeviceVector3DArray(dV, nVox);
    allocateDeviceVector3DArray(dScratchV, nVox);
    CUDAUtilities::CopyVectorFieldToDevice(v, dV);
    
    // create result array
    float *dDiv;
    allocateDeviceArray((void**)&dDiv, memSize);

    // compute CPU result
    RealImage div(sizeVec, 
		  Vector3D<Real>(0,0,0), // origin
		  spacingVec);

    HField3DUtils::divergence(v, div, spacingVec);

    // compute GPU result
    cudaHField3DUtils::divergence(dDiv,
				  dV,
				  dScratchV,
				  sizeVec, 
				  spacingVec);
    

    RealImage hDiv(sizeVec, Vector3D<Real>(0,0,0), spacingVec);
    copyArrayFromDevice(hDiv.getDataPointer(), dDiv, nVox);

    if(!TestUtils::Test(hDiv, div, 0.1, "div.mha")){
      std::cout << "divergence CPU vs. GPU test Failed!" << std::endl;
      return TEST_FAIL;
    }else{
      std::cout << "Passed divergence CPU vs. GPU test" << std::endl;
    }
    
  }
  return TEST_PASS;

}


// test pointwise dot product cpu vs. gpu
int runTests4(){
  for(Real spacing = 1.0; spacing <= 2.0; spacing += 1.0){
    std::cout << "Testing spacing " << spacing << std::endl;
    
    Vector3D<Real> spacingVec(spacing, spacing, spacing);
    Vector3D<unsigned int> sizeVec(32,32,32);
    unsigned int nVox = sizeVec.productOfElements();
    unsigned int memSize = nVox*sizeof(Real);

    // create vector field v
    VectorField v1(sizeVec), v2(sizeVec);
    TestUtils::GenWavy(v1, 1.0, 4.0);
    TestUtils::GenWavy(v2, 0.5, 2.0);

    // copy v1 and v2 to device
    cplVector3DArray dV1, dV2;
    allocateDeviceVector3DArray(dV1, nVox);
    allocateDeviceVector3DArray(dV2, nVox);
    CUDAUtilities::CopyVectorFieldToDevice(v1, dV1);
    CUDAUtilities::CopyVectorFieldToDevice(v2, dV2);
    
    // create result array
    float *dDot;
    allocateDeviceArray((void**)&dDot, memSize);

    // compute CPU result
    RealImage dot(sizeVec, 
		  Vector3D<Real>(0,0,0), // origin
		  spacingVec);

    HField3DUtils::pointwiseL2DotProd(v1,v2,dot);
    
    // compute GPU result
    cplVector3DOpers::DotProd(dDot, dV1, dV2, nVox);

    // copy GPU results to host
    RealImage hDot(sizeVec, Vector3D<Real>(0,0,0), spacingVec);
    copyArrayFromDevice(hDot.getDataPointer(), dDot, nVox);

    if(!TestUtils::Test(hDot, dot, 0.1, "dot.mha")){
      std::cout << "dot prod CPU vs. GPU test Failed!" << std::endl;
      return TEST_FAIL;
    }else{
      std::cout << "Passed dot prod CPU vs. GPU test" << std::endl;
    }
    
  }
  return TEST_PASS;
  
}

int main(int argc, char *argv[]){
   if(runTests1() != TEST_PASS){
     return -1;
   }

  if(runTests2() != TEST_PASS){
    return -1;
  }

  if(runTests3() != TEST_PASS){
    return -1;
  }

  if(runTests4() != TEST_PASS){
    return -1;
  }

  return 0;
}
