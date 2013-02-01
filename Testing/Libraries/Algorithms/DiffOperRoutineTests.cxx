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

#include "DiffOper.h"

#ifdef CUDA_ENABLED
#include "cudaInterface.h"
#include "DiffOperGPU.h"
#include "CUDAUtilities.h"
#endif // CUDA_ENABLED

// test precomputed vs. on-the-fly
int runTests1(){

  for(Real spacing = 1.0; spacing <= 2.0; spacing += 1.0){

    std::cout << "Testing spacing " << spacing << std::endl;

    Vector3D<Real> spacingVec(spacing, spacing, spacing);
    Vector3D<unsigned int> sizeVec(32,32,32);
    // Real voxelVol = (Real)(spacing*spacing*spacing);
    Real result;

    // Create the operators

    // parameters
    Real alpha = 0.01;
    Real beta = 0.01;
    Real gamma = 0.001;

    DiffOper opPre(sizeVec, spacingVec);
    opPre.SetAlpha(alpha);
    opPre.SetBeta(beta);
    opPre.SetGamma(gamma);
    opPre.SetDivergenceFree(false);
    opPre.SetUseEigenLUT(true);
    opPre.Initialize();
    
    DiffOper opFly(sizeVec, spacingVec);
    opFly.SetAlpha(alpha);
    opFly.SetBeta(beta);
    opFly.SetGamma(gamma);
    opFly.SetDivergenceFree(false);
    opFly.SetUseEigenLUT(false);
    opFly.Initialize();
    
    // Set up data

    // create images
    RealImage bullseye(Vector3D<unsigned int>(32,32,32), // size
			  Vector3D<Real>(0,0,0), // origin
			  Vector3D<Real>(spacing,spacing,spacing)); // spacing
    TestUtils::GenBullseye(bullseye, .8,.6,.2);

    // calc gradient
    VectorField g(sizeVec);
    Array3DUtils::computeGradient(bullseye, g, spacingVec, true);

    // Apply operators

    VectorField opVFFly(sizeVec);
    opFly.CopyIn(g);
    opFly.ApplyOperator();
    opFly.CopyOut(opVFFly);

    VectorField opVFPre(sizeVec);
    opPre.CopyIn(g);
    opPre.ApplyOperator();
    opPre.CopyOut(opVFPre);

    result = TestUtils::VecSquaredDiff(opVFPre, opVFFly);
    if(result > 0.1){
      std::cerr << "Error, images not equal after forward operator: " << result << std::endl;
      ApplicationUtils::SaveHFieldITK("applyLPre", "mha", opVFPre);
      ApplicationUtils::SaveHFieldITK("applyLFly", "mha", opVFFly);
      return TEST_FAIL;
    }
    std::cout << "Passed forward operator test: " << result << std::endl;

    // Apply inverse

    opPre.CopyIn(g);
    opPre.ApplyInverseOperator();
    opPre.CopyOut(opVFPre);

    opFly.CopyIn(g);
    opFly.ApplyInverseOperator();
    opFly.CopyOut(opVFFly);

    result = TestUtils::VecSquaredDiff(opVFPre, opVFFly);
    if(result > 0.1){
      std::cerr << "Error, images not equal after inverse operator: " << result << std::endl;
      ApplicationUtils::SaveHFieldITK("applyLPre", "mha", opVFPre);
      ApplicationUtils::SaveHFieldITK("applyLFly", "mha", opVFFly);
      return TEST_FAIL;
    }
    std::cout << "Passed inverse operator test: " << result << std::endl;

    
  } // end loop over spacing
  
  return TEST_PASS;

}

// test that we get the same image back if we apply a forward then reverse transformation
int runTests2(){
  
  for(Real spacing = 1.0; spacing <= 2.0; spacing += 1.0){

    std::cout << "Testing spacing " << spacing << std::endl;

    Vector3D<Real> spacingVec(spacing, spacing+0.25, spacing);
    Vector3D<unsigned int> sizeVec(32,32,32);
    Real result;

    // use defaults
    DiffOperParam param;
    DiffOper op(sizeVec, spacingVec, param);

    // create images
    RealImage bullseye(sizeVec, // size
		       Vector3D<Real>(0,0,0), // origin
		       spacingVec); // spacing
    TestUtils::GenBullseye(bullseye, .8,.6,.2);

    // calc gradient
    VectorField g(sizeVec);
    Array3DUtils::computeGradient(bullseye, g, spacingVec, true);

    // Apply operators

    VectorField opVF(sizeVec);
    op.CopyIn(g);
    op.ApplyInverseOperator();
    op.ApplyOperator();
    op.CopyOut(opVF);

     result = TestUtils::VecSquaredDiff(opVF, g);
     if(result > 0.1){
       std::cerr << "Error, images not equal after forward-inverse operator test: " << result << std::endl;
       ApplicationUtils::SaveHFieldITK("LLinv.mha", opVF);
       ApplicationUtils::SaveHFieldITK("orig.mha", g);
       return TEST_FAIL;
     }
     std::cout << "Passed forward-inverse operator test: " << result << std::endl;

#ifdef CUDA_ENABLED
     // run gpu-forward/gpu-reverse test
     DiffOperGPU gpuOp;
     param.UseEigenLUT() = false;
     gpuOp.SetSize(sizeVec, spacingVec, param);
     cplVector3DArray dG;
     allocateDeviceVector3DArray(dG, sizeVec.productOfElements());
     CUDAUtilities::CopyVectorFieldToDevice(g, dG);
     gpuOp.ApplyOperator(dG);
     gpuOp.ApplyInverseOperator(dG);
     CUDAUtilities::CopyVectorFieldFromDevice(dG, opVF);
     result = TestUtils::VecSquaredDiff(opVF, g);
     if(result > 0.1){
       std::cerr << "Error, images not equal after forward-inverse operator GPU test: " << result << std::endl;
       ApplicationUtils::SaveHFieldITK("LLinvGPU.mha", opVF);
       ApplicationUtils::SaveHFieldITK("orig.mha", g);
       return TEST_FAIL;
     }
     std::cout << "Passed GPU forward-inverse operator test: " << result << std::endl;
     // run cpu-forward/gpu-reverse test
     // run gpu-forward/gpu-reverse test
     VectorField cpuForward(sizeVec);
     op.CopyIn(g);
     op.ApplyOperator();
     op.CopyOut(cpuForward);
     CUDAUtilities::CopyVectorFieldToDevice(cpuForward, dG);
     gpuOp.ApplyInverseOperator(dG);
     CUDAUtilities::CopyVectorFieldFromDevice(dG, opVF);
     result = TestUtils::VecSquaredDiff(opVF, g);
     if(result > 0.1){
       std::cerr << "Error, images not equal after cpu-forward/gpu-inverse operator test: " << result << std::endl;
       ApplicationUtils::SaveHFieldITK("cpuLgpuLinv.mha", opVF);
       ApplicationUtils::SaveHFieldITK("orig.mha", g);
       return TEST_FAIL;
     }
     std::cout << "Passed CPU-forward/GPU-inverse operator test: " << result << std::endl;
#endif // CUDA_ENABLED

  } // end loop over spacing
  
  return TEST_PASS;

}

int main(int argc, char *argv[]){
  if(runTests1() != TEST_PASS){
    return -1;
  }
  if(runTests2() != TEST_PASS){
    return -1;
  }
  return 0;
}
