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


#include <string>

#include "TestUtils.h"

#include "FluidWarpParameters.h"
#include "FluidWarp.h"
#include "ApplicationUtils.h"
#include "HField3DUtils.h"

// If this is true (1), write out the resulting images instead of
// comparing them to generated data
#define WRITE_IMAGES 0

#define EPS 10.0

// // take filename such as /some/path/file.ext
// // return file_baseExt.ext
// std::string createErrorName(const char *fullPath, const char *baseExt){
//   std::string path;
//   std::string base;
//   std::string ext;

//   TestUtils::SplitName(fullPath, path, base, ext);

//   std::string rtn  = path + base + "_" + baseExt;
//   if(ext != ""){
//     rtn  = rtn + "." + ext;
//   }

//   return rtn;
// }

// bool test(RealImage &img, const char *filename){
//   Real result = TestUtils::SquaredDiff(img, filename);
//   if(result > EPS){
//     std::cerr << "Error: data does not match " << filename << " : " << result << std::endl;
//     std::string errName = createErrorName(filename, "err");
//     ApplicationUtils::SaveImageITK(errName.c_str(), img);
//     return false;
//   }
//   return true;
// }

// bool test(VectorField &vf, const char *filename){
//   Real result = TestUtils::SquaredDiff(vf, filename);
//   if(result > EPS){
//     std::cerr << "Error: data does not match " << filename << " : " << result << std::endl;
//     std::string errName = createErrorName(filename, "_err");
//     ApplicationUtils::SaveHFieldITK(errName.c_str(), vf);
//     return false;
//   }
//   return true;
// }

int runTests1(const char *testingDataDir){
  
  // set up the parameters
  
  FluidWarpParameters params;
  params.numIterations = 20;
  params.alpha = 0.01;
  params.beta = 0.01;
  params.gamma = 0.0001;
  params.maxPerturbation = 0.5;
  params.jacobianScale = false;
  params.divergenceFree = false;
  
  
  // read in the images
  Array3D<float> *images[4];
  char buff[1024];
  for(int i=0;i<4;i++){
    RealImage *tmpIm = new RealImage();
    sprintf(buff, "%s/%s/BullseyeTest%02d.mha", 
	    testingDataDir,
	    "Input/Bullseyes",
	    i);
    std::cout << "Reading image: " << buff << std::endl;
    ApplicationUtils::LoadImageITK(buff, *tmpIm);
    images[i] = tmpIm;
  }
  char dataPath[1024];
  sprintf(dataPath, "%s/%s", testingDataDir, "Tests/Libraries/Algorithms/FluidWarp");
  
  Vector3D<unsigned int> imSize = images[0]->getSize();
  
  // generate the empty output vector fields
  VectorField *vf[4];
  VectorField *vfInv[4];
  for(int i=0;i<4;i++){
    VectorField *v = new VectorField(imSize);
    vf[i] = v;
    VectorField *vInv = new VectorField(imSize);
    vfInv[i] = vInv;
  }
  
  // create the FluidWarp object
  FluidWarp fluidWarper;
  //fluidWarper.setOutputMode(FluidWarp::FW_OUTPUT_MODE_VERBOSE);
  
  // will hold deformed images
  RealImage defImage(imSize);
  
  const char *prefix;

  // ### shrinkRegion ###
  std::cout << "Testing shrinkRegion" << std::endl;
  
  HField3DUtils::setToIdentity(*vf[0]);            
  HField3DUtils::setToIdentity(*vfInv[0]);            
  fluidWarper.shrinkRegion(*images[0],
 			   params,
 			   *vf[0],
 			   *vfInv[0]);

  prefix = "ShrinkRegion";
  if(WRITE_IMAGES){
    // save the deformation field
    sprintf(buff, "%s/%sDeformation.mha", dataPath, prefix);
    ApplicationUtils::SaveHFieldITK(buff, *vf[0]);
    
    // create and save deformed image
    HField3DUtils::apply(*images[0], *vf[0], defImage);
    sprintf(buff, "%s/%sDeformedImage.mha", dataPath, prefix);
    ApplicationUtils::SaveImageITK(buff, defImage);
  }else{
    sprintf(buff, "%s/%sDeformation.mha", dataPath, prefix);
    if(!TestUtils::Test(*vf[0], buff,EPS)) return TEST_FAIL;

    HField3DUtils::apply(*images[0], *vf[0], defImage);
    sprintf(buff, "%s/%sDeformedImage.mha", dataPath, prefix);
    if(!TestUtils::Test(defImage, buff,EPS)) return TEST_FAIL;
  }

  // ### shrinkRegionForward ###
  std::cout << "Testing shrinkRegionForward" << std::endl;

  HField3DUtils::setToIdentity(*vf[0]);            
  HField3DUtils::setToIdentity(*vfInv[0]);            
  fluidWarper.shrinkRegionForward(*images[0],
 				  params,
 				  *vf[0]);
  
  prefix = "ShrinkRegionForward";
  if(WRITE_IMAGES){
    // save the deformation field
    sprintf(buff, "%s/%sDeformation.mha", dataPath, prefix);
    ApplicationUtils::SaveHFieldITK(buff, *vf[0]);
    
    // create and save deformed image
    HField3DUtils::apply(*images[0], *vf[0], defImage);
    sprintf(buff, "%s/%sDeformedImage.mha", dataPath, prefix);
    ApplicationUtils::SaveImageITK(buff, defImage);
  }else{
    // test the deformation field
    sprintf(buff, "%s/%sDeformation.mha", dataPath, prefix);
    if(!TestUtils::Test(*vf[0], buff,EPS)) return TEST_FAIL;
    
    // test deformed image
    HField3DUtils::apply(*images[0], *vf[0], defImage);
    sprintf(buff, "%s/%sDeformedImage.mha", dataPath, prefix);
    if(!TestUtils::Test(defImage, buff,EPS)) return TEST_FAIL;
  }

  // ### computeHFieldAsymmetric ###
  std::cout << "Testing computeHFieldAsymmetric" << std::endl;

  HField3DUtils::setToIdentity(*vf[0]);            
  HField3DUtils::setToIdentity(*vfInv[0]);            
  fluidWarper.computeHFieldAsymmetric(*images[0],
  				      *images[1],
  				      params,
  				      *vf[0],
  				      *vfInv[0]);
//    fluidWarper.computeHFieldAsymmetric(*images[0],
//   				      *images[1],
//   				      params,
//   				      *vf[0]);

  prefix = "ComputeHFieldAsymmetric";
  if(WRITE_IMAGES){
    // save the deformation field and inverse
    sprintf(buff, "%s/%sDeformation.mha", dataPath, prefix);
    ApplicationUtils::SaveHFieldITK(buff, *vf[0]);
    sprintf(buff, "%s/%sDeformationInv.mha", dataPath, prefix);
    ApplicationUtils::SaveHFieldITK(buff, *vfInv[0]);
    
    // create and save deformed images
    HField3DUtils::apply(*images[1], *vf[0], defImage);
    sprintf(buff, "%s/%sDeformedImage.mha", dataPath, prefix);
    ApplicationUtils::SaveImageITK(buff, defImage);
    HField3DUtils::apply(*images[0], *vfInv[0], defImage);
    sprintf(buff, "%s/%sDeformedImageInv.mha", dataPath, prefix);
    ApplicationUtils::SaveImageITK(buff, defImage);
  }else{
    // test the deformation field and inverse
    sprintf(buff, "%s/%sDeformation.mha", dataPath, prefix);
    if(!TestUtils::Test(*vf[0], buff, EPS)) return TEST_FAIL;
    sprintf(buff, "%s/%sDeformationInv.mha", dataPath, prefix);
    if(!TestUtils::Test(*vfInv[0], buff, EPS)) return TEST_FAIL;
    
    // test deformed images
    HField3DUtils::apply(*images[1], *vf[0], defImage);
    sprintf(buff, "%s/%sDeformedImage.mha", dataPath, prefix);
    if(!TestUtils::Test(defImage, buff, EPS)) return TEST_FAIL;
    HField3DUtils::apply(*images[0], *vfInv[0], defImage);
    sprintf(buff, "%s/%sDeformedImageInv.mha", dataPath, prefix);
    if(!TestUtils::Test(defImage, buff, EPS)) return TEST_FAIL;
  }

  
  // ### computeHField2Symmetric ###
  std::cout << "Testing computeHField2Symmetric" << std::endl;

  HField3DUtils::setToIdentity(*vf[0]);            
  HField3DUtils::setToIdentity(*vfInv[0]);            
  HField3DUtils::setToIdentity(*vf[1]);            
  HField3DUtils::setToIdentity(*vfInv[1]);            
  fluidWarper.computeHField2Symmetric(*images[0],
				      *images[1],
				      params,
				      *vf[0],
				      *vf[1],
				      *vfInv[0],
				      *vfInv[1]);

  prefix = "ComputeHField2Symmetric";
  if(WRITE_IMAGES){
    // save the deformation field and inverse
    sprintf(buff, "%s/%sDeformation0.mha", dataPath, prefix);
    ApplicationUtils::SaveHFieldITK(buff, *vf[0]);
    sprintf(buff, "%s/%sDeformation0Inv.mha", dataPath, prefix);
    ApplicationUtils::SaveHFieldITK(buff, *vfInv[0]);
    sprintf(buff, "%s/%sDeformation1.mha", dataPath, prefix);
    ApplicationUtils::SaveHFieldITK(buff, *vf[1]);
    sprintf(buff, "%s/%sDeformation1Inv.mha", dataPath, prefix);
    ApplicationUtils::SaveHFieldITK(buff, *vfInv[1]);
    
    // create and save deformed images
    HField3DUtils::apply(*images[0], *vf[0], defImage);
    sprintf(buff, "%s/%sMeanImage0.mha", dataPath, prefix);
    ApplicationUtils::SaveImageITK(buff, defImage);
    HField3DUtils::apply(*images[1], *vf[1], defImage);
    sprintf(buff, "%s/%sMeanImage1.mha", dataPath, prefix);
    ApplicationUtils::SaveImageITK(buff, defImage);
  }else{
    // test the deformation field and inverse
    sprintf(buff, "%s/%sDeformation0.mha", dataPath, prefix);
    if(!TestUtils::Test(*vf[0], buff, EPS)) return TEST_FAIL;
    sprintf(buff, "%s/%sDeformation0Inv.mha", dataPath, prefix);
    if(!TestUtils::Test(*vfInv[0], buff, EPS)) return TEST_FAIL;
    sprintf(buff, "%s/%sDeformation1.mha", dataPath, prefix);
    if(!TestUtils::Test(*vf[1], buff, EPS)) return TEST_FAIL;
    sprintf(buff, "%s/%sDeformation1Inv.mha", dataPath, prefix);
    if(!TestUtils::Test(*vfInv[1], buff, EPS)) return TEST_FAIL;
    
    // test deformed images
    HField3DUtils::apply(*images[0], *vf[0], defImage);
    sprintf(buff, "%s/%sMeanImage0.mha", dataPath, prefix);
    if(!TestUtils::Test(defImage, buff, EPS)) return TEST_FAIL;
    HField3DUtils::apply(*images[1], *vf[1], defImage);
    sprintf(buff, "%s/%sMeanImage1.mha", dataPath, prefix);
    if(!TestUtils::Test(defImage, buff, EPS)) return TEST_FAIL;
  }
  
  // ### computeHFieldNSymmetric ###
  std::cout << "Testing computeHFieldNSymmetric" << std::endl;

  for(int i=0;i<4;i++){
    HField3DUtils::setToIdentity(*vf[i]);
    HField3DUtils::setToIdentity(*vfInv[i]);
  }
  RealImage meanImage(imSize);
  fluidWarper.computeHFieldNSymmetric(4, images,
				      params,
				      meanImage,
				      vf,
				      vfInv);
  prefix = "ComputeHFieldNSymmetric";
  if(WRITE_IMAGES){
    for(int i=0;i<4;i++){
      // save the deformation field and inverse
      sprintf(buff, "%s/%sDeformation%d.mha", dataPath, prefix, i);
      ApplicationUtils::SaveHFieldITK(buff, *vf[i]);
      sprintf(buff, "%s/%sDeformation%dInv.mha", dataPath, prefix, i);
      ApplicationUtils::SaveHFieldITK(buff, *vfInv[i]);
      
      // create and save deformed images
      HField3DUtils::apply(*images[i], *vf[i], defImage);
      sprintf(buff, "%s/%sMeanImage%d.mha", dataPath, prefix, i);
      ApplicationUtils::SaveImageITK(buff, defImage);
    }
    sprintf(buff, "%s/%sMeanImage.mha", dataPath, prefix);
    ApplicationUtils::SaveImageITK(buff, meanImage);
  }else{
    for(int i=0;i<4;i++){
      // test the deformation field and inverse
      sprintf(buff, "%s/%sDeformation%d.mha", dataPath, prefix, i);
      if(!TestUtils::Test(*vf[i], buff, EPS)) return TEST_FAIL;
      sprintf(buff, "%s/%sDeformation%dInv.mha", dataPath, prefix, i);
      if(!TestUtils::Test(*vfInv[i], buff, EPS)) return TEST_FAIL;
      
      // test deformed images
      HField3DUtils::apply(*images[i], *vf[i], defImage);
      sprintf(buff, "%s/%sMeanImage%d.mha", dataPath, prefix, i);
      if(!TestUtils::Test(defImage, buff, EPS)) return TEST_FAIL;
    }
    sprintf(buff, "%s/%sMeanImage.mha", dataPath, prefix);
    if(!TestUtils::Test(meanImage, buff, EPS)) return TEST_FAIL;
  }
  
  return TEST_PASS;

}

int main(int argc, char *argv[]){
  if(runTests1(argv[1]) != TEST_PASS){
    std::cerr << "Test Failed." << std::endl; 
    return -1;
  }
  std::cout << "All Tests Passed." << std::endl;
  return 0;
}
