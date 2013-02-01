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

#include "VectorField3D.hxx"
#include "Image3D.hxx"
#include "ImageUtils.h"
#include "ApplicationUtils.h"

#include "MetamorphosisTestUtils.h"

int runTests(const char *testingDataDir){
  

  Vector3D<Real> spacingVec(1.0, 1.0, 1.0);
  Vector3D<unsigned int> sizeVec(32,32,32);
  // Real voxelVol = spacingVec.productOfElements();
  Real result;
  
  // Create the bullseye image

  char buff[1024];
  RealImage bullseyeNew;
  sprintf(buff, "%s/%s/BullseyeTestBlur00.mha", 
	  testingDataDir,
	  "Input/Bullseyes");
  std::cout << "Reading image: " << buff << std::endl;
  ApplicationUtils::LoadImageITK(buff, bullseyeNew);
  

//   RealImage bullseyeNew(Vector3D<unsigned int>(32,32,32), // size
// 			Vector3D<Real>(0,0,0), // origin
// 			spacingVec); // spacing
  Image3D bullseyeOld;
  TestUtils::GenBullseye(bullseyeNew, .8,.6,.2);
  MetamorphosisTestUtils::ConvertToImage3D(bullseyeNew, bullseyeOld);
  RealImage bullseyeOrig(bullseyeNew);

  // test conversion
  result = MetamorphosisTestUtils::diff(bullseyeNew, bullseyeOld);
  if(result != 0.0){
    std::cerr << "Error, initial diff of images not zero: " << result << std::endl;
    return TEST_FAIL;
  }
  
  // upsample images
  ImageUtils::sincUpsample(bullseyeNew, (unsigned int)2);
  Image3D *upsampledOld = upsampleSinc(&bullseyeOld,(unsigned int)2);
  
  result = MetamorphosisTestUtils::diff(bullseyeNew, *upsampledOld);
  if(result != 0.0){
    std::cerr << "Error, upsampled images not equal: " << result << std::endl;
    writeMetaImage("upsampleOld.mha", upsampledOld);
    ApplicationUtils::SaveImageITK("upsampleNew.mha", bullseyeNew);
    return TEST_FAIL;
  }

  // test gaussian downsampling
  ApplicationUtils::SaveImageITK("OrigImage.mha",bullseyeOrig);
  RealImage downsampledInt;
  Vector3D<unsigned int> newSizeInt(16, 16, 16);
  ImageUtils::gaussianDownsample(bullseyeOrig, downsampledInt, newSizeInt);
  ApplicationUtils::SaveImageITK("IntDownsample.mha",downsampledInt);
  RealImage downsampledNonInt;
  Vector3D<unsigned int> newSizeNonInt(17, 17, 17);
  ImageUtils::gaussianDownsample(bullseyeOrig, downsampledNonInt, newSizeNonInt);
  ApplicationUtils::SaveImageITK("NonIntDownsample.mha",downsampledNonInt);
  
  
  return TEST_PASS;
  
}

int main(int argc, char *argv[]){
  if(runTests(argv[1]) != TEST_PASS){
    return -1;
  }
  return 0;
}
