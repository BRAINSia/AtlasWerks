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
#include "HField3DUtils.h"

#include "MetamorphosisTestUtils.h"

int runTests(){
  

  // ### sincUpdample ###
  {
    Vector3D<Real> spacingVec(1.0, 1.0, 1.0);
    Vector3D<unsigned int> sizeVec(32,32,32);
    //Real voxelVol = spacingVec.productOfElements();
    Real result;
    
    // Create the bullseye image
    
    RealImage bullseye(sizeVec, // size
		       Vector3D<Real>(0,0,0), // origin
		       spacingVec); // spacing
    TestUtils::GenBullseye(bullseye, .8,.6,.2);
    
    // test gradient
    VectorField gNew(sizeVec);
    Array3DUtils::computeGradient(bullseye, gNew, spacingVec, true);
    VectorField3D gOld(sizeVec.x,sizeVec.y,sizeVec.z,
		       spacingVec.x,spacingVec.y,spacingVec.z);
    // test vector field conversion
    MetamorphosisTestUtils::ConvertToVectorField3D(gNew, gOld);
    result = MetamorphosisTestUtils::diff(gNew, gOld);
    if(result != 0.0){
      std::cerr << "Error, vector conversion/diff calculation failed: " << result << std::endl;
      return TEST_FAIL;
    }
    
    // upsample hfields
    VectorField upsampledNew;
    HField3DUtils::sincUpsample(gNew, upsampledNew, (unsigned int)2);
    VectorField3D *upsampledOld = upsampleSinc(&gOld,(unsigned int)2);
    
    result = MetamorphosisTestUtils::diff(upsampledNew, *upsampledOld);
    
    
    // the old code has a bug, so we're not acutally testing this
    //   if(result != 0.0){
    //     std::cerr << "Error, upsampled hfields not equal: " << result << std::endl;
    //     writeComponentImages("upsampledOld", *upsampledOld);
    //     ApplicationUtils::SaveHFieldITK("upsampledNew", "mha", upsampledNew);
    //     MetamorphosisTestUtils::writeDiffMag("upsampleDiffMag.mha", upsampledNew, *upsampledOld);
    //     return TEST_FAIL;
    //   }
  
  }

  // ### trilerp ###
  
  {

    // not really testing anything

    Vector3D<unsigned int> imSizeBig(34,32,32);
    Vector3D<unsigned int> imSizeSmall(32,32,32);
    VectorField vf1(imSizeBig);
    TestUtils::GenDilation(vf1, 2.0);
    VectorField vf2(imSizeSmall);
    TestUtils::GenDilation(vf2, 1.0);
    HField3DUtils::addIdentity(vf2);
    TestUtils::WriteHField("DilationSmall.mha", vf2);
    for(unsigned int z=0;z<imSizeSmall.z;z++){
      for(unsigned int y=0;y<imSizeSmall.y;y++){
	for(unsigned int x=0;x<imSizeSmall.x;x++){
	  vf1(x,y,z) = vf2(x,y,z);
	}
      }
    }
    TestUtils::WriteHField("DilationBig.mha", vf1);

    VectorField resultPID(imSizeSmall);
    HField3DUtils::compose(vf1, vf2, resultPID, HField3DUtils::BACKGROUND_STRATEGY_PARTIAL_ID);
    TestUtils::WriteHField("DoubleDilationPartialID.mha", resultPID);

    VectorField resultID(imSizeSmall);
    HField3DUtils::compose(vf1, vf2, resultID, HField3DUtils::BACKGROUND_STRATEGY_ID);
    TestUtils::WriteHField("DoubleDilationID.mha", resultID);

    VectorField identity(imSizeSmall);
    HField3DUtils::setToIdentity(identity);
    HField3DUtils::compose(vf2,identity,resultPID,HField3DUtils::BACKGROUND_STRATEGY_PARTIAL_ID);
    TestUtils::Test(vf2,resultPID,0.1,"PID_Ident_Test.mha");
    HField3DUtils::compose(vf2,identity,resultID,HField3DUtils::BACKGROUND_STRATEGY_ID);
    TestUtils::Test(vf2,resultID,0.1,"ID_Ident_Test.mha");
    TestUtils::Test(resultPID,resultID,0.1,"IDvsPID_Test.mha");

  }


  return TEST_PASS;
  
}

int main(int argc, char *argv[]){
  if(runTests() != TEST_PASS){
    return -1;
  }
  return 0;
}
