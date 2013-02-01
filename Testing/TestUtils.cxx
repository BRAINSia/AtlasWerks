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

void TestUtils::GenBullseye(RealImage &TestData, double r1, double r2, double r3){
  Real hVal = 1.0;
  Real lVal = 0.0;
  Array3DToyImages::createSphere(TestData,r1,lVal,hVal);
  Array3DToyImages::fillSphere(TestData,r2,lVal);
  Array3DToyImages::fillSphere(TestData,r3,hVal);
}

void TestUtils::GenXGradient(RealImage &TestData, double min, double max){
  Vector3D<unsigned int> size = TestData.getSize();
  for(unsigned int z=0;z<size.z;z++){
    for(unsigned int y=0;y<size.y;y++){
      for(unsigned int x=0;x<size.x;x++){
	TestData(x,y,z) = min + max*(x/(size.x-1));
      }
    }
  }
}

void TestUtils::GenDilation(VectorField &TestData, double max){
  Vector3D<unsigned int> size = TestData.getSize();
  Vector3D<Real> center(static_cast<Real>(size.x)/2.0,
		      static_cast<Real>(size.y)/2.0,
		      static_cast<Real>(size.z)/2.0);
  double maxDist = center.normL2();
  for(unsigned int z=0;z<size.z;z++){
    for(unsigned int y=0;y<size.y;y++){
      for(unsigned int x=0;x<size.x;x++){
	Vector3D<Real> curPt(static_cast<Real>(x),
			     static_cast<Real>(y),
			     static_cast<Real>(z));
	Vector3D<Real> vec = curPt-center;
	double dist = vec.normL2();
	if(dist != 0){
	  vec.normalize();
	  vec *= max*(dist/maxDist);
	  TestData(x,y,z) = vec;
	}else{
	  TestData(x,y,z) = Vector3D<Real>(0,0,0);
	}
      }
    }
  }
}

void TestUtils::GenDilation2(VectorField &TestData, double max){
  Vector3D<unsigned int> size = TestData.getSize();
  Vector3D<Real> center(static_cast<Real>(size.x)/2.0,
		      static_cast<Real>(size.y)/2.0,
		      static_cast<Real>(size.z)/2.0);
  double maxDist = center.normL2();
  for(unsigned int z=0;z<size.z;z++){
    for(unsigned int y=0;y<size.y;y++){
      for(unsigned int x=0;x<size.x;x++){
	Vector3D<Real> curPt(static_cast<Real>(x),
			     static_cast<Real>(y),
			     static_cast<Real>(z));
	Vector3D<Real> vec = curPt-center;
	double dist = vec.normL2();
	dist = 1.0 - dist/maxDist;
	if(dist != 0){
	  vec.normalize();
	  vec *= max*dist;
	  TestData(x,y,z) = vec;
	}else{
	  TestData(x,y,z) = Vector3D<Real>(0,0,0);
	}
      }
    }
  }
}

void TestUtils::GenWavy(VectorField &TestData, double amp, double p){
  SizeType size = TestData.getSize();
  Vector3D<Real> center(static_cast<Real>(size.x)/2.0,
			static_cast<Real>(size.y)/2.0,
			static_cast<Real>(size.z)/2.0);
  double theta, r, phi;
  for(unsigned int z=0;z<size.z;z++){
    for(unsigned int y=0;y<size.y;y++){
      for(unsigned int x=0;x<size.x;x++){
	Vector3D<Real> curPt(static_cast<Real>(x),
			     static_cast<Real>(y),
			     static_cast<Real>(z));
	Vector3D<Real> vec = curPt-center;
	r = vec.normL2();
	theta = std::atan2(sqrt(vec.z*vec.z + vec.y*vec.y),(double)vec.x);
	phi = std::atan2((double)vec.y,(double)vec.z);
	r += amp*std::cos(p*theta);
	vec.x = r * std::cos(theta) - vec.x;
	vec.y = r * std::sin(theta) * std::sin(phi) - vec.y;
	vec.z = r*std::sin(theta) * std::cos(phi) - vec.z;
	TestData(x,y,z) = vec;
      }
    }
  }
}

void TestUtils::WriteHField(const char *filename,
			    const VectorField &hField)
{
  std::string path,base,ext,outname;
  ApplicationUtils::SplitName(filename, path, base, ext);
  outname = base+"_orig."+ext;
  ApplicationUtils::SaveHFieldITK(outname.c_str(), hField);
  VectorField uField(hField);
  HField3DUtils::HtoDisplacement(uField);
  outname = base+"_def."+ext;
  ApplicationUtils::SaveHFieldITK(outname.c_str(), uField);
  RealImage mag(hField.getSize());
  HField3DUtils::pointwiseL2Norm(uField,mag);
  outname = base+"_mag."+ext;
  ApplicationUtils::SaveImageITK(outname.c_str(), mag);
}
   


 Real TestUtils::SquaredDiff(const RealImage &testData, 
 			    const char *baselineFileName)
 {
     RealImage baseline;
     ApplicationUtils::LoadImageITK(baselineFileName,baseline);
     return Array3DUtils::squaredDifference(testData, baseline);
 }

Real TestUtils::SquaredDiff(const VectorField &testData, 
			    const char *baselineFileNameBase,
			    const char *baselineExtension)
{
  VectorField baseline;
  if(baselineExtension == NULL){
    ApplicationUtils::LoadHFieldITK(baselineFileNameBase,
				    baseline);
  }else{
    ApplicationUtils::LoadHFieldITK(baselineFileNameBase,
				    baselineExtension,
				    baseline);
  }
  return TestUtils::VecSquaredDiff(testData, baseline);
}

// squared difference between vector fields
Real TestUtils::VecSquaredDiff(const VectorField &v1,
			       const VectorField &v2)
{
  VectorField tmp(v1);
  tmp.pointwiseSubtract(v2);
  tmp.pointwiseMultiplyBy(tmp);
  Vector3D<Real> sum = Array3DUtils::sumElements(tmp);
  return (sum.x+sum.y+sum.z);
}

bool 
TestUtils::
Test(const RealImage &testIm, 
     const RealImage &baseline, 
     Real eps,
     const char *errName)
{
  Real err = Array3DUtils::squaredDifference(testIm, baseline);
  std::cout << "Test error is " << err << std::endl;
  if(err > eps){
    if(errName != NULL){
      std::cout << "Test failed, writing images:" << std::endl;
      std::string path, base, ext;
      ApplicationUtils::SplitName(errName, path, base, ext);
      std::string outname = base + "_err." + ext;
      std::cout << "   writing " << outname << std::endl;
      ApplicationUtils::SaveImageITK(outname.c_str(), testIm);
      outname = base + "_orig." + ext;
      std::cout << "   writing " << outname << std::endl;
      ApplicationUtils::SaveImageITK(outname.c_str(), baseline);
      RealImage diff(testIm);
      diff.pointwiseSubtract(baseline);
      outname = base + "_diff." + ext;
      std::cout << "   writing " << outname << std::endl;
      ApplicationUtils::SaveImageITK(outname.c_str(), diff);
    }
    return false;
  }
  return true;
}

bool
TestUtils::
Test(const VectorField &testVF, 
     const VectorField &baseline, 
     Real eps,
     const char *errName)
{
  Real err = TestUtils::VecSquaredDiff(testVF, baseline);
  std::cout << "Test error is " << err << std::endl;
  if(err > eps){
    if(errName != NULL){
      std::cout << "Test failed, writing images." << std::endl;
      std::string path, base, ext;
      ApplicationUtils::SplitName(errName, path, base, ext);
      std::string outname = base + "_err." + ext;
      std::cout << "   writing " << outname << std::endl;
      ApplicationUtils::SaveHFieldITK(outname.c_str(), testVF);
      outname = base + "_orig." + ext;
      std::cout << "   writing " << outname << std::endl;
      ApplicationUtils::SaveHFieldITK(outname.c_str(), baseline);
      VectorField diff(testVF);
      diff.pointwiseSubtract(baseline);
      RealImage diffMag;
      HField3DUtils::pointwiseL2Norm(diff, diffMag);
      outname = base + "_diff." + ext;
      std::cout << "   writing " << outname << std::endl;
      ApplicationUtils::SaveImageITK(outname.c_str(), diffMag);
    }
    return false;
  }
  return true;
}

bool 
TestUtils::
Test(const RealImage &testData, 
     const char *fileName,
     Real eps,
     bool writeImagesOnFail)
{
  RealImage baseline;
  ApplicationUtils::LoadImageITK(fileName,baseline);
  bool rtn;
  if(writeImagesOnFail){
    rtn = TestUtils::Test(testData, baseline, eps, fileName); 
  }else{
    rtn = TestUtils::Test(testData, baseline, eps); 
  }
  return rtn;
}

bool 
TestUtils::
Test(const VectorField &testData, 
     const char *fileName,
     Real eps,
     bool writeImagesOnFail)
{
  VectorField baseline;
  ApplicationUtils::LoadHFieldITK(fileName,baseline);
  bool rtn;
  if(writeImagesOnFail){
    rtn = TestUtils::Test(testData, baseline, eps, fileName); 
  }else{
    rtn = TestUtils::Test(testData, baseline, eps); 
  }
  return rtn;
}
