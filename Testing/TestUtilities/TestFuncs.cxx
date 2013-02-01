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

#include <cmath>

#include "Array3D.h"
#include "Array3DIO.h"
#include "Array3DUtils.h"
#include "ImageUtils.h"
#include "Array3DToyImages.h"
#include "ApplicationUtils.h"
#include "HField3DUtils.h"
#include "TestUtils.h"

#define RAND (((double)rand())/((double)RAND_MAX))

typedef float Real;
typedef Image<Real> RealImage;
typedef Array3D<Vector3D<Real> > VectorField;

void GenData(){
  int arraySz = 256;
  double inc = 1.0/5.0;
  
  // set this so results are reproduceable
  srand(17);
  
  // for each test image
  for(int imNum=0;imNum<4;imNum++){
    // generate concentric spheres
    double sphereSize = 4.0/5.0 + inc*(RAND - 1.0/2.0);
    std::cout << "image " << imNum << " sphere sizes: " << sphereSize << ", ";
    Array3D<Real> bullseye(arraySz,arraySz,arraySz);
    Array3DToyImages::createSphere(bullseye,sphereSize,0.0,1.0);
    sphereSize = sphereSize-inc + inc*(RAND - 1.0/2.0);
    std::cout << sphereSize << ", ";
    Array3DToyImages::fillSphere(bullseye,sphereSize,0.0);
    sphereSize = sphereSize-inc + inc*(RAND - 1.0/2.0);
    std::cout << sphereSize << std::endl;
    Array3DToyImages::fillSphere(bullseye,sphereSize,1.0);
    RealImage bullseyeIm(bullseye);
    char fname[100];
    sprintf(fname,"BullseyeTest%02d.mha",imNum);
    ApplicationUtils::SaveImageITK(fname,bullseyeIm);
  }
}

void GenGridData(){
  Vector3D<unsigned int> size(32,32,32);
  Vector3D<unsigned int> gridSpacing(4,4,4);
  Array3D<Real> grid;
  Array3DToyImages::createGrid(grid, size, gridSpacing);
  RealImage image(grid);
  ApplicationUtils::SaveImageITK("GridImage.mha", image);
}
 
void GenWavyData(){
  VectorField data(SizeType(32,32,32));
  TestUtils::GenWavy(data, 2.0, 4.0);
  ApplicationUtils::SaveHFieldITK("WavyVec.mha", data);
}
 
bool TestImages(RealImage &im1, RealImage &im2, 
		Real eps=1.0)
{
  double norm = Array3DUtils::sumOfSquaredElements(im1);
  double sum = Array3DUtils::sumElements(im1);
  std::cout << "squared norm of image 1 is " << norm << std::endl;
  std::cout << "sum norm of image 1 is " << sum << std::endl;
  norm = Array3DUtils::sumOfSquaredElements(im2);
  sum = Array3DUtils::sumElements(im2);
  std::cout << "squared norm of image 2 is " << norm << std::endl;
  std::cout << "sum norm of image 2 is " << sum << std::endl;
  double err = ImageUtils::squaredError(im1,im2);
  std::cout << "Sum-of-squares intensity difference is " << err << std::endl;
  return err < eps;
}

bool TestHFields(VectorField &vf1, VectorField &vf2,
		 Real eps=1.0)
{
  Vector3D<Real> sum = Array3DUtils::sumOfSquaredElements(vf1);
  std::cout << "Squared norm if h-field 1 is " << sum.x+sum.y+sum.z << std::endl;
  sum = Array3DUtils::sumOfSquaredElements(vf2);
  std::cout << "Squared norm if h-field 2 is " << sum.x+sum.y+sum.z << std::endl;

  Vector3D<Real> d = Array3DUtils::squaredDifference(vf1, vf2);
  Real err = d.x + d.y + d.z;
  std::cout << "Sum-of-squares intensity difference is " << err << std::endl;
  return err < eps;
}

void WriteDiff(VectorField &vf1, VectorField &vf2,
	       const char *base, const char *ext=NULL)
{
  VectorField diff = vf1;
  diff.pointwiseSubtract(vf2);

  if(ext == NULL){
    ApplicationUtils::SaveHFieldITK(base, diff);
  }else{
    ApplicationUtils::SaveHFieldITK(base, ext, diff);
  }
}

void WriteDiff(RealImage &im1, RealImage &im2,
	       const char *outputName)
{
  RealImage diff = im1;
  diff.pointwiseSubtract(im2);

  ApplicationUtils::SaveImageITK(outputName, diff);
}

void PrintUsage(const char *cmdName){
  std::cout << "Usage: " << cmdName
	    << " -gen "
	    << "| -genGrid "
	    << "| -genWavy "
	    << "| -imgSqrDiff filename1 filename2 [output] "
	    << "| -defSqrDiff {def1base def1ext def2base def2ext | def1 def2} [-o outputBase outputExt | outputName] "
	    << "| -subFromIdent {defbase defext | def} -o {outbase outext | out} "
	    << "| -vecToMag {defbase defext | def} outname"
	    << std::endl;
  std::exit(-1);
}

int main(int argc, char *argv[]){

  if(argc < 2){
    PrintUsage(argv[0]);
  }
  
  int curArg = 1;
  while(curArg < argc){
    if(strcmp("-gen",argv[curArg]) == 0){
      GenData();
      curArg++;
    }else if(strcmp("-genGrid",argv[curArg]) == 0){
      GenGridData();
      curArg++;
    }else if(strcmp("-genWavy",argv[curArg]) == 0){
      GenWavyData();
      curArg++;
    }else if(strcmp("-imgSqrDiff",argv[curArg]) == 0){
      if(curArg+2 >= argc){
	std::cout << "Error, -imgSqrDiff requires two arguments" << std::endl;
	std::cout << "# of args: " << argc << ", cur arg:" << curArg << std::endl;
	PrintUsage(argv[0]);
      }else{
	RealImage im1, im2;
	ApplicationUtils::LoadImageITK(argv[curArg+1], im1);
	ApplicationUtils::LoadImageITK(argv[curArg+2], im2);
	curArg+=3;
	if(curArg < argc){
	  WriteDiff(im1, im2, argv[curArg]);
	}
	if(TestImages(im1,im2)){
	  exit(0);
	}else{
	  exit(-1);
	}
      }
    }else if(strcmp("-defSqrDiff",argv[curArg]) == 0){
      if(curArg+2 >= argc){
	std::cout << "Error, -defSqrDiff requires at least two  arguments" << std::endl;
	std::cout << "# of args: " << argc << ", cur arg:" << curArg << std::endl;
	PrintUsage(argv[0]);
	exit(-1);
      }else{
	curArg++;
	int startArg = curArg;
	while(curArg < argc && strcmp("-o",argv[curArg]) != 0) curArg++;
	VectorField vf1,vf2;
	int nFNameArgs = curArg-startArg;
	if(nFNameArgs == 4){
	  ApplicationUtils::LoadHFieldITK(argv[startArg],argv[startArg+1],vf1);
	  ApplicationUtils::LoadHFieldITK(argv[startArg+2],argv[startArg+3],vf2);
	}else if(nFNameArgs == 2){
	  ApplicationUtils::LoadHFieldITK(argv[startArg],vf1);
	  ApplicationUtils::LoadHFieldITK(argv[startArg+1],vf2);
	}
	if(curArg+2 < argc){
	  WriteDiff(vf1, vf2, argv[curArg+1], argv[curArg+2]);
	}else if(curArg+1 < argc){
	  WriteDiff(vf1, vf2, argv[curArg+1]);
	}
	if(TestHFields(vf1,vf2)){
	  exit(0);
	}else{
	  exit(-1);
	}
      }
    }else if(strcmp("-subFromIdent",argv[curArg]) == 0){
      if(curArg+2 >= argc){
	std::cout << "Error, -subFromIdent requires at least two  arguments" << std::endl;
	std::cout << "# of args: " << argc << ", cur arg:" << curArg << std::endl;
	PrintUsage(argv[0]);
	exit(-1);
      }else{
	curArg++;
	int startArg = curArg;
	while(curArg < argc && strcmp("-o",argv[curArg]) != 0) curArg++;
	VectorField vf;
	int nFNameArgs = curArg-startArg;
	if(nFNameArgs == 2){
	  ApplicationUtils::LoadHFieldITK(argv[startArg],argv[startArg+1],vf);
	}else if(nFNameArgs == 1){
	  ApplicationUtils::LoadHFieldITK(argv[startArg],vf);
	}
	HField3DUtils::HtoDisplacement(vf);
	if(curArg+2 < argc){
	  ApplicationUtils::SaveHFieldITK(argv[curArg+1],argv[curArg+2],vf);
	}else if(curArg+1 < argc){
	  ApplicationUtils::SaveHFieldITK(argv[curArg+1],vf);
	}
	exit(0);
      }
    }else if(strcmp("-vecToMag",argv[curArg]) == 0){
      if(argc < curArg+3 || argc > curArg+4){
	std::cout << "Error, -vecToMag requires at least two and at most three arguments" << std::endl;
	std::cout << "# of args: " << argc-curArg << std::endl;
	PrintUsage(argv[0]);
	exit(-1);
      }else{
	VectorField vf;
	if(argc == curArg+4){
	  ApplicationUtils::LoadHFieldITK(argv[curArg+1],argv[curArg+2],vf);
	}else if(argc == curArg+3){
	  ApplicationUtils::LoadHFieldITK(argv[curArg+1],vf);
	}
	RealImage mag;
	HField3DUtils::pointwiseL2Norm(vf, mag);
	ApplicationUtils::SaveImageITK(argv[argc-1],mag);
	exit(0);
      }
    }else{
      PrintUsage(argv[0]);
    }
  }

  
}
