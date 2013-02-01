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

#include <iostream>

#include "LDMM.h"
#include "ApplicationUtils.h"
#include "HField3DUtils.h"

// ###########################################
// sanity check, test mass-scaling deformation
int main(int argc, char *argv[]){

  if(argc < 4)
    {
      std::cerr << "Usage: MassConservationTest inputImage vectorFieldFormatString numTimesteps [-singleStep]"
		<< std::endl;
      return EXIT_FAILURE;
    }

  const char *inputImageFile = argv[1];
  const char *inputVecFormat = argv[2];
  unsigned int nTimeSteps = atoi(argv[3]);
  bool singleStep = false;

  int curArg = 4;
  while(argc > curArg){
    if(argv[curArg][0] == '-'){
      if(strcmp(argv[curArg], "-singleStep") == 0){
	std::cout << "Using single-step jacobian computation" << std::endl;
	singleStep = true;
	curArg++;
      }else if(strcmp(argv[curArg], "-unused2") == 0){
	curArg++;
      }else{
	std::cerr << "Error, Unknown option " << argv[curArg] << std::endl;
	std::exit(-1);
      }
    }else{
      std::cerr << "Error, Unknown option " << argv[curArg] << std::endl;
      std::exit(-1);
    }
  }
  
  RealImage *I0 = new RealImage();

  // read in I0
  ApplicationUtils::LoadImageITK(inputImageFile, *I0);
  Vector3D<unsigned int> size = I0->getSize();
  Vector3D<Real> spacing = I0->getSpacing();

  // allocate memory
  VectorField *curV = new VectorField(size);
  VectorField *hField = new VectorField(size);
  VectorField *scratchV = new VectorField(size);
  HField3DUtils::setToIdentity(*hField);
  RealImage *defImg = new RealImage(size);
  RealImage *Dphit = new RealImage(size);
  RealImage *scratchI = new RealImage(size);
  RealImage *scratchI2 = new RealImage(size);
  char buff[1024];

  Real mass = Array3DUtils::mass(*I0);
  std::cout << "Original mass = " << mass << std::endl;

  for(unsigned int t=0;t<nTimeSteps;t++){

    // read in the next vector field

    sprintf(buff, inputVecFormat, t);
    std::cout << "reading vector field " << buff << std::endl;
    ApplicationUtils::LoadHFieldITK(buff, *curV);

    // compose with current vector field
    
    HField3DUtils::composeHVInv(*hField, *curV, *scratchV, spacing,
				HField3DUtils::BACKGROUND_STRATEGY_PARTIAL_ID);
    *hField = *scratchV;

    // deform original image

    HField3DUtils::apply(*I0, *hField, *defImg);

    // save deformed image
    sprintf(buff,"DefImage%02d.mha", t+1);
    ApplicationUtils::SaveImageITK(buff,*defImg);
    Real unscaledMass = Array3DUtils::mass(*defImg);

    // update jacobian

    if(singleStep){
      HField3DUtils::jacobian(*hField,*Dphit,spacing);
    }else{
      // compute update
      HField3DUtils::setToIdentity(*scratchV);
      scratchV->pointwiseSubtract(*curV);
      // compute the determinant of jacobian of the velocity
      // deformation
      HField3DUtils::jacobian(*scratchV,*scratchI,spacing);
      
      if(t == 0){
	*Dphit = *scratchI;
      }else{
	// deform the old jac
	HField3DUtils::apply(*Dphit,*scratchV,*scratchI2);
	*Dphit = *scratchI2;
	Dphit->pointwiseMultiplyBy(*scratchI);
      }
    }

    // save jacobian determinant
    sprintf(buff,"JacDet%02d.mha", t+1);
    ApplicationUtils::SaveImageITK(buff,*Dphit);
    

    // compute scaled deformation
    defImg->pointwiseMultiplyBy(*Dphit);

    // save mass-conserved image
    sprintf(buff,"MassConservedDefImage%02d.mha", t+1);
    ApplicationUtils::SaveImageITK(buff,*defImg);

    // compute mass
    mass = Array3DUtils::mass(*defImg);
    std::cout << "Iter " << t << " unscaled mass = " << unscaledMass << std::endl;
    std::cout << "Iter " << t << " mass = " << mass << std::endl;
  }

}
