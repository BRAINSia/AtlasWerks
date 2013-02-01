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

#include "LDMMGeodesicShooting.h"

LDMMGeodesicShooting::
LDMMGeodesicShooting(SizeType imSize, 
		     SpacingType imSpacing)
  : mImSize(imSize),
    mImSpacing(imSpacing),
    mSaveAlphas(false),
    mSaveImages(false),
    mSaveVecs(false)
{
  
}

LDMMGeodesicShooting::
~LDMMGeodesicShooting()
{
}

const RealImage&
LDMMGeodesicShooting::
GetAlphaT(unsigned int t)
{
  if(!mSaveAlphas){
    throw AtlasWerksException(__FILE__,__LINE__,"Cannot get alphas, alphas not saved");
  }
  if(t > mAlphaVec.size()){
    std::string msg = 
      StringUtils::strPrintf("Cannot get alpha %d, only %d timesteps have been saved", 
			     t, mAlphaVec.size());
    throw AtlasWerksException(__FILE__,__LINE__,msg.c_str());
  }
  return *mAlphaVec[t];
}

const RealImage&
LDMMGeodesicShooting::
GetIT(unsigned int t)
{
  if(!mSaveImages){
    throw AtlasWerksException(__FILE__,__LINE__,"Cannot get images, images not saved");
  }
  if(t > mImVec.size()){
    std::string msg = 
      StringUtils::strPrintf("Cannot get image %d, only %d images have been saved", 
			     t, mImVec.size());
    throw AtlasWerksException(__FILE__,__LINE__,msg.c_str());
  }
  return *mImVec[t];
}

const VectorField&
LDMMGeodesicShooting::
GetVT(unsigned int t)
{
  if(!mSaveVecs){
    throw AtlasWerksException(__FILE__,__LINE__,"Cannot get velocities, velocities not saved");
  }
  if(t > mVVec.size()){
    std::string msg = 
      StringUtils::strPrintf("Cannot get velocity %d, only %d timesteps have been saved", 
			     t, mVVec.size());
    throw AtlasWerksException(__FILE__,__LINE__,msg.c_str());
  }
  return *mVVec[t];
}

void
LDMMGeodesicShooting::
GetJacDetPhi0To1(RealImage &jacDet)
{
  if(!mSaveVecs){
    throw AtlasWerksException(__FILE__,__LINE__,"Cannot compute jacobian det., velocities not saved");
  }
  
  RealImage scratchI(mImSize);
  RealImage scratchI2(mImSize);
  VectorField scratchV(mImSize);

  for(int t = mVVec.size()-1; t >= 0; t--){

    // compute determinant of jacobian of current deformation:
    // |D(h_{t-1})| = (|D(h_t)|(x+v_t))*|D(x+v_t)|

    // get identity in world coords
    HField3DUtils::setToIdentity(scratchV);
    scratchV.scale(mImSpacing);
    // and add velocity in world coords
    scratchV.pointwiseAdd(*mVVec[t]);
    
    // scratchI = |D(x+v(x))|
    HField3DUtils::jacobian(scratchV,scratchI,mImSpacing);
    
    /////HField3DUtils::jacobian(scratchV,scratchI,Vector3D<Real>(1.0,1.0,1.0));
    if(t == (int)mVVec.size()-1){
      jacDet = scratchI;
    }else{
      // deform current det. of jac.
      HField3DUtils::applyU(jacDet, *mVVec[t], scratchI2, mImSpacing);
      jacDet = scratchI2;
      // scale by new deformation jacobian
      jacDet.pointwiseMultiplyBy(scratchI);
    }
  }
}

  
void
LDMMGeodesicShooting::
GetJacDetPhi1To0(RealImage &jacDet)
{
  if(!mSaveVecs){
    throw AtlasWerksException(__FILE__,__LINE__,"Cannot compute jacobian det., velocities not saved");
  }

  RealImage scratchI(mImSize);
  RealImage scratchI2(mImSize);
  VectorField scratchV(mImSize);

  for(unsigned int t = 0; t < mVVec.size(); t++){

    // get identity in world coords
    HField3DUtils::setToIdentity(scratchV);
    scratchV.scale(mImSpacing);
    // and subtract velocity in world coords
    scratchV.pointwiseSubtract(*mVVec[t]);
    
    // scratchI = |D(x-v(x))|
    HField3DUtils::jacobian(scratchV,scratchI,mImSpacing);
    
    /////HField3DUtils::jacobian(scratchV,scratchI,Vector3D<Real>(1.0,1.0,1.0));
    if(t == 0){
      jacDet = scratchI;
    }else{
      // deform current det. of jac.
      HField3DUtils::applyU(jacDet, *mVVec[t], scratchI2, mImSpacing);
      jacDet = scratchI2;
      // scale by new deformation jacobian
      jacDet.pointwiseMultiplyBy(scratchI);
    }
  }
  
  
}


