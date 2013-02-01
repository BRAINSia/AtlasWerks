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

#include "LDMMVelShootingDefData.h"

//
// ################ LDMMVelShootingDefData Implementation ################ //
//
LDMMVelShootingDefData::
LDMMVelShootingDefData(const RealImage *I0, 
		       const RealImage *I1,
		       const LDMMParam &param)
  : DeformationDataInterface(*I0),
    mParam(param),
    mNTimeSteps(param.NTimeSteps()),
    mI0Orig(I0),
    mI1Orig(I1),
    mI0Scaled(new RealImage()),
    mI1Scaled(new RealImage()),
    mI0Ptr(mI0Orig),
    mI1Ptr(mI1Orig),
    mScaleI0(true),
    mScaleI1(true),
    mV0(NULL),
    mPhiT0(NULL),
    mPhi0T(NULL)
{

  if(mImSize != mI1Orig->getSize()){
    throw AtlasWerksException(__FILE__,__LINE__,"Error, I0 and I1 are different sizes");
  }
  if((mImOrigin - mI1Orig->getOrigin()).length() > ATLASWERKS_EPS){
    throw AtlasWerksException(__FILE__,__LINE__,"Error, I0 and I1 have different origins");
  }
  if((mImSpacing - mI1Orig->getSpacing()).length() > ATLASWERKS_EPS){
    throw AtlasWerksException(__FILE__,__LINE__,"Error, I0 and I1 have different spacings");
  }

  mCurSize = mImSize;
  mCurSpacing = mImSpacing;

  mV0 = new VectorField();
  mPhiT0 = new VectorField();
  mPhi0T = new VectorField();
}

LDMMVelShootingDefData::
~LDMMVelShootingDefData()
{
  delete mI0Scaled;
  delete mI1Scaled;
  delete mV0;
  delete mPhiT0;
  delete mPhi0T;
}

void 
LDMMVelShootingDefData::
GetI0At1(RealImage& iDef)
{
  iDef.resize(mCurSize);
  iDef.setOrigin(mImOrigin);
  iDef.setSpacing(mCurSpacing);
  HField3DUtils::apply(*mI0Ptr, *mPhiT0, iDef);
}

void 
LDMMVelShootingDefData::
GetI1At0(RealImage& iDef)
{
  iDef.resize(mCurSize);
  iDef.setOrigin(mImOrigin);
  iDef.setSpacing(mCurSpacing);
  HField3DUtils::apply(*mI1Ptr, *mPhi0T, iDef);
}

void 
LDMMVelShootingDefData::
GetI0AtT(RealImage& iDef, unsigned int tIdx)
{
  throw AtlasWerksException(__FILE__,__LINE__,"Error, computing intermediate images not supported yet");
}

void 
LDMMVelShootingDefData::
GetI1AtT(RealImage& iDef, unsigned int tIdx)
{
  throw AtlasWerksException(__FILE__,__LINE__,"Error, computing intermediate images not supported yet");
}

void 
LDMMVelShootingDefData::
GetDef1To0(VectorField &h)
{
  h = this->PhiT0();
}

void 
LDMMVelShootingDefData::
GetDef0To1(VectorField &h)
{
  h = this->Phi0T();
}

RealImage& 
LDMMVelShootingDefData::
Alpha(unsigned int t)
{
  throw AtlasWerksException(__FILE__,__LINE__,"Error, Alpha computation not supported yet");
}

void 
LDMMVelShootingDefData::
GetVField(VectorField &v, unsigned int t)
{
  if(t != 0){
    throw AtlasWerksException(__FILE__,__LINE__,"Error, computing vector fields t > 0 not supported yet");
  }
  v = *mV0;
}

void 
LDMMVelShootingDefData::
SetScaleLevel(const MultiscaleManager &scaleManager)
{
  
  DeformationDataInterface::SetScaleLevel(scaleManager);

  // handle images
  if(scaleManager.FinalScaleLevel()){
    // Use original image
    mI0Ptr = mI0Orig;
    mI1Ptr = mI1Orig;
  }else{
    // downsample from original images
    if(mScaleI0){
      scaleManager.DownsampleToLevel(*mI0Orig, mCurScaleLevel, *mI0Scaled);
      mI0Ptr = mI0Scaled;
    }
    if(mScaleI1){
      scaleManager.DownsampleToLevel(*mI1Orig, mCurScaleLevel, *mI1Scaled);
      mI1Ptr = mI1Scaled;
    }
  }

  // handle v0
  if(scaleManager.InitialScaleLevel()){
    // initialize vFields
    mV0->resize(mCurSize);
    mV0->fill(0.0);
    mPhi0T->resize(mCurSize);
    HField3DUtils::setToIdentity(*mPhi0T);
    mPhiT0->resize(mCurSize);
    HField3DUtils::setToIdentity(*mPhiT0);
  }else{
    // upsample v0
    scaleManager.UpsampleToLevel(*mV0, mCurScaleLevel);
    // upsample vector fields
    HField3DUtils::hToVelocity(*mPhi0T);
    HField3DUtils::hToVelocity(*mPhiT0);
    scaleManager.UpsampleToLevel(*mPhi0T, mCurScaleLevel);
    scaleManager.UpsampleToLevel(*mPhiT0, mCurScaleLevel);
    HField3DUtils::velocityToH(*mPhi0T);
    HField3DUtils::velocityToH(*mPhiT0);
  }

}
