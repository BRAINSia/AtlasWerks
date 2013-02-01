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


#include "LDMMDeformationData.h"

#include <sstream>

//
// ################ LDMMDeformationData Implementation ################ //
//

LDMMDeformationData::
LDMMDeformationData(const RealImage *I0, 
		    const RealImage *I1,
		    const LDMMParam &param)
  : DeformationDataInterface(*I0),
    mParam(param),
    mNTimeSteps(param.NTimeSteps()),
    mI0Orig(I0),
    mI1Orig(I1),
    mI0Scaled(new RealImage()),
    mI1Scaled(new RealImage()),
    mDefToMean(NULL),
    mComputeAlphas(false),
    mComputeAlpha0(false),
    mScaleI0(true),
    mScaleI1(true),
    mSaveDefToMean(false),
    mUsedInMeanCalc(true)
{
  mI0Ptr = mI0Orig;
  mI1Ptr = mI1Orig;

  mImSize = mI0Orig->getSize();
  mImOrigin = mI0Orig->getOrigin();
  mImSpacing = mI0Orig->getSpacing();

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

  // Vector fields
  for(unsigned int tIdx=0; tIdx<mNTimeSteps;tIdx++){
      mV.push_back(new VectorField());
#if USE_LV
      mLV.push_back(new VectorField());
#endif
  }
}

LDMMDeformationData::
~LDMMDeformationData()
{
  for(unsigned int tIdx=0; tIdx<mNTimeSteps;tIdx++){
      delete mV[tIdx];
#if USE_LV
      delete mLV[tIdx];
#endif
  }
  mV.clear();
#if USE_LV
  mLV.clear();
#endif
  for(unsigned int tIdx=0; tIdx<mAlpha.size();tIdx++){
    delete mAlpha[tIdx];
  }
  mAlpha.clear();
  if(mDefToMean){
    delete mDefToMean;
    mDefToMean = NULL;
  }
}

RealImage& 
LDMMDeformationData::
Alpha(unsigned int t)
{
  if(t > mAlpha.size()){
    std::stringstream ss;
    ss << "Error, requesting alpha " << t << ", which does not seem to have been computed";
    throw AtlasWerksException(__FILE__, __LINE__, ss.str());
  }
  return *mAlpha[t];
}

RealImage& 
LDMMDeformationData::
Alpha0()
{
  if(mAlpha.size() < 1){
    throw AtlasWerksException(__FILE__, __LINE__, "Error, requesting alpha0, "
			   ", which does not seem to have been computed");
  }
  return *mAlpha[0];
}

void 
LDMMDeformationData::
SetScaleLevel(const MultiscaleManager &scaleManager)
{
  DeformationDataInterface::SetScaleLevel(scaleManager);

  // handle vFields
  if(scaleManager.InitialScaleLevel()){
    // initialize vFields
    for(unsigned int tIdx=0; tIdx<mNTimeSteps;tIdx++){
        mV[tIdx]->resize(mCurSize);
        mV[tIdx]->fill(0.0);
#if USE_LV
        mLV[tIdx]->resize(mCurSize);
        mLV[tIdx]->fill(0.0);
#endif
    }
  }else{
    // upsample vFields
    for(unsigned int tIdx=0; tIdx<mNTimeSteps;tIdx++){
        scaleManager.UpsampleToLevel(*mV[tIdx], mCurScaleLevel);
#if USE_LV
        scaleManager.UpsampleToLevel(*mLV[tIdx], mCurScaleLevel);
#endif
    }
  }

  // handle images
  if(scaleManager.FinalScaleLevel()){
    // Use original image
    mI0Ptr = mI0Orig;
    mI1Ptr = mI1Orig;
  }else{
    // downsample from original images
    if(mScaleI0){
      mI0Scaled->resize(mCurSize);
      scaleManager.DownsampleToLevel(*mI0Orig, mCurScaleLevel, *mI0Scaled);
      mI0Ptr = mI0Scaled;
    }
    if(mScaleI1){
      mI1Scaled->resize(mCurSize);
      scaleManager.DownsampleToLevel(*mI1Orig, mCurScaleLevel, *mI1Scaled);
      mI1Ptr = mI1Scaled;
    }
  }

  if(mSaveDefToMean){
    mDefToMean->resize(mCurSize);
    mDefToMean->setSpacing(mCurSpacing);
  }

  this->InitAlphas();
  
}

void 
LDMMDeformationData::
InitAlphas()
{
  if(mComputeAlphas){
    // create alpha images if necessary
    if(mAlpha.size() < mNTimeSteps){
      for(unsigned int tIdx=mAlpha.size(); tIdx<mNTimeSteps;tIdx++){
	mAlpha.push_back(new RealImage());
      }
    }
    // resize images
    for(unsigned int tIdx=0; tIdx<mNTimeSteps;tIdx++){
      mAlpha[tIdx]->resize(mCurSize);
      mAlpha[tIdx]->setOrigin(mImOrigin);
      mAlpha[tIdx]->setSpacing(mCurSpacing);
    }
  }else if(mComputeAlpha0){
    if(mAlpha.size() < 1){
      mAlpha.push_back(new RealImage());
    }
    mAlpha[0]->resize(mCurSize);
    mAlpha[0]->setOrigin(mImOrigin);
    mAlpha[0]->setSpacing(mCurSpacing);
  }
}

void 
LDMMDeformationData::
SaveDefToMean(bool save, RealImage *defToMeanIm){
  mSaveDefToMean = save; 
  if(mSaveDefToMean && !mDefToMean){
    if(!defToMeanIm){ 
      defToMeanIm = new RealImage(mCurSize, mImOrigin, mCurSpacing);
    }
    mDefToMean = defToMeanIm;
  }
  if(!mSaveDefToMean && mDefToMean){
    if(defToMeanIm){
      throw AtlasWerksException(__FILE__, __LINE__, "Error, calling SaveDefToMean(false) with a non-null image, this doesn't make any sense");
    }
    delete mDefToMean;
    mDefToMean = NULL;
  }
}

RealImage* 
LDMMDeformationData::
GetDefToMean()
{
  if(mSaveDefToMean){
    return mDefToMean;
  }else{
    throw AtlasWerksException(__FILE__, __LINE__, "Error, can only call GetDefToMean if mSaveDefToMean is true");
  }
}

void
LDMMDeformationData::
GetI0At1(RealImage& iDef)
{
  this->GetI0AtT(iDef, mNTimeSteps);
}

void 
LDMMDeformationData::
GetI1At0(RealImage& iDef)
{
  this->GetI1AtT(iDef, 0);
}

void
LDMMDeformationData::
GetI0AtT(RealImage& iDef, unsigned int tIdx)
{
  iDef.resize(mCurSize);
  iDef.setOrigin(mImOrigin);
  iDef.setSpacing(mCurSpacing);
  VectorField hField(mCurSize);
  GetDefTTo0(hField, tIdx);
  HField3DUtils::apply(*mI0Ptr, hField, iDef);
}

void 
LDMMDeformationData::
GetI1AtT(RealImage& iDef, unsigned int tIdx)
{
  iDef.resize(mCurSize);
  iDef.setOrigin(mImOrigin);
  iDef.setSpacing(mCurSpacing);
  VectorField hField(mCurSize);
  GetDefTTo1(hField, tIdx);
  HField3DUtils::apply(*mI1Ptr, hField, iDef);
}

void 
LDMMDeformationData::
GetDef0To1(VectorField &hField)
{
  this->GetDefTTo1(hField, 0);
}

void 
LDMMDeformationData::
GetDef1To0(VectorField &hField)
{
  this->GetDefTTo0(hField, mNTimeSteps);
}

void 
LDMMDeformationData::
GetDef0ToT(VectorField &hField, unsigned int tIdx)
{
  hField.resize(mCurSize);
  HField3DUtils::setToIdentity(hField);
  VectorField scratchV(mCurSize);
  for(unsigned int i = 0; i < tIdx; i++){
    HField3DUtils::
      composeVH(*mV[i], hField, scratchV, mI0Ptr->getSpacing(),
		HField3DUtils::BACKGROUND_STRATEGY_PARTIAL_ZERO);
    hField = scratchV;
  }
}

void 
LDMMDeformationData::
GetDefTTo0(VectorField &hField, unsigned int tIdx)
{
  hField.resize(mCurSize);
  HField3DUtils::setToIdentity(hField);
  VectorField scratchV(mCurSize);
  for(unsigned int i = 0; i < tIdx; i++){
    HField3DUtils::
      composeHVInv(hField, *mV[i], scratchV, mI0Ptr->getSpacing(),
		   HField3DUtils::BACKGROUND_STRATEGY_PARTIAL_ID);
    hField = scratchV;
  }
}

void 
LDMMDeformationData::
GetDef1ToT(VectorField &hField, unsigned int tIdx)
{
  hField.resize(mCurSize);
  HField3DUtils::setToIdentity(hField);
  VectorField scratchV(mCurSize);
  for(int i = mNTimeSteps-1; i >= (int)tIdx; i--){
    HField3DUtils::composeVHInv(*mV[i], hField, scratchV, mI0Ptr->getSpacing(),
				HField3DUtils::BACKGROUND_STRATEGY_PARTIAL_ZERO);
    hField = scratchV;
  }
}

void 
LDMMDeformationData::
GetDefTTo1(VectorField &hField, unsigned int tIdx)
{
  hField.resize(mCurSize);
  HField3DUtils::setToIdentity(hField);
  VectorField scratchV(mCurSize);
  for(int i = mNTimeSteps-1; i >= (int)tIdx; i--){
    HField3DUtils::composeHV(hField, *mV[i], scratchV, mI0Ptr->getSpacing(),
			     HField3DUtils::BACKGROUND_STRATEGY_PARTIAL_ID);
    hField = scratchV;
  }
}

void 
LDMMDeformationData::
AddEnergy(const Energy &e)
{
  // call superclass version
  DeformationDataInterface::AddEnergy(e);
  
  // deal with energy increase / step size modification
  if(mEnergyHistory.LastEnergyChange() > 0){
    mEnergyHistory.AddEvent(EnergyIncreaseEvent());
    LOGNODETHREAD(logWARNING) << "Increasing energy detected in deformation " << this->GetName();
    if(mParam.AutoStepReduce()){
      mStepSize /= 2.0;
      LOGNODETHREAD(logINFO) << "Reducing StepSize to " << mStepSize;
      mEnergyHistory.AddEvent(StepSizeEvent(mStepSize));
    }
  }
}

void
LDMMDeformationData::
InterpV(VectorField &v, Real tIdx)
{
  // previous integer time
  int tp = static_cast<int>(tIdx);
  Real frac = tIdx - static_cast<Real>(tp);
  if(tp < 0){
    LOGNODETHREAD(logWARNING) << "Time out of range for interp " 
			      << tp << ", from time " 
			      << tIdx
			      << std::endl;
    tp = 0;
    frac = 0.f;
  }

  // no vector field at final timepoint, so if between T-1 and T,
  // assume it stays the same as T-1
  if(tp >= (int)mNTimeSteps-1){
    if(tp >= (int)mNTimeSteps){
      LOGNODETHREAD(logWARNING) << "Time out of range for interp " 
				<< tp << ", from time " 
				<< tIdx
				<< std::endl;
    }
    tp = mNTimeSteps-1;
    frac = 0.f;
  }

  v = *mV[tp];
  if(frac > 0.f){
    v.scale(1.0-frac);
    VectorField tmp = *mV[tp+1];
    tmp.scale(frac);
    v.pointwiseAdd(tmp);
  }

}
