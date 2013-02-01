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


#include "GreedyDeformationData.h"

GreedyDeformationData::
GreedyDeformationData(const RealImage *I0, 
		      const RealImage *I1,
		      const GreedyParam &param)
  : DeformationDataInterface(*I0),
    mParam(param),
    mI0Orig(I0),
    mI1Orig(I1),
    mI0Scaled(new RealImage()),
    mI1Scaled(new RealImage()),
    mI0Ptr(mI0Orig),
    mI1Ptr(mI1Orig),
    mDef1To0(new VectorField()),
    mDef0To1(NULL),
    mComputeInverseHField(false),
    mSaveDefToMean(false),
    mDefToMean(NULL),
    mScaleI0(true),
    mScaleI1(true),
    mInitialAffine(NULL)
{
}

GreedyDeformationData::
~GreedyDeformationData()
{
  delete mI0Scaled;
  delete mI1Scaled;
  delete mDef1To0;
  if(mDef0To1){
    delete mDef0To1;
    mDef0To1 = NULL;
  }
  if(mDefToMean){
    delete mDefToMean;
    mDefToMean = NULL;
  }
}

void
GreedyDeformationData::
GetI0At1(RealImage& iDef)
{
  iDef.resize(mCurSize);
  iDef.setOrigin(mImOrigin);
  iDef.setSpacing(mCurSpacing);
  HField3DUtils::apply(*mI0Ptr, *mDef1To0, iDef);
}

void 
GreedyDeformationData::
GetI1At0(RealImage& iDef)
{
  iDef.resize(mCurSize);
  iDef.setOrigin(mImOrigin);
  iDef.setSpacing(mCurSpacing);
  if(mComputeInverseHField){
    HField3DUtils::apply(*mI1Ptr, *mDef0To1, iDef);
  }else{
    VectorField hInv(mCurSize);
    HField3DUtils::computeInverseZerothOrder(*mDef1To0, hInv);
    HField3DUtils::apply(*mI0Ptr, hInv, iDef);
  }
}

void 
GreedyDeformationData::
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
GreedyDeformationData::
GetDefToMean()
{
  if(mSaveDefToMean){
    return mDefToMean;
  }else{
    throw AtlasWerksException(__FILE__, __LINE__, "Error, can only call GetDefToMean if mSaveDefToMean is true");
  }
}

void 
GreedyDeformationData::
GetDef1To0(VectorField &h)
{
  h = this->Def1To0();
}

void 
GreedyDeformationData::
GetDef0To1(VectorField &h)
{
  if(!mComputeInverseHField){
    throw AtlasWerksException(__FILE__,__LINE__,"Error, inverse HField not computed");
  }
  h = this->Def0To1();
}

bool 
GreedyDeformationData::
ComputeInverseHField()
{
  return mComputeInverseHField;
}

void 
GreedyDeformationData::
ComputeInverseHField(bool computeInv)
{
  mComputeInverseHField = computeInv;

  if(mComputeInverseHField && mDef0To1 == NULL){
    mDef0To1 = new VectorField();
  }
}

void 
GreedyDeformationData::
SetInitialAffine(const Affine3D &aff)
{
  mInitialAffine = new Affine3D();
  *mInitialAffine = aff;
}

const Affine3D&
GreedyDeformationData::
GetInitialAffine()
{
  return *mInitialAffine;
}

void 
GreedyDeformationData::
AddEnergy(const Energy &e)
{
  // call superclass version
  DeformationDataInterface::AddEnergy(e);

  Real energyChange = mEnergyHistory.LastEnergyChange();
  // test for NaN
  if(energyChange != energyChange){
    LOGNODETHREAD(logERROR) 
      << "NaN detected in deformation " << this->GetName();
  }
  // deal with energy increase / step size modification
  if(energyChange > 0){
    mEnergyHistory.AddEvent(EnergyIncreaseEvent());
    LOGNODETHREAD(logWARNING) << "Increasing energy detected in deformation " << this->GetName();
    if(mParam.AutoStepReduce()){
      LOGNODETHREAD(logINFO) << "Reducing StepSize";
      this->StepSize(this->StepSize()/2.0);
    }
  }
}

void 
GreedyDeformationData::
SetScaleLevel(const MultiscaleManager &scaleManager)
{

  DeformationDataInterface::SetScaleLevel(scaleManager);

  // handle hField
  if(scaleManager.InitialScaleLevel()){
    // initialize hFields
    mDef1To0->resize(mCurSize);
    if(mInitialAffine){
      HField3DUtils::initializeFromAffine(*mDef1To0, *mInitialAffine, mImOrigin, mCurSpacing);
    }else{
      HField3DUtils::setToIdentity(*mDef1To0);
    }
    if(mComputeInverseHField){
      mDef0To1->resize(mCurSize);
      if(mInitialAffine){
	HField3DUtils::initializeFromAffineInv(*mDef0To1, *mInitialAffine, mImOrigin, mCurSpacing);
      }else{
	HField3DUtils::setToIdentity(*mDef0To1);
      }
    }
  }else{
    // convert to vField for upsampling
    HField3DUtils::hToVelocity(*mDef1To0);
    scaleManager.UpsampleToLevel(*mDef1To0, mCurScaleLevel);
    HField3DUtils::velocityToH(*mDef1To0);
    if(mComputeInverseHField){
      // convert to vField for upsampling
      HField3DUtils::hToVelocity(*mDef0To1);
      scaleManager.UpsampleToLevel(*mDef0To1, mCurScaleLevel);
      HField3DUtils::velocityToH(*mDef0To1);
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
      //mI0Scaled->resize(curSize);
      scaleManager.DownsampleToLevel(*mI0Orig, mCurScaleLevel, *mI0Scaled);
      mI0Ptr = mI0Scaled;
    }
    if(mScaleI1){
      //mI1Scaled->resize(curSize);
      scaleManager.DownsampleToLevel(*mI1Orig, mCurScaleLevel, *mI1Scaled);
      mI1Ptr = mI1Scaled;
    }
  }

  if(mSaveDefToMean){
    mDefToMean->resize(mCurSize);
    mDefToMean->setSpacing(mCurSpacing);
  }

}



















