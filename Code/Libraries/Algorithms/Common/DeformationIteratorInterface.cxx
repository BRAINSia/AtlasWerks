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

#include "DeformationIteratorInterface.h"

/* ================ ScaleLevelDataInfo ================ */

ScaleLevelDataInfo::
ScaleLevelDataInfo(SizeType size, OriginType origin, SpacingType spacing)
  : mImSize(size),
    mImOrigin(origin),
    mImSpacing(spacing),
    mNVox(size.productOfElements()),
    mCurScaleLevel(0),
    mCurVox(0)
{
}

ScaleLevelDataInfo::
ScaleLevelDataInfo(RealImage imTemplate)
  : mImSize(imTemplate.getSize()),
    mImOrigin(imTemplate.getOrigin()),
    mImSpacing(imTemplate.getSpacing()),
    mNVox(mImSize.productOfElements()),
    mCurScaleLevel(0),
    mCurVox(0)
{
}

void 
ScaleLevelDataInfo::
SetScaleLevel(const MultiscaleManager &scaleManager)
{
  mCurScaleLevel = scaleManager.CurScaleLevel();
  mCurSize = scaleManager.CurScaleSize();
  mCurSpacing = scaleManager.CurScaleSpacing();
  mCurVox = mCurSize.productOfElements();
}


/* ================ DeformationDataInterface ================ */

DeformationDataInterface::
DeformationDataInterface(SizeType size, 
			 OriginType origin, 
			 SpacingType spacing)
  : ScaleLevelDataInfo(size, origin, spacing),
    mCurIter(0)
{
}

DeformationDataInterface::
DeformationDataInterface(RealImage imTemplate)
  : ScaleLevelDataInfo(imTemplate),
    mCurIter(0)
{
}

void
DeformationDataInterface::
StepSize(Real step)
{
  mStepSize = step; 
  mEnergyHistory.AddEvent(StepSizeEvent(mStepSize));
}

void 
DeformationDataInterface::
AddEnergy(const Energy &e)
{
  IterationEvent it(mCurScaleLevel, mCurIter, e);
  mEnergyHistory.AddEvent(it);
}

bool
DeformationDataInterface::
HasEnergy()
{
  return (mEnergyHistory.LastEnergy() != NULL);
}

const Energy& 
DeformationDataInterface::
LastEnergy()
{
  Energy *e = mEnergyHistory.LastEnergy();
  if(!e){
    throw AtlasWerksException(__FILE__,__LINE__,"Error, no energy available");
  }
  return *e;
}

/* ================ DeformationIteratorInterface ================ */

DeformationIteratorInterface::
DeformationIteratorInterface(SizeType size, 
			     OriginType origin, 
			     SpacingType spacing)
  : ScaleLevelDataInfo(size, origin, spacing)
{
}
