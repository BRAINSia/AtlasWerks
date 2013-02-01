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

#ifndef __DEFORMATION_ITERATOR_INTERFACE__
#define __DEFORMATION_ITERATOR_INTERFACE__

#include "AtlasWerksTypes.h"
#include "MultiscaleManager.h"
#include "Energy.h"
#include "EnergyHistory.h"

class ScaleLevelDataInfo {
public:
  ScaleLevelDataInfo(SizeType size, OriginType origin, SpacingType spacing);
  ScaleLevelDataInfo(RealImage imTemplate);
  virtual void SetScaleLevel(const MultiscaleManager &scaleManager);
protected:
  // size/origin info
  SizeType mImSize;
  OriginType mImOrigin;
  SpacingType mImSpacing;
  unsigned int mNVox;
  // scale level info
  unsigned int mCurScaleLevel;
  SizeType mCurSize;
  SpacingType mCurSpacing;
  unsigned int mCurVox;
};

/**
 * Class holds per-deformation data, including debugging information
 */
class DeformationDataInterface : public ScaleLevelDataInfo 
{
public:
  DeformationDataInterface(SizeType size, OriginType origin, SpacingType spacing);
  DeformationDataInterface(RealImage imTemplate);
  virtual ~DeformationDataInterface(){};
  virtual const RealImage& I0() = 0;
  virtual const RealImage& I1() = 0;
  virtual void GetI0At1(RealImage& iDef) = 0;
  virtual void GetI1At0(RealImage& iDef) = 0;
  virtual void GetDef0To1(VectorField &hField) = 0;
  virtual void GetDef1To0(VectorField &hField) = 0;

  virtual EnergyHistory& GetEnergyHistory(){ return mEnergyHistory; }
  virtual void AddEnergy(const Energy &e);
  virtual bool HasEnergy();
  virtual const Energy& LastEnergy();

  const std::string& GetName(){ return mName; }
  void SetName(const std::string &name){ mName = name; }

  unsigned int GetCurIter(){ return mCurIter; }
  void SetCurIter(unsigned int iter){ mCurIter = iter; }

  Real StepSize(){ return mStepSize; }
  void StepSize(Real step);

protected:
  // name to identify this deformation
  std::string mName;
  // step size
  Real mStepSize;
  
  unsigned int mCurIter;

  EnergyHistory mEnergyHistory;
};

/**
 * Performs a single iteration on the data passed to Iterate()
 */
class DeformationIteratorInterface : public ScaleLevelDataInfo 
{
public:
  DeformationIteratorInterface(SizeType size, 
			       OriginType origin, 
			       SpacingType spacing);
  virtual ~DeformationIteratorInterface(){};
  virtual void Iterate(DeformationDataInterface &deformaitonData)=0;
};

#endif
