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

#ifndef __LDMM_WARP_H__
#define __LDMM_WARP_H__

#include <fstream>

#include "AtlasWerksTypes.h"
#include "LDMMWarpParam.h"
#include "WarpInterface.h"
#include "LDMMEnergy.h"

/* ################ LDMMWarp ################ */


template<class LDMMIteratorT>
class LDMMWarp 
  : public WarpInterface 
{
  
public:

  typedef typename LDMMIteratorT::DeformationDataType DeformationDataType;
  typedef typename LDMMIteratorT::ParamType IteratorParamType;

  LDMMWarp(const RealImage *I0,
	   const RealImage *I1,
	   const LDMMWarpParam &param,
	   std::string warpName = "");
  ~LDMMWarp();

  const LDMMEnergy& GetEnergy(){ return mDeformationData.LastEnergy(); }
  
  virtual void GetDefImage(RealImage &defIm);
  virtual void GetInvDefImage(RealImage &defIm);
  virtual void GetDefImage(RealImage &defIm, unsigned int tIdx);
  virtual void GetInvDefImage(RealImage &defIm, unsigned int tIdx);
  virtual void GetDefField(VectorField& hField);
  virtual void GetInvDefField(VectorField& hField);
  virtual void GetAlpha0(RealImage &alpha);
  virtual void GetAlpha(RealImage &alpha, int tIdx);
  virtual void GetVField(VectorField &v, int tIdx);
  
  /** Perform the warping */
  void RunWarp();

  void GenerateOutput();

protected:

  /** Set parameters for the given scale level */
  virtual void SetScaleLevel(const MultiscaleManager &scaleManager);

  virtual void BeginWarp(){};
  virtual void FinishWarp(){};

  virtual void BeginIteration(unsigned int scale, unsigned int iter){};
  virtual void FinishIteration(unsigned int scale, unsigned int iter){};
  
  void WriteDefImage(std::string title = "");
  void WriteInvDefImage(std::string title = "");
  void WriteDefField(std::string title = "");
  void WriteInvDefField(std::string title = "");
  void WriteAlpha0(std::string title = "");
  void WriteAlphas(std::string title = "");
  void WriteVelocityFields(std::string title = "");
  void WriteIntermediateImages(std::string title = "");
  void WriteEnergy(std::string title = "");

  // data
  unsigned int mNTimeSteps;
  unsigned int mNScaleLevels;
  
  DeformationDataType mDeformationData;
  const LDMMWarpParam &mParam;
  MultiscaleManager *mScaleManager;
  LDMMIteratorT *mIterator;
  const LDMMScaleLevelParam *mCurScaleParam;

  // Image Info
  SizeType mCurSize;
  SpacingType mCurSpacing;

};

#ifndef SWIG
#include "LDMMWarp.txx"
#endif // !SWIG

#endif // __LDMM_WARP_H__
