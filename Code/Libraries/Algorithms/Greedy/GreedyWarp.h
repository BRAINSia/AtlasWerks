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

#ifndef __GREEDY_WARP_H__
#define __GREEDY_WARP_H__

#include "AtlasWerksTypes.h"
#include "GreedyWarpParam.h"
#include "WarpInterface.h"

typedef AffineTransform3D<Real> Affine3D;

/**
 * Class for registering two images using the `Greedy' LDMM algorithm.
 * GreedyIteratorCPU or GreedyIteratorGPU can be chosen as the
 * template parameter in order to do the warp on the CPU or GPU.
 */
template<class GreedyIteratorT>
class GreedyWarp : public WarpInterface 
{
  
public:

  typedef typename GreedyIteratorT::DeformationDataType DeformationDataType;
  typedef typename GreedyIteratorT::ParamType IteratorParamType;

  /**
   * Constructor
   * \param I0 `Moving' input image
   * \param I1 `Static/Template' input image
   * \param param parameters controlling registration
   * \param aff optional initial affine transformation
   */
  GreedyWarp(const RealImage *I0,
	     const RealImage *I1,
	     const GreedyWarpParam &param,
	     Affine3D *aff=NULL,
	     std::string warpName="");

  ~GreedyWarp();

  /** Perform the warping */
  void RunWarp();

  /**
   * Write out results per parameter file settings
   */
  void GenerateOutput();
  
  /**
   * Get the DeformationData structure for the warp
   */
  DeformationDataType& DeformationData(){ return mDeformationData; }

  /**
   * Get the deformed image (I0 registered to I1) (should only be
   * called after registration is complete)
   */
  void GetI0At1(RealImage &im){ mDeformationData.GetI0At1(im); }

  /**
   * Get the inverse deformed image (I1 registered to I0) (should only
   * be called after registration is complete)
   */
  void GetI1At0(RealImage &im){ mDeformationData.GetI1At0(im); }
  
  /**
   * Get the deformation field (pointing from I1 to I0, registers I0 to I1)
   */
  void GetDef1To0(VectorField &vf){ mDeformationData.GetDef1To0(vf); }

  /**
   * Get the inverse deformation field (pointing from I0 to I1, registers I1 to I0)
   */
  void GetDef0To1(VectorField &vf){ mDeformationData.GetDef0To1(vf); }
  
protected:

  /** Set parameters for the given scale level */
  virtual void SetScaleLevel(const MultiscaleManager &scaleManager);


  /** Write out I0 deformed to time 1 with the give prefix*/
  void WriteDefImage(std::string title="");

  /** Write out I1 deformed to time 0 with the give prefix*/
  void WriteInvDefImage(std::string title="");

  /** Write out the deformation field from time 1 to time 0 with the
      give prefix*/
  void WriteDefField1To0(std::string title="");

  /** Write out the deformation field from time 0 to time 1 with the
      give prefix*/
  void WriteDefField0To1(std::string title="");

  /** Write out the energy history as an XML file*/
  void WriteEnergy(std::string title = "");

  /** Write the current output images/hfields */
  void GenerateCurOutput(const std::string &prefix, 
			 bool writeDefImage, 
			 bool writeInvDefImage, 
			 bool writeDefField, 
			 bool writeInvDefField);

  /** Write I0 and I1 at this scale level */
  void WriteScaleLevelImages(int scaleLevel);

  // data
  unsigned int mNScaleLevels;
  
  DeformationDataType mDeformationData;
  const GreedyWarpParam &mParam;
  MultiscaleManager *mScaleManager;
  GreedyIteratorT *mIterator;
  const IteratorParamType *mCurScaleParam;
  
  // Image Info
  SizeType mCurSize;
  SpacingType mCurSpacing;
  
};

#ifndef SWIG
#include "GreedyWarp.txx"
#endif // !SWIG

#endif // __GREEDY_WARP_H__
