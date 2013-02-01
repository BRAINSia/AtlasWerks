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


#ifndef __WARP_INTERRFACE_H__
#define __WARP_INTERRFACE_H__

#include "AtlasWerksTypes.h"
#include "WeightedImageSet.h"
#include "EnergyHistory.h"

class WarpInterface
{
  
public:
  
  WarpInterface(const RealImage *I0, const RealImage *I1);
  virtual ~WarpInterface();
  
  virtual void RunWarp() = 0;

  virtual void GenerateOutput() = 0;
  
protected:

  //
  // Data
  //

  const RealImage *mI0Orig;
  const RealImage *mI1Orig;

  SizeType mImSize;
  OriginType mImOrigin;
  SpacingType mImSpacing;

  EnergyHistory mEnergyHistory;
  
};

#endif // __WARP_INTERRFACE_H__
