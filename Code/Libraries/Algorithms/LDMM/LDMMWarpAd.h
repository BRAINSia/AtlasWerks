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

#ifndef __LDMM_WARP_AD__
#define __LDMM_WARP_AD__

#include "LDMMWarpParam.h"
#include "LDMMDeformationData.h"
#include "LDMMWarp.h"


/**
 * Class manages LDMM adjoint shooting registration
 */
template<class LDMMAdShootingIteratorT>
class LDMMWarpAd : 
  public LDMMWarp<LDMMAdShootingIteratorT> 
{
  
public:

  /**
   * Constructor
   *
   * \param I0 `Moving' image
   *
   * \param I1 `Static/Template' image
   *
   * \param param Parameters controlling registration
   */
  LDMMWarpAd(const RealImage *I0,
	      const RealImage *I1,
	      const LDMMWarpParam &param,
	      std::string warpName="");
  ~LDMMWarpAd();

protected:

  virtual void FinishWarp();
  
};


#ifndef SWIG
#include "LDMMWarpAd.txx"
#endif // !SWIG

#endif // __LDMM_WARP_AD__
