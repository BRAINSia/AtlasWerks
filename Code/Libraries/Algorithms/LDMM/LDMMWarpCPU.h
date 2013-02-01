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

#ifndef __LDMM_WARP_CPU__
#define __LDMM_WARP_CPU__

#include "LDMMIteratorCPU.h"
#include "LDMMWarpParam.h"
#include "LDMMDeformationData.h"
#include "LDMMWarp.h"


/**
 * Class manages LDMM registration on the CPU
 */
class LDMMWarpCPU : 
  public LDMMWarp<LDMMIteratorCPU> 
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
  LDMMWarpCPU(const RealImage *I0,
	      const RealImage *I1,
	      const LDMMWarpParam &param,
	      std::string warpName="");
  ~LDMMWarpCPU();

protected:

  virtual void FinishWarp();
  virtual void BeginIteration(unsigned int scale, unsigned int iter);
  void ComputeAlphas();
  
};

#endif // __LDMM_WARP_CPU__
