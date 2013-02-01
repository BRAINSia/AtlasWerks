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

#ifndef __LDMM_WARP_GPU_H__
#define __LDMM_WARP_GPU_H__

#include "AtlasWerksTypes.h"
#include "LDMMWarp.h"
#include "LDMMWarpParam.h"
#include "LDMMIteratorGPU.h"
#include "VectorMath.h"
#include "cudaReduce.h"
#include "WarpInterface.h"
#include <cudaGaussianFilter.h>

/**
 * Class handles LDMM registration on the GPU
 */
class LDMMWarpGPU : 
  public LDMMWarp<LDMMIteratorGPU> 
{
  
public:
  
  /**
   * Constructor
   *
   * \param I0 `Moving' image
   *
   * \param IT `Static/Template' image
   *
   * \param param Parameters controlling registration
   */
  LDMMWarpGPU(const RealImage *I0,
	      const RealImage *IT,
	      const LDMMWarpParam &param,
	      std::string warpName="");
  
  ~LDMMWarpGPU();
  
protected:

  virtual void SetScaleLevel(const MultiscaleManager &scaleManager);

  virtual void FinishWarp();
  virtual void BeginIteration(unsigned int scale, unsigned int iter);
  
  void InitDeviceData();
  void FreeDeviceData();
  void ComputeAlphas();

  //
  // Data
  //

  unsigned int mNVox;
  unsigned int mCurVox;

  /** Reduce plan (for testing) */
  cplReduce *mdRd; 

};

#endif // __LDMM_WARP_GPU_H__
