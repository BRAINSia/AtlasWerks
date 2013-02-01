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


#include "LDMMWarpCPU.h"

/** ################ LDMMWarpCPU ################ **/

LDMMWarpCPU::
LDMMWarpCPU(const RealImage *I0,
	    const RealImage *I1,
	    const LDMMWarpParam &param,
	    std::string warpName)
  : LDMMWarp<LDMMIteratorCPU>(I0, I1, param, warpName)
{
  if(param.WriteAlpha0()){
    mDeformationData.ComputeAlpha0(true);
  }
  if(param.WriteAlphas()){
    mDeformationData.ComputeAlphas(true);
  }
}

LDMMWarpCPU::
~LDMMWarpCPU()
{
}

void 
LDMMWarpCPU::
FinishWarp()
{
  if(mParam.WriteAlphas() || mParam.WriteAlpha0()){
    this->ComputeAlphas();
  }
}

void 
LDMMWarpCPU::
BeginIteration(unsigned int scale, unsigned int iter)
{
  // write extra output if needed
  if( (mParam.WriteAlphas() || mParam.WriteAlpha0()) &&
      mParam.GetScaleLevel(scale).OutputEveryNIterations() > 0 
      && (iter+1)%mParam.GetScaleLevel(scale).OutputEveryNIterations() == 0)
    {
      LOGNODETHREAD(logDEBUG) << "Computing alphas to be written out as intermediate data";
      this->ComputeAlphas();
    }
}

void
LDMMWarpCPU::
ComputeAlphas()
{
  LOGNODETHREAD(logDEBUG) << "Computing alphas";
  // compute the alphas
  mIterator->Iterate(mDeformationData, true);
}
