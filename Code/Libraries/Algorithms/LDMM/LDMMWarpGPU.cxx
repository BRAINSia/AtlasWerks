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


#include "LDMMWarpGPU.h"
#include "StringUtils.h"
#include "cudaInterface.h"
#include "cudaHField3DUtils.h"
#include "CUDAUtilities.h"
#include <cutil_comfunc.h>
#include <cudaUpsample3D.h>
//#include <gaussFilter.h>
#include <log.h>
#include <cudaDownSample.h>
#include <cutil_inline.h>

LDMMWarpGPU::
LDMMWarpGPU(const RealImage *I0,
	    const RealImage *I1,
	    const LDMMWarpParam &param,
	    std::string warpName)
  : LDMMWarp<LDMMIteratorGPU>(I0, I1, param, warpName),
    mdRd(NULL)
{

  mNVox = mImSize.productOfElements();

  if(param.WriteAlpha0()){
    mDeformationData.ComputeAlpha0(true);
  }
  if(param.WriteAlphas()){
    mDeformationData.ComputeAlphas(true);
  }
 
  this->InitDeviceData();
}

LDMMWarpGPU::
~LDMMWarpGPU()
{
  this->FreeDeviceData();
}

void 
LDMMWarpGPU::
InitDeviceData()
{
  mdRd = new cplReduce();
  checkCUDAError("LDMMWarpGPU::InitDeviceData");
}

void 
LDMMWarpGPU::
FreeDeviceData()
{
}

void
LDMMWarpGPU::
SetScaleLevel(const MultiscaleManager &scaleManager)
{
  LDMMWarp<LDMMIteratorGPU>::SetScaleLevel(scaleManager);
}

void 
LDMMWarpGPU::
FinishWarp()
{
  if(mParam.WriteAlphas() || mParam.WriteAlpha0()){
    this->ComputeAlphas();
  }
}

void 
LDMMWarpGPU::
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
LDMMWarpGPU::
ComputeAlphas()
{
  LOGNODETHREAD(logDEBUG) << "Computing alphas";
  // compute the alphas, they're stored in mDeformationData
  mIterator->Iterate(mDeformationData, true);
}

