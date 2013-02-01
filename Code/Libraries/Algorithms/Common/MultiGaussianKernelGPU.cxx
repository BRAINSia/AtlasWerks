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

#include "MultiGaussianKernelGPU.h"

// for testing
#include "ApplicationUtils.h"
#include "CUDAUtilities.h"

MultiGaussianKernelGPU::
MultiGaussianKernelGPU()
  : mdKernelC(NULL),
    mdScratch(NULL),
    // TEST
    mCallCount(0)
    // END TEST
{
  
}

MultiGaussianKernelGPU::
~MultiGaussianKernelGPU()
{
  FreeDeviceData();
}

void 
MultiGaussianKernelGPU::
GenerateGaussianSum()
{
  mGaussian.fill(0);

  Vector3D<float> offset;
  offset.x = mSize.x % 2 == 0 ? 0.5 : 0;
  offset.y = mSize.y % 2 == 0 ? 0.5 : 0;
  offset.z = mSize.z % 2 == 0 ? 0.5 : 0;

  SizeType idxOff;
  idxOff.x = mSize.x % 2 == 0 ? 1 : 0;
  idxOff.y = mSize.y % 2 == 0 ? 1 : 0;
  idxOff.z = mSize.z % 2 == 0 ? 1 : 0;

  SizeType halfSizeCeil;
  halfSizeCeil.x = 
    static_cast<unsigned int>
    (std::ceil(mSize.x/2.0));
  halfSizeCeil.y = 
    static_cast<unsigned int>
    (std::ceil(mSize.y/2.0));
  halfSizeCeil.z = 
    static_cast<unsigned int>
    (std::ceil(mSize.z/2.0));

  SizeType halfSizeFloor = mSize / 2;
  
  for(unsigned int gaussIdx=0; gaussIdx < mNGaussians; gaussIdx++){
    
    Vector3D<float> &sigma = mSigma[gaussIdx];
    float weight = mWeight[gaussIdx];
    LOGNODE(logDEBUG) << "Weight of gaussian " << gaussIdx << " is " << weight;
    Vector3D<float> sig2(2*sigma.x*sigma.x,
			 2*sigma.y*sigma.y,
			 2*sigma.z*sigma.z);
    
    for(unsigned int z=0;z<halfSizeCeil.z;++z){
      for(unsigned int y=0;y<halfSizeCeil.y;++y){
	for(unsigned int x=0;x<halfSizeCeil.x;++x){
	  // get the current value at this location
	  float v = mGaussian(x,y,z);
	  // calculate this gaussian contribution and add to current value
	  float fx = x+offset.x;
	  float fy = y+offset.y;
	  float fz = z+offset.z;
	  float p = fx*fx/sig2.x + fy*fy/sig2.y + fz*fz/sig2.z;
	  v += weight*exp(-p);
	  // assign this value to all symmetric locations
	  // mGaussian(halfSizeFloor.x+x,halfSizeFloor.y+y,halfSizeFloor.z+z) = v;
	  // mGaussian(halfSizeFloor.x+x,halfSizeFloor.y+y,halfSizeFloor.z-z-idxOff.z) = v;
	  // mGaussian(halfSizeFloor.x+x,halfSizeFloor.y-y-idxOff.y,halfSizeFloor.z+z) = v;
	  // mGaussian(halfSizeFloor.x+x,halfSizeFloor.y-y-idxOff.y,halfSizeFloor.z-z-idxOff.z) = v;
	  // mGaussian(halfSizeFloor.x-x-idxOff.x,halfSizeFloor.y+y,halfSizeFloor.z+z) = v;
	  // mGaussian(halfSizeFloor.x-x-idxOff.x,halfSizeFloor.y+y,halfSizeFloor.z-z-idxOff.z) = v;
	  // mGaussian(halfSizeFloor.x-x-idxOff.x,halfSizeFloor.y-y-idxOff.y,halfSizeFloor.z+z) = v;
	  // mGaussian(halfSizeFloor.x-x-idxOff.x,halfSizeFloor.y-y-idxOff.y,halfSizeFloor.z-z-idxOff.z) = v;

	  // assign this value to all symmetric locations
	  mGaussian(x,y,z) = v;
	  mGaussian(x,y,(mSize.z-z-idxOff.z)%mSize.z) = v;
	  mGaussian(x,(mSize.y-y-idxOff.y)%mSize.y,z) = v;
	  mGaussian(x,(mSize.y-y-idxOff.y)%mSize.y,(mSize.z-z-idxOff.z)%mSize.z) = v;
	  mGaussian((mSize.x-x-idxOff.x)%mSize.x,y,z) = v;
	  mGaussian((mSize.x-x-idxOff.x)%mSize.x,y,(mSize.z-z-idxOff.z)%mSize.z) = v;
	  mGaussian((mSize.x-x-idxOff.x)%mSize.x,(mSize.y-y-idxOff.y)%mSize.y,z) = v;
	  mGaussian((mSize.x-x-idxOff.x)%mSize.x,(mSize.y-y-idxOff.y)%mSize.y,(mSize.z-z-idxOff.z)%mSize.z) = v;
	}
      }
    }

  } // end loop over gaussians

  // normalize

  double sum = 0.f;
  for (unsigned i=0; i< mNVox; ++i)
    sum += mGaussian(i);
  
  for (unsigned i=0; i< mNVox; ++i)
    mGaussian(i) /= sum;
  
  ApplicationUtils::SaveImageITK("GaussianTest.mha", mGaussian);
}

void 
MultiGaussianKernelGPU::
SetSize(const SizeType &size, 
	const SpacingType &spacing,
	const KernelParam &param)
{

  if(!param.IsMultiGaussianKernelParam()){
    throw AtlasWerksException(__FILE__, __LINE__, 
			      "Error, kernel param is not "
			      "multi-gaussian param");
  }
  
  const MultiGaussianKernelParam *MGParam = param.AsMultiGaussianKernelParam();
  
  mNGaussians = MGParam->Gaussian().size();
  mSize = size;
  mSpacing = spacing;
  mNVox = mSize.productOfElements();

  // TEST
  mItersBetweenUpdate = MGParam->ItersBetweenUpdate();
  // END TEST

  // parse sigmas
  float weightSum = 0.f;
  for(unsigned int i=0;i<mNGaussians;i++){
    float s = MGParam->Gaussian()[i].Sigma();
    float w = MGParam->Gaussian()[i].Weight();
    mSigma.push_back(Vector3D<float>(s,s,s));
    mWeight.push_back(w);
    weightSum += w;
    // TEST
    mWeightChange.push_back(MGParam->Gaussian()[i].WeightChange());
    // END TEST
  }
  
  // normalize weights
  for(unsigned int i=0;i<mNGaussians;i++){
    mWeight[i] /= weightSum;
  }

  if(mNGaussians == 0){
    throw AtlasWerksException(__FILE__, __LINE__, 
			      "Error, must have at least one sigma vaule "
			      "specified for MultiGaussianKernel");
  }
  
  LOGNODE(logDEBUG) << "MultiGaussianKernel is sum of " << mNGaussians << " gaussians";

  mGaussian.resize(mSize);

  this->GenerateGaussianSum();
  
  InitDeviceData();

}

void 
MultiGaussianKernelGPU::
InitDeviceData()
{
  // free old data
  FreeDeviceData();

  // allocate scratch array
  dmemAlloc(mdScratch, mNVox);

  // allocate real gaussian kernel on GPU
  float* dKernel;
  dmemAlloc(dKernel, mNVox);
  copyArrayToDevice(dKernel, mGaussian.getDataPointer(), mNVox);
  cplVectorOpers::MulC_I(dKernel, (float)mNVox, mNVox);

  // Copy to GPU
  dmemAlloc(mdKernelC, mNVox);

  // Compute the guassian kernel on frequency domain
  mdFFTFilter.setSize(mSize);
  mdFFTFilter.getFFTWrapper().forwardFFT(mdKernelC, dKernel);

  dmemFree(dKernel);
}

void 
MultiGaussianKernelGPU::
FreeDeviceData()
{
  dmemFree(mdKernelC);
  dmemFree(mdScratch);
}

void 
MultiGaussianKernelGPU::
ApplyInverseOperator(cplVector3DArray& dF)
{
  // TEST
  if(mItersBetweenUpdate > 0 && 
     mCallCount > 0 && 
     mCallCount % mItersBetweenUpdate == 0)
    {
      for(int gaussIdx=0;gaussIdx < mNGaussians; gaussIdx++){
	mWeight[gaussIdx] += mWeightChange[gaussIdx];
      }
      this->GenerateGaussianSum();
      InitDeviceData();
    }
  mCallCount++;
  // END TEST
  mdFFTFilter.convolve(dF.x, dF.x, mdKernelC, mSize);
  //copyArrayDeviceToDevice(dF.x, mdScratch, mNVox);
  mdFFTFilter.convolve(dF.y, dF.y, mdKernelC, mSize);
  mdFFTFilter.convolve(dF.z, dF.z, mdKernelC, mSize);
}
