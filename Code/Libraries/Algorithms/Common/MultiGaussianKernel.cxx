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

#include "MultiGaussianKernel.h"


template<class T>
MultiGaussianKernelT<T>::
MultiGaussianKernelT(const SizeType &size,
		     const SpacingType &spacing,
		     const MultiGaussianKernelParam &param)
{
  mNGaussians = param.Gaussian().size();
  // parse sigmas
  float weightSum = 0.f;
  for(unsigned int i=0;i<mNGaussians;i++){
    float s = param.Gaussian()[i].Sigma();
    float w = param.Gaussian()[i].Weight();
    mSigma.push_back(Vector3D<float>(s,s,s));
    mWeight.push_back(w);
    weightSum += w;
  }

  // normalize weights
  for(unsigned int i=0;i<mNGaussians;i++){
    mWeight[i] /= weightSum;
  }

  LOGNODE(logDEBUG) << "MultiGaussianKernel is sum of " << mNGaussians << " gaussians";
  
  if(mSigma.size() == 0){
    throw AtlasWerksException(__FILE__, __LINE__, 
			      "Error, must have at least one sigma vaule "
			      "specified for MultiGaussianKernel");
  }
  
  mSize = size;
  mSpacing = spacing;
  mVFBuffer = new KernelInternalVF(mSize);
  mSumBuffer = new KernelInternalVF(mSize);
  mXComponentImage = new RealImage(mSize);
  mYComponentImage = new RealImage(mSize);
  mZComponentImage = new RealImage(mSize);
}

template<class T>
MultiGaussianKernelT<T>::
~MultiGaussianKernelT()
{
  delete mVFBuffer;
  delete mSumBuffer;
  delete mXComponentImage;
  delete mYComponentImage;
  delete mZComponentImage;
}

template<class T>
void 
MultiGaussianKernelT<T>::
CopyIn(const VectorField &vf)
{
  if(vf.getSize() != mSize){
    throw AtlasWerksException(__FILE__, __LINE__, "Error, incorrect image size");
  }
  for(unsigned int z=0;z<mSize.z;++z){
    for(unsigned int y=0;y<mSize.y;++y){
      for(unsigned int x=0;x<mSize.x;++x){
	(*mVFBuffer)(x,y,z) = vf(x,y,z);
      }
    }
  }
}
  
template<class T>
void 
MultiGaussianKernelT<T>::
CopyOut(VectorField &vf)
{
  if(mSize != vf.getSize()){
    vf.resize(mSize);
  }
  for(unsigned int z=0;z<mSize.z;++z){
    for(unsigned int y=0;y<mSize.y;++y){
      for(unsigned int x=0;x<mSize.x;++x){
	vf(x,y,z) = (*mVFBuffer)(x,y,z);
      }
    }
  }
}
  
template<class T>
void 
MultiGaussianKernelT<T>::
ApplyInverseOperator()
{
  mSumBuffer->fill(0.0);
  
  // split into components
  for(unsigned int z=0;z<mSize.z;++z){
    for(unsigned int y=0;y<mSize.y;++y){
      for(unsigned int x=0;x<mSize.x;++x){
	(*mXComponentImage)(x,y,z) = (*mVFBuffer)(x,y,z).x;
	(*mYComponentImage)(x,y,z) = (*mVFBuffer)(x,y,z).y;
	(*mZComponentImage)(x,y,z) = (*mVFBuffer)(x,y,z).z;
      }
    }
  }
    
  for(unsigned int gaussIdx=0; gaussIdx < mNGaussians; gaussIdx++){
    
    float weight = mWeight[gaussIdx];
    Vector3D<float> sigma = mSigma[gaussIdx];
    Vector3D<unsigned int> kernelSize(2*static_cast<unsigned int>(std::ceil(sigma.x)),
				      2*static_cast<unsigned int>(std::ceil(sigma.y)),
				      2*static_cast<unsigned int>(std::ceil(sigma.z)));
    
    mXGaussFilter.SetInput(*mXComponentImage);
    mXGaussFilter.setSigma(sigma.x, sigma.y, sigma.z);
    mXGaussFilter.setKernelSize(kernelSize.x, kernelSize.y, kernelSize.z);
    mXGaussFilter.Update();
    const RealImage &xOut = mXGaussFilter.GetOutput();
    
    mYGaussFilter.SetInput(*mYComponentImage);
    mYGaussFilter.setSigma(sigma.x, sigma.y, sigma.z);
    mYGaussFilter.setKernelSize(kernelSize.x, kernelSize.y, kernelSize.z);
    mYGaussFilter.Update();
    const RealImage &yOut = mYGaussFilter.GetOutput();
    
    mZGaussFilter.SetInput(*mZComponentImage);
    mZGaussFilter.setSigma(sigma.x, sigma.y, sigma.z);
    mZGaussFilter.setKernelSize(kernelSize.x, kernelSize.y, kernelSize.z);
    mZGaussFilter.Update();
    const RealImage &zOut = mZGaussFilter.GetOutput();

    // combine components
    for(unsigned int z=0;z<mSize.z;++z){
      for(unsigned int y=0;y<mSize.y;++y){
	for(unsigned int x=0;x<mSize.x;++x){
	  (*mSumBuffer)(x,y,z).x += weight*xOut(x,y,z);
	  (*mSumBuffer)(x,y,z).y += weight*yOut(x,y,z);
	  (*mSumBuffer)(x,y,z).z += weight*zOut(x,y,z);
	}
      }
    }

  }

  *mVFBuffer = *mSumBuffer;
}
  
template class MultiGaussianKernelT<float>;
template class MultiGaussianKernelT<double>;
