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


#include "ForwardSolve.h"
#include "ImageUtils.h"
#include "ApplicationUtils.h"
#include "GaussianFilter3D.h"
#include <limits>

ForwardSolve::
ForwardSolve(const Vector3D<unsigned int> &size,
	     const Vector3D<double> &origin,
	     const Vector3D<double> &spacing)
  : mSize(size),
    mOrigin(origin),
    mSpacing(spacing),
    mTAdjI0(NULL),
    mHField(NULL),
    mScratchI1(NULL),
    mScratchI2(NULL)
{
  // allocate space
  mTAdjI0 = new RealImage(mSize, mOrigin, mSpacing);
  mHField = new VectorField(mSize);
  mScratchI1 = new RealImage(mSize, mOrigin, mSpacing);
  mScratchI2 = new RealImage(mSize, mOrigin, mSpacing);
  // set default params
  ForwardSolveParam param;
  SetParams(param);
}
  
ForwardSolve::
ForwardSolve(const Vector3D<unsigned int> &size, 
	     const Vector3D<double> &origin,
	     const Vector3D<double> &spacing,
	     const ForwardSolveParam &param)
  : mSize(size),
    mOrigin(origin),
    mSpacing(spacing),
    mTAdjI0(NULL),
    mHField(NULL),
    mScratchI1(NULL),
    mScratchI2(NULL)
{
  // allocate space
  mTAdjI0 = new RealImage(mSize, mOrigin, mSpacing);
  mHField = new VectorField(mSize);
  mScratchI1 = new RealImage(mSize, mOrigin, mSpacing);
  mScratchI2 = new RealImage(mSize, mOrigin, mSpacing);
  // set params
  SetParams(param);
}
  
ForwardSolve::
~ForwardSolve()
{
  delete mTAdjI0;
  delete mHField;
  delete mScratchI1;
  delete mScratchI2;
}

void 
ForwardSolve::
SetParams(const ForwardSolveParam &param)
{
  SetPrintError(param.PrintError());
  SetGaussianRegularize(param.GaussianRegularize());
  SetGaussianSigma(param.GaussianSigma());
  SetGaussianKernelSize(param.GaussianKernelSize());
  SetIterationFileFormat(param.IterationOutputFileFormat());
  SetNIterations(param.NIterations());
  SetStepSize(param.StepSize());
  SetInitializeWithTInv(param.InitializeWithTInv());
}

ForwardSolveParam 
ForwardSolve::
GetParams()
{
  ForwardSolveParam param;
  param.PrintError() = GetPrintError();
  param.GaussianRegularize() = GetGaussianRegularize();
  param.GaussianSigma() = GetGaussianSigma();
  param.GaussianKernelSize() = GetGaussianKernelSize();
  param.IterationOutputFileFormat() = GetIterationFileFormat();
  param.NIterations() = GetNIterations();
  param.StepSize() = GetStepSize();
  param.InitializeWithTInv() = GetInitializeWithTInv();
  return param;
}

void
ForwardSolve::
Solve(const RealImage &I0,
      const VectorField &v,
      RealImage &I1,
      unsigned int maxIter)
{
  mNIterations = maxIter;
  Solve(I0, v, I1);
}

void
ForwardSolve::
Solve(const RealImage &I0,
      const VectorField &v,
      RealImage &I1)
{
  // initial estimage of final image
  if(mInitializeWithTInv){
    VectorField vInv = v;
    vInv.scale(-1.0);
    HField3DUtils::applyU(I0,vInv,I1,mSpacing);
  }else{
    // start with zero image instead of initial estimate?
    I1.fill(0.0);
  }

  // for forward splatting we will need an hfield-version of v
  *mHField = v;
  mHField->scale(1.0/mSpacing);
  HField3DUtils::addIdentity(*mHField);
  // forward splat I0 (create mTAdjI0)
  HField3DUtils::forwardApply(I0, *mHField, *mTAdjI0, (Real)0.0, false);

  ApplicationUtils::SaveImageITK("TAdjI0.nhdr", *mTAdjI0);

  double lastErr = std::numeric_limits<double>::max();
  for(unsigned int iter=0;iter < mNIterations; iter++){

    if(mIterFormat.size() > 0){
      ApplicationUtils::
	SaveImageITK(StringUtils::strPrintf(mIterFormat.c_str(), iter).c_str(), I1);
    }

    // pull back I1
    HField3DUtils::applyU(I1,v,*mScratchI1, mSpacing);

    double err = ImageUtils::squaredError(*mScratchI1, I0);
    if(mPrintError){
      std::cout << "Squared error=" << err << std::endl;
    }
    if(err > lastErr){
      std::cout << "exiting at iteration " << iter << " due to increasing error" << std::endl;
      break;
    }
    lastErr = err;
    
    // forward splat I1
    HField3DUtils::forwardApply(*mScratchI1, *mHField, *mScratchI2, (Real)0.0, false);
    
    mScratchI2->pointwiseSubtract(*mTAdjI0);
    
    if(mGaussianRegularize){
      // update estimate of I1
      GaussianFilter3D filter;
      filter.SetInput(*mScratchI2);
      filter.setSigma(mGaussianSigma,mGaussianSigma,mGaussianSigma);
      filter.setKernelSize(mGaussianKernelSize,mGaussianKernelSize,mGaussianKernelSize);
      filter.Update();
      *mScratchI2 = filter.GetOutput();
      mScratchI2->scale(mStepSize);
      I1.scale(1.0-mStepSize*0.01);
    }
    I1.pointwiseSubtract(*mScratchI2);
  }
}
      
void
ForwardSolve::
RunForwardSolve(const RealImage &I0,
		const VectorField &v,
		RealImage &I1,
		unsigned int maxIters)
{
  Vector3D<unsigned int> imSize = I0.getSize();
  Vector3D<double> imOrigin = I0.getOrigin();
  Vector3D<double> imSpacing = I0.getSpacing();
  ForwardSolve solver(imSize, imOrigin, imSpacing);
  I1.resize(imSize);
  I1.setOrigin(imOrigin);
  I1.setSpacing(imSpacing);
  solver.Solve(I0, v, I1, maxIters);
}

