// Description: Downsize  Array2D with gaussian filter
// Author: Matthieu Jomier <mat-dev@jomier.com>

#include "DownsampleFilter2D.h"

DownsampleFilter2D::DownsampleFilter2D()
{
  //default parameters
  SigmaX = 1;
  SigmaY = 1;
  //  SigmaZ = 1;
  KernelSizeX = 1;
  KernelSizeY = 1;
  //KernelSizeZ = 1;
  DownsizeFactorX = DownsizeFactorY =2;
}

DownsampleFilter2D::~DownsampleFilter2D()
{
}


void DownsampleFilter2D::SetInput(const Array2D<float>& _inarray)
{
  inarray = &_inarray;
}

Array2D<float>& DownsampleFilter2D::GetOutput()
{
  return outarray;
}

		 
void DownsampleFilter2D::SetSigma(float _sigmax,float _sigmay)
{
  SigmaX = _sigmax;
  SigmaY = _sigmay;
  //  SigmaZ = _sigmaz;
}

void DownsampleFilter2D::SetSize(int _kernelsizex,int _kernelsizey)
{
  KernelSizeX = _kernelsizex;
  KernelSizeY = _kernelsizey;
  //  KernelSizeZ = _kernelsizez;
}

void DownsampleFilter2D::SetFactor(int _factor)
{
  DownsizeFactorX = DownsizeFactorY = _factor;
}

void DownsampleFilter2D::SetFactor(int _factorX, int _factorY)
{
  DownsizeFactorX = _factorX;
  DownsizeFactorY = _factorY;
  //  DownsizeFactorZ = _factorZ;
}


void DownsampleFilter2D::Update()
{
  //Compute Gaussian filter
  GaussianFilter2D* filter = new GaussianFilter2D();
  filter->SetInput(*inarray);
  filter->setSigma(SigmaX,SigmaY);
  filter->setFactor(DownsizeFactorX,DownsizeFactorY);
  filter->setKernelSize(KernelSizeX,KernelSizeY);
  filter->Update();

  //Downsample image
  DownsizeFilter2D* downsize = new DownsizeFilter2D();
  downsize->SetInput(filter->GetOutput());
  downsize->setSize(DownsizeFactorX, DownsizeFactorY);
  downsize->Update();

  outarray = downsize->GetOutput();


  delete filter;
  delete downsize;
}

Vector2D<unsigned int> DownsampleFilter2D::GetNewSize()
{
  return outarray.getSize();
}

