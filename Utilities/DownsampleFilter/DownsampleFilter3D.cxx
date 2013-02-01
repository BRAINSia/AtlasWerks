// Description: Downsize  Array3D with gaussian filter
// Author: Matthieu Jomier <mat-dev@jomier.com>

#include "DownsampleFilter3D.h"

DownsampleFilter3D::DownsampleFilter3D()
{
  //default parameters
  SigmaX = 1;
  SigmaY = 1;
  SigmaZ = 1;
  KernelSizeX = 1;
  KernelSizeY = 1;
  KernelSizeZ = 1;
  DownsizeFactorX = DownsizeFactorY = DownsizeFactorZ =2;
}

DownsampleFilter3D::~DownsampleFilter3D()
{
}


void DownsampleFilter3D::SetInput(const Array3D<float>& _inarray)
{
  inarray = &_inarray;
}

Array3D<float>& DownsampleFilter3D::GetOutput()
{
  return outarray;
}

		 
void DownsampleFilter3D::SetSigma(float _sigmax,float _sigmay, float _sigmaz)
{
  SigmaX = _sigmax;
  SigmaY = _sigmay;
  SigmaZ = _sigmaz;
}

void DownsampleFilter3D::SetSize(int _kernelsizex,int _kernelsizey, int _kernelsizez)
{
  KernelSizeX = _kernelsizex;
  KernelSizeY = _kernelsizey;
  KernelSizeZ = _kernelsizez;
}

void DownsampleFilter3D::SetFactor(int _factor)
{
  DownsizeFactorX = DownsizeFactorY = DownsizeFactorZ = _factor;
}

void DownsampleFilter3D::SetFactor(int _factorX, int _factorY, int _factorZ)
{
  DownsizeFactorX = _factorX;
  DownsizeFactorY = _factorY;
  DownsizeFactorZ = _factorZ;
}


void DownsampleFilter3D::Update()
{
  //Compute Gaussian filter
  GaussianFilter3D* filter = new GaussianFilter3D();
  filter->SetInput(*inarray);
  filter->setSigma(SigmaX,SigmaY,SigmaZ);
  filter->setFactor(DownsizeFactorX,DownsizeFactorY,DownsizeFactorZ);
  filter->setKernelSize(KernelSizeX,KernelSizeY,KernelSizeZ);
  filter->Update();

  //Downsample image
  DownsizeFilter3D* downsize = new DownsizeFilter3D();
  downsize->SetInput(filter->GetOutput());
  downsize->setSize(DownsizeFactorX, DownsizeFactorY, DownsizeFactorZ);
  downsize->Update();

  outarray = downsize->GetOutput();


  delete filter;
  delete downsize;
}

Vector3D<unsigned int> DownsampleFilter3D::GetNewSize()
{
  return outarray.getSize();
}

