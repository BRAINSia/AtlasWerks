// Description: Downsize Array_3D image with gaussian filter
// Author: Matthieu Jomier <mat-dev@jomier.com>	

#ifndef DOWNSAMPLEFILTER3D_H
#define DOWNSAMPLEFILTER3D_H

#include "Array3D.h"
#include "GaussianFilter3D.h"
#include "DownsizeFilter3D.h"
#include "Vector3D.h"

class DownsampleFilter3D
{
public:
  DownsampleFilter3D();
  ~DownsampleFilter3D();
	
  void SetInput(const Array3D<float>&);
  Array3D<float>& GetOutput();
  void Update();
  void SetSigma(float,float,float);
  void SetSize(int,int,int);
  void SetFactor(int);
  void SetFactor(int,int,int);
  Vector3D<unsigned int> GetNewSize();

private:
  float SigmaX;
  float SigmaY;
  float SigmaZ;
  int KernelSizeX;
  int KernelSizeY;
  int KernelSizeZ;
  int DownsizeFactorX;
  int DownsizeFactorY;
  int DownsizeFactorZ;
  const Array3D<float>* inarray;
  Array3D<float> outarray;
};

#endif
