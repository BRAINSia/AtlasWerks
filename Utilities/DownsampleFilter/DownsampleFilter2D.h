// Description: Downsize Array_2D image with gaussian filter
// Author: Matthieu Jomier <mat-dev@jomier.com>	

#ifndef DOWNSAMPLEFILTER2D_H
#define DOWNSAMPLEFILTER2D_H

#include "Array2D.h"
#include "GaussianFilter2D.h"
#include "DownsizeFilter2D.h"
#include "Vector2D.h"

class DownsampleFilter2D
{
public:
  DownsampleFilter2D();
  ~DownsampleFilter2D();
	
  void SetInput(const Array2D<float>&);
  Array2D<float>& GetOutput();
  void Update();
  void SetSigma(float,float);
  void SetSize(int,int);
  void SetFactor(int);
  void SetFactor(int,int);
  Vector2D<unsigned int> GetNewSize();

private:
  float SigmaX;
  float SigmaY;
  //  float SigmaZ;
  int KernelSizeX;
  int KernelSizeY;
  //int KernelSizeZ;
  int DownsizeFactorX;
  int DownsizeFactorY;
  //int DownsizeFactorZ;
  const Array2D<float>* inarray;
  Array2D<float> outarray;
};

#endif
