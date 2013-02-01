// Description: 3D gaussian blur
// Author: Matthieu Jomier <mat-dev@jomier.com>	  
#ifndef GAUSSIANFILTER3D_H
#define GAUSSIANFILTER3D_H

#include "Array3D.h"
#include "Vector2D.h"

class GaussianFilter3D
{
public:
  GaussianFilter3D();
  ~GaussianFilter3D();
  void SetInput(const Array3D<float>& );
  Array3D<float>& GetOutput();
  void Update();
  void setSigma(float, float, float);
  void setFactor(int,int,int);
  void setKernelSize(int, int, int);

private:
  float** filter_generate_gauss_full(float,int);
  void filter_symmetric_full(const Array3D<float> in, 
                             Array3D<float>& out, 
                             float sigma[3], 
                             int kernelSize[3]);
	
  const Array3D<float>* inarray;
  Array3D<float> outarray;
  int dim[3];
  float sigma[3];
  int DownsizeFactorX;
  int DownsizeFactorY;
  int DownsizeFactorZ;
  int kernelSize[3];
};

#endif

