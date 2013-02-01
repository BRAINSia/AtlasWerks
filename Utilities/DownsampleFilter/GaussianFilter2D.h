// Description: 2D gaussian blur
// Author: Matthieu Jomier <mat-dev@jomier.com>	  
// Jacob Hinkle: converted to 2D, 2008
#ifndef GAUSSIANFILTER2D_H
#define GAUSSIANFILTER2D_H

#include "Array2D.h"
//#include "Vector2D.h"

class GaussianFilter2D
{
public:
  GaussianFilter2D();
  ~GaussianFilter2D();
  void SetInput(const Array2D<float>& );
  Array2D<float>& GetOutput();
  void Update();
  void setSigma(float, float);
  void setFactor(int,int);
  void setKernelSize(int, int);

private:
  float** filter_generate_gauss_full(float,int);
  void filter_symmetric_full(const Array2D<float> in, 
                             Array2D<float>& out, 
                             float sigma[2], 
                             int kernelSize[2]);
	
  const Array2D<float>* inarray;
  Array2D<float> outarray;
  int dim[2];
  float sigma[2];
  int DownsizeFactorX;
  int DownsizeFactorY;
  //  int DownsizeFactorZ;
  int kernelSize[2];
};

#endif

