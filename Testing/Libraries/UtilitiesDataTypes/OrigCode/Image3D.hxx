#ifndef __Image3D_hxx
#define __Image3D_hxx

#include <iostream>
#include <fstream>
#include <fftw3.h>
typedef float Real;

struct Image3D
{
  unsigned int xDim, yDim, zDim;
  Real deltaX, deltaY, deltaZ;
  Real xOrig, yOrig, zOrig;
  Real * data;

  Image3D()
  {
    xDim = 0;
    yDim = 0;
    zDim = 0;
    deltaX = 1.0;
    deltaY = 1.0;
    deltaZ = 1.0;
    data = NULL;
  }

  Image3D(unsigned int _xDim, unsigned int _yDim, unsigned int _zDim,
          Real _deltaX = 1.0, Real _deltaY = 1.0, Real _deltaZ = 1.0)
  {
    xDim = _xDim;
    yDim = _yDim;
    zDim = _zDim;

    deltaX = _deltaX;
    deltaY = _deltaY;
    deltaZ = _deltaZ;

    data = new Real[xDim * yDim * zDim];
  }

  // Gets linear interpolated value at x,y,z
  Real getVal(Real x, Real y, Real z) const;

  void minMax(Real & min, Real & max)
  {
    int i, size = xDim * yDim * zDim;
    min = data[0];
    max = data[0];
    for(i = 1; i < size; i++)
    {
      if(data[i] < min)
         min = data[i];
      if(data[i] > max)
        max = data[i];
    }
  }
};

void copyImage(Image3D & result, const Image3D & im);

// Returns result = result + im
void addImage(Image3D & result, const Image3D & im);

// Returns result = result + a
void addImage(Image3D & result, Real a);

// Returns result = result - im
void subImage(Image3D & result, const Image3D & im);

// Returns im = a * im
void multImage(Image3D & im, Real a);

// Returns im1 = im1 * im2
void multImage(Image3D & im1, Image3D & im2);

Real l2normSqr(const Image3D & im);

Real l2DotProd(const Image3D & im1, const Image3D & im2);

Image3D * downsample(Image3D * image, unsigned int factor);
Image3D * upsample(Image3D * image, unsigned int factor);
Image3D * upsampleSinc(Image3D * image, unsigned int factor);

Image3D * gaussBlur(Image3D * image, int width, Real sigma);

Image3D * readMetaImage(const char * filename);
void writeMetaImage(const char * filePrefix, Image3D * image);

#endif
