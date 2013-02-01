#ifndef __VectorField3D_hxx
#define __VectorField3D_hxx

#include "Image3D.hxx"
#include <vector>

struct VectorField3D
{
  int xDim, yDim, zDim;
  Real deltaX, deltaY, deltaZ;
  Real * data;

  VectorField3D()
  {
    xDim = 0;
    yDim = 0;
    zDim = 0;
    deltaX = 1.0;
    deltaY = 1.0;
    deltaZ = 1.0;
    data = NULL;
  }

  VectorField3D(int _xDim, int _yDim, int _zDim,
                Real _deltaX = 1.0, Real _deltaY = 1.0, Real _deltaZ = 1.0)
  {
    xDim = _xDim;
    yDim = _yDim;
    zDim = _zDim;

    deltaX = _deltaX;
    deltaY = _deltaY;
    deltaZ = _deltaZ;

    data = new Real[xDim * yDim * zDim * 3];
    int i;
    for(i = 0; i < xDim * yDim * zDim * 3; i++)
      data[i] = 0.0;
  }

  // Gets linear interpolated values at x,y,z
  void getVals(Real & xVal, Real & yVal, Real & zVal,
               Real x, Real y, Real z) const;

  void read(const char * filename)
  {
    std::ifstream in(filename, std::ifstream::binary);
    in.read((char *) data, sizeof(Real) * xDim * yDim * zDim * 3);
  }

  void write(const char * filename)
  {
    std::ofstream out(filename, std::ofstream::binary);
    out.write((char *) data, sizeof(Real) * xDim * yDim * zDim * 3);
  }

  void minMax(Real & min, Real & max)
  {
    int i, size = xDim * yDim * zDim * 3;
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

// Returns v = v * im
void multVectorField(VectorField3D & v, const Image3D & im);

// Returns v = v * a
void multVectorField(VectorField3D & v, Real a);

// Returns result = result + v
void addVectorField(VectorField3D & result, const VectorField3D & v);

// Returns result = result - v
void subVectorField(VectorField3D & result, const VectorField3D & v);

// Nikhil: Returns maximum of the norms of all the vectors in a vector field
Real maxL2norm(VectorField3D * v);

Real l2normSqr(const VectorField3D & v);
Real l2DotProd(const VectorField3D & v1, const VectorField3D & v2);

// Compose v2(v1(x)), v1 and v2 are first optionally scaled
void compose(VectorField3D & result, VectorField3D & v1, VectorField3D & v2,
             Real v1Scale = 1.0, Real v2Scale = 1.0);

void integrate(VectorField3D & h,
               std::vector<VectorField3D *>::iterator start,
               std::vector<VectorField3D *>::iterator end);
void integrate(VectorField3D & h, std::vector<VectorField3D *> & v);
void integrateInv(VectorField3D & h, std::vector<VectorField3D *> & v);
void invert(VectorField3D & hInv, const VectorField3D & h);

VectorField3D * upsample(VectorField3D * v, int factor);
VectorField3D * upsampleSinc(VectorField3D * v, int factor);
VectorField3D * downsample(VectorField3D * v, int factor);

//Nikhil: add begin
VectorField3D * readVectorField3D(const char * filename);
//Nikhil: add end

void writeVectorField3D(const char * filename, const VectorField3D & v);
void writeComponentImages(const char * filename, const VectorField3D & v);

#endif
