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

#include "itkImageFileWriter.h"
#include "itkImageFileReader.h"
#include "itkImage.h"

#include "VectorField3D.hxx"

// Gets linear interpolated values at x,y,z
void VectorField3D::getVals(Real & xVal, Real & yVal, Real & zVal,
                            Real x, Real y, Real z) const
{
  Real d000, d001, d010, d011, d100, d101, d110, d111;
  Real d00x, d01x, d10x, d11x, d0yx, d1yx;

  Real voxelX = x / deltaX;
  Real voxelY = y / deltaY;
  Real voxelZ = z / deltaZ;

  int xInt = static_cast<int>(voxelX + xDim) % xDim;
  int yInt = static_cast<int>(voxelY + yDim) % yDim;
  int zInt = static_cast<int>(voxelZ + zDim) % zDim;

  //std::cout << xInt << ", " << yInt << ", " << zInt << std::endl;

  int xIntP = (xInt + 1) % xDim;
  int yIntP = (yInt + 1) % yDim;
  int zIntP = (zInt + 1) % zDim;

  Real xFrac = voxelX - static_cast<Real>(xInt);
  Real yFrac = voxelY - static_cast<Real>(yInt);
  Real zFrac = voxelZ - static_cast<Real>(zInt);

  //std::cout << xFrac << ", " << yFrac << ", " << zFrac << std::endl;

  if(xInt >= 0 && xInt < xDim && yInt >= 0 && yInt < yDim &&
     zInt >= 0 && zInt < zDim)
  {
    // component 0
    d000 = data[(xDim * (yDim * zInt + yInt) + xInt) * 3];
    d001 = data[(xDim * (yDim * zInt + yInt) + xIntP) * 3];
    d010 = data[(xDim * (yDim * zInt + yIntP) + xInt) * 3];
    d011 = data[(xDim * (yDim * zInt + yIntP) + xIntP) * 3];

    d100 = data[(xDim * (yDim * zIntP + yInt) + xInt) * 3];
    d101 = data[(xDim * (yDim * zIntP + yInt) + xIntP) * 3];
    d110 = data[(xDim * (yDim * zIntP + yIntP) + xInt) * 3];
    d111 = data[(xDim * (yDim * zIntP + yIntP) + xIntP) * 3];

    d00x = d000 + xFrac * (d001 - d000);
    d01x = d010 + xFrac * (d011 - d010);
    d10x = d100 + xFrac * (d101 - d100);
    d11x = d110 + xFrac * (d111 - d110);
    d0yx = d00x + yFrac * (d01x - d00x);
    d1yx = d10x + yFrac * (d11x - d10x);

    xVal = d0yx + zFrac * (d1yx - d0yx);

    // component 1
    d000 = data[(xDim * (yDim * zInt + yInt) + xInt) * 3 + 1];
    d001 = data[(xDim * (yDim * zInt + yInt) + xIntP) * 3 + 1];
    d010 = data[(xDim * (yDim * zInt + yIntP) + xInt) * 3 + 1];
    d011 = data[(xDim * (yDim * zInt + yIntP) + xIntP) * 3 + 1];

    d100 = data[(xDim * (yDim * zIntP + yInt) + xInt) * 3 + 1];
    d101 = data[(xDim * (yDim * zIntP + yInt) + xIntP) * 3 + 1];
    d110 = data[(xDim * (yDim * zIntP + yIntP) + xInt) * 3 + 1];
    d111 = data[(xDim * (yDim * zIntP + yIntP) + xIntP) * 3 + 1];

    d00x = d000 + xFrac * (d001 - d000);
    d01x = d010 + xFrac * (d011 - d010);
    d10x = d100 + xFrac * (d101 - d100);
    d11x = d110 + xFrac * (d111 - d110);
    d0yx = d00x + yFrac * (d01x - d00x);
    d1yx = d10x + yFrac * (d11x - d10x);

    yVal = d0yx + zFrac * (d1yx - d0yx);

    // component 2
    d000 = data[(xDim * (yDim * zInt + yInt) + xInt) * 3 + 2];
    d001 = data[(xDim * (yDim * zInt + yInt) + xIntP) * 3 + 2];
    d010 = data[(xDim * (yDim * zInt + yIntP) + xInt) * 3 + 2];
    d011 = data[(xDim * (yDim * zInt + yIntP) + xIntP) * 3 + 2];

    d100 = data[(xDim * (yDim * zIntP + yInt) + xInt) * 3 + 2];
    d101 = data[(xDim * (yDim * zIntP + yInt) + xIntP) * 3 + 2];
    d110 = data[(xDim * (yDim * zIntP + yIntP) + xInt) * 3 + 2];
    d111 = data[(xDim * (yDim * zIntP + yIntP) + xIntP) * 3 + 2];

    d00x = d000 + xFrac * (d001 - d000);
    d01x = d010 + xFrac * (d011 - d010);
    d10x = d100 + xFrac * (d101 - d100);
    d11x = d110 + xFrac * (d111 - d110);
    d0yx = d00x + yFrac * (d01x - d00x);
    d1yx = d10x + yFrac * (d11x - d10x);

    zVal = d0yx + zFrac * (d1yx - d0yx);
  }
}

// Returns v = v * im
void multVectorField(VectorField3D & v, const Image3D & im)
{
  int size = v.xDim * v.yDim * v.zDim;
  int i, index;

  for(i = 0, index = 0; i < size; i++, index+=3)
  {
    v.data[index] *= im.data[i];
    v.data[index+1] *= im.data[i];
    v.data[index+2] *= im.data[i];
  }
}

// Returns v = v * a
void multVectorField(VectorField3D & v, Real a)
{
  int size = v.xDim * v.yDim * v.zDim * 3;
  int i;

  for(i = 0; i < size; i++)
    v.data[i] *= a;
}

// Returns result = result + v
void addVectorField(VectorField3D & result, const VectorField3D & v)
{
  int size = v.xDim * v.yDim * v.zDim * 3;
  int i;

  for(i = 0; i < size; i++)
    result.data[i] += v.data[i];
}

// Returns result = result - v
void subVectorField(VectorField3D & result, const VectorField3D & v)
{
  int size = v.xDim * v.yDim * v.zDim * 3;
  int i;

  for(i = 0; i < size; i++)
    result.data[i] -= v.data[i];
}

// Nikhil: 10 June 2009 maxL2norm
// Returns maximum of the norms of all the vectors in a vector field
Real maxL2norm( VectorField3D* v)
{
  int i, j, k;
  int xDim = v->xDim;
  int yDim = v->yDim;
  int zDim = v->zDim;

  Real maxNorm = 0.0;
  Real tempNorm = 0.0;
  int index = 0;
  
  for(k = 0; k < zDim; k++)
  {
    for(j = 0; j < yDim; j++)
    {
      for(i = 0; i < xDim; i++)
      {
	tempNorm = sqrt((v->data[index] * v->data[index]) + (v->data[index+1] * v->data[index+1]) + (v->data[index+2] * v->data[index+2]));
	if(tempNorm>maxNorm)
		maxNorm = tempNorm;
	
        index += 3;
      }
    }
  }
  return maxNorm;
}


Real l2normSqr(const VectorField3D & v)
{
  int i, size = v.xDim * v.yDim * v.zDim * 3;
  Real norm = 0.0;
  Real fact = v.deltaX * v.deltaY * v.deltaZ;

  for(i = 0; i < size; i++)
    norm += v.data[i] * v.data[i] * fact;

  return norm;
}

Real l2DotProd(const VectorField3D & v1, const VectorField3D & v2)
{
  int i, size = v1.xDim * v1.yDim * v1.zDim * 3;
  Real dotProd = 0.0;
  Real fact = v1.deltaX * v1.deltaY * v1.deltaZ;

  for(i = 0; i < size; i++)
    dotProd += v1.data[i] * v2.data[i] * fact;

  return dotProd;
}

// Compose v2(v1(x)), v1 and v2 are first optionally scaled
void compose(VectorField3D & result, VectorField3D & v1, VectorField3D & v2,
             Real v1Scale, Real v2Scale)
{
  int i, j, k;
  int vIndex;
  Real x, y, z;
  Real xVal, yVal, zVal;

  vIndex = 0;
  for(k = 0; k < v1.zDim; k++)
  {
    z = static_cast<Real>(k) * v1.deltaZ;
    for(j = 0; j < v1.yDim; j++)
    {
      y = static_cast<Real>(j) * v1.deltaY;
      for(i = 0; i < v1.xDim; i++)
      {
        x = static_cast<Real>(i) * v1.deltaX;
        v2.getVals(xVal, yVal, zVal, x + v1.data[vIndex] * v1Scale,
                   y + v1.data[vIndex + 1] * v1Scale, z + v1.data[vIndex + 2] * v1Scale);

        //std::cout << v1.data[vIndex] << ", " << v1.data[vIndex + 1] << ", " << v1.data[vIndex + 2] << std::endl;
        //std::cout << x << ", " << y << ", " << z << " = " << xVal << ", " << yVal << ", " << zVal << std::endl;
        result.data[vIndex] = v1Scale * v1.data[vIndex] + v2Scale * xVal;
        result.data[vIndex + 1] = v1Scale * v1.data[vIndex + 1] + v2Scale * yVal;
        result.data[vIndex + 2] = v1Scale * v1.data[vIndex + 2] + v2Scale * zVal;

        vIndex += 3;
      }
    }
  }
}

// Integrates time dependent vector field v to give deformation field h
void integrate(VectorField3D & h,
               std::vector<VectorField3D *>::iterator start,
               std::vector<VectorField3D *>::iterator end)
{
  int i, j, k;
  int vIndex;
  Real x, y, z;
  Real xVal, yVal, zVal;
  std::vector<VectorField3D *>::iterator it;
  VectorField3D * u;

  //int hsize = h.xDim * h.yDim * h.zDim * 3;

  // Initialize h to zero
  //for(i = 0; i < hsize; i++)
  //  h.data[i] = 0.0;

  for(it = start; it != end; it++)
  {
    u = *it;
    vIndex = 0;
    for(k = 0; k < h.zDim; k++)
    {
      z = static_cast<Real>(k) * h.deltaZ;
      for(j = 0; j < h.yDim; j++)
      {
        y = static_cast<Real>(j) * h.deltaY;
        for(i = 0; i < h.xDim; i++)
        {
          x = static_cast<Real>(i) * h.deltaX;
          u->getVals(xVal, yVal, zVal, x + h.data[vIndex],
                     y + h.data[vIndex + 1], z + h.data[vIndex + 2]);

          h.data[vIndex] += xVal;
          h.data[vIndex + 1] += yVal;
          h.data[vIndex + 2] += zVal;

          vIndex += 3;
        }
      }
    }
  }
}

void integrate(VectorField3D & h, std::vector<VectorField3D *> & v)
{
  integrate(h, v.begin(), v.end());
}

// Integrates vector field v backwards to give deformation field h
void integrateInv(VectorField3D & h, std::vector<VectorField3D *> & v)
{
  int t, i, j, k;
  int vIndex;
  int timeSteps = v.size();
  Real x, y, z;
  Real xVal, yVal, zVal;
  VectorField3D * u;

  //int hsize = h.xDim * h.yDim * h.zDim * 3;

  // Initialize h to zero
  //for(i = 0; i < hsize; i++)
  //  h.data[i] = 0.0;

  for(t = timeSteps - 1; t >= 0; t--)
  {
    u = v[t];
    vIndex = 0;
    for(k = 0; k < h.zDim; k++)
    {
      z = static_cast<Real>(k) * h.deltaZ;
      for(j = 0; j < h.yDim; j++)
      {
        y = static_cast<Real>(j) * h.deltaY;
        for(i = 0; i < h.xDim; i++)
        {
          x = static_cast<Real>(i) * h.deltaX;
          u->getVals(xVal, yVal, zVal, x + h.data[vIndex],
                     y + h.data[vIndex + 1], z + h.data[vIndex + 2]);

          h.data[vIndex] -= xVal;
          h.data[vIndex + 1] -= yVal;
          h.data[vIndex + 2] -= zVal;

          vIndex += 3;
        }
      }
    }
  }
}

// Finds the inverse of the deformation field h
void invert(VectorField3D & hInv, const VectorField3D & h)
{
  int i, j, k;
  int hsize = h.xDim * h.yDim * h.zDim * 3;
  int vIndex;
  Real x, y, z;
  Real xVal, yVal, zVal;

  // Initialize as negative vector field
  for(i = 0; i < hsize; i++)
    hInv.data[i] = -h.data[i];

  vIndex = 0;
  for(k = 0; k < h.zDim; k++)
  {
    z = static_cast<Real>(k) * h.deltaZ;
    for(j = 0; j < h.yDim; j++)
    {
      y = static_cast<Real>(j) * h.deltaY;
      for(i = 0; i < h.xDim; i++)
      {
        x = static_cast<Real>(i) * h.deltaX;
        h.getVals(xVal, yVal, zVal, hInv.data[vIndex] + x,
                  hInv.data[vIndex+1] + y, hInv.data[vIndex+2] + z);

        xVal -= hInv.data[vIndex];
        yVal -= hInv.data[vIndex+1];
        zVal -= hInv.data[vIndex+2];

        hInv.data[vIndex] -= xVal;
        hInv.data[vIndex+1] -= yVal;
        hInv.data[vIndex+2] -= zVal;

        vIndex += 3;
      }
    }
  }
}

VectorField3D * downsample(VectorField3D * v, int factor)
{
  int xDim = v->xDim;
  int yDim = v->yDim;
  int zDim = v->zDim;

  int newXDim = xDim / factor;
  int newYDim = yDim / factor;
  int newZDim = zDim / factor;

  int i, j, k;

  int size = xDim * yDim * zDim * 3;
  int newSize = size / (factor * factor * factor);

  VectorField3D * newV = new VectorField3D(newXDim, newYDim, newZDim);
  newV->deltaX = v->deltaX * factor;
  newV->deltaY = v->deltaY * factor;
  newV->deltaZ = v->deltaZ * factor;
  Real * newData = newV->data;

  for(i = 0; i < newSize; i++)
    newData[i] = 0.0;

  int index = 0;
  int newIndex;
  Real divFactor = static_cast<Real>(factor * factor * factor);
  for(k = 0; k < zDim; k++)
  {
    for(j = 0; j < yDim; j++)
    {
      for(i = 0; i < xDim; i++)
      {
        newIndex = (i / factor + newXDim * ((j / factor) + newYDim * (k / factor))) * 3;
        newData[newIndex] += v->data[index] / divFactor;
        newData[newIndex + 1] += v->data[index + 1] / divFactor;
        newData[newIndex + 2] += v->data[index + 2] / divFactor;

        index += 3;
      }
    }
  }

  return newV;
}

VectorField3D * upsample(VectorField3D * v, int factor)
{
  int xDim = v->xDim;
  int yDim = v->yDim;
  int zDim = v->zDim;

  int newXDim = xDim * factor;
  int newYDim = yDim * factor;
  int newZDim = zDim * factor;

  int i, j, k;

  VectorField3D * newV = new VectorField3D(newXDim, newYDim, newZDim);
  newV->deltaX = v->deltaX / static_cast<Real>(factor);
  newV->deltaY = v->deltaY / static_cast<Real>(factor);
  newV->deltaZ = v->deltaZ / static_cast<Real>(factor);
  Real * newData = newV->data;
  Real * data = v->data;

  Real xFrac, yFrac, zFrac;
  Real x, y, z;
  Real d000, d001, d010, d011, d100, d101, d110, d111;
  Real d00x, d01x, d10x, d11x, d0yx, d1yx;
  int xInt, yInt, zInt, xIntP, yIntP, zIntP;
  int index = 0;
  for(k = 0; k < newZDim; k++)
  {
    // Add zDim to z coordinate to make sure it is positive (it's removed by mod later)
    z = (static_cast<Real>(k) - 0.5) / static_cast<Real>(factor) + static_cast<Real>(zDim);
    zInt = static_cast<int>(z);
    zFrac = z  - static_cast<Real>(zInt);
    zInt = zInt % zDim;
    //zFrac = static_cast<Real>(k % factor) / static_cast<Real>(factor);
    for(j = 0; j < newYDim; j++)
    {
      y = (static_cast<Real>(j) - 0.5) / static_cast<Real>(factor) + static_cast<Real>(yDim);
      yInt = static_cast<int>(y);
      yFrac = y  - static_cast<Real>(yInt);
      yInt = yInt % yDim;
      //yFrac = static_cast<Real>(j % factor) / static_cast<Real>(factor);
      for(i = 0; i < newXDim; i++)
      {
        x = (static_cast<Real>(i) - 0.5) / static_cast<Real>(factor) + static_cast<Real>(xDim);
        xInt = static_cast<int>(x);
        xFrac = x  - static_cast<Real>(xInt);
        xInt = xInt % xDim;
        //xFrac = static_cast<Real>(i % factor) / static_cast<Real>(factor);

        //xInt = (i / factor + xDim) % xDim;
        //yInt = (j / factor + yDim) % yDim;
        //zInt = (k / factor + zDim) % zDim;

        xIntP = (xInt + 1) % xDim;
        yIntP = (yInt + 1) % yDim;
        zIntP = (zInt + 1) % zDim;

        // component 0
        d000 = data[(xDim * (yDim * zInt + yInt) + xInt) * 3];
        d001 = data[(xDim * (yDim * zInt + yInt) + xIntP) * 3];
        d010 = data[(xDim * (yDim * zInt + yIntP) + xInt) * 3];
        d011 = data[(xDim * (yDim * zInt + yIntP) + xIntP) * 3];
        
        d100 = data[(xDim * (yDim * zIntP + yInt) + xInt) * 3];
        d101 = data[(xDim * (yDim * zIntP + yInt) + xIntP) * 3];
        d110 = data[(xDim * (yDim * zIntP + yIntP) + xInt) * 3];
        d111 = data[(xDim * (yDim * zIntP + yIntP) + xIntP) * 3];

        d00x = d000 + xFrac * (d001 - d000);
        d01x = d010 + xFrac * (d011 - d010);
        d10x = d100 + xFrac * (d101 - d100);
        d11x = d110 + xFrac * (d111 - d110);
        d0yx = d00x + yFrac * (d01x - d00x);
        d1yx = d10x + yFrac * (d11x - d10x);

        newData[index++] = d0yx + zFrac * (d1yx - d0yx);

        // component 1
        d000 = data[(xDim * (yDim * zInt + yInt) + xInt) * 3 + 1];
        d001 = data[(xDim * (yDim * zInt + yInt) + xIntP) * 3 + 1];
        d010 = data[(xDim * (yDim * zInt + yIntP) + xInt) * 3 + 1];
        d011 = data[(xDim * (yDim * zInt + yIntP) + xIntP) * 3 + 1];
        
        d100 = data[(xDim * (yDim * zIntP + yInt) + xInt) * 3 + 1];
        d101 = data[(xDim * (yDim * zIntP + yInt) + xIntP) * 3 + 1];
        d110 = data[(xDim * (yDim * zIntP + yIntP) + xInt) * 3 + 1];
        d111 = data[(xDim * (yDim * zIntP + yIntP) + xIntP) * 3 + 1];

        d00x = d000 + xFrac * (d001 - d000);
        d01x = d010 + xFrac * (d011 - d010);
        d10x = d100 + xFrac * (d101 - d100);
        d11x = d110 + xFrac * (d111 - d110);
        d0yx = d00x + yFrac * (d01x - d00x);
        d1yx = d10x + yFrac * (d11x - d10x);

        newData[index++] = d0yx + zFrac * (d1yx - d0yx);

        // component 2
        d000 = data[(xDim * (yDim * zInt + yInt) + xInt) * 3 + 2];
        d001 = data[(xDim * (yDim * zInt + yInt) + xIntP) * 3 + 2];
        d010 = data[(xDim * (yDim * zInt + yIntP) + xInt) * 3 + 2];
        d011 = data[(xDim * (yDim * zInt + yIntP) + xIntP) * 3 + 2];
        
        d100 = data[(xDim * (yDim * zIntP + yInt) + xInt) * 3 + 2];
        d101 = data[(xDim * (yDim * zIntP + yInt) + xIntP) * 3 + 2];
        d110 = data[(xDim * (yDim * zIntP + yIntP) + xInt) * 3 + 2];
        d111 = data[(xDim * (yDim * zIntP + yIntP) + xIntP) * 3 + 2];

        d00x = d000 + xFrac * (d001 - d000);
        d01x = d010 + xFrac * (d011 - d010);
        d10x = d100 + xFrac * (d101 - d100);
        d11x = d110 + xFrac * (d111 - d110);
        d0yx = d00x + yFrac * (d01x - d00x);
        d1yx = d10x + yFrac * (d11x - d10x);

        newData[index++] = d0yx + zFrac * (d1yx - d0yx);
      }
    }
  }

  return newV;
}

VectorField3D * upsampleSinc(VectorField3D * v, int factor)
{
  unsigned int xDim = v->xDim;
  unsigned int yDim = v->yDim;
  unsigned int zDim = v->zDim;

  unsigned int newXDim = xDim * factor;
  unsigned int newYDim = yDim * factor;
  unsigned int newZDim = zDim * factor;

  VectorField3D * newV = new VectorField3D(newXDim, newYDim, newZDim);
  newV->deltaX = v->deltaX / static_cast<Real>(factor);
  newV->deltaY = v->deltaY / static_cast<Real>(factor);
  newV->deltaZ = v->deltaZ / static_cast<Real>(factor);
  
  Real * scratch, * newScratch,* newVecFieldComplexData,* oldVecFieldComplexData;
  unsigned int sizeComplexOld = (xDim+1) * (yDim+1) * (zDim+1) * 2 * 3;
  unsigned int sizeComplexNew = (newXDim+1) * (newYDim+1) * (newZDim+1) * 2 * 3;
  scratch = new Real[sizeComplexOld];
  newScratch = new Real[sizeComplexNew];
  oldVecFieldComplexData = new Real[sizeComplexOld];
  newVecFieldComplexData = new Real[sizeComplexNew];
  fftwf_plan fftwForwardPlan;
  fftwf_plan fftwBackwardPlan;
  fftwf_plan_with_nthreads(2);
  
  unsigned int index, INDEX;  
  unsigned int x, y, z;
  int X, Y, Z;
  index = 0;INDEX=0;
  for ( z = 0; z < zDim+1; ++z)
  {
    for ( y = 0; y < yDim+1; ++y)
    {
      for ( x = 0; x < xDim+1; ++x)
      {
	index = 3*(x+((xDim+1) * (y + (yDim+1) * z)));
	if(x==0||y==0||z==0)
	{
		oldVecFieldComplexData[2*index] = static_cast<Real>(0); 
		oldVecFieldComplexData[2*index+2] = static_cast<Real>(0); 
		oldVecFieldComplexData[2*index+4] = static_cast<Real>(0); 	  	
	}
	else
	{
  		INDEX = 3*(x-1+(xDim * (y-1 + yDim * (z-1))));
		oldVecFieldComplexData[2*index] = v->data[INDEX]; 
	  	oldVecFieldComplexData[2*index+2] = v->data[INDEX+1]; 
	  	oldVecFieldComplexData[2*index+4] = v->data[INDEX+2]; 
	}
	oldVecFieldComplexData[2*index+1] = static_cast<Real>(0); 
	oldVecFieldComplexData[2*index+3] = static_cast<Real>(0); 
	oldVecFieldComplexData[2*index+5] = static_cast<Real>(0); 
      }
    }
  }  
  int dims[3];
  dims[0] = zDim+1;
  dims[1] = yDim+1;
  dims[2] = xDim+1;
  int newDims[3];
  newDims[0] = newZDim+1;
  newDims[1] = newYDim+1;
  newDims[2] = newXDim+1;

  fftwForwardPlan = fftwf_plan_many_dft(3, dims, 3, (fftwf_complex *)oldVecFieldComplexData, 0, 3, 1, (fftwf_complex *)(scratch),0, 3, 1, -1 , FFTW_ESTIMATE);
  fftwBackwardPlan = fftwf_plan_many_dft(3, newDims, 3, (fftwf_complex *)(newScratch), 0, 3, 1, (fftwf_complex *)newVecFieldComplexData, 0, 3, 1, +1, FFTW_ESTIMATE);
  std::cout << "Plans of interpolation for upsampling vector field done" << std::endl;

  fftwf_execute(fftwForwardPlan);

  //Pad the frequency response with zeros
  index = 0;INDEX = 0;
  unsigned int lowerX, upperX, lowerY, upperY, lowerZ, upperZ;
  lowerX = static_cast<unsigned int>(xDim/2+1); upperX = static_cast<unsigned int>((factor-0.5)*xDim);
  lowerY = static_cast<unsigned int>(yDim/2+1); upperY = static_cast<unsigned int>((factor-0.5)*yDim);
  lowerZ = static_cast<unsigned int>(zDim/2+1); upperZ = static_cast<unsigned int>((factor-0.5)*zDim);

  for ( z = 0; z < newZDim; ++z)
  {
    for ( y = 0; y < newYDim; ++y)
	{
	  for ( x = 0; x < newXDim; ++x)
	  {
		index = 6*(x+((newXDim+1) * (y + (newYDim+1) * z)));
		
		if(x < lowerX && y<lowerY && z< lowerZ)
		{
			X = x; Y = y; Z = z;
			INDEX = 6*(X+((xDim+1) * (Y + (yDim+1) * Z)));
			newScratch[index] = scratch[INDEX];
			newScratch[index+1] = scratch[INDEX+1];
			newScratch[index+2] = scratch[INDEX+2];
			newScratch[index+3] = scratch[INDEX+3];
			newScratch[index+4] = scratch[INDEX+4];
			newScratch[index+5] = scratch[INDEX+5];

		}			
		else if(x > upperX && y>upperY && z> upperZ)
		{
			X = x-(factor-1)*xDim; Y = y-(factor-1)*yDim; Z = z-(factor-1)*zDim;
			INDEX = 6*(X+((xDim+1) * (Y + (yDim+1) * Z)));
			newScratch[index] = scratch[INDEX];
			newScratch[index+1] = scratch[INDEX+1];
			newScratch[index+2] = scratch[INDEX+2];
			newScratch[index+3] = scratch[INDEX+3];
			newScratch[index+4] = scratch[INDEX+4];
			newScratch[index+5] = scratch[INDEX+5];
		}
		else if(x < lowerX && y<lowerY && z> upperZ)
		{
			X = x; Y = y; Z = z-(factor-1)*zDim;
			INDEX = 6*(X+((xDim+1) * (Y + (yDim+1) * Z)));
			newScratch[index] = scratch[INDEX];
			newScratch[index+1] = scratch[INDEX+1];
			newScratch[index+2] = scratch[INDEX+2];
			newScratch[index+3] = scratch[INDEX+3];
			newScratch[index+4] = scratch[INDEX+4];
			newScratch[index+5] = scratch[INDEX+5];		}
		else if(x < lowerX && y>upperY && z< lowerZ)
		{
			X = x; Y = y-(factor-1)*yDim; Z = z;
			INDEX = 6*(X+((xDim+1) * (Y + (yDim+1) * Z)));
			newScratch[index] = scratch[INDEX];
			newScratch[index+1] = scratch[INDEX+1];
			newScratch[index+2] = scratch[INDEX+2];
			newScratch[index+3] = scratch[INDEX+3];
			newScratch[index+4] = scratch[INDEX+4];
			newScratch[index+5] = scratch[INDEX+5];
		}
		else if(x > upperX && y<lowerY && z< lowerZ)
		{
			X = x-(factor-1)*xDim; Y = y; Z = z;
			INDEX = 6*(X+((xDim+1) * (Y + (yDim+1) * Z)));
			newScratch[index] = scratch[INDEX];
			newScratch[index+1] = scratch[INDEX+1];
			newScratch[index+2] = scratch[INDEX+2];
			newScratch[index+3] = scratch[INDEX+3];
			newScratch[index+4] = scratch[INDEX+4];
			newScratch[index+5] = scratch[INDEX+5];
 		}
		else if(x < lowerX && y>upperY && z> upperZ)
		{
			X = x; Y = y-(factor-1)*yDim; Z = z-(factor-1)*zDim;
			INDEX = 6*(X+((xDim+1) * (Y + (yDim+1) * Z)));
			newScratch[index] = scratch[INDEX];
			newScratch[index+1] = scratch[INDEX+1];
			newScratch[index+2] = scratch[INDEX+2];
			newScratch[index+3] = scratch[INDEX+3];
			newScratch[index+4] = scratch[INDEX+4];
			newScratch[index+5] = scratch[INDEX+5];		
		}
		else if(x > upperX && y<lowerY && z> upperZ)
		{
			X = x-(factor-1)*xDim; Y = y; Z = z-(factor-1)*zDim;
			INDEX = 6*(X+((xDim+1) * (Y + (yDim+1) * Z)));
			newScratch[index] = scratch[INDEX];
			newScratch[index+1] = scratch[INDEX+1];
			newScratch[index+2] = scratch[INDEX+2];
			newScratch[index+3] = scratch[INDEX+3];
			newScratch[index+4] = scratch[INDEX+4];
			newScratch[index+5] = scratch[INDEX+5];
		}
		else if(x > upperX && y>upperY && z< lowerZ)
		{
			X = x-(factor-1)*xDim; Y = y-(factor-1)*yDim; Z = z;
			INDEX = 6*(X+((xDim+1) * (Y + (yDim+1) * Z)));
			newScratch[index] = scratch[INDEX];
			newScratch[index+1] = scratch[INDEX+1];
			newScratch[index+2] = scratch[INDEX+2];
			newScratch[index+3] = scratch[INDEX+3];
			newScratch[index+4] = scratch[INDEX+4];
			newScratch[index+5] = scratch[INDEX+5];
		}		
		else
		{
			newScratch[index] = static_cast<Real>(0);
			newScratch[index+1] = static_cast<Real>(0);
			newScratch[index+2] = static_cast<Real>(0);
			newScratch[index+3] = static_cast<Real>(0);
			newScratch[index+4] = static_cast<Real>(0);
			newScratch[index+5] = static_cast<Real>(0);
		}
	  }
	}
  }
  fftwf_execute(fftwBackwardPlan);
  
  for ( z = 0; z < newZDim+1; ++z)
  {
    for ( y = 0; y < newYDim+1; ++y)
    {
      for ( x = 0; x < newXDim+1; ++x)
      {
	index = 3*(x+((newXDim+1) * (y + (newYDim+1) * z)));
	if(!(x==0||y==0||z==0))
	{
	  INDEX = 3*(x-1+(newXDim * (y-1 + newYDim * (z-1))));
	  //Ignoring the imaginary part as we know that the result would be real
    	  newV->data[INDEX] = newVecFieldComplexData[2*index];
  	  newV->data[INDEX+1] = newVecFieldComplexData[2*index+2];
  	  newV->data[INDEX+2] = newVecFieldComplexData[2*index+4]; 
	}
      }
    }
  }

  multVectorField(*newV,  1/static_cast<Real>(xDim*yDim*zDim));  
  delete [] scratch;
  delete [] newScratch;
  delete [] oldVecFieldComplexData;
  delete [] newVecFieldComplexData;  
  return newV;
}

//Nikhil: add begin
VectorField3D * readVectorField3D(const char * filename)
{
  if(filename == NULL)
    return NULL;
  
  typedef itk::Vector<Real, 3> VectorType;
  typedef itk::Image<VectorType, 3> VectorImageType;
  typedef itk::ImageFileReader< VectorImageType > ReaderType;
  ReaderType::Pointer reader = ReaderType::New();

  reader->SetFileName(filename);
  try
  {
    reader->Update();
  }
  catch(itk::ExceptionObject & e)
  {
    std::cerr << "Error reading vector image: " << e << std::endl;
    return NULL;
  }

  VectorImageType::Pointer itkImage = reader->GetOutput();
  VectorImageType::SizeType size = itkImage->GetLargestPossibleRegion().GetSize();
  VectorField3D * vectorImage = new VectorField3D(size[0], size[1], size[2]);

  unsigned int i, j, k, index = 0;
  for(k = 0; k < size[2]; k++)
  {
    for(j = 0; j < size[1]; j++)
    {
      for(i = 0; i < size[0]; i++)
      {
        VectorImageType::IndexType itkIndex;
        itkIndex[0] = i; itkIndex[1] = j; itkIndex[2] = k;
        vectorImage->data[index++] = itkImage->GetPixel(itkIndex)[0];
	vectorImage->data[index++] = itkImage->GetPixel(itkIndex)[1];
	vectorImage->data[index++] = itkImage->GetPixel(itkIndex)[2];
      }
    }
  }

  return vectorImage;
}
//Nikhil: add end
void writeVectorField3D(const char * filename, const VectorField3D & v)
{
  typedef itk::Vector<Real, 3> VectorType;
  typedef itk::Image<VectorType, 3> VectorImageType;
  VectorImageType::Pointer image = VectorImageType::New();

  VectorImageType::SizeType size;
  size[0] = v.xDim;
  size[1] = v.yDim;
  size[2] = v.zDim;
  VectorImageType::IndexType orig;
  orig[0] = 0;
  orig[1] = 0;
  orig[2] = 0;
  VectorImageType::RegionType region;
  region.SetSize(size);
  region.SetIndex(orig);
  image->SetRegions(region);
  image->Allocate();
  VectorImageType::SpacingType spacing;
  spacing[0] = v.deltaX;
  spacing[1] = v.deltaY;
  spacing[2] = v.deltaZ;
  image->SetSpacing(spacing);

  VectorImageType::IndexType index;
  VectorImageType::PixelType val;
  int i, j, k, vIndex = 0;
  for(k = 0; k < v.zDim; k++)
  {
    for(j = 0; j < v.yDim; j++)
    {
      for(i = 0; i < v.xDim; i++)
      {
        index[0] = i;
        index[1] = j;
        index[2] = k;

        val[0] = v.data[vIndex++];
        val[1] = v.data[vIndex++];
        val[2] = v.data[vIndex++];
        image->SetPixel(index, val);
      }
    }
  }

  itk::ImageFileWriter<VectorImageType>::Pointer VolWriter
    = itk::ImageFileWriter<VectorImageType>::New();
  VolWriter->SetFileName(filename);
  VolWriter->SetInput(image);
  //VolWriter->UseCompressionOn();
  try
  {
    VolWriter->Update();
  }
  catch (itk::ExceptionObject &ex)
  {
    std::cout << ex << std::endl;
  }
}

void writeComponentImages(const char * filename, const VectorField3D & v)
{
  int i;
  int size = v.xDim * v.yDim * v.zDim;

  Image3D imX(v.xDim, v.yDim, v.zDim, v.deltaX, v.deltaY, v.deltaZ);
  Image3D imY(v.xDim, v.yDim, v.zDim, v.deltaX, v.deltaY, v.deltaZ);
  Image3D imZ(v.xDim, v.yDim, v.zDim, v.deltaX, v.deltaY, v.deltaZ);

  for(i = 0; i < size; i++)
  {
    imX.data[i] = v.data[i*3];
    imY.data[i] = v.data[i*3 + 1];
    imZ.data[i] = v.data[i*3 + 2];
  }

  char outFilename[1024];
  sprintf(outFilename, "%s_x.nhdr", filename);
  writeMetaImage(outFilename, &imX);
  sprintf(outFilename, "%s_y.nhdr", filename);
  writeMetaImage(outFilename, &imY);
  sprintf(outFilename, "%s_z.nhdr", filename);
  writeMetaImage(outFilename, &imZ);
}
