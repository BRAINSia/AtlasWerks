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

#include "MetamorphosisRoutines.hxx"

// Compute gradient of u with cyclic boundary
// Assumes gradU is already allocated
void grad(const Image3D & u, VectorField3D & gradU)
{
  int i, j, k;
  int prevI, prevJ, prevK;
  int nextI, nextJ, nextK;
  int sliceSize = u.xDim * u.yDim;
  int sliceIndex, rowIndex, inSliceIndex;
  int index;

  Real xFact = 0.5 / u.deltaX;
  Real yFact = 0.5 / u.deltaY;
  Real zFact = 0.5 / u.deltaZ;

  Real * gradData = gradU.data;

  sliceIndex = 0;
  rowIndex = 0;
  index = 0;
  for(k = 0; k < static_cast<int>(u.zDim); k++)
  {
    prevK = ((u.zDim + k - 1) % u.zDim) * sliceSize;
    nextK = ((k + 1) % u.zDim) * sliceSize;

    inSliceIndex = 0;
    for(j = 0; j < static_cast<int>(u.yDim); j++)
    {
      prevJ = ((u.yDim + j - 1) % u.yDim) * u.xDim + sliceIndex;
      nextJ = ((j + 1) % u.yDim) * u.xDim + sliceIndex;

      for(i = 0; i < static_cast<int>(u.xDim); i++)
      {
        prevI = (u.xDim + i - 1) % u.xDim;
        nextI = (i + 1) % u.xDim;

        gradData[index] =
          (u.data[rowIndex + nextI] - u.data[rowIndex + prevI]) * xFact;
        gradData[index + 1] =
          (u.data[nextJ + i] - u.data[prevJ + i]) * yFact;
        gradData[index + 2] = 
          (u.data[nextK + inSliceIndex] - u.data[prevK + inSliceIndex]) * zFact;

        index += 3;
        inSliceIndex++;
      }

      rowIndex += u.xDim;
    }

    sliceIndex += sliceSize;
  }
}

// Pull back of an image by a vector field
void pullBack(const Image3D & im, const VectorField3D & v, Image3D & result)
{
  unsigned int i, j, k;
  int xInt, yInt, zInt;
  int xIntP, yIntP, zIntP;
  int index, vIndex;
  Real x, y, z;
  Real xFrac, yFrac, zFrac;

  Real d000, d001, d010, d011, d100, d101, d110, d111;
  Real d00x, d01x, d10x, d11x, d0yx, d1yx;

  index = 0;
  vIndex = 0;
  for(k = 0; k < im.zDim; k++)
  {
    for(j = 0; j < im.yDim; j++)
    {
      for(i = 0; i < im.xDim; i++)
      {
        x = static_cast<Real>(i + im.xDim) + v.data[vIndex] / v.deltaX;
        y = static_cast<Real>(j + im.yDim) + v.data[vIndex + 1] / v.deltaY;
        z = static_cast<Real>(k + im.zDim) + v.data[vIndex + 2] / v.deltaZ;

        xInt = static_cast<int>(x);
        yInt = static_cast<int>(y);
        zInt = static_cast<int>(z);

        xFrac = x - static_cast<Real>(xInt);
        yFrac = y - static_cast<Real>(yInt);
        zFrac = z - static_cast<Real>(zInt);

        // Wrap around indices -- ASSUMES NO FARTHER THAN ONE WRAP AROUND
        xInt = xInt % im.xDim;
        yInt = yInt % im.yDim;
        zInt = zInt % im.zDim;

        xIntP = (xInt + 1) % im.xDim;
        yIntP = (yInt + 1) % im.yDim;
        zIntP = (zInt + 1) % im.zDim;

        d000 = im.data[im.xDim * (im.yDim * zInt + yInt) + xInt];
        d001 = im.data[im.xDim * (im.yDim * zInt + yInt) + xIntP];
        d010 = im.data[im.xDim * (im.yDim * zInt + yIntP) + xInt];
        d011 = im.data[im.xDim * (im.yDim * zInt + yIntP) + xIntP];
        
        d100 = im.data[im.xDim * (im.yDim * zIntP + yInt) + xInt];
        d101 = im.data[im.xDim * (im.yDim * zIntP + yInt) + xIntP];
        d110 = im.data[im.xDim * (im.yDim * zIntP + yIntP) + xInt];
        d111 = im.data[im.xDim * (im.yDim * zIntP + yIntP) + xIntP];

        d00x = d000 + xFrac * (d001 - d000);
        d01x = d010 + xFrac * (d011 - d010);
        d10x = d100 + xFrac * (d101 - d100);
        d11x = d110 + xFrac * (d111 - d110);
        d0yx = d00x + yFrac * (d01x - d00x);
        d1yx = d10x + yFrac * (d11x - d10x);

        result.data[index] = d0yx + zFrac * (d1yx - d0yx);
        index++;
        vIndex += 3;
      }
    }
  }
}

// Pull back of an image by a vector field
void pullBackClamp(const Image3D & im, const VectorField3D & v, Image3D & result)
{
  unsigned int i, j, k;
  int xInt, yInt, zInt;
  int xIntP, yIntP, zIntP;
  int index, vIndex;
  Real x, y, z;
  Real xFrac, yFrac, zFrac;

  Real d000, d001, d010, d011, d100, d101, d110, d111;
  Real d00x, d01x, d10x, d11x, d0yx, d1yx;

  Real xMax = static_cast<Real>(im.xDim - 1);
  Real yMax = static_cast<Real>(im.yDim - 1);
  Real zMax = static_cast<Real>(im.zDim - 1);

  index = 0;
  vIndex = 0;
  for(k = 0; k < im.zDim; k++)
  {
    for(j = 0; j < im.yDim; j++)
    {
      for(i = 0; i < im.xDim; i++)
      {
        x = static_cast<Real>(i) + v.data[vIndex] / v.deltaX;
        y = static_cast<Real>(j) + v.data[vIndex + 1] / v.deltaY;
        z = static_cast<Real>(k) + v.data[vIndex + 2] / v.deltaZ;

        if(x < 0.0)
          x = 0.0;
        if(x > xMax)
          x = xMax;
        if(y < 0.0)
          y = 0.0;
        if(y > yMax)
          y = yMax;
        if(z < 0.0)
          z = 0.0;
        if(z > zMax)
          z = zMax;
        
        xInt = static_cast<int>(x);
        yInt = static_cast<int>(y);
        zInt = static_cast<int>(z);

        xFrac = x - static_cast<Real>(xInt);
        yFrac = y - static_cast<Real>(yInt);
        zFrac = z - static_cast<Real>(zInt);

        // Wrap around indices -- ASSUMES NO FARTHER THAN ONE WRAP AROUND
        xInt = xInt % im.xDim;
        yInt = yInt % im.yDim;
        zInt = zInt % im.zDim;

        xIntP = (xInt + 1) % im.xDim;
        yIntP = (yInt + 1) % im.yDim;
        zIntP = (zInt + 1) % im.zDim;

        d000 = im.data[im.xDim * (im.yDim * zInt + yInt) + xInt];
        d001 = im.data[im.xDim * (im.yDim * zInt + yInt) + xIntP];
        d010 = im.data[im.xDim * (im.yDim * zInt + yIntP) + xInt];
        d011 = im.data[im.xDim * (im.yDim * zInt + yIntP) + xIntP];
        
        d100 = im.data[im.xDim * (im.yDim * zIntP + yInt) + xInt];
        d101 = im.data[im.xDim * (im.yDim * zIntP + yInt) + xIntP];
        d110 = im.data[im.xDim * (im.yDim * zIntP + yIntP) + xInt];
        d111 = im.data[im.xDim * (im.yDim * zIntP + yIntP) + xIntP];

        d00x = d000 + xFrac * (d001 - d000);
        d01x = d010 + xFrac * (d011 - d010);
        d10x = d100 + xFrac * (d101 - d100);
        d11x = d110 + xFrac * (d111 - d110);
        d0yx = d00x + yFrac * (d01x - d00x);
        d1yx = d10x + yFrac * (d11x - d10x);

        result.data[index] = d0yx + zFrac * (d1yx - d0yx);
        index++;
        vIndex += 3;
      }
    }
  }
}

// Pull back of an image and a vector field by a vector field
void pullBack(const Image3D & im, const VectorField3D & v, Image3D & resultI,
              const VectorField3D & u, VectorField3D & resultU)
{
  unsigned int i, j, k;
  int xInt, yInt, zInt;
  int xIntP, yIntP, zIntP;
  int index, vIndex;
  Real x, y, z;
  Real xFrac, yFrac, zFrac;

  Real d000, d001, d010, d011, d100, d101, d110, d111;
  Real d00x, d01x, d10x, d11x, d0yx, d1yx;

  index = 0;
  vIndex = 0;
  for(k = 0; k < im.zDim; k++)
  {
    for(j = 0; j < im.yDim; j++)
    {
      for(i = 0; i < im.xDim; i++)
      {
        x = static_cast<Real>(i + im.xDim) + v.data[vIndex] / v.deltaX;
        y = static_cast<Real>(j + im.yDim) + v.data[vIndex + 1] / v.deltaY;
        z = static_cast<Real>(k + im.zDim) + v.data[vIndex + 2] / v.deltaZ;

        xInt = static_cast<int>(x);
        yInt = static_cast<int>(y);
        zInt = static_cast<int>(z);

        xFrac = x - static_cast<Real>(xInt);
        yFrac = y - static_cast<Real>(yInt);
        zFrac = z - static_cast<Real>(zInt);

        // Wrap around indices -- ASSUMES NO FARTHER THAN ONE WRAP AROUND
        xInt = xInt % im.xDim;
        yInt = yInt % im.yDim;
        zInt = zInt % im.zDim;

        xIntP = (xInt + 1) % im.xDim;
        yIntP = (yInt + 1) % im.yDim;
        zIntP = (zInt + 1) % im.zDim;

        // Compute image result
        d000 = im.data[im.xDim * (im.yDim * zInt + yInt) + xInt];
        d001 = im.data[im.xDim * (im.yDim * zInt + yInt) + xIntP];
        d010 = im.data[im.xDim * (im.yDim * zInt + yIntP) + xInt];
        d011 = im.data[im.xDim * (im.yDim * zInt + yIntP) + xIntP];
        
        d100 = im.data[im.xDim * (im.yDim * zIntP + yInt) + xInt];
        d101 = im.data[im.xDim * (im.yDim * zIntP + yInt) + xIntP];
        d110 = im.data[im.xDim * (im.yDim * zIntP + yIntP) + xInt];
        d111 = im.data[im.xDim * (im.yDim * zIntP + yIntP) + xIntP];

        d00x = d000 + xFrac * (d001 - d000);
        d01x = d010 + xFrac * (d011 - d010);
        d10x = d100 + xFrac * (d101 - d100);
        d11x = d110 + xFrac * (d111 - d110);
        d0yx = d00x + yFrac * (d01x - d00x);
        d1yx = d10x + yFrac * (d11x - d10x);

        resultI.data[index] = d0yx + zFrac * (d1yx - d0yx);

        // Compute vector field result (component 0)
        d000 = u.data[(u.xDim * (u.yDim * zInt + yInt) + xInt) * 3];
        d001 = u.data[(u.xDim * (u.yDim * zInt + yInt) + xIntP) * 3];
        d010 = u.data[(u.xDim * (u.yDim * zInt + yIntP) + xInt) * 3];
        d011 = u.data[(u.xDim * (u.yDim * zInt + yIntP) + xIntP) * 3];
        
        d100 = u.data[(u.xDim * (u.yDim * zIntP + yInt) + xInt) * 3];
        d101 = u.data[(u.xDim * (u.yDim * zIntP + yInt) + xIntP) * 3];
        d110 = u.data[(u.xDim * (u.yDim * zIntP + yIntP) + xInt) * 3];
        d111 = u.data[(u.xDim * (u.yDim * zIntP + yIntP) + xIntP) * 3];

        d00x = d000 + xFrac * (d001 - d000);
        d01x = d010 + xFrac * (d011 - d010);
        d10x = d100 + xFrac * (d101 - d100);
        d11x = d110 + xFrac * (d111 - d110);
        d0yx = d00x + yFrac * (d01x - d00x);
        d1yx = d10x + yFrac * (d11x - d10x);

        resultU.data[vIndex] = d0yx + zFrac * (d1yx - d0yx);

        // component 1
        d000 = u.data[(u.xDim * (u.yDim * zInt + yInt) + xInt) * 3 + 1];
        d001 = u.data[(u.xDim * (u.yDim * zInt + yInt) + xIntP) * 3 + 1];
        d010 = u.data[(u.xDim * (u.yDim * zInt + yIntP) + xInt) * 3 + 1];
        d011 = u.data[(u.xDim * (u.yDim * zInt + yIntP) + xIntP) * 3 + 1];
        
        d100 = u.data[(u.xDim * (u.yDim * zIntP + yInt) + xInt) * 3 + 1];
        d101 = u.data[(u.xDim * (u.yDim * zIntP + yInt) + xIntP) * 3 + 1];
        d110 = u.data[(u.xDim * (u.yDim * zIntP + yIntP) + xInt) * 3 + 1];
        d111 = u.data[(u.xDim * (u.yDim * zIntP + yIntP) + xIntP) * 3 + 1];

        d00x = d000 + xFrac * (d001 - d000);
        d01x = d010 + xFrac * (d011 - d010);
        d10x = d100 + xFrac * (d101 - d100);
        d11x = d110 + xFrac * (d111 - d110);
        d0yx = d00x + yFrac * (d01x - d00x);
        d1yx = d10x + yFrac * (d11x - d10x);

        resultU.data[vIndex + 1] = d0yx + zFrac * (d1yx - d0yx);

        // component 2
        d000 = u.data[(u.xDim * (u.yDim * zInt + yInt) + xInt) * 3 + 2];
        d001 = u.data[(u.xDim * (u.yDim * zInt + yInt) + xIntP) * 3 + 2];
        d010 = u.data[(u.xDim * (u.yDim * zInt + yIntP) + xInt) * 3 + 2];
        d011 = u.data[(u.xDim * (u.yDim * zInt + yIntP) + xIntP) * 3 + 2];
        
        d100 = u.data[(u.xDim * (u.yDim * zIntP + yInt) + xInt) * 3 + 2];
        d101 = u.data[(u.xDim * (u.yDim * zIntP + yInt) + xIntP) * 3 + 2];
        d110 = u.data[(u.xDim * (u.yDim * zIntP + yIntP) + xInt) * 3 + 2];
        d111 = u.data[(u.xDim * (u.yDim * zIntP + yIntP) + xIntP) * 3 + 2];

        d00x = d000 + xFrac * (d001 - d000);
        d01x = d010 + xFrac * (d011 - d010);
        d10x = d100 + xFrac * (d101 - d100);
        d11x = d110 + xFrac * (d111 - d110);
        d0yx = d00x + yFrac * (d01x - d00x);
        d1yx = d10x + yFrac * (d11x - d10x);

        resultU.data[vIndex + 2] = d0yx + zFrac * (d1yx - d0yx);

        index++;
        vIndex += 3;
      }
    }
  }
}

// Push an image forward by a vector field
void pushForward(const Image3D & im, const VectorField3D & v, Image3D & result)
{
  unsigned int i, j, k;
  int xInt, yInt, zInt;
  int xIntP, yIntP, zIntP;
  int index, vIndex;
  Real x, y, z;
  Real xFrac, yFrac, zFrac;

  Real w000, w001, w010, w011, w100, w101, w110, w111;
  int i000, i001, i010, i100, i011, i101, i110, i111;

  for(i = 0; i < im.xDim * im.yDim * im.zDim; i++)
    result.data[i] = 0.0;//im.data[i];

  index = 0;
  vIndex = 0;
  for(k = 0; k < im.zDim; k++)
  {
    for(j = 0; j < im.yDim; j++)
    {
      for(i = 0; i < im.xDim; i++)
      {
        x = static_cast<Real>(i + im.xDim) + v.data[vIndex] / v.deltaX;
        y = static_cast<Real>(j + im.yDim) + v.data[vIndex + 1] / v.deltaY;
        z = static_cast<Real>(k + im.zDim) + v.data[vIndex + 2] / v.deltaZ;

        xInt = static_cast<int>(x);
        yInt = static_cast<int>(y);
        zInt = static_cast<int>(z);

        xFrac = x - static_cast<Real>(xInt);
        yFrac = y - static_cast<Real>(yInt);
        zFrac = z - static_cast<Real>(zInt);

        // Wrap around indices -- ASSUMES NO FARTHER THAN ONE WRAP AROUND
        xInt = xInt % im.xDim;
        yInt = yInt % im.yDim;
        zInt = zInt % im.zDim;
        xIntP = (xInt + 1) % im.xDim;
        yIntP = (yInt + 1) % im.yDim;
        zIntP = (zInt + 1) % im.zDim;

        w000 = (1.0 - xFrac) * (1.0 - yFrac) * (1.0 - zFrac);
        w001 = (1.0 - xFrac) * (1.0 - yFrac) * zFrac;
        w010 = (1.0 - xFrac) * yFrac * (1.0 - zFrac);
        w100 = xFrac * (1.0 - yFrac) * (1.0 - zFrac);
        w011 = (1.0 - xFrac) * yFrac * zFrac;
        w101 = xFrac * (1.0 - yFrac) * zFrac;
        w110 = xFrac * yFrac * (1.0 - zFrac);
        w111 = xFrac * yFrac * zFrac;

        i000 = xInt + im.xDim * (yInt + im.yDim * zInt);
        i001 = xInt + im.xDim * (yInt + im.yDim * zIntP);
        i010 = xInt + im.xDim * (yIntP + im.yDim * zInt);
        i100 = xIntP + im.xDim * (yInt + im.yDim * zInt);
        i011 = xInt + im.xDim * (yIntP + im.yDim * zIntP);
        i101 = xIntP + im.xDim * (yInt + im.yDim * zIntP);
        i110 = xIntP + im.xDim * (yIntP + im.yDim * zInt);
        i111 = xIntP + im.xDim * (yIntP + im.yDim * zIntP);

        result.data[i000] += w000 * im.data[index];
        result.data[i001] += w001 * im.data[index];
        result.data[i010] += w010 * im.data[index];
        result.data[i100] += w100 * im.data[index];
        result.data[i011] += w011 * im.data[index];
        result.data[i101] += w101 * im.data[index];
        result.data[i110] += w110 * im.data[index];
        result.data[i111] += w111 * im.data[index];

        index++;
        vIndex += 3;
      }
    }
  }
}

// Compute translational component of a vector field
void translationalComponent(const VectorField3D & v,
                            Real & xTrans, Real & yTrans, Real & zTrans)
{
  int i, j, k;
  int index = 0;

  xTrans = 0.0;
  yTrans = 0.0;
  zTrans = 0.0;
  for(k = 0; k < v.zDim; k++)
  {
    for(j = 0; j < v.yDim; j++)
    {
      for(i = 0; i < v.xDim; i++)
      {
        xTrans += v.data[index++];
        yTrans += v.data[index++];
        zTrans += v.data[index++];
      }
    }
  }

  xTrans /= static_cast<Real>(v.xDim * v.yDim * v.zDim);
  yTrans /= static_cast<Real>(v.xDim * v.yDim * v.zDim);
  zTrans /= static_cast<Real>(v.xDim * v.yDim * v.zDim);
}
