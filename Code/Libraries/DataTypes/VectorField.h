/* ================================================================
 *
 * AtlasWerks Project
 *
 * Copyright (c) Sarang C. Joshi. All rights reserved.  See
 * Copyright.txt or for details.
 *
 * This software is distributed WITHOUT ANY WARRANTY; without even the
 * implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 * PURPOSE.  See the above copyright notice for more information.
 *
 * ================================================================ */

#ifndef VECTORFIELD_H
#define VECTORFIELD_H

#include "Vector3D.h"
#include "Array3D.h"
#include "AtlasWerksTypes.h"

#ifndef SWIG

#include <iosfwd>
#include <string.h>

#endif // SWIG

class VectorField : public Array3D<Vector3D<VectorFieldCoordType> >
{
public:
  VectorField(unsigned int szx, unsigned int szy, unsigned int szz) : Array3D(szx, szy, szz) {};
  VectorField(Vector3D<unsigned int> size) : Array3D(size) {};
  VectorField(bool isDef=0);

  void toH(double delta = 1.0);
  void toV();

  void invertIterative(unsigned int numIters=5);

  // TODO: match HField3DUtils format here
  template <typename T>
  void trilerp(const T& x, const T& y, const T& z,
              T& hx, T& hy, T& hz) const;
  // Background strategy will be assumed based on whether this is a
  //velocity or an hfield
  //BackgroundStrategy backgroundStrategy = BACKGROUND_STRATEGY_ID);

  // Apply this displacement or deformation to an image by composition
  // or by splatting
  void composeImage(RealImage img);
  void splatImage(RealImage img);

private:
  bool _isDeformation;
};

#endif // VECTORFIELD_H
