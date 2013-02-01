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

#include "VectorField.h"
#include "AtlasWerksTypes.h"

#include "HField3DUtils.h"

VectorField::VectorField(bool isDef)
  : _isDeformation(isDef)
{
}

void
VectorField::toH(double delta)
{
  if (_isDeformation)
    return; // nothing to do

  for (unsigned int z=0;z < _size.z;++z)
    for (unsigned int y=0;y < _size.y;++y)
      for (unsigned int x=0;x < _size.x;++x)
        {
        (*this)(x, y, z).x = x + (*this)(x,y,z).x * delta;
        (*this)(x, y, z).y = y + (*this)(x,y,z).y * delta;
        (*this)(x, y, z).z = z + (*this)(x,y,z).z * delta;
        }

  _isDeformation = true;
}

void
VectorField::toV()
{
  if (!_isDeformation)
    return; // nothing to do

  for (unsigned int z=0;z < _size.z;++z)
    for (unsigned int y=0;y < _size.y;++y)
      for (unsigned int x=0;x < _size.x;++x)
        {
        (*this)(x, y, z).x = (*this)(x,y,z).x - x;
        (*this)(x, y, z).y = (*this)(x,y,z).y - y;
        (*this)(x, y, z).z = (*this)(x,y,z).z - z;
        }

  _isDeformation = false;
}

void
VectorField::invertIterative(unsigned int numIters)
{  
  // This implements the fixed point algorithm published in Chen2008
  // for computing the inverse of a deformation field 
  // The iteration is v^{n+1}(x) = -u(x + v^n(x)) where u is (*this)
  // They find that 5 iterations is usually enough for convergence

  bool switchedToV = _isDeformation;
  toV(); // This algorithm is written for displacement fields,
               // so switch between if necessary

  VectorField v(false);
  v.resize(_size);
  VectorField tmp(false);
  tmp.resize(_size);

  // Initialize with small deformation approximation
  for (unsigned int z=0;z < _size.z;++z)
    for (unsigned int y=0;y < _size.y;++y)
      for (unsigned int x=0;x < _size.x;++x)
        v.set(x,y,z,-get(x,y,z));

  for (unsigned int iter=0; iter < numIters; ++iter)
    {
    // Evaluate RHS
    HField3DUtils::composeHV(*this,v,tmp);

    // Need to make negative, and put result back into v
    for (unsigned int z=0;z < _size.z;++z)
      for (unsigned int y=0;y < _size.y;++y)
        for (unsigned int x=0;x < _size.x;++x)
          v.set(x,y,z,-tmp.get(x,y,z));
    }

  // Copy result back into *this
  copyData(v.getDataPointer());

  // We may have switched this to a Vfield before, if so don't mess
  // with it
  if (switchedToV)
    this->toH();
}

template <typename T>
void VectorField::trilerp(const T& x, const T& y, const T& z,
               T& hx, T& hy, T& hz) const
{
  // Trilerp self at a single point (just call the HField3DUtils) but
  // automatically determine background strategy

  if (_isDeformation) HField3DUtils::trilerp(*this, x,y,z,hx,hy,hz,HField3DUtils::BACKGROUND_STRATEGY_PARTIAL_ID);
  else HField3DUtils::trilerp(*this,x,y,z,hx,hy,hz,HField3DUtils::BACKGROUND_STRATEGY_PARTIAL_ZERO);
}

