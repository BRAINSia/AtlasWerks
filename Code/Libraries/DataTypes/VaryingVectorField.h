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

#ifndef VARYINGVECTORFIELD_H
#define VARYINGVECTORFIELD_H

#include "Vector3D.h"
#include "Array3D.h"
#include "AtlasWerksTypes.h"

#include "VectorField.h"

#ifndef SWIG

#include <iosfwd>
#include <string.h>

#endif // SWIG

class VaryingVectorField : public std::vector<VectorField>
{
public:
  VaryingVectorField(unsigned int numSteps);

  // Flip arrows by inverting velocity fields (or negating them)
  void flipArrows();

  // Need to be able to integrate to get a deformation to time t
  void flowToTime(double t, VectorField &h) const;

private:
  bool _arrowsFwd; // whether these velocity fields point forward or
                   // back towards zero

  unsigned int _numSteps;
};

#endif // VARYINGVECTORFIELD_H
