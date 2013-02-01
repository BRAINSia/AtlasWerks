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

#ifndef BIDIVARYINGVECTORFIELD_H
#define BIDIVARYINGVECTORFIELD_H

#include "Vector3D.h"
#include "Array3D.h"
#include "AtlasWerksTypes.h"

#include "VaryingVectorField.h"

#ifndef SWIG

#include <iosfwd>
#include <string.h>

#endif // SWIG

// This is just a bidirectional varying vector field

class BidiVaryingVectorField
{
public:
  BidiVaryingVectorField(unsigned int numStepsFwd, unsigned int numStepsRev);

  // Flip arrows by inverting velocity fields (or negating them)
  void flipArrows();

  // Integrate, but allows negative values
  void flowToTime(double t, VectorField &h) const;

  VaryingVectorField vfwd, vrev;
private:
  bool _arrowsFwd; // whether these velocity fields point forward or
                   // back towards zero

  unsigned int _numStepsFwd, _numStepsRev;
};

#endif // VARYINGVECTORFIELD_H
