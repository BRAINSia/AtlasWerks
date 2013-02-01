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

#ifndef JOINTLYVARYINGVECTORFIELD_H
#define JOINTLYVARYINGVECTORFIELD_H

#include "Vector3D.h"
#include "Array3D.h"
#include "AtlasWerksTypes.h"

#include "BidiVaryingVectorField.h"

#ifndef SWIG

#include <iosfwd>
#include <string.h>

#endif // SWIG

// A collection of bidi vector fields, representing the results of a
// multiple regression, or a multiflow estimation.  The main method is
// joint flow, which interpolates all the Vfields at the same time.
class JointlyVaryingVectorField : public std::vector<BidiVaryingVectorField>
{
public:
  JointlyVaryingVectorField();

  // trilerp a point radially from one radius down to (0,0,...)
  // TODO: make this syntax match that of ordinary trilerp
  Vector3D<double> jointlyFlowPoint(Vector3D<double> spatialcoords, 
                                    std::vector<double> paramcoords) const;
  void jointlyFlow(Vector3D<double> spatialcoords, std::vector<double> paramcoords,
                   VectorField &h, unsigned int numsteps, double endradius = 0.0) const;

private:
  unsigned int _numFlows;
  unsigned int _sizeX, _sizeY, _sizeZ;
};

#endif // JOINTLYVECTORFIELD_H
