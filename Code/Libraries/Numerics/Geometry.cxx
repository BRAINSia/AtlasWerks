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

#include "Vector3D.h"
#include "Geometry.h"

void 
Geometry
::computeOrthonormalFrame(const Vector3D<Geometry::CoordinateType>& givenVector,
			  Vector3D<Geometry::CoordinateType>& b1,
			  Vector3D<Geometry::CoordinateType>& b2,
			  Vector3D<Geometry::CoordinateType>& b3) 
{
  // first compute b3, same sense as the givenVector
  b3 = givenVector;
  b3.normalize();

  // compute b1, a vector perpendicular to b3
  // the dot product of b3 and b1 should be 0
  // take care not to divide by zero
  if (b3.x != 0) 
    {
      b1.set(-(b3.y + b3.z)/b3.x, 1, 1);
    }
  else if (b3.y != 0) 
    {
      b1.set(1, -(b3.x + b3.z)/b3.y, 1);    
    }
  else if (b3.z != 0) 
    {
      b1.set(1, 1, -(b3.x + b3.y)/b3.z);
    }
  b1.normalize();

  // b2 is prependicular to b1 and b3
  b2 = b3.cross(b1);
  b2.normalize();
}
