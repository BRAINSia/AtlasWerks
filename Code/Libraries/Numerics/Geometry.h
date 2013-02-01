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

#ifndef Geometry_h
#define Geometry_h

#include "Vector3D.h"

class Geometry 
{
public:
  // this should be templated but vc++ can't handle it
  typedef double CoordinateType;

  // returns the point t of the way from p1 to p2
  // where t parameterizes the line segment from p1 to p2, t = [0,1]
  static 
  Vector3D<CoordinateType> 
  midpoint(const Vector3D<CoordinateType>& p1,
	   const Vector3D<CoordinateType>& p2,
	   const double& t = 0.5);

  //
  // returns true if the lines intersect at one point
  //
  static
  bool
  lineLineIntersection2D(const CoordinateType& line1Point1X,
                         const CoordinateType& line1Point1Y,
                         const CoordinateType& line1Point2X,
                         const CoordinateType& line1Point2Y,
                         const CoordinateType& line2Point1X,
                         const CoordinateType& line2Point1Y,
                         const CoordinateType& line2Point2X,
                         const CoordinateType& line2Point2Y);
                         
  // returns the point where the line segment from linePoint1 to linePoint2
  // intersects the plane given by planePoint and planeNormal
  static 
  Vector3D<CoordinateType> 
  linePlaneIntersection(const Vector3D<CoordinateType>& planeNormal,
			const Vector3D<CoordinateType>& planePoint,
			const Vector3D<CoordinateType>& linePoint1,
			const Vector3D<CoordinateType>& linePoint2);

  // creates a 3D orthonormal frame using the given vector as one
  // cardinal direction
  static 
  void 
  computeOrthonormalFrame(const Vector3D<CoordinateType>& normal,
			  Vector3D<CoordinateType>& b1,
			  Vector3D<CoordinateType>& b2,
			  Vector3D<CoordinateType>& b3);
};

#include "Geometry.inl"

#endif
