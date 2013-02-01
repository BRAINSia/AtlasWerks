#ifndef GEOMETRY_INL
#define GEOMETRY_INL

#include <cmath>

inline
Vector3D<Geometry::CoordinateType>
Geometry
::midpoint(const Vector3D<Geometry::CoordinateType>& p1,
	   const Vector3D<Geometry::CoordinateType>& p2,
	   const double& t) 
{
  return p1 + (p2 - p1) * t;
}

inline
bool
Geometry
::lineLineIntersection2D(const Geometry::CoordinateType& x1,
                         const Geometry::CoordinateType& y1,
                         const Geometry::CoordinateType& x2,
                         const Geometry::CoordinateType& y2,
                         const Geometry::CoordinateType& x3,
                         const Geometry::CoordinateType& y3,
                         const Geometry::CoordinateType& x4,
                         const Geometry::CoordinateType& y4)
{
  // http://astronomy.swin.edu.au/~pbourke/geometry/lineline2d/

  double uaNumerator = (x4-x3) * (y1-y3) - (y4-y3) * (x1-x3);
  double ubNumerator = (x2-x1) * (y1-y3) - (y2-y1) * (x1-x3);

  double denominator = (y4-y3) * (x2-x1) - (x4-x3) * (y2-y1);

  if (denominator == Geometry::CoordinateType(0))
    {
      // the lines are parallel
      
      if (uaNumerator == Geometry::CoordinateType(0) && 
          ubNumerator == Geometry::CoordinateType(0))
        {
          // the lines are coincident
        }           
      return false;
    }
  
  double ua = uaNumerator / denominator;
  double ub = ubNumerator / denominator;

  return (ua >= 0 && ua <= 1) && (ub >= 0 && ub <= 1);
}

inline
Vector3D<Geometry::CoordinateType>
Geometry
::linePlaneIntersection(const Vector3D<Geometry::CoordinateType>& planeNormal, 
			const Vector3D<Geometry::CoordinateType>& planePoint,
			const Vector3D<Geometry::CoordinateType>& linePoint1, 
			const Vector3D<Geometry::CoordinateType>& linePoint2)
{ 
  // http://astronomy.swin.edu.au/~pbourke/geometry/planeline/
  double denominator = planeNormal.dot(linePoint2 - linePoint1);
  double numerator = planeNormal.dot(planePoint - linePoint1);
  return midpoint(linePoint1, linePoint2, numerator / denominator);
}
 
#endif
