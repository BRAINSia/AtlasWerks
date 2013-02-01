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

#ifndef SURFACEUTILS_H
#define SURFACEUTILS_H

#include "Line3D.h"
#include "Array2D.h"
#include "Vector3D.h"
#include "Surface.h"
#include "AffineTransform3D.h"
#include <list>

class SurfaceUtils {
 public:
  typedef std::list<Line3D>    LineList;
  typedef LineList::iterator   LineIterator;  
  typedef std::vector< Vector3D<double> > VertexList;

  static 
  Surface 
  createUnitOctahedron();

  static
  void
  computeVolumeAndCentroid(const Surface& surface, 
                           double& volume,
                           Vector3D<double>& centroid);

  static 
  double 
  computeVolume(const Surface& surface);

  static
  Vector3D<double>
  computeCentroid(const Surface& surface);

  static
  double
  computeDistancePointToSurface(const Vector3D<double>& point,
                                const Surface& surface);

  static
  double
  computeHausdorffDistance(const Surface& surfaceA,
                           const Surface& surfaceB);

  static
  double
  computeMeanDistance(const Surface& surfaceA,
                      const Surface& surfaceB);

  static 
  bool 
  isTriangular(const Surface& surface);

  static 
  void 
  makeTriangular(Surface& surface);

  static 
  void 
  removeDoubles(Surface& surface, 
		const double& tolerance = 0.001);

  static 
  void 
  clean(Surface& surface, 
	const double& tolerance = 0.001);

  static 
  void 
  intersectWithPlane(const Surface& surface,
		     const Vector3D<double>& planePoint,
		     const Vector3D<double>& planeNormal,
		     LineList& lines);

  static 
  void 
  worldToImageIndexXY(Surface& surface,
		      Vector3D<double> origin,
                      Vector3D<double> spacing);

  static 
  void 
  worldToImageIndex(Surface& surface,
                    Vector3D<double> origin,
                    Vector3D<double> spacing);
  
  static
  void
  refineSurface( Surface& surface );
  
  static 
  void 
  imageIndexToWorldXY(Surface& surface,
		      Vector3D<double> origin,
                      Vector3D<double> spacing);

  static 
  void 
  imageIndexToWorld(Surface& surface,
                    Vector3D<double> origin,
                    Vector3D<double> spacing);

  static
  void
  applyAffine(Surface& surface,
              const AffineTransform3D<double>& a);
 
  static
  void
  computeCentroidOfFacets(const Surface& surface, 
                           Vector3D<double>& centroid);

  static
  void
  triangleIntersectWithRay( const Vector3D<double> orig,
                            const Vector3D<double> dir, 
                            const Vector3D<double> v0,
                            const Vector3D<double> v1, 
                            const Vector3D<double> v2,
                            bool& insideTri, 
                            bool& confused,
                            bool& intersect,
                            Vector3D<double>& Pt );  


  static
  void
  surfaceIntersectWithRay(const Surface surface,
		           const Vector3D<double> RayPoint,
		           const Vector3D<double> RayVector,
		           Vector3D<double>& Pt,
                   Vector3D<double>& V0,
                   Vector3D<double>& V1,
                   Vector3D<double>& V2,
                   bool& hole);  

  static
  void
  getDistanceFromOrig(const Surface surface,
                      Array2D<double>& TetaPhi,
                      bool& hole); 
  
  static
  void
  getDistanceBetweenTwoSurfaces(const Surface surface1,
                                const Surface surface2,
                                Array2D<double>& TetaPhi,
                                bool& hole); 
  static
  bool
  PointInTriangle(const Vector3D<double>& p,
                   const Vector3D<double>& a,
                   const Vector3D<double>& b,
                   const Vector3D<double>& c); 

  static
  bool
  SameSide(const Vector3D<double>& p1,
           const Vector3D<double>& p2,
           const Vector3D<double>& a,
           const Vector3D<double>& b); 

  static
  double
  getMeanDistanceBetweenTwoSurfaces(Array2D<double>& TetaPhi); 


  static
  Surface
  Sphere(const double& radius, 
         const int& level = 2); 

  static
  Array2D<Vector3D<double> >
  getArray2DinColors(Array2D<double>& TetaPhi,
                     const std::string& colorFileName,
                     const int visualisationMode); 

  static
  void
  readColorFile(std::istream &input, 
              Vector3D<double> colorArray[256]); 
  static
  void
  readColorFile(const std::string& filename,
              Vector3D<double> colorArray[256]); 


  static
  void
  perturbVertexPositions(Surface& s,
                         const double& maxXPert,
                         const double& maxYPert,
                         const double& maxZPert);
};
#endif

