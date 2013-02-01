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

#include <cassert>

#include "SurfaceUtils.h"
#include "Surface.h"
#include "Geometry.h"
#include "Array2D.h"
#include "Vector2D.h"
#include "Random.h"

const double PI = 3.14159265;

Surface
SurfaceUtils
::createUnitOctahedron()
{
  Surface surface;

  int XPLUS = surface.addVertex( 1,  0,  0);
  int XMIN  = surface.addVertex(-1,  0,  0);
  int YPLUS = surface.addVertex( 0,  1,  0);
  int YMIN  = surface.addVertex( 0, -1,  0);
  int ZPLUS = surface.addVertex( 0,  0,  1);
  int ZMIN  = surface.addVertex( 0,  0, -1);

  surface.addFacet(ZPLUS, XPLUS, YPLUS);
  surface.addFacet(ZPLUS, YPLUS, XMIN);
  surface.addFacet(ZPLUS, XMIN , YMIN);
  surface.addFacet(ZPLUS, YMIN , XPLUS);
  surface.addFacet(ZMIN , YPLUS, XPLUS);
  surface.addFacet(ZMIN , XPLUS, YMIN);
  surface.addFacet(ZMIN , YMIN , XMIN);
  surface.addFacet(ZMIN , XMIN , YPLUS);
  
  return(surface);  
}

bool 
SurfaceUtils
::isTriangular(const Surface& surface)
{
  for (Surface::ConstFacetIter facetIter = surface.facets.begin();
       facetIter != surface.facets.end();
       ++facetIter)
    {
      if (facetIter->size() != 3) return false;
    }
  return true;
}

void 
SurfaceUtils
::makeTriangular(Surface& surface)
{
  Surface::FacetList oldFacets(surface.facets);
  surface.facets.clear();

  for (Surface::FacetIter facetIter = oldFacets.begin(); 
       facetIter != oldFacets.end();
       ++facetIter)
    {
      assert (facetIter->size() > 2);

      // if it's already a triangle, add it
      if (facetIter->size() == 3) 
	{
	  surface.facets.push_back(*facetIter);
	}

      // otherwise peel off triangles in a stupid manner
      // clockwise
      else
	{
	  for (unsigned int s = facetIter->size(); s > 2; --s)
	    {
	      Surface::Facet f(3);
	      f[0] = (*facetIter)[0];
	      f[1] = (*facetIter)[s - 2];
	      f[2] = (*facetIter)[s - 1];
	      surface.facets.push_back(f);
	    }
	}
    }
}

void
SurfaceUtils
::removeDoubles(Surface& surface, const double& tolerence)
{
}

void 
SurfaceUtils
::clean(Surface& surface, const double& tolerence)
{
  
}

void 
SurfaceUtils
::intersectWithPlane(const Surface& surface,
		     const Vector3D<double>& planePoint,
		     const Vector3D<double>& planeNormal,
		     LineList& lines)
{
  lines.clear();
  Line3D line;

  // useful for debugging
  unsigned int outsideCount  = 0;
  unsigned int straddleCount = 0;
  unsigned int lineCount     = 0;
  unsigned int planeCount    = 0;
  unsigned int pointCount    = 0;
  
  //
  // check each surface facet for intersection with plane
  //
  for (Surface::ConstFacetIter facetIter = surface.facets.begin();
       facetIter != surface.facets.end();
       ++facetIter)
    {
      Vector3D<double> vertex1 = surface.vertices[(*facetIter)[0]];
      Vector3D<double> vertex2 = surface.vertices[(*facetIter)[1]];
      Vector3D<double> vertex3 = surface.vertices[(*facetIter)[2]];

      // the sign of v1, v2, v3 will decide which side of 
      // the plane each vertex is on
      double v1 = planeNormal.dot(planePoint - vertex1);
      double v2 = planeNormal.dot(planePoint - vertex2);
      double v3 = planeNormal.dot(planePoint - vertex3);

      if (v1 * v2 > 0 && v2 * v3 > 0)
	{
	  // this triangle is completely on one side of
	  // the plane, don't add any lines
	  ++outsideCount;
	  continue;
	}      
      else if (v1 * v2 < 0 || v2 * v3 < 0 || v3 * v1 < 0)
	{
	  // the triangle straddles the plain, add intersection
	  Vector3D<double> *pt = &line.p1;
	  int vertexCount = 0;

	  if (v1 * v2 < 0)
	    {
	      // the line vertex1->vertex2 passes through the plane
	      *pt = Geometry::linePlaneIntersection(planeNormal,
						    planePoint,
						    vertex1,
						    vertex2);
	      pt = &line.p2;
	      ++vertexCount;
	    }
	  if (v2 * v3 < 0)
	    {
	      // the line vertex2->vertex3 passes through the plane
	      *pt = Geometry::linePlaneIntersection(planeNormal,
						    planePoint,
						    vertex2,
						    vertex3);
	      pt = &line.p2;
	      ++vertexCount;
	    }
	  if (v3 * v1 < 0)
	    {
	      // the line vertex3->vertex1 passes through the plane
	      *pt = Geometry::linePlaneIntersection(planeNormal,
						    planePoint,
						    vertex3,
						    vertex1);
	      pt = &line.p2;
	      ++vertexCount;
	    }
	  
	  //
	  // handle case where one line passes through plane and
	  // one vertex is on plane
	  //
	  if (v1 == 0)
	    {
	      *pt = vertex1;
	      pt  = &line.p2;
	      ++vertexCount;
	    }
	  if (v2 == 0)
	    {
	      *pt = vertex2;
	      pt  = &line.p2;
	      ++vertexCount;
	    }
	  if (v3 == 0)
	    {
	      *pt = vertex3;
	      pt  = &line.p2;
	      ++vertexCount;
	    }
	  assert(vertexCount == 2);
	  if (line.p2 < line.p1) std::swap(line.p2, line.p1);
	  lines.push_back(line);
	  ++straddleCount;
	}
      else if (v1 == 0 && v2 == 0 && v3 == 0)
	{
	  // the triangle lies in the plane, don't add any lines
	  ++planeCount;
	}
      else if (v1 == 0 && v2 == 0)
	{
	  // the line vertex1->vertex2 lies in plane	  
	  assert(v3 != 0);
	  line.p1 = vertex1;
	  line.p2 = vertex2;
	  if (line.p2 < line.p1) std::swap(line.p2, line.p1);
	  lines.push_back(line);
	  ++lineCount;
	}
      else if (v2 == 0 && v3 == 0)
	{
	  // the line vertex2->vertex3 lies in plane	  
	  assert(v1 != 0);
	  line.p1 = vertex2;
	  line.p2 = vertex3;
	  if (line.p2 < line.p1) std::swap(line.p2, line.p1);
	  lines.push_back(line);
	  ++lineCount;
	}
      else if (v3 == 0 && v1 == 0)
	{
	  // the line vertex3->vertex1 lies in plane	  
	  assert(v2 != 0);
	  line.p1 = vertex3;
	  line.p2 = vertex1;
	  if (line.p2 < line.p1) std::swap(line.p2, line.p1);
	  lines.push_back(line);
	  ++lineCount;
	}
      else
	{
	  // triangle intersects plane at a point, don't add any lines
	  assert((v1 == 0 && v2 * v3 > 0) ||
		 (v2 == 0 && v3 * v1 > 0) ||
		 (v3 == 0 && v1 * v2 > 0));	  
	  ++pointCount;
	}
    }

//   std::cerr << "Facets: " << surface.facets.size() << std::endl;
//   std::cerr << "Lines : " << lines.size() << std::endl;
//   std::cerr << "\toutside : " << outsideCount << std::endl;  
//   std::cerr << "\tstraddle: " << straddleCount << std::endl;  
//   std::cerr << "\tplane   : " << planeCount << std::endl;  
//   std::cerr << "\tline    : " << lineCount << std::endl;  
//   std::cerr << "\tpoint   : " << pointCount << std::endl;  

  //
  // remove doubles
  //
  lines.sort();
  lines.unique();  

  //std::cerr << "Unique Lines : " << lines.size() << std::endl;  
}

///////////////////////////////////////////////////////////////////////////
//
// refineSurface
//
// Splits every triangle into four triangles
// for each triangle ABD, refine into four triangles:
//
// D      A        E               D      A        E
// ----------------                ----------------
// \      /\      /                \      /\      /
//  \    /  \    /                  \   c/__\b   /
//   \  /    \  /         to         \  /\  /\  /
//    \/______\/                      \/__\/__\/
//   B \      /C                     B \   a  /C
//      \    /                          \    /
//       \  /                            \  /
//        \/                              \/
//         F                               F
//
// since we are refining the entire surface, just save neighborhood
// generation to the end.
//
///////////////////////////////////////////////////////////////////////////
void SurfaceUtils::refineSurface(Surface& surface)
{
  //if (mVerbose)
   // cout << "in refineSurface" << endl;

  int A, B, C,
    a, b, c,
    t1, t2, t3;

  // create fixed copy of surface
  Surface origSurf = surface;

  //
  // Subdivide all triangles
  //
  for (unsigned int i=0; i < origSurf.numFacets(); i++)
    {

      // store original triangle vertex indices
      A = origSurf.facets[i][0];
      B = origSurf.facets[i][1];
      C = origSurf.facets[i][2];

      /*if (mVerbose)
        {
          cout << "splitting triangle " << i << endl;
          cout << "Initial Vertices and Neighborhoods:"
               << "\n    A (" << A << "): " //<< mNbhd[A]
               << "\n    B (" << B << "): " //<< mNbhd[B]
               << "\n    C (" << C << "): " //<< mNbhd[C]
               << endl;
        }*/

      // compute the midpoints of each edge of the triangle, and
      // add as new vertices
      a = surface.addVertex((origSurf.vertices[B]+origSurf.vertices[C])/2.0);
      b = surface.addVertex((origSurf.vertices[A]+origSurf.vertices[C])/2.0);
      c = surface.addVertex((origSurf.vertices[A]+origSurf.vertices[B])/2.0);

      /*if (mVerbose)
        cout << "new vertices " << a << ", " << b << ", " << c << endl;
*/
      // replace target triangle with four new triangles
      // (indices: i, t1, t2, t3)
      t1 = surface.addFacet(B, a, c);
      t2 = surface.addFacet(a, b, c);
      t3 = surface.addFacet(a, C, b);
      // change orig to Acb
      surface.facets[i][1] = c;
      surface.facets[i][2] = b;
    }
}

void 
SurfaceUtils
::worldToImageIndexXY(Surface& surface,
		              Vector3D<double> origin,
                      Vector3D<double> spacing)
{
    surface.translate(-origin);
    surface.scale(1/spacing);
}

void 
SurfaceUtils
::imageIndexToWorldXY(Surface& surface,
		              Vector3D<double> origin,
                      Vector3D<double> spacing)
{
    surface.scale(spacing);
    surface.translate(origin);
}

void 
SurfaceUtils
::worldToImageIndex(Surface& surface,
                    Vector3D<double> origin,
                    Vector3D<double> spacing)
{
    surface.translate(-origin);
    surface.scale(1.0/spacing);
}

void 
SurfaceUtils
::imageIndexToWorld(Surface& surface,
                    Vector3D<double> origin,
                    Vector3D<double> spacing)
{
    surface.scale(spacing);
    surface.translate(origin);
}

void
SurfaceUtils
::applyAffine(Surface& surface,
              const AffineTransform3D<double>& a)
{
    unsigned int numVertices = surface.numVertices();
    for (unsigned int i = 0; i < numVertices; ++i)
    {
      Vector3D<double> tmp(surface.vertices[i]);
      a.transformVector(tmp, surface.vertices[i]);
    }
}

double
SurfaceUtils
::computeVolume(const Surface& surface)
{
    double volume;
    Vector3D<double> centroid;
    SurfaceUtils::computeVolumeAndCentroid(surface, volume, centroid);
    return volume;
}

Vector3D<double>
SurfaceUtils
::computeCentroid(const Surface& surface)
{
  double volume;
  Vector3D<double> centroid;
  SurfaceUtils::computeVolumeAndCentroid(surface, volume, centroid);
  return centroid;
}

namespace {
// This function is only visible in this file.

// This takes into account the surface integral over a whole face.  It
// returns twice the value of the integral so that it is a drop-in
// replacement for the three lines updating centroid in
// computeVolumeAndCentroid.
//
// Useful if the volume has large non-axis-aligned triangles.

Vector3D<double>
facetCentroidContribution( Vector3D<double> AA, Vector3D<double> BB,
                           Vector3D<double> CC)
{

    double x = -((AA[0]*AA[0] + BB[0]*BB[0] + BB[0]*CC[0] + 
                  CC[0]*CC[0] + AA[0]*(BB[0] + CC[0]))*
                 (AA[2]*(BB[1] - CC[1]) + BB[2]*CC[1] - 
                  BB[1]*CC[2] + AA[1]*(-BB[2] + CC[2])))/6.;

    double y = ((AA[1]*AA[1] + BB[1]*BB[1] + BB[1]*CC[1] + 
                 CC[1]*CC[1] + AA[1]*(BB[1] + CC[1]))*
                (AA[2]*(BB[0] - CC[0]) + BB[2]*CC[0] - 
                 BB[0]*CC[2] + AA[0]*(-BB[2] + CC[2])))/6.;

    double z = -((AA[1]*(BB[0] - CC[0]) + BB[1]*CC[0] - BB[0]*CC[1] + 
                  AA[0]*(-BB[1] + CC[1]))*
                 (AA[2]*AA[2] + BB[2]*BB[2] + BB[2]*CC[2] + 
                  CC[2]*CC[2] + AA[2]*(BB[2] + CC[2])))/6.;

    return Vector3D<double>(x, y, z);

}

}; // End of unnamed namespace hiding above func outside this file.

void
SurfaceUtils
::computeVolumeAndCentroid(const Surface& surface, 
                           double& volume,
                           Vector3D<double>& centroid)
{
  // Assume without checking that surface is closed.
  
  // compute volume using Stokes' theorem
  volume = 0;
  centroid = Vector3D<double>(0,0,0);

  for (Surface::ConstFacetIter facetIter = surface.facets.begin(); 
       facetIter != surface.facets.end(); 
       ++facetIter) 
  {
    const Surface::Vertex& vertex0 = surface.vertices[(*facetIter)[0]];
    const Surface::Vertex& vertex1 = surface.vertices[(*facetIter)[1]];
    const Surface::Vertex& vertex2 = surface.vertices[(*facetIter)[2]];

    // get triangle edges
    Vector3D<double> edge01 = vertex1 - vertex0;
    Vector3D<double> edge20 = vertex0 - vertex2;

    // get normal vector for facet
    // - will be the same regardless of what pair is taken
    // - direction depends on order -- triangles *must* have
    // consistent orientation, with v0, v1, v2 *counterclockwise* 
    // as viewed from outside.
    // - length will be twice the area of the triangle
    Surface::Normal facetNormal = -edge01.cross(edge20);

    // triangleCentroid dot facetNormal == 2 * (integral over tri of unit
    // normal dotted with coords of point)
    Vector3D<double> facetCentroid = (vertex0 + vertex1 + vertex2) / 3;

    volume += facetCentroid.dot(facetNormal);
    centroid.x += facetCentroid.x * facetCentroid.x * facetNormal.x;
    centroid.y += facetCentroid.y * facetCentroid.y * facetNormal.y;
    centroid.z += facetCentroid.z * facetCentroid.z * facetNormal.z;

  }

  // Includes factor of 3 because the divergence of (x,y,z) is 3, and
  // a factor of 2 because the length of each normal is twice the area
  // of the triangle.
  volume /= 6;

  // A factor of 2 because, e.g., div(x^2, 0, 0) = 2x, and another
  // factor of 2 for the triangle area issue.
  centroid /= (4 * volume);

}

double
SurfaceUtils::
computeDistancePointToSurface(const Vector3D<double>& point,
                              const Surface& surface)
{
  //initialize 
  double minDist = surface.vertices.begin()->distance(point);
    
  for (Surface::ConstVertexIter vertexIter = surface.vertices.begin(); 
       vertexIter != surface.vertices.end(); 
       ++vertexIter) 
  {
    double distance = vertexIter->distance(point);
    if( distance < minDist )
    {
      minDist = distance;
    }
  }
  return minDist;
}

//
// compute the distance of Hausdorff between
// the byu surfaces A and B
// a : vertex of surface A
// b : vertex of surface B
//
// d = Max [ Sup Inf d(a,b), Sup Inf d (b, a)]
// d = Max [ Sup d(a,B) , Sup d(b,A)]
//
// dp 2004

double
SurfaceUtils::
computeHausdorffDistance(const Surface& surfaceA,
                         const Surface& surfaceB)
{
  double hausdorff = 0;
  double InfAB = 0, InfBA = 0;  //Inferior value 
  double SupAB = 0, SupBA = 0;  //Superior value
  
  //compute superior value from surface A to B
  for (Surface::ConstVertexIter vertexIterA = surfaceA.vertices.begin(); 
       vertexIterA != surfaceA.vertices.end(); 
       ++vertexIterA) 

  {
    InfAB = computeDistancePointToSurface(*vertexIterA, surfaceB);
    if( InfAB > SupAB ) { SupAB = InfAB; }
  }
  
  //compute superior value from surface B to A
  for (Surface::ConstVertexIter vertexIterB = surfaceB.vertices.begin(); 
       vertexIterB != surfaceB.vertices.end(); 
       ++vertexIterB) 
  {
    InfBA = computeDistancePointToSurface(*vertexIterB, surfaceA);
    if( InfBA > SupBA ) { SupBA = InfBA; }
  }
  
  //maximun value between the two superior value computed
  if(SupAB>SupBA)
    hausdorff = SupAB;
  else
    hausdorff = SupBA;

  return hausdorff;
}

double
SurfaceUtils::
computeMeanDistance(const Surface& surfaceA, const Surface& surfaceB)
{
  double cumDist = 0;
  int numPoints = 0;
  
  for (Surface::ConstVertexIter vertexIterA = surfaceA.vertices.begin(); 
       vertexIterA != surfaceA.vertices.end(); 
       ++vertexIterA) 

  {
    cumDist += computeDistancePointToSurface(*vertexIterA, surfaceB);
    ++numPoints;
  }
  
  for (Surface::ConstVertexIter vertexIterB = surfaceB.vertices.begin(); 
       vertexIterB != surfaceB.vertices.end(); 
       ++vertexIterB) 
  {
    cumDist += computeDistancePointToSurface(*vertexIterB, surfaceA);
    ++numPoints;
  }

  return cumDist / numPoints;

}







void
SurfaceUtils
::computeCentroidOfFacets(const Surface& surface, 
                      Vector3D<double>& centroid)
{
  centroid = Vector3D<double>(0,0,0);

  for (Surface::ConstFacetIter facetIter = surface.facets.begin(); 
       facetIter != surface.facets.end(); 
       ++facetIter) 
  {
    const Surface::Vertex& vertex0 = surface.vertices[(*facetIter)[0]];
    const Surface::Vertex& vertex1 = surface.vertices[(*facetIter)[1]];
    const Surface::Vertex& vertex2 = surface.vertices[(*facetIter)[2]];

    Vector3D<double> facetCentroid = (vertex0 + vertex1 + vertex2) / 3;

    centroid.x += facetCentroid.x;
    centroid.y += facetCentroid.y;
    centroid.z += facetCentroid.z;

  }

  centroid /= surface.numFacets();

}



bool
SurfaceUtils
::SameSide(const Vector3D<double>& p1,
           const Vector3D<double>& p2,
           const Vector3D<double>& a,
           const Vector3D<double>& b)
{
    Vector3D<double> cp1 = (b - a).cross(p1 - a);
    Vector3D<double> cp2 = (b - a).cross(p2 - a);
    double DotProduct = cp1.dot(cp2); 
    if (DotProduct >= 0)
        return true;
    else
        return false;
}



bool
SurfaceUtils
:: PointInTriangle(const Vector3D<double>& p,
                   const Vector3D<double>& a,
                   const Vector3D<double>& b,
                   const Vector3D<double>& c)
{
    if (SameSide(p, a, b, c) && SameSide(p, b, a, c) && SameSide(p, c, a, b))
        return true;
    else
        return false;
}



void
SurfaceUtils
::triangleIntersectWithRay( const Vector3D<double> orig,
                            const Vector3D<double> dir, 
                            const Vector3D<double> v0,
                            const Vector3D<double> v1, 
                            const Vector3D<double> v2,
                            bool& insideTri, 
                            bool& confused,
                            bool& intersect,
                            Vector3D<double>& Pt )

{
    //find vector Normal to the plan formed by v0 v1 and v2
    Vector3D<double> planeNormal;
    planeNormal = (v1 - v0).cross(v2 - v0);

    //take one vertex for planePoint
    Vector3D<double> planePoint = v0;

    //take two points of the Ray
    Vector3D<double> linePoint1 = orig;
    Vector3D<double> linePoint2 = orig + dir;

    //if den is zero, ray lies in plane of triangle or is parallel
    double den = planeNormal.dot(linePoint2 - linePoint1);
    double num = planeNormal.dot(planePoint - linePoint1);

    if(den == 0)
    {
        intersect = false;
        insideTri = false;

        if(num==0)
            confused = true;
        else //paralell
            confused = false;
    }
    else
    {
        confused = false;

        Pt = linePoint1 + (linePoint2 - linePoint1) * (num / den);

        //the ray is not a line, so is this point in the good direction ?
        Vector3D<double> origPt = Pt - orig;
        Vector3D<double> sign = dir * origPt;
        if(sign.x < 0 || sign.y < 0 || sign.z < 0)
            intersect = false;
        else
        {
            intersect = true;
            /*
            //is it inside the triangle ? 
            //if YES then the sum of the angles with the three vertices is equal to 2Pi
            double PtV0PtV1 = (v0 - Pt).dot(v1 - Pt);
            double PtV1PtV2 = (v1 - Pt).dot(v2 - Pt);
            double PtV2PtV0 = (v2 - Pt).dot(v0 - Pt);

            double NormPtV0 = (v0 - Pt).length();
            double NormPtV1 = (v1 - Pt).length();
            double NormPtV2 = (v2 - Pt).length();

            double sumAngle = 0.0;
            if ( (PtV0PtV1 / (NormPtV0 * NormPtV1)) <= 1 && (PtV0PtV1 / (NormPtV0 * NormPtV1)) >= -1 &&
                 (PtV1PtV2 / (NormPtV1 * NormPtV2)) <= 1 && (PtV1PtV2 / (NormPtV1 * NormPtV2)) >= -1 &&
                 (PtV2PtV0 / (NormPtV2 * NormPtV0)) <= 1 && (PtV2PtV0 / (NormPtV2 * NormPtV0)) >= -1 )

            sumAngle = acos(PtV0PtV1 / (NormPtV0 * NormPtV1)) 
                     + acos(PtV1PtV2 / (NormPtV1 * NormPtV2)) 
                     + acos(PtV2PtV0 / (NormPtV2 * NormPtV0));

            if ((sumAngle <= (2.0*PI) + 0.01) || (sumAngle >= (2.0*PI) - 0.01))
                insideTri = true;
            else
                insideTri = false;
            */
            insideTri = PointInTriangle(Pt, v0, v1, v2);
        }
    }
}



void 
SurfaceUtils
::surfaceIntersectWithRay(const Surface surface,
		           const Vector3D<double> RayPoint,
		           const Vector3D<double> RayVector,
		           Vector3D<double>& Pt,
                   Vector3D<double>& V0,
                   Vector3D<double>& V1,
                   Vector3D<double>& V2,
                   bool& hole)
{ 
    bool _insideTri = false , _confused = false , _intersect = false;
    Vector3D<double> _Pt;
    
    // check each surface facet for intersection with Ray
    hole = true;
    bool stop = false;

        for (Surface::ConstFacetIter facetIter = surface.facets.begin();
            facetIter != surface.facets.end() && stop == false;
            ++facetIter)
        {
            Vector3D<double> vertex0 = surface.vertices[(*facetIter)[0]];
            Vector3D<double> vertex1 = surface.vertices[(*facetIter)[1]];
            Vector3D<double> vertex2 = surface.vertices[(*facetIter)[2]];

            triangleIntersectWithRay(RayPoint, 
                                     RayVector, 
                                     vertex0,
                                     vertex1, 
                                     vertex2,
                                     _insideTri, 
                                     _confused,
                                     _intersect,
                                     _Pt);
            if(_insideTri == true)
            {
                Pt = _Pt;
                V0 = vertex0;
                V1 = vertex1;
                V2 = vertex2;
                hole = false;
                stop = true;
            }
        }
}



void 
SurfaceUtils
::getDistanceFromOrig(const Surface surface,
                      Array2D<double>& TetaPhi,
                      bool& hole)
{ 
    //get the origin of the ray, the centroid of the surface
    Vector3D<double> RayPoint = computeCentroid(surface);

    //get the direction of the ray (1/34 = 0.0294)
    double teta, phi, value = 0.0294;
    int k, j;
    bool _hole;
    hole = false;
    Vector3D<double> RayVector, _Pt, _V0, _V1, _V2;
    
    //10 values for teta from 0 to 1.8PI, 
    //and 6 values for phi from -Pi/2 to Pi/2
    for (j = 0; j < 68; j++)
    {
        for (k = 0; k < 35; k++)
        {
            teta = (j+0.002) * value * PI;
            phi = ((k+0.002) * value * PI) - (PI / 2);
            RayVector.x = cos(phi) * sin(teta);
            RayVector.y = cos(phi) * cos(teta);
            RayVector.z = sin(phi);
            
            surfaceIntersectWithRay(surface,
		                     RayPoint,
		                     RayVector,
		                     _Pt,
                             _V0,
                             _V1,
                             _V2,
                             _hole);
            if(_hole == true)
            {
                hole = true;
                TetaPhi.set(j, k, 0);
                std::cout << "after allocation Tetaphi 0 because hole " << std::endl;
                std::cout << "teta : " << teta << " phi : " << phi << std::endl;
            }
            else
            {
                TetaPhi.set(j, k, RayPoint.distance(_Pt));
            }
        }
    }
}




void 
SurfaceUtils
::getDistanceBetweenTwoSurfaces(const Surface surface1,
                                const Surface surface2,
                                Array2D<double>& TetaPhi,
                                bool& hole)
{ 
    //get the 2 arrays
    Array2D<double> TetaPhi1, TetaPhi2;
    TetaPhi1.resize(68, 35);
    TetaPhi2.resize(68, 35);
    bool hole1, hole2;
    getDistanceFromOrig(surface1, TetaPhi1, hole1);
    getDistanceFromOrig(surface2, TetaPhi2, hole2);

    if(hole1 == true || hole2 == true)
        hole = true;
    else
        hole = false;

    int j, k;
    
    for(j = 0; j < 68; j++)
    {
        for (k = 0; k < 35; k++)
        {
            if((TetaPhi1.get(j, k) <= 0.01 && TetaPhi1.get(j, k) >= -0.01) || 
                (TetaPhi2.get(j, k) <= 0.01 && TetaPhi2.get(j, k) >= -0.01))
                TetaPhi.set(j, k, 911);
            else
            {
                //std::cout<<"TetaPhi1 : "<<TetaPhi1.get(j, k)<<" TetaPhi2 : "<<TetaPhi2.get(j, k)<<std::endl;
                TetaPhi.set(j, k, (TetaPhi1.get(j, k) - TetaPhi2.get(j, k)) );
            }
        }
    }
}



void
SurfaceUtils
::readColorFile(std::istream &input, 
              Vector3D<double> colorArray[256]) 
{
    if (input.fail())
    {
        throw std::runtime_error("invalid input stream");
    }

    int k;
    for (k = 0 ; k < 256 ; k++)
    {
       input >> colorArray[k].x >> colorArray[k].y >> colorArray[k].z;
    }

    if (input.fail()) 
    {
        throw std::runtime_error("error reading vertices.");
    }
}




void
SurfaceUtils
::readColorFile(const std::string& filename,
              Vector3D<double> colorArray[256])
{
    std::ifstream input(filename.c_str(), std::ios::binary);
    if (input.bad()) 
    {
        throw std::runtime_error("error opening file");
    }
    readColorFile(input, colorArray);
    input.close();
}


Array2D<Vector3D<double> >
SurfaceUtils
::getArray2DinColors(Array2D<double>& TetaPhi,
                     const std::string& colorFileName,
                     const int visualisationMode)
{   
    //rescale the values of TetaPhi between 0 and 255 integers -> TetaPhiGrey
    Array2D<double> TetaPhiGrey;//int
    TetaPhiGrey.resize(68, 35);    
    int j, k;//, value;
    double max = -1000, min = 1000, value;
    //max and min
    for(j = 0; j < 68; j++)
    {
        for (k = 0; k < 35; k++)
        {
            if (TetaPhi.get(j, k) != 911 && TetaPhi.get(j, k) > max)
                max = TetaPhi.get(j, k);
            if (TetaPhi.get(j, k) < min)
                min = TetaPhi.get(j, k);
        }
    }
    std::cout<<"max : "<<max<<std::endl;
    std::cout<<"min : "<<min<<std::endl;

    if(visualisationMode == 0)//absolut -4/4
    {
        for(j = 0; j < 68; j++)
        {
            for (k = 0; k < 35; k++)
            {
                if(TetaPhi.get(j, k) <= 915 && TetaPhi.get(j, k) >= 908)
                    TetaPhiGrey.set(j, k, 911);
                else
                {
                    value =  (TetaPhi.get(j, k)) * 255 / (8) + 255 / 2 ;
                    TetaPhiGrey.set(j, k, value);
                    //std::cout<<TetaPhiGrey.get(j, k)<<" ";
                }
            }
            //std::cout<<std::endl;
        }
        std::cout<<"visualisationMode ABSOLUT -4/4"<<std::endl;
    }
    if(visualisationMode == 1)//relative
    {
        for(j = 0; j < 68; j++)
        {
            for (k = 0; k < 35; k++)
            {
                if(TetaPhi.get(j, k) <= 915 && TetaPhi.get(j, k) >= 908)
                    TetaPhiGrey.set(j, k, 911);
                else
                {
                    value =  (TetaPhi.get(j, k)) * 255 / (max - min) - min * 255 / (max - min) ;
                    TetaPhiGrey.set(j, k, value);
                    //std::cout<<TetaPhiGrey.get(j, k)<<" ";
                }
            }
            //std::cout<<std::endl;
        }
        std::cout<<"visualisationMode RELATIVE"<<std::endl;
    }
    if(visualisationMode == 2)//mid
    {
        if(max < 0 && min < 0)
        {
            for(j = 0; j < 68; j++)
            {
                for (k = 0; k < 35; k++)
                {
                    if(TetaPhi.get(j, k) <= 915 && TetaPhi.get(j, k) >= 908)
                        TetaPhiGrey.set(j, k, 911);
                    else
                    {
                        value = - (127 * TetaPhi.get(j, k) / min) + 127;
                        TetaPhiGrey.set(j, k, value);
                        //std::cout<<TetaPhiGrey.get(j, k)<<" ";
                    }
                }
                //std::cout<<std::endl;
            } 
            std::cout<<"max<0 min<0"<<std::endl;
        }
        if(max > 0 && min > 0)
        {
            for(j = 0; j < 68; j++)
            {
                for (k = 0; k < 35; k++)
                {
                    if(TetaPhi.get(j, k) <= 915 && TetaPhi.get(j, k) >= 908)
                        TetaPhiGrey.set(j, k, 911);
                    else
                    {
                        value = (128 * TetaPhi.get(j, k) / max) + 127;
                        TetaPhiGrey.set(j, k, value);
                        //std::cout<<TetaPhiGrey.get(j, k)<<" ";
                    }
                }
                //std::cout<<std::endl;
            }
            std::cout<<"max>0 min>0"<<std::endl;
        }
        if(max > 0 && min < 0)
        {
            if(fabs(min) < max)
            {
                for(j = 0; j < 68; j++)
                {
                    for (k = 0; k < 35; k++)
                    {
                        if(TetaPhi.get(j, k) <= 915 && TetaPhi.get(j, k) >= 908)
                            TetaPhiGrey.set(j, k, 911);
                        else
                        {
                            value = (128 * TetaPhi.get(j, k) / max) + 127 ;
                            TetaPhiGrey.set(j, k, value);
                            //std::cout<<TetaPhiGrey.get(j, k)<<" ";
                        }
                    }
                    //std::cout<<std::endl;
                }
                std::cout<<"max>0 min<0  abs(min)<max"<<std::endl;
            }
            else
            {
                for(j = 0; j < 68; j++)
                {
                    for (k = 0; k < 35; k++)
                    {
                        if(TetaPhi.get(j, k) <= 915 && TetaPhi.get(j, k) >= 908)
                            TetaPhiGrey.set(j, k, 911);
                        else
                        {
                            value = -(127 * TetaPhi.get(j, k) / min) + 127 ;
                            TetaPhiGrey.set(j, k, value);
                            //std::cout<<TetaPhiGrey.get(j, k)<<" ";
                        }
                    }
                    //std::cout<<std::endl;
                }
                std::cout<<"max>0 min<0  abs(min)>max"<<std::endl;
            }
        }
        std::cout<<"visualisationMode MID"<<std::endl;
    }



    //get TetaPhiColor from TetaPhiGrey and colorFileName
    //get colorArray
    Vector3D<double> colorArray[256];
    readColorFile(colorFileName, colorArray);

    Vector3D<double> holeColor;
    Array2D<Vector3D<double> > TetaPhiColor;  
    TetaPhiColor.resize(68, 35);
    for(j = 0; j < 68; j++)
    {
        for (k = 0; k < 35; k++)
        {
            //std::cout<<"inside for  j : "<<j<<" k : "<<k<<std::endl;
            if(TetaPhiGrey.get(j, k) <= 915 && TetaPhiGrey.get(j, k) >= 905)
            {
                std::cout<<"HOLE IN WHITE"<<std::endl;
                holeColor.x = 255;
                holeColor.y = 255;
                holeColor.z = 255;
                TetaPhiColor.set(j, k, holeColor);
            }
            else
            {
                //std::cout<<"before allocation TetaPhiColor   TetaPhiGrey : "<<TetaPhiGrey.get(j, k)<<std::endl;
                TetaPhiColor.set(j, k, (colorArray[(int)TetaPhiGrey.get(j, k)])*255 );
                //std::cout<<"after allocation TetaPhiColor"<<std::endl;
                //std::cout<<"TetaPhiColor(" << j<<", "<<k<<") : "<<TetaPhiColor.get(j, k)<<std::endl;
            }
        }

    }
    return TetaPhiColor;
}


  


double
SurfaceUtils
::getMeanDistanceBetweenTwoSurfaces(Array2D<double>& TetaPhi)
{ 
    int NumElements = TetaPhi.getNumElements();
    double mean = 0;
    int j, k;
    for(j = 0; j < 68; j++)
    {
        for (k = 0; k < 35; k++)
        {
            mean = mean + TetaPhi.get(j, k);
        }
    }
    return (mean / NumElements);
}




Surface
SurfaceUtils
::Sphere(const double& radius, 
         const int& level)
{
    Surface surface = createUnitOctahedron();

    int k = 0;
    while(k < level)
    {
        //refine Octahedron
        refineSurface(surface);
    
        //vertices at equal distance R (rayon) from centroid 
        for (Surface::VertexIter vertexIter = surface.vertices.begin(); 
             vertexIter != surface.vertices.end(); 
             ++vertexIter) 
        {
            (*vertexIter).normalize();
        }
        k++;
    }
    for (Surface::VertexIter vertexIter = surface.vertices.begin(); 
         vertexIter != surface.vertices.end(); 
         ++vertexIter) 
    {
        *vertexIter = radius * (*vertexIter);
    }
    return surface;
}



//
// uniform perturbation of surface vertices
//
void
SurfaceUtils
::perturbVertexPositions(Surface& s,
                         const double& maxXPert,
                         const double& maxYPert,
                         const double& maxZPert)
{
  for (Surface::VertexIter i = s.vertices.begin();
       i != s.vertices.end();
       ++i)
    {
      i->x += Random::sampleUniform(-maxXPert,maxXPert);
      i->y += Random::sampleUniform(-maxYPert,maxYPert);
      i->z += Random::sampleUniform(-maxZPert,maxZPert);
    }
}

