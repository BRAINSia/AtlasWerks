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

#include "Contour.h"
#include <vector>

void 
Contour
::clear()
{
  vertices.clear();
}

void 
Contour
::updateMinMax(double& thickness)
{
  if (!vertices.empty())
    {
      max.x = vertices[0].x;
      min.x = vertices[0].x;
      max.y = vertices[0].y;
      min.y = vertices[0].y;
      max.z = vertices[0].z;
      min.z = vertices[0].z;

      for (ConstVertexIter vertIter = vertices.begin();
	   vertIter != vertices.end();
	   ++vertIter)
	{
	  if (vertIter->x < min.x) min.x = vertIter->x;
	  if (vertIter->y < min.y) min.y = vertIter->y;
	  //if (vertIter->z < min.z) min.z = vertIter->z;
	  if (vertIter->x > max.x) max.x = vertIter->x;
	  if (vertIter->y > max.y) max.y = vertIter->y;
	  //if (vertIter->z > max.z) max.z = vertIter->z;
	}

      // the max.z of a contour in PLUNC is the half of the thickness 
      // between this contour and the contour up  
      
      max.z += thickness/2;
      
      // the min.z of a contour in PLUNC is the half of the thickness 
      // between this contour and the contour down
      
      min.z -= thickness/2;
    }
}

double
Contour
::perimeter()
{
  double perimeter = 0;
  for (unsigned int i = 0; i < vertices.size() - 1; ++i)
    {
      perimeter += vertices[i].distance(vertices[i+1]);
    }  
  return perimeter;
}

// - eliminate duplicate points
// - eliminate crossed contour loops
// - enforce contour direction  
void
Contour
::clean()
{
  double tolerence = 0.005;
  int numDuplicates = _eliminateDuplicatePoints(tolerence);
  std::cerr << "Num. Duplicates: " << numDuplicates << std::endl;
  //_eliminateLoops();
  //_setContourDirection();
}

//
// eliminate duplicate points while maintaining order of vertices
//
int
Contour
::_eliminateDuplicatePoints(const double& tolerence)
{
  VertexList uniqueVertices;
  int numDuplicates = 0;
  for (VertexIter i = vertices.begin(); i != vertices.end(); ++i)
    {
      bool unique = true;
      for (VertexIter j = uniqueVertices.begin(); 
           j != uniqueVertices.end(); 
           ++j)
        {
          if (i->distance(*j) < tolerence)
            {
              unique = false;
              break;
            }
        }
      if (unique)
        {
          uniqueVertices.push_back(*i);
        }
      else
        {
          ++numDuplicates;
        }
    }
  vertices.assign(uniqueVertices.begin(), uniqueVertices.end());
  return numDuplicates;
}

void
Contour
::writePoints(std::ostream& output)
{
  for (VertexIter i = vertices.begin(); i != vertices.end(); ++i)
    {
      output << "(" << i->x << ", " << i->y << ", " << i->z << ")" 
             << std::endl;
    }  
}

//
// eliminate loops while maintaining order of vertices
//
// int
// Contour
// ::_eliminateLoops(const double& tolerence)
// {
//   int numLoops = 0;
//   for (int i = 0; i < vertices.size() - 1; ++i)
//     {
//       for (int j = i + 2; j < vertices.size() - 1; ++j)
//         {
//           if (Geometry::
//               lineLineIntersection2D(vertices[i].x,   vertices[i].y,
//                                      vertices[i+1].x, vertices[i+1].y,
//                                      vertices[j].x,   vertices[j].y,
//                                      vertices[j+1].x, vertices[j+1].y))
//             {
              
//             }
//         }
//     }
// }
