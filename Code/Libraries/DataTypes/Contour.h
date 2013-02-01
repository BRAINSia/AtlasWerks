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

#ifndef CONTOUR_H
#define CONTOUR_H

#include <vector>
#include "Vector3D.h"

class Contour
{
public:
  typedef Vector3D<double>                 Vertex;
  typedef std::vector<Vertex>              VertexList;
  typedef VertexList::iterator             VertexIter;
  typedef VertexList::const_iterator       ConstVertexIter;

  int                            sliceNumber;
  VertexList                     vertices;
  Vector3D<double>               max;
  Vector3D<double>               min;
  
  void clear();
  void clean();
  void updateMinMax(double& thickness);
  void writePoints(std::ostream& output);
  double perimeter();

private:
  int _eliminateDuplicatePoints(const double& tolerence);
  int _eliminateLoops();
};

#endif
