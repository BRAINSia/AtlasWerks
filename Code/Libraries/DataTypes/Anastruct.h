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

#ifndef Anastruct_h
#define Anastruct_h

// gen.h is where the plunc anastruct is defined
#include <gen.h>
#include <vector>
#include <string>
#include "Vector3D.h"
#include "Contour.h"
#include <cassert>

//#include "PlatformCompatibility.h"

class Anastruct
{
public:
  typedef ANASTRUCT                    PLUNCAnastruct;
  typedef CONTOUR                      PLUNCContour;
  typedef std::vector<Contour>         ContourList;
  typedef ContourList::iterator        ContourIter;
  typedef ContourList::const_iterator  ConstContourIter;

  Anastruct();
  Anastruct(const Anastruct& rhs);
  Anastruct(const PLUNCAnastruct& rhs);

  Anastruct& operator=(const Anastruct& rhs);
  Anastruct& operator=(const PLUNCAnastruct& rhs);

  void translate(const Vector3D<double>& t);
  void translate(const double& tx, 
		 const double& ty,
		 const double& tz);
  
  void clear();
  void clean();
  void updateMinMax();

  ContourList              contours;
  std::string              label;
  Vector3D<double>         max;
  Vector3D<double>         min;

  void copyToPLUNCAnastruct(PLUNCAnastruct& rhs) const;
  void copyFromPLUNCAnastruct(const PLUNCAnastruct& rhs);
};

#endif
