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

#ifndef AnastructUtils_h
#define AnastructUtils_h


// 'identifier' : identifier was truncated to 'number' characters in the
// debug information
#ifdef WIN32
#pragma warning ( disable : 4786 )
#endif

#include "Anastruct.h"
#include "Vector3D.h"
#include "Line3D.h"
#include "Surface.h"
#include <vector>
#include <list>

class AnastructUtils
{
public:
  typedef std::list<Line3D>    LineList;
  typedef LineList::iterator   LineIterator;


  typedef Vector3D<float>       vectfloatdptr;
  typedef std::vector<vectfloatdptr>  vectfloatptr;
  typedef std::vector<vectfloatptr>   vectthreeD;

  typedef std::vector<float>  distance;
  typedef std::vector<distance>  distancelist;



  static
  void
  anastructToSurfacePowerCrust(const Anastruct& anastruct, Surface& surface);
  static
  void
  anastructToSurfacePowerCrust2(const Anastruct& anastruct, Surface& surface);

  static
  void
  anastructToNewanastruct(const Anastruct& oldanastruct, Anastruct& newanastruct);
  
  static
  void
  anastructToNewanastruct2(const Anastruct& oldanastruct, Anastruct& newanastruct);

  static
  vectthreeD
  InsertNewPoint(vectthreeD points, int pointIndexbegin, int contourIndex,
                 int numPoints, int numContours);

  static
  distancelist 
  ComputeDistance(vectthreeD points, int numContours);

  static 
  void
  anastructToSurfaceVTK(const Anastruct& anastruct, Surface& surface);

  static 
  void
  anastructToSurfaceVTK2(const Anastruct& anastruct, Surface& surface);

  static 
  void
  anastructToSurfaceDelaunayVTK(const Anastruct& anastruct, Surface& surface);

  static 
  void
  surfaceToAnastruct(const Surface& surface, 
		     Anastruct& anastruct,
		     unsigned int numContours,
		     int    *const sliceNumbers,
		     double *const zPositions,
                     const std::string& name = "");

  static 
  void
  surfaceToAnastruct(const Surface& surface, 
		     Anastruct& anastruct,
                     const std::string& name = "");

  static
  void
  surfaceToAnastruct(const Surface& surface,
                     Anastruct& anastruct,
                     const Vector3D<double>& imOrigin,
                     const Vector3D<double>& imSpacing,
                     const std::string& name = "");
  static 
  void 
  worldToImageIndex(Anastruct& anastruct,
                    const Vector3D<double>& imageOrigin,
                    const Vector3D<double>& imageSpacing);

  static 
  void 
  worldToImageIndexXY(Anastruct& anastruct,
		      const double& imageOriginX,
		      const double& imageOriginY,
		      const double& imageSpacingX,
		      const double& imageSpacingY);

  static 
  void 
  imageIndexToWorld(Anastruct& anastruct,
                    const Vector3D<double>& imageOrigin,
                    const Vector3D<double>& imageSpacing);

  static 
  void 
  imageIndexToWorldXY(Anastruct& anastruct,
		      const double& imageOriginX,
		      const double& imageOriginY,
		      const double& imageSpacingX,
		      const double& imageSpacingY);

  static
  void
  scale(Anastruct& anastruct, const double& scale);

  static
  void
  readPLUNCAnastruct(Anastruct& anastruct,
		     const char * const fileName);
  static
  void
  writePLUNCAnastruct(Anastruct& anastruct,
                      const char * const fileName);

  static
  void
  deletePLUNCAnastruct(Anastruct::PLUNCAnastruct &anastruct);

  static
  void
  capContour(Anastruct& anastruct);

private:
  static 
  void 
  _extractContours(const Surface& surface,
		   const double& zPosition,
		   int sliceNumber,
		   std::vector<Contour>& contour,
		   const double& tolerance = 0.001);
};

#endif
