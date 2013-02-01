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

#include "Anastruct.h"

Anastruct
::Anastruct()
{}

Anastruct
::Anastruct(const Anastruct& rhs)
  : contours(rhs.contours),
    label(rhs.label),
    max(rhs.max),
    min(rhs.min)
{}

Anastruct
::Anastruct(const PLUNCAnastruct& rhs)
{
  copyFromPLUNCAnastruct(rhs);  
}

Anastruct& 
Anastruct
::operator=(const Anastruct& rhs)
{
  if (this != &rhs)
    {
      contours = rhs.contours;
      label = rhs.label;
      max = rhs.max;
      min = rhs.min;
    }
  return *this;
}

Anastruct& 
Anastruct
::operator=(const PLUNCAnastruct& rhs)
{
  copyFromPLUNCAnastruct(rhs);  
  return *this;
}

void
Anastruct
::clear()
{
  contours.clear();
}

void
Anastruct
::updateMinMax()
{
  if (contours.empty()) return;
  
  double thickness = 0;
  thickness = contours[1].vertices[0].z - contours[0].vertices[0].z;

  for (ContourIter contourIter = contours.begin();
       contourIter != contours.end();
       ++contourIter)
    {
      contourIter->updateMinMax(thickness);
    }

  max.x = contours[0].max.x;
  min.x = contours[0].min.x;
  max.y = contours[0].max.y;
  min.y = contours[0].min.y;
  max.z = contours[0].max.z;
  min.z = contours[0].min.z;

  for (ConstContourIter constContourIter = contours.begin();
       constContourIter != contours.end();
       ++constContourIter)
    {
      if (constContourIter->max.x > max.x) max.x = constContourIter->max.x;
      if (constContourIter->max.y > max.y) max.y = constContourIter->max.y;
      if (constContourIter->max.z > max.z) max.z = constContourIter->max.z;

      if (constContourIter->min.x < min.x) min.x = constContourIter->min.x;
      if (constContourIter->min.y < min.y) min.y = constContourIter->min.y;
      if (constContourIter->min.z < min.z) min.z = constContourIter->min.z;
    }
}

void
Anastruct
::copyFromPLUNCAnastruct(const PLUNCAnastruct& rhs)
{
  label.assign(rhs.label);
  max.set(rhs.max.x, rhs.max.y, rhs.max.z);
  min.set(rhs.min.x, rhs.min.y, rhs.min.z);
  
  //
  // copy over each contour in the array
  //
  contours.resize(rhs.contour_count);
  for (int contourIndex = 0; 
       contourIndex < rhs.contour_count; 
       ++contourIndex)
    {
      contours[contourIndex].sliceNumber = 
	rhs.contours[contourIndex].slice_number;

      contours[contourIndex].max.set(rhs.contours[contourIndex].max.x,
				     rhs.contours[contourIndex].max.y,
				     rhs.contours[contourIndex].max.z);
      contours[contourIndex].min.set(rhs.contours[contourIndex].min.x,
				     rhs.contours[contourIndex].min.y,
				     rhs.contours[contourIndex].min.z);

      int vertexCount = rhs.contours[contourIndex].vertex_count;
      contours[contourIndex].vertices.resize(vertexCount);
      for (int vertexIndex = 0; vertexIndex < vertexCount; ++vertexIndex)
	{
	  contours[contourIndex].vertices[vertexIndex].set(rhs.contours[contourIndex].x[vertexIndex],
							   rhs.contours[contourIndex].y[vertexIndex],
							   rhs.contours[contourIndex].z);
	}
    }
}

void
Anastruct
::copyToPLUNCAnastruct(PLUNCAnastruct& rhs) const
{
  // copy over the name
  for (unsigned int labelIndex = 0; labelIndex <= label.size(); ++labelIndex)
    {
      rhs.label[labelIndex] = label.c_str()[labelIndex];
    }
  
  rhs.max.x = max.x;
  rhs.max.y = max.y;
  rhs.max.z = max.z;

  rhs.min.x = min.x;
  rhs.min.y = min.y;
  rhs.min.z = min.z;
  
  //
  // copy over each contour in the array
  //
  if (rhs.contours)
    {
      // delete old memory held by this contour
      for (int i = 0; i < rhs.contour_count; ++i)
	{
	  if (rhs.contours[i].vertex_count && rhs.contours[i].x) 
	    {
	      delete [] rhs.contours[i].x;
	    }
	  if (rhs.contours[i].vertex_count && rhs.contours[i].y) 
	    {
	      delete [] rhs.contours[i].y;
	    }
	}
      delete [] rhs.contours;
    }
  rhs.contours = new CONTOUR[contours.size()];
  rhs.contour_count = contours.size();

  for (int contourIndex = 0; 
       contourIndex < rhs.contour_count; 
       ++contourIndex)
    {
      rhs.contours[contourIndex].slice_number =
	contours[contourIndex].sliceNumber;
      if (!contours[contourIndex].vertices.empty())
	{
	  rhs.contours[contourIndex].z = 
	    contours[contourIndex].vertices[0].z;
	}

      rhs.contours[contourIndex].max.x = contours[contourIndex].max.x;
      rhs.contours[contourIndex].max.y = contours[contourIndex].max.y;
      rhs.contours[contourIndex].max.z = contours[contourIndex].max.z;

      rhs.contours[contourIndex].min.x = contours[contourIndex].min.x;
      rhs.contours[contourIndex].min.y = contours[contourIndex].min.y;
      rhs.contours[contourIndex].min.z = contours[contourIndex].min.z;
	 
      rhs.contours[contourIndex].density = 1.0; 
      int vertexCount = contours[contourIndex].vertices.size();
      rhs.contours[contourIndex].vertex_count = vertexCount;
      rhs.contours[contourIndex].x = new float[vertexCount];
      rhs.contours[contourIndex].y = new float[vertexCount];

      for (int vertexIndex = 0; vertexIndex < vertexCount; ++vertexIndex)
	{
	  rhs.contours[contourIndex].x[vertexIndex] =
	    contours[contourIndex].vertices[vertexIndex].x;
	  rhs.contours[contourIndex].y[vertexIndex] =
	    contours[contourIndex].vertices[vertexIndex].y;
	}
    }
}

void 
Anastruct
::translate(const Vector3D<double>& t) 
{
  translate(t.x, t.y, t.z);
}

void 
Anastruct
::translate(const double& tx, 
	    const double& ty, 
	    const double& tz) 
{
  for (ContourIter citer = contours.begin();
       citer != contours.end();
       ++citer)
    {
      for (Contour::VertexIter vertIter = citer->vertices.begin(); 
           vertIter != citer->vertices.end(); 
           ++vertIter)
        {
          vertIter->translate(tx, ty, tz);
        }		
    }
  updateMinMax();
}

void
Anastruct
::clean()
{
  // - eliminate duplicate points in each contour
  // - eliminate crossed contour loops
  // - enforce contour direction  
  for (ContourIter citer = contours.begin();
       citer != contours.end();
       ++citer)
    {
      citer->clean();
    }
}
