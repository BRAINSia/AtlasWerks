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

#ifndef IMANA_H
#define IMANA_H

//////////////////////////////////////////////////////////////////////
//
// File: ImAna.h
//
// The class defines an Image Anatomical Structure and is internally
// represented by an Anastruct, a Surface and their properties :
// visibility, color, aspect in the 3D window and opacity.
//
// D. Prigent (04/07/2004)
//////////////////////////////////////////////////////////////////////

// 'identifier' : identifier was truncated to 'number' characters in the
// debug information
#ifdef WIN32
#pragma warning ( disable : 4786 )
#endif

#include <vector>
#include "Surface.h"
#include "Anastruct.h"
#include "Vector3D.h"

////////////////////////////////
// Image Anatomical Structure //
////////////////////////////////

// A Image Anatonical Structure (ImAna) can be defined with 
// an Anastruct or a Surface. When you past an Anastruct, it is creating 
// the Surface. 
// Properties are set by default if they are not passed in the constructor
// All the variables and the function are public for a fastest access.

class ImAna
{
public:
  typedef double                ColorType;
  typedef Vector3D<ColorType>   RGBType;

  enum SurfaceRepresentationType{surface_representation,
                                 wireframe_representation,
                                 contours_representation};
  
  Anastruct anastruct;
  Surface surface;
  bool visible;
  RGBType color;//RGB
  SurfaceRepresentationType aspect;
  float opacity;//[0,1]
  
  ImAna();
  ImAna(const Anastruct& newAnastruct, 
        const bool& visibility = true, 
        const RGBType& anastructColor = RGBType(1, 0, 0), //red : default color
        const SurfaceRepresentationType& anastructAspect =
           surface_representation, 
        const float& anastructOpacity=1.0);
  
  ImAna(const Anastruct& newAnastruct, 
        const Surface& newSurface, 
        const bool& visibility = true,
        const RGBType& anastructColor = RGBType(1, 0, 0), //red : default color
        const SurfaceRepresentationType& anastructAspect=
           surface_representation, 
        const float& anastructOpacity=1.0);
  ~ImAna();
  ImAna& operator=(const ImAna& rhs);
  
private:
  
};// ImAna class

#endif
