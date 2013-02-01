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

//////////////////////////////////////////
//
// File ImAna.cxx
//
//////////////////////////////////////////
#ifndef IMANA_CXX
#define IMANA_CXX

#include "ImAna.h"
#include <AnastructUtils.h>

/////////////////
// constructor //
/////////////////

ImAna
::ImAna()
: visible(true),
  color(0,0,0),//black
  aspect(surface_representation),
  opacity(1.0)
{}

////////////////////////////////
// constructor with anastruct //
////////////////////////////////

ImAna
::ImAna(const Anastruct& newAnastruct, 
        const bool& visibility, 
        const RGBType& anastructColor, 
        const SurfaceRepresentationType& anastructAspect, 
        const float& anastructOpacity)
{
  anastruct = newAnastruct;
  AnastructUtils::anastructToSurfacePowerCrust(anastruct,surface);
  visible   = visibility;
  color     = anastructColor;
  aspect    = anastructAspect;
  opacity   = anastructOpacity;
}

//////////////////////////////
// constructor with surface //
//////////////////////////////

ImAna
::ImAna(const Anastruct& newAnastruct, 
        const Surface& newSurface, 
        const bool& visibility, 
        const RGBType& anastructColor, 
        const SurfaceRepresentationType& anastructAspect, 
        const float& anastructOpacity)
{
  anastruct = newAnastruct;
  surface   = newSurface;
  visible   = visibility;
  color     = anastructColor;
  aspect    = anastructAspect;
  opacity   = anastructOpacity;
}

//////////////////
//  destructor  //
//////////////////

ImAna
::~ImAna()
{}

////////////////
// operator = //
////////////////

ImAna&
ImAna
::operator=(const ImAna& rhs)
{
  if (this == &rhs) return *this;
  
  anastruct = rhs.anastruct;
  surface   = rhs.surface;
  visible   = rhs.visible;
  aspect    = rhs.aspect;
  opacity   = rhs.opacity;
  
  return *this;
  
}

#endif
