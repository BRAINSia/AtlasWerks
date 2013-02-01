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

//////////////////////////////////////////////////////////////////////
//
// File: Window2D.h
//
//////////////////////////////////////////////////////////////////////
#ifndef _WINDOW_2D_H_
#define _WINDOW_2D_H_

// 'identifier' : identifier was truncated to 'number' characters in the
// debug information
#ifdef WIN32
#pragma warning ( disable : 4786 )
#endif

#include "IView.h"

class Window2D : public IView<float, float> 
{
public:  
  Window2D( int xPos, int yPos, int width, int height, char* label );
  ~Window2D();
  
private:
}; // class Window2D

#endif //_WINDOW_2D_H_
