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

#ifndef Window_CS_HDR
#define Window_CS_HDR 

// 'identifier' : identifier was truncated to 'number' characters in the
// debug information
#ifdef WIN32
#pragma warning ( disable : 4786 )
#endif

#include <FL/gl.h>
#include <FL/Fl_Gl_Window.H>
#include <FL/Fl.H>
#include <FL/Fl_Widget.H>

class ImMapGUI;

class WindowCS : public Fl_Gl_Window {

  void draw(void);
  int handle(int);
  
protected:

  ImMapGUI* GUI;
  
  int diffDrag;
  int minSlider,maxSlider;
  int drag,dragMin,dragMax;
  int initMax,initMin;
  int startDrag;

public :
  
  WindowCS(int X, int Y, int W, int H, const char *L);
  ~WindowCS();
  
  int GetMin();
  int GetMax();
  void SetMin(int _min);
  void SetMax(int _max);

  void SetGUI(ImMapGUI* _GUI);
  
};

#endif
