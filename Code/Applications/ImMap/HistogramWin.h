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

#ifndef HISTOGRAM_WIN_H
#define  HISTOGRAM_WIN_H

#include <FL/gl.h>
#include <FL/Fl.H>
#include <FL/Fl_Gl_Window.H>
#include <vector>

class HistogramWin: public Fl_Gl_Window {

public :

  HistogramWin(double X, double Y, double W, double H, const char *L);

  void setWindow(std::vector<double> histogram, 
				 float relativeMin, float relativeMax,
				 float absolutMin, float absolutMax);

  void clear();

  void draw(void);
  int handle(int);

  void updateRelativeMin(float newMin);
  void updateRelativeMax(float newMax);

  float updateRelativeMin(){return _relativeMin;};
  float updateRelativeMax(){return _relativeMax;};

  void setMinMaxChangedCallback(void(*callbackFcn)(void* arg),void *callbackArg);

  void setLUTcolor(float red, float green, float blue);

private :

  GLubyte LUT[256][1][3];

  float _LUTcolorRed, _LUTcolorGreen, _LUTcolorBlue;
  std::vector<double> _histogram;

  // window dimensions
  double  _windowWidth;
  double  _windowHeight;

  // Max of the histo
  double _maxPopulation;

  float _relativeMin, _relativeMax;
  float _absolutMin, _absolutMax, _absolutDiff;
  bool _draggingMin,_draggingMax, _draggingBoth;
  double  _lastMouseXPosition;

  bool initialized;

  void _updateMinMaxDraggingFlags ( bool isPress, double mouseX, double mouseY);
  void _dragMinMax(double mouseX, double mouseY);
  
  float _windowToHistogramValue(double windowX);
  double   _histogramValueToWindow(float histogramValue);

  void (*_MinMaxChangedCallback)(void *arg);
  void *_MinMaxChangedCallbackArg;

};

#endif

