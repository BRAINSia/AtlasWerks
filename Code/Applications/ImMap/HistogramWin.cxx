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

#include "HistogramWin.h"
#include <iostream>
//#include <Debugging.h>

HistogramWin::HistogramWin(double X, double Y, double W, double H, const char *L)
  : Fl_Gl_Window(X,Y,W,H,L)
{
  _maxPopulation = 0;
  _relativeMin = _relativeMax = 0.0;
  _absolutMin = _absolutMax = 0.0;
  _windowWidth = W;
  _windowHeight = H;

  _MinMaxChangedCallback = 0;
  _MinMaxChangedCallbackArg = 0;

  initialized = false;

  for (int i = 0 ; i < 256 ; i++){
    LUT[i][0][0]=LUT[i][0][1]=LUT[i][0][2]=i;
	
  } 
  
}

void HistogramWin::setWindow(std::vector<double> histogram, 
                             float relativeMin, 
                             float relativeMax,
                             float absolutMin, 
                             float absolutMax)
{
  _histogram=histogram;
  _relativeMin=relativeMin;
  _relativeMax=relativeMax;
  _absolutMin=absolutMin;
  _absolutMax=absolutMax;
  _absolutDiff=absolutMax-absolutMin+1;

  initialized = true;
  _draggingMin = false;
  _draggingMax = false;
  _draggingBoth = false;
  _LUTcolorRed = _LUTcolorGreen = _LUTcolorBlue = 1.0;
  redraw();
}

void HistogramWin::clear()
{
	initialized = false;
	redraw();
}

void HistogramWin::setLUTcolor(float red, float green, float blue)
{
  _LUTcolorRed = red;
  _LUTcolorGreen = green;
  _LUTcolorBlue = blue;

  for (int i = 0 ; i < 256 ; i++)
  {
    LUT[i][0][0]=static_cast<GLubyte>(i*_LUTcolorRed);
    LUT[i][0][1]=static_cast<GLubyte>(i*_LUTcolorGreen);
    LUT[i][0][2]=static_cast<GLubyte>(i*_LUTcolorBlue);	
  } 
  redraw();
}

void HistogramWin::updateRelativeMin(float newMin){

  _relativeMin=newMin;
  redraw();
  
}

void HistogramWin::updateRelativeMax(float newMax){

  _relativeMax=newMax;
  redraw();
 
}

void HistogramWin::setMinMaxChangedCallback(void(*callbackFcn)(void* arg),void *callbackArg)
{
	_MinMaxChangedCallback = callbackFcn;
	_MinMaxChangedCallbackArg = callbackArg;	
}

float HistogramWin::_windowToHistogramValue(double windowX)
{
	//there is a gap of 20 pixels between the line and the border of the window
	return (((windowX-10) * _absolutDiff)/(_windowWidth-20)) + _absolutMin;
}

double HistogramWin::_histogramValueToWindow(float histogramValue)
{
	if(_relativeMax>1)
		return static_cast<double>( (((histogramValue - _absolutMin) * (_windowWidth-20))/_absolutDiff)+10 );  
	else
		return static_cast<double>( (((histogramValue - _absolutMin) * (_windowWidth-20))/(_absolutDiff))+10 );
}

void HistogramWin::draw() {
  if (!valid()) {
    glClearColor(0.0, 0.0, 0.0, 0.0);
    glClear(GL_COLOR_BUFFER_BIT);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glViewport(0, 0, w(), h());
	ortho();

  }

  glMatrixMode(GL_MODELVIEW);
  glClearColor(0,0,0,0);
  glClear(GL_COLOR_BUFFER_BIT);

  if (initialized){

    // Search the max of the histo in its "valid" part
    _maxPopulation=0;

	double i =0;
    if(_relativeMax>1)  
   {
    for ( i = (int) _relativeMin  ; i < (int) _relativeMax+1 ; i ++ ){
	if(i==0 || i==1)
		continue;
      if (_histogram[i-(double)_absolutMin]>_maxPopulation) _maxPopulation=_histogram[i-(double)_absolutMin];
	//std::cout<<"i: "<<i<<"  : _absolutMin"<<_absolutMin<<"  hist val"<<_histogram[i-(double)_absolutMin]<<"  relative max : "<<_relativeMax<<"     "<<(int)((_relativeMax-_relativeMin+1)*100)<<std::endl;
    }
/*
	for(i=0;i<200;i+=1)
	{
		std::cout<<"iteration : "<<i<<"      val : "<<_histogram[i]<<std::endl;
	}
    
  */  
    glLineWidth(1.0);
    glColor3f(0,0,1);
    glBegin(GL_LINE_STRIP);
    for ( i = 1 ; i < _absolutDiff ; i ++){
		if ((i<(_relativeMin-_absolutMin)) || (i>(_relativeMax-_absolutMin))){
			glVertex2f(_histogramValueToWindow(i+_absolutMin),40);
		}
		else{
			if (_histogram[i]==0){
				glVertex2f(_histogramValueToWindow(i+_absolutMin),40);
			}
			else{
				glVertex2f(_histogramValueToWindow(i+_absolutMin),((_windowHeight-50)*(_histogram[i]/(float)_maxPopulation))+40 );
			}
		}
    }
    glEnd();
}
else
{
     for ( i = (int) _relativeMin*100  ; i < (int)((_relativeMax)*100 + 1) ; i ++ ){
	if (i==0)
		continue;
      if (_histogram[i-(double)_absolutMin]>_maxPopulation) _maxPopulation=_histogram[i-(double)_absolutMin];
        //std::cout<<"i: "<<i<<"  : _absolutMin"<<_absolutMin<<"  hist val"<<_histogram[i-(double)_absolutMin]<<"  relative max : "<<_relativeMax<<"     "<<(int)((_relativeMax-_relativeMin+1)*100)<<std::endl;
    }
/*
        for(i=0;i<200;i+=1)
        {
                std::cout<<"iteration : "<<i<<"      val : "<<_histogram[i]<<std::endl;
        }
    std::cout<<"absolute diff : "<<_absolutDiff<<"  absolute min : "<<_absolutMin<<"  relativeMin: "<<_relativeMin<<"  Relative Max: "<<_relativeMax<<std::endl;
*/
    glLineWidth(1.0);
    glColor3f(0,0,1);
    glBegin(GL_LINE_STRIP);
    for ( i = 1 ; i < (int)(_absolutDiff * 100) ; i ++){
                if ((i<((_relativeMin-_absolutMin)*100)) || (i>((_relativeMax-_absolutMin)*100))){
                        glVertex2f(_histogramValueToWindow(i+_absolutMin)/100+(10),40);
                }
                else{
                        if (_histogram[i]==0){
                                glVertex2f(_histogramValueToWindow(i+_absolutMin)/100+(10),40);
                        }
                        else{
				//std::cout<<"iter : "<<i<<std::endl;
                                glVertex2f((_histogramValueToWindow(i+_absolutMin)/100)+(10),((_windowHeight-50)*(_histogram[i]/(float)_maxPopulation))+40 );
                        }
                }
    }
    glEnd();
}

    // Draw 2 lines : at the beginning and at th end

	
    glColor3f(1,1,1);
	glLineWidth(1.0);
    glBegin(GL_LINES);
       glVertex2f(_histogramValueToWindow(_relativeMin),_windowHeight - 10);
       glVertex2f(_histogramValueToWindow(_relativeMin),30);
	   glVertex2f(_histogramValueToWindow(_relativeMin),30);
       glVertex2f(_histogramValueToWindow(_relativeMax),30);
       glVertex2f(_histogramValueToWindow(_relativeMax),30);
       glVertex2f(_histogramValueToWindow(_relativeMax),_windowHeight - 10);
    glEnd();

	//Draw arrows to show that lines can move

	glColor3f(1,1,1);
	glLineWidth(1.0);
    glBegin(GL_POLYGON);
	glVertex2f(_histogramValueToWindow(_relativeMin),_windowHeight - 10);
	glVertex2f(_histogramValueToWindow(_relativeMin)+7,_windowHeight - 5);
	glVertex2f(_histogramValueToWindow(_relativeMin),_windowHeight - 14);
    glVertex2f(_histogramValueToWindow(_relativeMin)-7,_windowHeight - 5);
	glEnd();


	double windowMiddle = _histogramValueToWindow(_relativeMin + ((_relativeMax - _relativeMin)/2));
	glColor3f(1,1,1);
	glLineWidth(1.0);
	glBegin(GL_POLYGON);
    glVertex2f(windowMiddle - 14, 30);
    glVertex2f(windowMiddle - 5 , 37);
	glVertex2f(windowMiddle - 10, 30);
    glVertex2f(windowMiddle - 5 , 23);
	glEnd();

	glColor3f(1,1,1);
	glLineWidth(1.0);
	glBegin(GL_POLYGON);
    glVertex2f(windowMiddle + 14, 30);
    glVertex2f(windowMiddle + 5 , 37);
	glVertex2f(windowMiddle + 10, 30);
    glVertex2f(windowMiddle + 5 , 23);
	glEnd();

	glColor3f(1,1,1);
	glLineWidth(1.0);
	glBegin(GL_POLYGON);
	glVertex2f(_histogramValueToWindow(_relativeMax),_windowHeight - 10);
    glVertex2f(_histogramValueToWindow(_relativeMax)+7,_windowHeight - 5);
	glVertex2f(_histogramValueToWindow(_relativeMax),_windowHeight - 14);
    glVertex2f(_histogramValueToWindow(_relativeMax)-7,_windowHeight - 5);
	glEnd();
      
    

    // Draw a white rectangle to end the LUT

	glColor3f(_LUTcolorRed, _LUTcolorGreen, _LUTcolorBlue);
     glBegin(GL_QUADS);
       glVertex2f(_histogramValueToWindow(_relativeMax),5);
       glVertex2f(_histogramValueToWindow(_relativeMax),22.0);
       glVertex2f((_windowWidth)-10,22);
       glVertex2f((_windowWidth)-10,5);
    glEnd();

    glPixelStorei(GL_UNPACK_ALIGNMENT,1 );
    glRasterPos2f(_histogramValueToWindow(_relativeMin),5);
	glPixelZoom(((_relativeMax-_relativeMin+1)/_absolutDiff)*0.63,17);
    glDrawPixels(256,1,GL_RGB,GL_UNSIGNED_BYTE,LUT);

  }
  

}
void HistogramWin::_updateMinMaxDraggingFlags(bool isPress, double mouseX, double mouseY)
{
	if(isPress)
	{
		double relativeMinLine = _histogramValueToWindow(_relativeMin);
		double relativeMaxLine = _histogramValueToWindow(_relativeMax);
		//int relativeDiff = relativeMaxLine - relativeMinLine;
		
		if ( mouseX <= relativeMinLine ) 
		{
			_draggingMin = true;
		}
		
		if ( mouseX >= relativeMaxLine )
		{
			_draggingMax = true;
		}
		if ( ( mouseX > relativeMinLine) &&
			 ( mouseX < relativeMaxLine))
		{
			_draggingBoth = true;
		}
	}
	else
	{
		_draggingMin = false;
		_draggingMax = false;
		_draggingBoth = false;

	}	
}


void HistogramWin::_dragMinMax(double mouseX, double mouseY)
{
  //	int clickTolerence = 50 ;
	if (_draggingMin || _draggingMax || _draggingBoth)
	{
		float moveX = (((mouseX - _lastMouseXPosition)*_absolutDiff)/(_windowWidth-20));
		float relativeDiff = _relativeMax - _relativeMin;
		
		if (_draggingMin)
		{
			if ((_relativeMin+moveX) < _absolutMin)
			{
				_relativeMin = _absolutMin;
				_lastMouseXPosition  = mouseX;
			}
			else
			{
				if ((_relativeMin+moveX) > _relativeMax)
				{
					_relativeMin = _relativeMax  ;
					_lastMouseXPosition  = mouseX;
				}
				else
				{
				_relativeMin += moveX;
				_lastMouseXPosition  = mouseX;
				}
			}
		}
		if (_draggingMax)
		{
			if ((_relativeMax+moveX) > _absolutMax)
			{
				_relativeMax = _absolutMax;
				_lastMouseXPosition  = mouseX;
			}
			else
			{
				if ((_relativeMax+moveX) < _relativeMin)
				{
					_relativeMax = _relativeMin;
					_lastMouseXPosition  = mouseX;
				}
				else
				{
					_relativeMax += moveX;
					_lastMouseXPosition  = mouseX;
				}
			}
		}
		
		if (_draggingBoth)
		{
			if ((_relativeMin+moveX) < _absolutMin)
			{
				_relativeMax = _absolutMin + relativeDiff;
				_relativeMin = _absolutMin;
				_lastMouseXPosition  = mouseX;
			}
			else
			{
				if ((_relativeMax+moveX) > _absolutMax)
				{
					_relativeMax = _absolutMax ;
					_relativeMin = _absolutMax - relativeDiff;
					_lastMouseXPosition  = mouseX;
				}
				else
				{
					_relativeMax += moveX;
					_relativeMin += moveX;
					_lastMouseXPosition  = mouseX;
				}
			}
		}
		redraw();
		if (_MinMaxChangedCallback != 0)
		{
			_MinMaxChangedCallback(_MinMaxChangedCallbackArg);
		}
	}

}


int HistogramWin::handle(int event) {

   
  switch(event) {
    
    
  case FL_PUSH:
    {
			double button = Fl::event_button();
			double mouseX = Fl::event_x();
			double mouseY = Fl::event_y();
			mouseY = _windowHeight - mouseY;
			if (button == 1)
			{
				_lastMouseXPosition  = mouseX;
				_updateMinMaxDraggingFlags(true,mouseX, mouseY);
			}
			return 1;
			break;
		}
  case FL_DRAG:
	  {
			double button = Fl::event_button();
			double mouseX = Fl::event_x();
			double mouseY = Fl::event_y();
			mouseY = _windowHeight - mouseY;
			if (button == 1)
			{
				_dragMinMax(mouseX, mouseY);
			}
			return 1;
			break;
		}
  case FL_RELEASE:    
        {
			double button = Fl::event_button();
			double mouseX = Fl::event_x();
			double mouseY = Fl::event_y();
			mouseY = _windowHeight - mouseY;
			if (button == 1)
			{
				_updateMinMaxDraggingFlags(false,mouseX, mouseY);
			}
			return 1;
			break;
		}
  case FL_FOCUS :
  case FL_UNFOCUS :
    // Return 1 if you want keyboard events, 0 otherwise
    return 1;
  case FL_KEYBOARD:
    // keypress, key is in Fl::event_key(), ascii in Fl::event_text()
    // Return 1 if you understand/use the keyboard event, 0 otherwise...
    return 1;
  case FL_SHORTCUT:
    // shortcut, key is in Fl::event_key(), ascii in Fl::event_text()
    // Return 1 if you understand/use the shortcut event, 0 otherwise...
    return 1;
  default:
    // pass other events to the base class...
    return Fl_Gl_Window::handle(event);
  }
}
