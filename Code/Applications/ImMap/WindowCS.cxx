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

# include "WindowCS.h"
#include "ImMap.h"



// WindowCS : Window Color Scale

// This OpenGL Window display a slider that looks like a fltk Slider.
// By default the slider fill all the window.
// You can chnage the lower or upper bound by dragging it.
// You can drag too the slider itself
// By default     minSlider=0;
//                maxSlider=WINDOW_H-1;


int WINDOW_H = 450;
int WINDOW_W = 30;


int approx(int first, double second){
  if ( first <= (second + 2) && first >= (second - 2)){
    return 1;
  }
  else{
    return 0;
  }
}

WindowCS::WindowCS(int X, int Y, int W, int H, const char *L)
  : Fl_Gl_Window(X,Y,W,H,L)
{
  minSlider=0;
  maxSlider=WINDOW_H-1;

  drag=dragMin=dragMax=0;
  startDrag=0;

}

WindowCS::~WindowCS(){
}

void WindowCS::draw() {
  if (!valid()) {
    
    glClearColor(0.0, 0.0, 0.0, 0.0);
    glClear(GL_COLOR_BUFFER_BIT);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glViewport(0, 0, w(), h()); 
    glOrtho(0,30,0,450,-1,1);
    
  }
 
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
  glOrtho(0,WINDOW_W,0,WINDOW_H,-1,1);
  
  glClearColor(207.0/255.0,205.0/255.0,126.0/255.0,0);
  glClear(GL_COLOR_BUFFER_BIT);

// We draw a rectangle corresponding to the slider

  glColor3f(0.75,0.75,0.9); 
  glRecti(0,minSlider,WINDOW_W,maxSlider);
  
// The following code is to imitate a FLTK slider 

  glLineWidth(1);
  glColor3f(0.2,0.2,0.2); 
  glBegin(GL_LINE_STRIP);
    glVertex2f(0,minSlider);
    glVertex2f(WINDOW_W-1,minSlider);
    glVertex2f(WINDOW_W-1,maxSlider);
  glEnd();
  
  glLineWidth(1);
  glColor3f(0.5,0.5,0.5); 
  glBegin(GL_LINE_STRIP);
    glVertex2f(1,minSlider+1);
    glVertex2f(WINDOW_W-2,minSlider+1);
    glVertex2f(WINDOW_W-2,maxSlider-1);
  glEnd();

  glLineWidth(2);
  glColor3f(0.87,0.87,0.87); 
  glBegin(GL_LINE_STRIP);
    glVertex2f(WINDOW_W-1,maxSlider);
    glVertex2f(1,maxSlider);
    glVertex2f(1,minSlider+1);
  glEnd();

  glLineWidth(2);
  glColor3f(0.1,0.1,0.1); 
  glBegin(GL_TRIANGLES);
    glVertex2f(WINDOW_W/4,maxSlider-20);
    glVertex2f(WINDOW_W/2,maxSlider-10);
    glVertex2f(3*WINDOW_W/4,maxSlider-20);

    glVertex2f(WINDOW_W/4,minSlider+20);
    glVertex2f(WINDOW_W/2,minSlider+10);
    glVertex2f(3*WINDOW_W/4,minSlider+20);
  glEnd();
  
  glBegin(GL_LINES);
    glVertex2f(4,minSlider+(maxSlider-minSlider)/2);
    glVertex2f(WINDOW_W-4,minSlider+(maxSlider-minSlider)/2);
    glVertex2f(4,minSlider+6*(maxSlider-minSlider)/10);
    glVertex2f(WINDOW_W-4,minSlider+6*(maxSlider-minSlider)/10);
    glVertex2f(4,minSlider+4*(maxSlider-minSlider)/10);
    glVertex2f(WINDOW_W-4,minSlider+4*(maxSlider-minSlider)/10);
  glEnd();
  
// Each time we redraw the slider we update the extrema

}

int  WindowCS::GetMin(){
  return minSlider;
}

int  WindowCS::GetMax(){
  return maxSlider;
}

void WindowCS::SetMin(int _min){
  minSlider = _min;
}

void WindowCS::SetMax(int _max){
  maxSlider = _max;
}

void WindowCS::SetGUI(ImMapGUI* _GUI){
  GUI = _GUI ;
}

int WindowCS::handle(int event) {
 
  switch(event) {
    
  case FL_PUSH: 
    
    int y_curs;
    y_curs=h()-Fl::event_y();
    
// To drag the Min bound
    if (approx(y_curs,minSlider*(h()/450.0)) || (y_curs>(minSlider*(h()/450.0)) && y_curs<(minSlider+20)*(h()/450.0))){
      dragMin=1;
      startDrag=y_curs;
      initMin=minSlider;
    }   
// To drag the Max bound
    if (approx(y_curs,maxSlider*(h()/450.0)) || ( y_curs<(maxSlider*(h()/450.0)) && y_curs>(maxSlider-20)*(h()/450.0))){
      dragMax=1;
      startDrag=y_curs;
      initMax=maxSlider;
    }
// To drag the slider
    if (y_curs>(minSlider+21)*(h()/450.0) && y_curs<(maxSlider-21)*(h()/450.0)){
      drag=1;
      startDrag=y_curs;
      initMax=maxSlider;
      initMin=minSlider;
    }
    
    
    return 1;
  case FL_DRAG:

    y_curs=h()-Fl::event_y();

    if (dragMin ){
      //minSlider=y_curs;
      minSlider=initMin+int((y_curs-startDrag)/(h()/450.0)+0.5);
      if (minSlider>(maxSlider-30)){ // to avoid overlaping
	minSlider=maxSlider-30;
      }
    }
    if (dragMax){
     // maxSlider=y_curs;
      maxSlider=initMax+int((y_curs-startDrag)/(h()/450.0)+0.5);
      if (maxSlider<(minSlider+30)){ // to avoid overlaping
	maxSlider=minSlider+30;
       }
    }

// The slider can't be drag anymore if one of the bound is equal to 
// the limit fot this given bound
    if (drag){
      
      diffDrag = initMax-initMin;
      
      if (maxSlider<(WINDOW_H-1)){
	minSlider=initMin+int((y_curs-startDrag)/(h()/450.0)+0.5);
      }
      if (minSlider>0){
	maxSlider=initMax+int((y_curs-startDrag)/(h()/450.0)+0.5);
      }
    }

// The bounds can't be dragged over their limits
    if (maxSlider>(WINDOW_H-1)) {
      maxSlider=WINDOW_H-1;
      if (drag) {
	minSlider=maxSlider-diffDrag;
      }
    }
    if (minSlider<0) {
      minSlider=0;
       if (drag) {
	maxSlider=minSlider+diffDrag;
       }
    } 

    redraw();
    GUI->histogramSliderCallback();
    return 1;
  case FL_RELEASE:
    dragMin=0;
    dragMax=0;
    drag=0;
    return 1;
  case FL_FOCUS :
   // cout << "Window 2D got focus" << endl;
  case FL_UNFOCUS :
   // cout << "Window 2D lost focus" << endl;
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
