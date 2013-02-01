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
// File BYUTable.cxx
//
//////////////////////////////////////////
#ifndef BYUTABLE_CXX
#define BYUTABLE_CXX

#include "BYUTable.h"
#include <iostream>
#include <stdio.h>
#include <FL/Fl_Window.H>
#include <FL/fl_draw.H>

BYUTable::BYUTable(int X, int Y, int W, int H, char* c)
: Fl_Table_Row(X,Y,W,H,c)
{
  _windowWidth = W;
  _windowHeight = H;
  
  _NbCols = 5;
  _NbRows = 0;
  
}

/////////////////
// constructor //
/////////////////

BYUTable::BYUTable(int X, int Y, int W, int H)
: Fl_Table_Row(X,Y,W,H)
{
  _windowWidth = W;
  _windowHeight = H;
  
  _NbCols = 5;
  _NbRows = 0;
  
}

////////////////
// destructor //
////////////////

BYUTable::~BYUTable()
{
}

//////////////////
// clearTable() //
//////////////////

void BYUTable::clearTable()
{
  clear();
  _objectNameList.clear();
  _objectVisibilityList.clear();
  _objectColorList.clear();
  _objectAspectList.clear();
  _objectOpacityList.clear();
  _objectImageIndex.clear();
  _objectImAnaIndex.clear();
  _NbRows = 0;
}

//////////////////////
// addObjectToTable //
//////////////////////

void BYUTable::addObjectToTable(const unsigned int& imageIndex, 
                                const unsigned int& imAnaIndex,
                                const std::string& objectName, 
                                const bool& viewObject, 
                                const Vector3D<double>& objectColor, 
                                const unsigned int& objectAspect, 
                                const float& objectOpacity)
{
  //save the imageIndex and the imAna for each row
  _objectImageIndex.push_back(imageIndex);
  _objectImAnaIndex.push_back(imAnaIndex);
  
  //resize the table
  _NbRows++;
  rows(_NbRows);
  cols(_NbCols);
  
  static char s[30];
  begin();		// start adding widgets to group
  {
    for ( unsigned int c = 0; c<_NbCols; c++ )
    {
      int X,Y,W,H;
      find_cell(CONTEXT_TABLE, _NbRows, c, X, Y, W, H);
      
      if (c == 0)
      {
        _objectNameList.push_back(objectName);
      }
      if (c == 1)
      {
        _objectVisibilityList.push_back(viewObject ? "yes" : "no");
      }
      if (c == 2)
      {
        _objectColorList.push_back(objectColor);
      }
      if (c == 3)
      {
        switch(objectAspect)
        {
        case 0 :
          {
            _objectAspectList.push_back("surface");
            break;
          }
        case 1 :
          {
            _objectAspectList.push_back("wireframe");
            break;
          }
        case 2 :
          {
            _objectAspectList.push_back("contours");
            break;
          }
        }
      }
      if (c == 4)
      {
        sprintf(s, "%f",objectOpacity);
        _objectOpacityList.push_back(s);
      }
    }
  }
  end();
}

///////////////
// draw_cell //
///////////////

void BYUTable::draw_cell(TableContext context, int R, int C, int X, int Y, int W, int H)
{
  switch ( context )
  {
    R = callback_row();
    C = callback_col();
    
    
    static char s[40];
    sprintf(s, "%d/%d", R, C);
    
  case CONTEXT_STARTPAGE:
    fl_color(FL_GRAY);
    fl_font(FL_HELVETICA, 11);
    // column 
    col_width(0,130);
    col_width(1,30);
    col_width(2,35);
    col_width(3,60);
    col_width(4,35);
    col_header(1);
    col_header_height(25);
    col_resize(0);
    //line
    row_height_all(25);
    row_header(0);
    row_resize(0);
    return;
    
    
  case CONTEXT_RC_RESIZE:
    {
      int X, Y, W, H;
      int index = 0;
      for ( int r = 0; r<rows(); r++ )
      {
        for ( int c = 0; c<cols(); c++ )
        {
          if ( index >= children() ) break;
          
          if (c != 0)
          {
            find_cell(CONTEXT_TABLE, r, c, X, Y, W, H);
            child(index++)->resize(X,Y,W,H);
          }
          
        }
      }
      init_sizes();			// tell group children resized
      return;
    }
    
  case CONTEXT_ENDPAGE:
    {
      return;
    }
    
  case CONTEXT_ROW_HEADER:
  case CONTEXT_COL_HEADER:
    fl_font(FL_HELVETICA, 12);
    
    if (C == 0)
    {
      fl_push_clip(X, Y, W, H);
      fl_draw_box(FL_THIN_UP_BOX, X, Y, W, H, color());
      fl_color(FL_BLACK);
      fl_draw("Object Name", X, Y, W, H, FL_ALIGN_CENTER);
      fl_pop_clip();
    }
    if (C == 1)
    {
      fl_push_clip(X, Y, W, H);
      fl_draw_box(FL_THIN_UP_BOX, X, Y, W, H, color());
      fl_color(FL_BLACK);
      fl_draw("View", X, Y, W, H, FL_ALIGN_CENTER);
      fl_pop_clip();
    }
    if (C == 2)
    {
      fl_push_clip(X, Y, W, H);
      fl_draw_box(FL_THIN_UP_BOX, X, Y, W, H, color());
      fl_color(FL_BLACK);
      fl_draw("Color", X, Y, W, H, FL_ALIGN_CENTER);
      fl_pop_clip();
    }
    if (C == 3)
    {
      fl_push_clip(X, Y, W, H);
      fl_draw_box(FL_THIN_UP_BOX, X, Y, W, H, color());
      fl_color(FL_BLACK);
      fl_draw("Aspect", X, Y, W, H, FL_ALIGN_CENTER);
      fl_pop_clip();
    }
    if (C == 4)
    {
      fl_push_clip(X, Y, W, H);
      fl_draw_box(FL_THIN_UP_BOX, X, Y, W, H, color());
      fl_color(FL_BLACK);
      fl_draw("alpha", X, Y, W, H, FL_ALIGN_CENTER);
      fl_pop_clip();
    }
    return;
    
  case CONTEXT_CELL:
    {
      selection_color(FL_YELLOW);
      if (C == 0)//name of the imAna
      {
        fl_push_clip(X, Y, W, H);
        {
          // BG COLOR
          fl_color( row_selected(R) ? selection_color() : FL_WHITE);
          fl_rectf(X, Y, W, H);
          
          // TEXT
          fl_color(FL_BLACK);
          fl_font(FL_HELVETICA, 11);
          fl_draw(_objectNameList[R].c_str(), X+2, Y, W, H, FL_ALIGN_LEFT );
          
          // BORDER
          fl_color(FL_LIGHT2); 
          fl_rect(X, Y, W, H);
        }
        fl_pop_clip();
        
        return;
      }
      if (C == 1)//visibility
      {
        fl_push_clip(X, Y, W, H);
        {
          // BG COLOR
          fl_color( row_selected(R) ? selection_color() : FL_WHITE);
          fl_rectf(X, Y, W, H);
          
          // TEXT
          fl_color(FL_BLACK);
          fl_font(FL_HELVETICA, 11);
          fl_draw(_objectVisibilityList[R].c_str(), X+2, Y, W, H, FL_ALIGN_LEFT );
          
          // BORDER
          fl_color(FL_LIGHT2); 
          fl_rect(X, Y, W, H);
        }
        fl_pop_clip();
        
        return;
      }
      if (C == 2)//color
      {
        fl_push_clip(X, Y, W, H);
        {
          // BG COLOR
          fl_color( fl_rgb_color((uchar)(255*_objectColorList[R].x),(uchar)(255*_objectColorList[R].y), (uchar)(255*_objectColorList[R].z)));
          fl_rectf(X, Y, W, H);
          
          // BORDER
          fl_color(FL_LIGHT2); 
          fl_rect(X, Y, W, H);
        }
        fl_pop_clip();
        
        return;
      }
      if (C == 3)//aspect
      {
        fl_push_clip(X, Y, W, H);
        {
          // BG COLOR
          fl_color( row_selected(R) ? selection_color() : FL_WHITE);
          fl_rectf(X, Y, W, H);
          
          // TEXT
          fl_color(FL_BLACK);
          fl_font(FL_HELVETICA, 11);
          fl_draw(_objectAspectList[R].c_str(), X+2, Y, W, H, FL_ALIGN_LEFT );
          
          // BORDER
          fl_color(FL_LIGHT2); 
          fl_rect(X, Y, W, H);
        }
        fl_pop_clip();
        
        return;
      }
      if (C == 4)//opacity
      {
        fl_push_clip(X, Y, W, H);
        {
          // BG COLOR
          fl_color( row_selected(R) ? selection_color() : FL_WHITE);
          fl_rectf(X, Y, W, H);
          
          // TEXT
          fl_color(FL_BLACK);
          fl_font(FL_HELVETICA, 11);
          fl_draw(_objectOpacityList[R].c_str(), X+2, Y, W, H, FL_ALIGN_LEFT );
          
          // BORDER
          fl_color(FL_LIGHT2); 
          fl_rect(X, Y, W, H);
        }
        fl_pop_clip();
        
        return;
      }
  }
  
  default:
    return;
    }
    
}

#endif
