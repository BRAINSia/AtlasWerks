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

#ifndef BYUTABLE_H
#define  BYUTABLE_H

//////////////////////////////////////////////////////////////////////
//
// File: BYUTable.h
//
// The class allows you to create a table which contains a list 
// of anastructs and their properties and by selection changed its  
//
// D. Prigent (04/26/2004)
//////////////////////////////////////////////////////////////////////

// 'identifier' : identifier was truncated to 'number' characters in the
// debug information
#ifdef WIN32
#pragma warning ( disable : 4786 )
#endif

#include <FL/Fl.H>
#include "Fl_Table.H"
#include "Fl_Table_Row.H"
#include <Vector3D.h>
#include <vector>
#include <stdio.h>
#include <string.h>

///////////////
// BYUTable  //
///////////////

// each anastruct is push in the table as a row.
// each row define the name, the visibility, the color,
// the aspect and the opacity of an anastruct : imAna 

class BYUTable: public Fl_Table_Row {
  
public :
  
  BYUTable(int X, int Y, int W, int H, char* c);
  BYUTable(int X, int Y, int W, int H);
  
  ~BYUTable();
  void addObjectToTable(const unsigned int& imageIndex, 
                        const unsigned int& imAnaIndex, 
                        const std::string& objectName, 
                        const bool& viewObject, 
                        const Vector3D<double>& objectColor, 
                        const unsigned int& objectAspect = 0, 
                        const float& objectOpacity = 1);
  void clearTable();
  
  unsigned int getNbRows(){return _NbRows;};
  int getImageIndex(unsigned int& rowIndex){return _objectImageIndex[rowIndex];};
  int getImAnaIndex(unsigned int& rowIndex){return _objectImAnaIndex[rowIndex];};
  
private :
  
  void draw_cell(TableContext context,int R=0, int C=0, int X=0, int Y=0, int W=0, int H=0);
  
  unsigned int _windowWidth, _windowHeight;
  
  unsigned int _NbRows, _NbCols;  
  
  std::vector< std::string > _objectNameList;
  std::vector< std::string > _objectVisibilityList;
  std::vector< Vector3D<double> > _objectColorList;
  std::vector< std::string > _objectAspectList;
  std::vector< std::string > _objectOpacityList;
  
  //keep correspondance between row and ImageIndex or ImAnaIndex
  std::vector<unsigned int>   _objectImageIndex;
  std::vector<unsigned int>   _objectImAnaIndex;
  
};// BYUTable class

#endif

