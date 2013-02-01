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

#include "DicomPreview.h"

int WINDOW_PIXEL_SIZE = 100;

/////////////////
// constructor //
/////////////////

DicomPreview
	::DicomPreview(int X, int Y, int W, int H, const char *L)
		:Fl_Gl_Window(X,Y,W,H,L)
{

 Xmin=Ymin=Zmin=-100;// square coord
 Xmax=Ymax=Zmax=100;
 
}

/////////////////
// SetFilename //
/////////////////

void DicomPreview :: SetFilename(char* FileName)
{
	_FileName=FileName;	
}

/////////////////
// LoadTexture //
/////////////////

void DicomPreview :: LoadTexture() 
{ 
	previewIO = new ImageIO;
	previewIO->LoadThisImage(_FileName, preview3D,ImageIO::dicom);
	
	int Xsize = preview3D.getSizeX();
	int Ysize = preview3D.getSizeY();

	ratio = Xsize/Ysize;

	Text.resize(4,Ysize,Xsize);
	
		for(int i = 0 ; i<Xsize ; i++ ){
			for(int j = 0 ; j<Ysize ; j++ ){
		unsigned char tmpval = (unsigned char)(preview3D.get(i,j,0)*255/4096);
		Text.set(0,j,i,tmpval);//R
		Text.set(1,j,i,tmpval);//G
		Text.set(2,j,i,tmpval);//B
		Text.set(3,j,i,tmpval);//A
			}
		}

glGenTextures (1, &texture[0]); 
glBindTexture (GL_TEXTURE_2D, texture[0]); 
glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER,GL_NEAREST); 
glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,GL_NEAREST); 
glTexImage2D(GL_TEXTURE_2D, 0, 3, Xsize, Ysize, 0, GL_RGBA, GL_UNSIGNED_BYTE, Text.getDataPointer()); 
delete previewIO;
}

//////////
// draw //
//////////

void DicomPreview::draw() {
  if (!valid()) {
	glClearColor(0.0, 0.0, 0.0, 0.0);
    glClear(GL_COLOR_BUFFER_BIT); 
    glMatrixMode(GL_MODELVIEW); 
    glLoadIdentity(); 
    glViewport(0, 0, w(), static_cast<GLsizei>(ratio*w())); 
    glOrtho(-WINDOW_PIXEL_SIZE, WINDOW_PIXEL_SIZE,-WINDOW_PIXEL_SIZE,WINDOW_PIXEL_SIZE,WINDOW_PIXEL_SIZE,-WINDOW_PIXEL_SIZE); 
	glEnable (GL_TEXTURE_2D);
  }
  	glClearColor(0.0, 0.0, 0.0, 0.0);
	glClear (GL_COLOR_BUFFER_BIT);
	LoadTexture();
	glMatrixMode(GL_MODELVIEW); 
    glLoadIdentity(); 
    glViewport(0, 0, w(), static_cast<GLsizei>(ratio*w())); 
    glOrtho(-WINDOW_PIXEL_SIZE, WINDOW_PIXEL_SIZE,-WINDOW_PIXEL_SIZE,WINDOW_PIXEL_SIZE,WINDOW_PIXEL_SIZE,-WINDOW_PIXEL_SIZE); 
	
	// display the texture in a square
	glEnable (GL_TEXTURE_2D); 
	glBindTexture (GL_TEXTURE_2D, texture[0]);
	
	glBegin(GL_QUADS); 
		glTexCoord2i(1,0);glVertex2i(Xmin,Ymin); 
		glTexCoord2i(0,0);glVertex2i(Xmin,Ymax); 
		glTexCoord2i(0,1);glVertex2i(Xmax,Ymax); 
		glTexCoord2i(1,1);glVertex2i(Xmax,Ymin); 
	glEnd();
	
	glDisable (GL_TEXTURE_2D); 
}







    


