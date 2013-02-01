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

#ifndef DICOMPREVIEW_H
#define DICOMPREVIEW_H

#include <FL/gl.h>
#include <FL/Fl_Gl_Window.H>

#include <ImageIO.h>

class DicomPreview : public Fl_Gl_Window{

  void draw();
  void LoadTexture();
  
public:

  DicomPreview(int X, int Y, int W, int H, const char *L);
  void SetFilename(char *);

private :
	
	Image<float> preview3D;
	ImageIO *previewIO;

	int Xmin,Ymin,Xmax,Ymax,Zmin,Zmax;
	
	unsigned texture[1];
	Array3D<unsigned char> Text;

	char* _FileName;

	float ratio;
	
	
};
#endif
