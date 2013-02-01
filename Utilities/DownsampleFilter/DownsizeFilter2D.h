/****************************************************************************
** File: DownsizeFilter2D.h												   **
** Description: Downsample 2D image							               **
** Author: Matthieu Jomier <mat-dev@jomier.com>						       **
** Version: 1.0															   **
**																		   **
** ----------------------------------------------------------------------- **
** History: 09/24/2003			| Main application v1.0					   **
**																		   **
*****************************************************************************/

#ifndef DOWNSIZEFILTER2D_H
#define DOWNSIZEFILTER2D_H

#include "Array2D.h"
#include "Vector2D.h"

class DownsizeFilter2D
{
public:
	DownsizeFilter2D();
	~DownsizeFilter2D();
	void SetInput(Array2D<float>& );
	Array2D<float>& GetOutput();
	void Update();
	void setSize(int);
	void setSize(int,int);
	Vector2D<int> GetNewSize();

private:	
	Array2D<float>* inarray;
	Array2D<float> outarray;
	Vector2D<int> newdim;
	int sizex,sizey;
};

#endif

