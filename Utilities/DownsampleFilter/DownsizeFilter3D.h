/****************************************************************************
** File: DownsizeFilter3D.h												   **
** Description: Downsample 3D image							               **
** Author: Matthieu Jomier <mat-dev@jomier.com>						       **
** Version: 1.0															   **
**																		   **
** ----------------------------------------------------------------------- **
** History: 09/24/2003			| Main application v1.0					   **
**																		   **
*****************************************************************************/

#ifndef DOWNSIZEFILTER3D_H
#define DOWNSIZEFILTER3D_H

#include "Array3D.h"
#include "Vector3D.h"

class DownsizeFilter3D
{
public:
	DownsizeFilter3D();
	~DownsizeFilter3D();
	void SetInput(Array3D<float>& );
	Array3D<float>& GetOutput();
	void Update();
	void setSize(int);
	void setSize(int,int,int);
	Vector3D<int> GetNewSize();

private:	
	Array3D<float>* inarray;
	Array3D<float> outarray;
	Vector3D<int> newdim;
	int sizex,sizey,sizez;
};

#endif

