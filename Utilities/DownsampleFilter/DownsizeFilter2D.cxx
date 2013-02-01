/****************************************************************************
** File: Downsizefilter2D.cxx
** Description: Downsample 2D image
** Author: Matthieu Jomier <mat-dev@jomier.com>
** Version: 1.0
**
** -----------------------------------------------------------------------
** History: 09/24/2003                  | Main application v1.0
**
*****************************************************************************/

#include "DownsizeFilter2D.h"

DownsizeFilter2D::DownsizeFilter2D()
{
	sizex = sizey =2;
}

DownsizeFilter2D::~DownsizeFilter2D()
{
}


void DownsizeFilter2D::SetInput(Array2D<float>& _inarray)
{
	inarray = &_inarray;
}

Array2D<float>& DownsizeFilter2D::GetOutput()
{
	return outarray;
}

Vector2D<int> DownsizeFilter2D::GetNewSize()
{
	return newdim;
}

void DownsizeFilter2D::Update()
{
	newdim.set((inarray->getSizeX()/sizex),(inarray->getSizeY()/sizey));
	outarray.resize(newdim[0],newdim[1]);
        //	for (int z=0;z<(newdim[2]);z++)
		for (int y=0;y<(newdim[1]);y++)
			for (int x=0;x<(newdim[0]);x++)
				outarray(x,y) = inarray->get(x*sizex,y*sizey);
}


void DownsizeFilter2D::setSize(int _size)
{
	sizex = sizey =_size;
	newdim.set((inarray->getSizeX()/sizex),(inarray->getSizeY()/sizey));
}

void DownsizeFilter2D::setSize(int _sizex,int _sizey)
{
	sizex = _sizex;
	sizey = _sizey;
        //	sizez = _sizez;
	newdim.set((inarray->getSizeX()/sizex),(inarray->getSizeY()/sizey));
}


