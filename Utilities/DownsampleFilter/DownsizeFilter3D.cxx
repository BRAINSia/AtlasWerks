/****************************************************************************
** File: Downsizefilter3D.cxx
** Description: Downsample 3D image
** Author: Matthieu Jomier <mat-dev@jomier.com>
** Version: 1.0
**
** -----------------------------------------------------------------------
** History: 09/24/2003                  | Main application v1.0
**
*****************************************************************************/

#include "DownsizeFilter3D.h"

DownsizeFilter3D::DownsizeFilter3D()
{
	sizex = sizey = sizez =2;
}

DownsizeFilter3D::~DownsizeFilter3D()
{
}


void DownsizeFilter3D::SetInput(Array3D<float>& _inarray)
{
	inarray = &_inarray;
}

Array3D<float>& DownsizeFilter3D::GetOutput()
{
	return outarray;
}

Vector3D<int> DownsizeFilter3D::GetNewSize()
{
	return newdim;
}

void DownsizeFilter3D::Update()
{
	newdim.set((inarray->getSizeX()/sizex),(inarray->getSizeY()/sizey),(inarray->getSizeZ()/sizez));
	outarray.resize(newdim[0],newdim[1],newdim[2]);
	for (int z=0;z<(newdim[2]);z++)
		for (int y=0;y<(newdim[1]);y++)
			for (int x=0;x<(newdim[0]);x++)
				outarray(x,y,z) = inarray->get(x*sizex,y*sizey,z*sizez);
}


void DownsizeFilter3D::setSize(int _size)
{
	sizex = sizey = sizez =_size;
	newdim.set((inarray->getSizeX()/sizex),(inarray->getSizeY()/sizey),(inarray->getSizeZ()/sizez));
}

void DownsizeFilter3D::setSize(int _sizex,int _sizey,int _sizez)
{
	sizex = _sizex;
	sizey = _sizey;
	sizez = _sizez;
	newdim.set((inarray->getSizeX()/sizex),(inarray->getSizeY()/sizey),(inarray->getSizeZ()/sizez));
}


