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


#include "vtkBYUReader.h"
#include "vtkPolyDataMapper.h"
#include "vtkRenderWindow.h"
#include "vtkCamera.h"
#include "vtkActor.h"
#include "vtkRenderer.h"
#include "vtkProperty.h"
#include "vtkPoints.h"
#include "vtkPolyLine.h"
#include "vtkUnstructuredGrid.h"
#include "vtkPointData.h"
#include "vtkPolyData.h"
#include "vtkCellArray.h"
#include "vtkFloatArray.h"
#include "vtkWindowToImageFilter.h"
#include "vtkPNGWriter.h"
#include "vtkInteractorStyleTrackballCamera.h"
#include <FL/Fl.H>
#include "vtkRenderWindowInteractor.h"

#include <iostream>
#include <stdexcept>
#include <cmath>

#include <StringUtils.h>

#include "Contour.h"
#include "SurfaceViewWindow.h"

/////////////////
// constructor //
/////////////////

SurfaceViewWindow
::SurfaceViewWindow(int x, int y, int w, int h, const char* l) :

Fl_VTK_Window(x, y, w, h, l)

{
  _verbose = true;
  
  
  //
  // set up the rendering window (vtk and fltk stuff)
  //
  
  _renderer = GetDefaultRenderer();
  _renderer->SetBackground(0, 0, 0);
  
  _renderWindow = vtkRenderWindow::New();
  _renderWindow->AddRenderer(_renderer);
  SetRenderWindow(_renderWindow);
  
  vtkInteractorStyleTrackballCamera *interactorStyle = 
    vtkInteractorStyleTrackballCamera::New();
  GetInteractor()->SetInteractorStyle(interactorStyle);  
  
  
  _viewSurfaces = true;
  _viewAnastructs = true;
  _viewROI = true;
  _ROIcreated = false;
  
  _ROIopacity = 0.25;
}


////////////////
// addSurface //
////////////////

int 
SurfaceViewWindow
::addSurface(int imageIndex,
             int imAnaIndex,
             const Surface& surface,
             const bool& visible,
             const double& r,
             const double& g,
             const double& b,
             const SurfaceRepresentationType& representation,
             const double& opacity)
{
  //
  // vtk pipeline
  // Surface -> vtkPolyDataMapper -> vtkActor
  //
  
  //
  // this can and should be done a lot faster by reserving the right
  // amount of memory for the lists instead of pushing back all the
  // time
  //
  
  // leave here for vcc compatibility
  unsigned int i;

  // We'll create the building blocks of polydata including data attributes.
  vtkPolyData   *surfacePolyData = vtkPolyData::New();
  vtkPoints     *points          = vtkPoints::New();
  vtkCellArray  *polys           = vtkCellArray::New();
  //vtkFloatArray *scalars         = vtkFloatArray::New();
  
  // Load the point, cell, and data attributes.
  vtkIdType tmpFacet[3];
  for (i = 0; i < surface.numVertices(); i++) 
  {
    points->InsertPoint(i, &surface.vertices[i].x);
  }
  
  for (i = 0; i < surface.numFacets(); i++) 
  {
    tmpFacet[0] = surface.facets[i][0];
    tmpFacet[1] = surface.facets[i][1];
    tmpFacet[2] = surface.facets[i][2];
    polys->InsertNextCell(3, tmpFacet);
  }
  
  // what does this do??
  //for (i = 0; i < surface.numVert(); i++) 
  //  {
  //    scalars->InsertTuple1(i,0);
  //  }
  
  // We now assign the pieces to the vtkPolyData.
  surfacePolyData->SetPoints(points);
  points->Delete();
  surfacePolyData->SetPolys(polys);
  polys->Delete();
  //surfacePolyData->GetPointData()->SetScalars(scalars);
  //scalars->Delete();
  
  vtkPolyDataMapper *surfaceMapper = vtkPolyDataMapper::New();
  surfaceMapper->SetInput(surfacePolyData);
  //surfaceMapper->ScalarVisibilityOff();
  vtkActor *surfaceActor = vtkActor::New();
  surfaceActor->SetMapper(surfaceMapper);
  surfaceActor->GetProperty()->SetColor(r, g, b);
  surfaceActor->GetProperty()->SetAmbient(0.3);  
  surfaceActor->GetProperty()->SetDiffuse(0.6);
  surfaceActor->GetProperty()->SetSpecular(0.4);
  surfaceActor->GetProperty()->SetSpecularPower(20);
  surfaceActor->GetProperty()->SetOpacity(opacity);
  _renderer->AddActor(surfaceActor);
  _surfaceVisible.push_back(visible);
  _representation.push_back(representation);
  index newSurface;
  newSurface.ImageIndex = imageIndex;
  newSurface.ImAnaIndex = imAnaIndex;
  _surfaceIndex.push_back(newSurface);
  //
  // keep track of this surface so we can manipulate its properties
  //
  _surfaces.push_back(surfaceActor);
  this->setSurfaceRepresentation(_surfaces.size() - 1,representation);
  return _surfaces.size() - 1;
}

////////////////
// addSurface //
////////////////

int 
SurfaceViewWindow
::addSurface(int imageIndex,
             int imAnaIndex,
             const char *const filename,
             const bool& visible,
             const double& r,
             const double& g,
             const double& b,
             const SurfaceRepresentationType& representation,
             const double& opacity)
{
  if (_verbose)
  {
    std::cerr << "adding surface: " << filename << std::endl;
  }
  
  //
  // vtk pipeline
  // file -> vtkBYUReader -> vtkPolyDataMapper -> vtkActor
  //
  vtkBYUReader *byuReader = vtkBYUReader::New();
  byuReader->SetGeometryFileName(filename);
  vtkPolyDataMapper *surfaceMapper = vtkPolyDataMapper::New();
  surfaceMapper->SetInput(byuReader->GetOutput());
  vtkActor *surfaceActor = vtkActor::New();
  surfaceActor->SetMapper(surfaceMapper);
  surfaceActor->GetProperty()->SetColor(r, g, b);
  surfaceActor->GetProperty()->SetAmbient(0.3);  
  surfaceActor->GetProperty()->SetDiffuse(0.6);
  surfaceActor->GetProperty()->SetSpecular(0.4);
  surfaceActor->GetProperty()->SetSpecularPower(20);
  surfaceActor->GetProperty()->SetOpacity(opacity);
  _renderer->AddActor(surfaceActor);
  _surfaceVisible.push_back(visible);
  _representation.push_back(representation);
  index newSurface;
  newSurface.ImageIndex = imageIndex;
  newSurface.ImAnaIndex = imAnaIndex;
  _surfaceIndex.push_back(newSurface);
  //
  // keep track of this surface so we can manipulate its properties
  //
  _surfaces.push_back(surfaceActor);
  this->setSurfaceRepresentation(_surfaces.size() - 1,representation);
  
  if (_verbose)
  {
    std::cerr << "added surface [" << _surfaces.size() 
      << "]: " << filename << std::endl;      
  }
  return _surfaces.size() - 1;
}

//////////////////
// addAnastruct //
//////////////////

int 
SurfaceViewWindow
::addAnastruct(const Anastruct& anastruct,
               const double& r,
               const double& g,
               const double& b)
{
  //
  // vtkPoints + set of vtkPolyLines get put into vtkUnstructuredGrid
  // then pipeline becomes
  // vtkUnstructuredGrid -> vtkDataSetMapper -> vtkActor
  // 
  
  //
  // create unstructured grid to hold all polylines
  //
  vtkUnstructuredGrid *polyLineGrid = vtkUnstructuredGrid::New();
  polyLineGrid->Allocate();
  
  // points for all contours
  vtkPoints *polyLinePoints = vtkPoints::New();
  
  //
  // add a polyline for each contour
  //
  for (Anastruct::ConstContourIter contourIter = anastruct.contours.begin();
       contourIter != anastruct.contours.end();
       ++contourIter)
  {
    //
    // create polyline
    //
    vtkPolyLine *polyLine = vtkPolyLine::New();
    
    int numPointsBeforeContour = polyLinePoints->GetNumberOfPoints();
    for (Contour::ConstVertexIter vertIter = contourIter->vertices.begin();
         vertIter != contourIter->vertices.end();
         ++vertIter)
     {
       int pointId = polyLinePoints->InsertNextPoint(vertIter->x,
                                                     vertIter->y,
                                                     vertIter->z);
       polyLine->GetPointIds()->InsertNextId(pointId);
     }
     
     // add again first point to connect loop (connect first and last point)
     if (contourIter->vertices.size() != 0)
     {
       polyLine->GetPointIds()->InsertNextId(numPointsBeforeContour);
     }
     
     //
     // add polyline to grid
     //
     polyLineGrid->InsertNextCell(polyLine->GetCellType(),
                                  polyLine->GetPointIds());      
  }
  
  // 
  // set points for grid
  //
  polyLineGrid->SetPoints(polyLinePoints);
  
  //
  // map grid to actor
  // 
  vtkDataSetMapper *anastructMapper = vtkDataSetMapper::New();
  anastructMapper->SetInput(polyLineGrid);
  vtkActor *anastructActor = vtkActor::New();
  anastructActor->SetMapper(anastructMapper);
  anastructActor->GetProperty()->SetColor(r, g, b);
  anastructActor->GetProperty()->SetAmbient(0.3);  
  anastructActor->GetProperty()->SetDiffuse(0.6);
  anastructActor->GetProperty()->SetSpecular(0.4);
  anastructActor->GetProperty()->SetSpecularPower(20);
  _renderer->AddActor(anastructActor);
  _anastructVisible.push_back(false);
  //
  // keep track of this actor so we can manipulate its properties
  //
  _anastructs.push_back(anastructActor);
  
  return _anastructs.size() - 1;
}

///////////////
// createROI //
///////////////

void
SurfaceViewWindow
::createROI()
{ 
  if (_ROIcreated) return;
  _roiGeometry = vtkCubeSource::New();
  _roiMapper = vtkDataSetMapper::New();
  _roiMapper->SetInput(_roiGeometry->GetOutput());
  _roi = vtkActor::New();
  _roi->SetMapper(_roiMapper);
  _renderer->AddActor(_roi);
  _ROIcreated = true;
  setViewROI(true);
}


///////////////
// updateROI //
///////////////

void 
SurfaceViewWindow
::updateROI(const Vector3D<double>& start,const Vector3D<double>& stop)
{
  double XLength = stop.x - start.x;
  double YLength = stop.y - start.y;
  double ZLength = stop.z - start.z;
  
  double XCenter = start.x + XLength/2;
  double YCenter = start.y + YLength/2;
  double ZCenter = start.z + ZLength/2;
  
  _roiGeometry->SetXLength(XLength);
  _roiGeometry->SetYLength(YLength);
  _roiGeometry->SetZLength(ZLength);
  _roiGeometry->SetCenter(XCenter, YCenter, ZCenter);
  _roi->GetProperty()->SetOpacity(_ROIopacity);
  _roi->GetProperty()->SetColor(0, 0, 1);
  _renderer->AddActor(_roi);
}

//////////////////
// clearSurface //
//////////////////

void
SurfaceViewWindow
::clearSurfaces()
{
  for (unsigned int surfaceIndex = 0; 
  surfaceIndex < _surfaces.size(); 
  surfaceIndex++)
  {
    _renderer->RemoveActor(_surfaces[surfaceIndex]);      
  }
  _surfaces.clear();
  _surfaceIndex.clear();
  _surfaceVisible.clear();
  _representation.clear();
}



//////////////////
// clearSurface //
//////////////////

void
SurfaceViewWindow
::clearSurface(unsigned int surfaceIndex)
{
  _renderer->RemoveActor(_surfaces[surfaceIndex]);
  _surfaces.erase(_surfaces.begin()+surfaceIndex);
  _surfaceIndex.erase(_surfaceIndex.begin()+surfaceIndex);
  _surfaceVisible.erase(_surfaceVisible.begin()+surfaceIndex);
  _representation.erase(_representation.begin()+surfaceIndex);
}

/////////////////////
// clearAnastructs //
/////////////////////

void
SurfaceViewWindow
::clearAnastructs()
{
  for (unsigned int anastructIndex = 0; 
  anastructIndex < _anastructs.size(); 
  anastructIndex++)
  {
    _renderer->RemoveActor(_anastructs[anastructIndex]);      
  }
  _anastructs.clear();
  _anastructVisible.clear();
}

//////////////
// clearROI //
//////////////

void
SurfaceViewWindow
::clearROI()
{
  _renderer->RemoveActor(_roi);      
}

////////////////////
// setViewSurface //
////////////////////

void 
SurfaceViewWindow
::setViewSurfaces(const bool& viewSurfaces)
{
  _viewSurfaces = viewSurfaces;
}

//////////////////////
// setViewAnastruct //
//////////////////////

void 
SurfaceViewWindow
::setViewAnastructs(const bool& viewAnastructs)
{
  _viewAnastructs = viewAnastructs;
}

////////////////
// setViewROI //
////////////////

void 
SurfaceViewWindow
::setViewROI(const bool& viewROI)
{
  _viewROI = viewROI;
}

//////////////////
// _updateProps //
//////////////////

void
SurfaceViewWindow
::_updateProps()
{
  _renderer->RemoveAllViewProps();
  
  if (_viewSurfaces)
  {
    for (unsigned int surfaceIndex = 0; 
	   surfaceIndex < _surfaces.size(); 
     surfaceIndex++)
     {
       if (_surfaceVisible[surfaceIndex])
       {
       _renderer->AddActor(_surfaces[surfaceIndex]);      
       }
     }
  }
  if (_viewAnastructs)
  {
    for (unsigned int anastructIndex = 0; 
	   anastructIndex < _anastructs.size(); 
     anastructIndex++)
     {
       if (_anastructVisible[anastructIndex])
       {
       _renderer->AddActor(_anastructs[anastructIndex]);    
       }
     }
  }
  if ((_viewROI)&&(_ROIcreated))
  {
    _renderer->AddActor(_roi);      
  }
  redraw();
}

////////////////////////
// setBackgroundColor //
////////////////////////

void 
SurfaceViewWindow
::setBackgroundColor(const double& r,
                     const double& g,
                     const double& b)
{
  _renderer->SetBackground(r, g, b);
}

void 
SurfaceViewWindow
::getBackgroundColor(double& r,
                     double& g,
                     double& b)
{
#if VTK_MAJOR_VERSION > 4
  double rgb[3];
#else
  float rgb[3];
#endif
  _renderer->GetBackground(rgb);
  r = rgb[0];
  g = rgb[1];
  b = rgb[2];
}

/////////////////////
// setSurfaceColor //
/////////////////////

void 
SurfaceViewWindow
::setSurfaceColor(int surfaceIndex,
                  const double& r,
                  const double& g,
                  const double& b)
{
  if (surfaceIndex < 0 || surfaceIndex >= (int)_surfaces.size())
  {
    throw std::invalid_argument("SurfaceViewWindow::setSurfaceColor: invalid surface index");
  }
  _surfaces[surfaceIndex]->GetProperty()->SetColor(r, g, b);
}

///////////////////////
// setAnastructColor //
///////////////////////

void 
SurfaceViewWindow
::setAnastructColor(int anastructIndex,
                    const double& r,
                    const double& g,
                    const double& b)
{
  if (anastructIndex < 0 || anastructIndex >= (int)_anastructs.size())
  {
    throw std::invalid_argument("anastructViewWindow::setAnastructColor: invalid anastruct index");
  }
  _anastructs[anastructIndex]->GetProperty()->SetColor(r, g, b);
}

///////////////////////
// setSurfaceOpacity //
///////////////////////

void 
SurfaceViewWindow
::setSurfaceOpacity(int surfaceIndex,
                    const double& opacity) 
{
  if (surfaceIndex < 0 || surfaceIndex >= (int)_surfaces.size())
  {
    throw std::invalid_argument("SurfaceViewWindow::setSurfaceOpacity: invalid surface index");
  }
  _surfaces[surfaceIndex]->GetProperty()->SetOpacity(opacity);
}

///////////////////
// setROIOpacity //
///////////////////

void 
SurfaceViewWindow
::setROIOpacity(const double& opacity) 
{
  _ROIopacity = opacity;
  _roi->GetProperty()->SetOpacity(_ROIopacity);
  _updateProps();
  updateDisplay();
}

//////////////////////////////
// setSurfaceRepresentation //
//////////////////////////////

void 
SurfaceViewWindow
::setSurfaceRepresentation(int surfaceIndex,
                           const SurfaceRepresentationType& representation)
{
  if (surfaceIndex < 0 || surfaceIndex >= (int)_surfaces.size())
  {
    throw std::invalid_argument("SurfaceViewWindow::setSurfaceRepresentation: invalid surface index");
  }
  _representation[surfaceIndex] = representation;
  switch(representation)
  {
  case surface_representation:
    _surfaces[surfaceIndex]->GetProperty()->SetRepresentationToSurface();
    if (surfaceIndex >= 0 && surfaceIndex < (int)_anastructs.size())
    {
    _anastructVisible[surfaceIndex]=false;
    _surfaceVisible[surfaceIndex]=true;
    _updateProps();
    }
    break;
  case wireframe_representation:
    _surfaces[surfaceIndex]->GetProperty()->SetRepresentationToWireframe();
     if (surfaceIndex >= 0 && surfaceIndex < (int)_anastructs.size())
    {
    _anastructVisible[surfaceIndex]=false;
    _surfaceVisible[surfaceIndex]=true;
    _updateProps();
    }
    break;
  case contours_representation:
    {
    _surfaces[surfaceIndex]->GetProperty()->SetRepresentationToPoints();
    if (surfaceIndex >= 0 && surfaceIndex < (int)_anastructs.size())
    {
    _anastructVisible[surfaceIndex]=true;
    _surfaceVisible[surfaceIndex]=false;
    _updateProps();
    } 
    }
    break;
  default:
    throw std::invalid_argument("SurfaceViewWindow::setSurfaceRepresentation: invalid representation");
  }
}

/////////////////////////
// setSurfaceLineWidth //
/////////////////////////

void 
SurfaceViewWindow
::setSurfaceLineWidth(int surfaceIndex,
                      const double& lineWidth) 
{
  if (surfaceIndex < 0 || surfaceIndex >= (int)_surfaces.size())
  {
    throw std::invalid_argument("SurfaceViewWindow::setSurfaceLineWidth: invalid surface index");
  }
  _surfaces[surfaceIndex]->GetProperty()->SetLineWidth(lineWidth);
}

/////////////////////////
// setSurfacePointSize //
/////////////////////////

void 
SurfaceViewWindow
::setSurfacePointSize(int surfaceIndex,
                      const double& pointSize) 
{
  if (surfaceIndex < 0 || surfaceIndex >= (int)_surfaces.size())
  {
    throw std::invalid_argument("SurfaceViewWindow::setSurfacePointSize: invalid surface index");
  }
  _surfaces[surfaceIndex]->GetProperty()->SetPointSize(pointSize);
}

//////////////////////////
// setSurfaceVisibility //
//////////////////////////

void SurfaceViewWindow
::setSurfaceVisibility(int surfaceIndex,
                       const bool& visible)
{
  if (surfaceIndex < 0 || surfaceIndex >= (int)_surfaces.size())
  {
    throw std::invalid_argument("SurfaceViewWindow::setSurfaceVisibility: invalid surface index");
  }
  if (_representation[surfaceIndex] == surface_representation)
  {
  _surfaceVisible[surfaceIndex]=visible;
  _anastructVisible[surfaceIndex]=false;
  }
  else if(_representation[surfaceIndex] == wireframe_representation)
  {
  _surfaceVisible[surfaceIndex]=visible;
  _anastructVisible[surfaceIndex]=false;
  }
  else if (_representation[surfaceIndex] == contours_representation)
  {
  _anastructVisible[surfaceIndex]=visible;
  _surfaceVisible[surfaceIndex]=false;
  }
  _updateProps();
}

/////////////////
// saveAsImage //
/////////////////

void 
SurfaceViewWindow
::saveWindowAsImage(const char *const filename) const
{
#if 1
  cerr << "Sorry, this function is broken." << std::endl;
#else
  std::string pngFilename = StringUtils::forceExtension(filename, "png");
  // set up pipeline
  vtkWindowToImageFilter *w2iFilter = vtkWindowToImageFilter::New();
  vtkPNGWriter *imageWriter = vtkPNGWriter::New();
  w2iFilter->SetInput(_renderWindow);
  w2iFilter->Update();
  imageWriter->SetInput(w2iFilter->GetOutput());
  imageWriter->SetFileName(pngFilename.c_str());
  
  // write the image
  _renderWindow->Render();
  imageWriter->Write();
#endif
}

////////////////////
// getNumSurfaces //
////////////////////

unsigned int
SurfaceViewWindow
::getNumSurfaces() const
{
  return _surfaces.size();
}

//////////////////////
// getNumAnastructs //
//////////////////////

unsigned int
SurfaceViewWindow
::getNumAnastructs() const
{
  return _anastructs.size();
}

//////////////////
// centerActors //
//////////////////

void 
SurfaceViewWindow
::centerActors()
{
  _renderer->ResetCamera();
}

///////////////
// jumpStart //
///////////////

void 
SurfaceViewWindow
::jumpStart()
{
  invalidate();
  redraw();
  //_renderWindow->Render();
}

///////////////////
// updateDisplay //
///////////////////

void
SurfaceViewWindow
::updateDisplay() 
{
_updateProps();
  //_renderWindow->Render();
}

////////////
// handle //
////////////

int
SurfaceViewWindow
::handle(int event)
{
  switch (event) 
  {
  case(FL_KEYBOARD):
    /*if (Fl::event_key() == 's') 
    {
      // view anastructs with surface
      setViewAnastructs(false);
      setViewSurfaces(true);
      _updateProps();
      return Fl_VTK_Window::handle(event);
    }
    else if (Fl::event_key() == 'w')
    {
      // dont view anastructs with wireframe
      setViewAnastructs(false);
      setViewSurfaces(true);
      _updateProps();
      return Fl_VTK_Window::handle(event);
    }
    else if (Fl::event_key() == 'z')
    {
      // view anastruct only
      setViewAnastructs(true);
      setViewSurfaces(false);
      _updateProps();
      updateDisplay();
      return 1;
    }
    else */
    if ((Fl::event_key() == 'e')&&(_ROIcreated))
    {
      // remove the roi
      if(_viewROI)
      {
        setViewROI(false);
      }
      else
      {
        setViewROI(true);
      }
      _updateProps();
      updateDisplay();
      return 1;
    }
    else if (Fl::event_key() == 'x')
    {
      std::cerr << "trying to jump start window..." << std::endl;
      jumpStart();
      return 1;
    }
    break;
  }
  
  // let vtkFlRenderWindowInteractor handle it
  // return vtkFlRenderWindowInteractor::handle(event);
  
  return Fl_VTK_Window::handle(event);
}

/////////////////////
// getSurfaceIndex //
/////////////////////

unsigned int 
SurfaceViewWindow
::getSurfaceIndex( int imageIndex,
             int imAnaIndex) 
{
  for(unsigned int surfaceIndex=0 ; surfaceIndex < _surfaces.size() ; surfaceIndex++)
  {
    if((_surfaceIndex[surfaceIndex].ImageIndex == imageIndex )
      &&(_surfaceIndex[surfaceIndex].ImAnaIndex == imAnaIndex ))
    {
      return surfaceIndex;
    } 
  }
  throw std::invalid_argument("SurfaceViewWindow::getSurfaceIndex: invalid index");
}
