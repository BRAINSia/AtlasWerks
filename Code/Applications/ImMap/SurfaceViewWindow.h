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

#ifndef SurfaceViewWindow_h
#define SurfaceViewWindow_h


#include <Fl_VTK_Window.H>

#include <vtkActor.h>
#include <vtkRenderWindow.h>
#include "vtkDataSetMapper.h"
#include <vtkCubeSource.h>
#include <vector>
#include "Surface.h"
#include "Anastruct.h"


class SurfaceViewWindow : public Fl_VTK_Window
{
public:
  enum SurfaceRepresentationType {surface_representation, 
				  wireframe_representation, 
				  contours_representation};

  // a la flglwindow
  SurfaceViewWindow(int x, int y, int w, int h, const char* l);

  int addSurface(int imageIndex,
                 int imAnaIndex,
                 const Surface& surface,
                 const bool& visible = true,
                 const double& r = 1,
                 const double& g = 0,
                 const double& b = 0,
                 const SurfaceRepresentationType& representation = surface_representation,
                 const double& opacity = 1.0);
  
  int addSurface(int imageIndex,
                 int imAnaIndex,
                 const char *const filename,
                 const bool& visible = true,
                 const double& r = 1,
                 const double& g = 0,
                 const double& b = 0,
                 const SurfaceRepresentationType& representation = surface_representation,
                 const double& opacity = 1.0);

  int addAnastruct(const Anastruct& anastruct,
		   const double& r = 1,
		   const double& g = 0, 
		   const double& b = 0);

  void createROI();
  void updateROI(const Vector3D<double>& start,const Vector3D<double>& stop);

  void clearSurfaces();
  void clearSurface(unsigned int surfaceIndex);
  void clearAnastructs();
  void clearROI();

  void setViewSurfaces(const bool& viewSurfaces);
  void setViewAnastructs(const bool& viewAnastructs);
  void setViewROI(const bool& viewROI);

  //
  // methods for manipulating appearence
  //
  void centerActors();
  
  void setBackgroundColor(const double& r,
			  const double& g,
			  const double& b);

  void getBackgroundColor(double& r,
			  double& g,
			  double& b);

  void setSurfaceColor(int surfaceIndex, 
  		       const double& r, 
		       const double& g, 
		       const double& b);
  void setAnastructColor(int anastructIndex, 
  		       const double& r, 
		       const double& g, 
		       const double& b);
  void setSurfaceOpacity(int surfaceIndex,
			 const double& opacity);
  void setROIOpacity(const double& opacity);
  void setSurfaceRepresentation(int surfaceIndex,
				const SurfaceRepresentationType& representation);
  void setSurfaceLineWidth(int surfaceIndex,
			   const double& lineWidth);
  void setSurfacePointSize(int surfaceIndex,
			   const double& pointSize);
  void setSurfaceVisibility(int surfaceIndex,
			   const bool& visible);

  unsigned int getNumSurfaces() const;
  unsigned int getNumAnastructs() const;

  void saveWindowAsImage(const char *const filename) const;
  void updateDisplay();

  //void setVerbose(bool verbose);
  //bool getVerbose() const;

  // override handle 
  int  handle( int event ); 
  
  unsigned int getSurfaceIndex(int imageIndex,
             int imAnaIndex);

  void jumpStart();
  
  private:
  bool _verbose;
  bool _ROIcreated;
  struct index {int ImageIndex, ImAnaIndex;};

  std::vector<vtkActor*> _surfaces;
  std::vector<vtkActor*> _anastructs;
  vtkActor*				 _roi;
  vtkCubeSource*         _roiGeometry;
  vtkDataSetMapper*      _roiMapper;


  bool                   _viewSurfaces;
  bool                   _viewAnastructs;
  bool                   _viewROI;

  vtkRenderer           *_renderer;
  vtkRenderWindow       *_renderWindow;

  void                   _updateProps();

  double				_ROIopacity;
  std::vector<bool>     _surfaceVisible;
  std::vector<bool>     _anastructVisible;
  std::vector<index>    _surfaceIndex;
  std::vector<SurfaceRepresentationType> _representation;

};
#endif
