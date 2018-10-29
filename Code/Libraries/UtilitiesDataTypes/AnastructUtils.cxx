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

#include "AnastructUtils.h"

// Debugging
#include <iostream>

#include <vector>
#include <list>
#include <cmath>
#include <cassert>
#include <cstdio>
#include <fcntl.h>
#include <stdexcept>
#include <iterator>
#include <unistd.h>
#include "SurfaceUtils.h"
#include "Geometry.h"
#include "Surface.h"
#include "gen.h"
#include "libplanio.h" 
#include <Random.h>

#include <vtkCellArray.h>
#include <vtkBYUWriter.h>
#include <vtkPowerCrustSurfaceReconstruction.h>
#include <vtkPolyDataMapper.h>
#include <vtkPolyDataNormals.h>
#include <vtkTriangleFilter.h>
#include <vtkVersion.h>

#ifdef WIN32
#include <io.h>
#endif

#include "vtkPoints.h"
#include "vtkCellArray.h"
#include "vtkPolyData.h"
#include "vtkVoxelContoursToSurfaceFilter.h"
#include "vtkSurfaceReconstructionFilter.h"
#include "vtkContourFilter.h"
#include "vtkReverseSense.h"
#include "vtkDelaunay3D.h"
#include "vtkUnstructuredGrid.h"
#include "vtkLinearSubdivisionFilter.h"

#if (VTK_MAJOR_VERSION>4)
  typedef double ptType;
#else
  typedef float ptType;
#endif

void
AnastructUtils
::anastructToSurfacePowerCrust(const Anastruct &anastruct, Surface& surface)
{
  Anastruct newanastruct;
  anastructToNewanastruct2(anastruct, newanastruct);  

  // convert to points
  vtkPoints* points = vtkPoints::New();
  vtkCellArray* verts = vtkCellArray::New();
  for (Anastruct::ConstContourIter contour = newanastruct.contours.begin();
       contour != newanastruct.contours.end(); ++contour) {
    for (Contour::ConstVertexIter vert = contour->vertices.begin();
	 vert != contour->vertices.end(); ++vert) {
      // put (*vert) in some data structure
      verts->InsertNextCell(1);
      double x = (*vert)[0] + Random::sampleUniform(-.01, .01);
      double y = (*vert)[1] + Random::sampleUniform(-.01, .01);
      double z = (*vert)[2] + Random::sampleUniform(-.01, .01);
      z *= 2;   // See if this improves the byu.
      verts->InsertCellPoint(points->InsertNextPoint( x, y, z ));
    }
  }

  vtkPolyData* polyData = vtkPolyData::New();
  polyData->SetPoints(points);
  polyData->SetVerts(verts);


  // run power crust
  vtkPowerCrustSurfaceReconstruction* surfaceGenerator = 
    vtkPowerCrustSurfaceReconstruction::New();
  surfaceGenerator->SetInputData(polyData);
  surfaceGenerator->Update();


  // have only triangles
  vtkTriangleFilter *surfaceTriangular =
    vtkTriangleFilter::New(); 
  surfaceTriangular-> SetInputData(surfaceGenerator->GetOutput());
  surfaceTriangular-> PassVertsOn();

  // Make sure all triangles have consistent orientation (implying
  // outward-pointing normals via right hand rule)
  vtkPolyDataNormals* polyDataNormals = vtkPolyDataNormals::New();
  polyDataNormals->SetInputData( surfaceTriangular->GetOutput() );
  polyDataNormals->SplittingOff();
  polyDataNormals->ComputePointNormalsOff();
  polyDataNormals->ComputeCellNormalsOn();
  polyDataNormals->Update();

  surface.clear();

  // convert the ouput of the filters into the  surface structure
  // vtkPolyData* polyDataTriangle = surfaceTriangular->GetOutput(); 
  vtkPolyData* polyDataForSurface = polyDataNormals->GetOutput();
  vtkCellArray* Celltriangle = polyDataForSurface->GetPolys();
  vtkPoints* Vertices =  polyDataForSurface->GetPoints();

  vtkIdType    size;
  vtkIdType*   cell;

  Celltriangle->InitTraversal();
  int stillMoreCells=1;

  while (stillMoreCells)
    { 

      stillMoreCells=Celltriangle->GetNextCell(size,cell);
     
      // give all the points to fill the vertices
      if(stillMoreCells)
	{      	 
	  ptType vert1[3];
	  ptType vert2[3];
	  ptType vert3[3];

	  Vertices->GetPoint(cell[0], vert1);
	  Vertices->GetPoint(cell[1], vert2);
	  Vertices->GetPoint(cell[2], vert3);

	  unsigned int vIdx1 = surface.addVertex(vert1[0], vert1[1], vert1[2]);
	  unsigned int vIdx2 = surface.addVertex(vert2[0], vert2[1], vert2[2]);
	  unsigned int vIdx3 = surface.addVertex(vert3[0], vert3[1], vert3[2]);

	  surface.addFacet(vIdx1, vIdx2, vIdx3);

	}

    }

  for (unsigned int i = 0; i < surface.vertices.size(); ++i) {
      surface.vertices[i].z *= .5;
  }

  std::cout << "num facets: " <<  surface.facets.size() << std::endl;
  std::cout << "num verts: " <<  surface.vertices.size() << std::endl;
	  
  points->Delete();  
  verts->Delete();	  
  polyData->Delete();	  
  surfaceGenerator->Delete();            
  surfaceTriangular->Delete();
#if OUTPUT_INTERMEDIATE_BYUS 
  writer->Delete();
#endif
}

void
AnastructUtils
::anastructToSurfacePowerCrust2(const Anastruct &anastruct, Surface& surface)
{
  Anastruct newanastruct;
  anastructToNewanastruct2(anastruct, newanastruct);  

  // convert to points
  vtkPoints* points = vtkPoints::New();
  vtkCellArray* verts = vtkCellArray::New();
  for (Anastruct::ConstContourIter contour = newanastruct.contours.begin();
       contour != newanastruct.contours.end(); ++contour) {
    for (Contour::ConstVertexIter vert = contour->vertices.begin();
	 vert != contour->vertices.end(); ++vert) {
      // put (*vert) in some data structure
      verts->InsertNextCell(1);
      double x = (*vert)[0] + Random::sampleUniform(-.01, .01);
      double y = (*vert)[1] + Random::sampleUniform(-.01, .01);
      double z = (*vert)[2] + Random::sampleUniform(-.01, .01);
      z *= 2;   // See if this improves the byu.
      verts->InsertCellPoint(points->InsertNextPoint( x, y, z ));
    }
  }

  vtkPolyData* polyData = vtkPolyData::New();
  polyData->SetPoints(points);
  polyData->SetVerts(verts);


  // run power crust
  vtkPowerCrustSurfaceReconstruction* surfaceGenerator = 
    vtkPowerCrustSurfaceReconstruction::New();
  surfaceGenerator->SetInputData(polyData);
  surfaceGenerator->Update();

  // have only triangles
  vtkTriangleFilter *surfaceTriangular =
    vtkTriangleFilter::New(); 
  surfaceTriangular-> SetInputData(surfaceGenerator->GetOutput());

  // Make sure all triangles have consistent orientation (implying
  // outward-pointing normals via right hand rule)
  vtkPolyDataNormals* polyDataNormals = vtkPolyDataNormals::New();
  polyDataNormals->SetInputData( surfaceTriangular->GetOutput() );
  polyDataNormals->SplittingOff();
  polyDataNormals->ComputePointNormalsOff();
  polyDataNormals->ComputeCellNormalsOn();
  polyDataNormals->Update();

  // convert the ouput of the filters into the  surface structure
  // vtkPolyData* polyDataTriangle = surfaceTriangular->GetOutput(); 
  vtkPolyData* polyDataForSurface = polyDataNormals->GetOutput();
  vtkCellArray* Celltriangle = polyDataForSurface->GetPolys();
  vtkPoints* Vertices =  polyDataForSurface->GetPoints();

  surface.clear();
  for (vtkIdType i = 0; i < Vertices->GetNumberOfPoints(); ++i) {
    ptType* vert = Vertices->GetPoint(i);
    surface.vertices.push_back(Surface::Vertex(vert[0],vert[1],vert[2]));
  }

  Celltriangle->InitTraversal();
  vtkIdType npts;
  vtkIdType* pts;
  while(Celltriangle->GetNextCell(npts,pts)) {
    surface.addFacet(pts[0],pts[1],pts[2]);    
  }

  for (unsigned int i = 0; i < surface.vertices.size(); ++i) {
      surface.vertices[i].z *= .5;
  }

  std::cout << "num facets: " <<  surface.facets.size() << std::endl;
  std::cout << "num verts: " <<  surface.vertices.size() << std::endl;
  
  points->Delete();  
  verts->Delete();	  
  polyData->Delete();	  
  surfaceGenerator->Delete();            
  surfaceTriangular->Delete();
#if OUTPUT_INTERMEDIATE_BYUS 
  writer->Delete();
#endif
}

void
AnastructUtils
::anastructToNewanastruct2(const Anastruct& oldanastruct, Anastruct& newanastruct)
{
  const int numContours = oldanastruct.contours.size();
  int *pointsPerContour = new int[numContours];

  AnastructUtils::vectthreeD points;

  points.resize(numContours);
  int contourIndex = 0;
  for (contourIndex = 0; contourIndex < numContours; ++contourIndex)
    {    
      int numPoints = oldanastruct.contours[contourIndex].vertices.size();
      pointsPerContour[contourIndex] = numPoints;     
      points[contourIndex].resize(numPoints);
      for (int pointIndex = 0; pointIndex < numPoints; ++pointIndex)
	{	
	  points[contourIndex][pointIndex] =
              oldanastruct.contours[contourIndex].vertices[pointIndex];
	}
    } 
 

  //fill the vector to have the distance between each points for each contour
  AnastructUtils::distancelist distance; 
  distance.resize(numContours);	
  for (contourIndex = 0; contourIndex < numContours; ++contourIndex)
    {    
      int numPoints = oldanastruct.contours[contourIndex].vertices.size();   
      distance[contourIndex].resize(numPoints);   
      distance[contourIndex].resize(numPoints);
      for (int pointIndex = 0; pointIndex < numPoints; ++pointIndex)
	{			  	 	 	 
	  distance[contourIndex][pointIndex] = 
	    sqrt( pow((points[contourIndex][(pointIndex) % numPoints][0]-
                       points[contourIndex][(pointIndex+1) % numPoints][0]),2)  +
                  pow((points[contourIndex][(pointIndex) % numPoints][1]-
                       points[contourIndex][(pointIndex+1) % numPoints][1]),2) +
                  pow((points[contourIndex][(pointIndex) % numPoints][2]-
                       points[contourIndex][(pointIndex+1) % numPoints][2]),2) );
	  }	
    }

  float distance_between_slices = 
	    sqrt( pow((points[0][1][0]- points[1][1][0]),2)  +
                  pow((points[0][1][1]- points[1][1][1]),2) +
                  pow((points[0][1][2]- points[1][1][2]),2) );	

  //to insert the new points in the points matrix to be able to create
  //the new anastruct
  AnastructUtils::vectthreeD newpoints;

  newpoints.resize(numContours); 
  for (contourIndex = 0; contourIndex < numContours; ++contourIndex)
    {     
      int numPoints = oldanastruct.contours[contourIndex].vertices.size();
      points[contourIndex].resize(numPoints);
      for (int pointIndex = 0; pointIndex < numPoints; ++pointIndex)
	{          
	  float ratio = distance[contourIndex][pointIndex] /
                        distance_between_slices;

	  while(ratio >= 2)
	    {	
	      numPoints++;	
	      distance[contourIndex].resize(numPoints);	
	      points = InsertNewPoint(points, pointIndex, contourIndex,
                                      numPoints, numContours);	    	
	      distance = ComputeDistance(points,numContours);
	      ratio = distance[contourIndex][pointIndex] / 
                      distance_between_slices; 	   
	    }
	}
    }
  
  // create the new anastruct from the new points of each contour
  newanastruct.contours.resize(numContours);
  for (contourIndex = 0; contourIndex < numContours; ++contourIndex)
    {    
      int numPoints = points[contourIndex].size();
      newanastruct.contours[contourIndex].vertices.resize(numPoints);
      for (int pointIndex = 0; pointIndex < numPoints; ++pointIndex)
	{		
	  newanastruct.contours[contourIndex].vertices[pointIndex] = 
              points[contourIndex][pointIndex];
	}
    }    
}

 
// this method makes problem for the bladder where we always have two
// points with the same coordinates in fact we take 0 for minimum.
void
AnastructUtils
::anastructToNewanastruct(const Anastruct& oldanastruct, Anastruct& newanastruct)
{
  const int numContours = oldanastruct.contours.size();
  int *pointsPerContour = new int[numContours];

  AnastructUtils::vectthreeD points;

  points.resize(numContours);
  int contourIndex = 0;
  for (contourIndex = 0; contourIndex < numContours; ++contourIndex)
    {    
      int numPoints = oldanastruct.contours[contourIndex].vertices.size();
      pointsPerContour[contourIndex] = numPoints;     
      points[contourIndex].resize(numPoints);
      for (int pointIndex = 0; pointIndex < numPoints; ++pointIndex)
	{	
	  points[contourIndex][pointIndex] = oldanastruct.contours[contourIndex].vertices[pointIndex];
	}
    } 
 

  //fill the vector to have the distance between each points for each contour
  AnastructUtils::distancelist distance; 
  float *min= new float[numContours]; 
  distance.resize(numContours);	
  for (contourIndex = 0; contourIndex < numContours; ++contourIndex)
    {    
      int numPoints = oldanastruct.contours[contourIndex].vertices.size();
    
      distance[contourIndex].resize(numPoints);
      min[contourIndex]= 
	sqrt( pow ((points[contourIndex][1][0]- points[contourIndex][2][0]) ,2) + 
	      pow((points[contourIndex][1][1]- points[contourIndex][2][1]),2) +  
	      pow((points[contourIndex][1][2]- points[contourIndex][2][2]),2) );    
      distance[contourIndex].resize(numPoints);
      for (int pointIndex = 0; pointIndex < numPoints; ++pointIndex)
	{			  	 	 	 
	  distance[contourIndex][pointIndex] = 
	    sqrt( pow((points[contourIndex][(pointIndex) % numPoints][0]- points[contourIndex][(pointIndex+1) % numPoints ][0]),2)  +  pow((points[contourIndex][(pointIndex) % numPoints][1]- points[contourIndex][(pointIndex+1) % numPoints][1]),2) + pow((points[contourIndex][(pointIndex) % numPoints][2]- points[contourIndex][(pointIndex+1) % numPoints][2]),2) );

	  if( distance[contourIndex][pointIndex] < 0.0005 )	    	   
	    { 
	      numPoints =numPoints-1; 
	      for( int  pointIndexIt = pointIndex; pointIndexIt < numPoints; ++pointIndexIt)
		{
		  points[contourIndex][(pointIndexIt) % numPoints]=points[contourIndex][(pointIndexIt+1) % numPoints];
		}
	      distance[contourIndex].resize(numPoints);
	    }

	  else
	    {
	      if(distance[contourIndex][pointIndex]< min[contourIndex])	    	    
		min[contourIndex]=distance[contourIndex][pointIndex];
	    }
	  }	
    }


  //to insert the new points in the points matrix to be able to create the new anastruct
  AnastructUtils::vectthreeD newpoints;

  newpoints.resize(numContours); 
  for (contourIndex = 0; contourIndex < numContours; ++contourIndex)
    {     
      int numPoints = oldanastruct.contours[contourIndex].vertices.size();
      points[contourIndex].resize(numPoints);
      for (int pointIndex = 0; pointIndex < numPoints; ++pointIndex)
	{          
	  float ratio= distance[contourIndex][pointIndex] /  min[contourIndex];
	  while(ratio >= 2)
	    {	
	      numPoints++;	
	      distance[contourIndex].resize(numPoints);	
	      points= InsertNewPoint(points,pointIndex,contourIndex,numPoints,numContours);	    	
	      distance = ComputeDistance(points,numContours);
	    
	      ratio= distance[contourIndex][pointIndex] /  min[contourIndex]; 	   
	    }
	}
    }
  
  // create the new anastruct from the new points of each contour
  newanastruct.contours.resize(numContours);
  for (contourIndex = 0; contourIndex < numContours; ++contourIndex)
    {    
      int numPoints = points[contourIndex].size();
      newanastruct.contours[contourIndex].vertices.resize(numPoints);
      for (int pointIndex = 0; pointIndex < numPoints; ++pointIndex)
	{		
	  newanastruct.contours[contourIndex].vertices[pointIndex]= points[contourIndex][pointIndex] ;   
	}	      
    }    
}

 

AnastructUtils::vectthreeD
AnastructUtils
::InsertNewPoint( vectthreeD points, int pointIndexbegin, int contourIndex , int numPoints, int numContours)
{
  Vector3D<float> point_to_insert = points[contourIndex][(pointIndexbegin)%(numPoints-1)]+  ( (points[contourIndex][(pointIndexbegin+1) %(numPoints-1)] - points[contourIndex][(pointIndexbegin) %(numPoints-1)])/2);
 
  points[contourIndex].insert (  points[contourIndex].begin()+  pointIndexbegin+1, point_to_insert   );

  return points;
}


AnastructUtils::distancelist
AnastructUtils
::ComputeDistance(vectthreeD  points, int numContours)
{
  AnastructUtils::distancelist distance;
  distance.resize(numContours);
  for (int contourIndex = 0; contourIndex < numContours; ++contourIndex)
    {
      distance[contourIndex].resize( points[contourIndex].size());
      for (unsigned int pointIndex = 0; pointIndex < points[contourIndex].size() ; ++pointIndex)
	{	 
	  distance[contourIndex][pointIndex] = 
	    sqrt(  (pow((points[contourIndex][(pointIndex) %  points[contourIndex].size()][0]-  points[contourIndex][(pointIndex+1) %  points[contourIndex].size()][0]),2)) + 
		   (pow((points[contourIndex][(pointIndex) %  points[contourIndex].size()][1]-  points[contourIndex][(pointIndex+1) %  points[contourIndex].size()][1]),2)) +  
		   (pow((points[contourIndex][(pointIndex) %  points[contourIndex].size()][2]-  points[contourIndex][(pointIndex+1) %  points[contourIndex].size()][2]),2) ));		 
	}
    }	
	   
  return distance;
}



void
AnastructUtils
::surfaceToAnastruct(const Surface& surface,
		     Anastruct& anastruct,
		     unsigned int numSlices,
		     int    *const sliceNumbers,
		     double *const contourZPositions,
                     const std::string& name)
{
  // clear old contours
  anastruct.clear();

  //
  // add contours for each slice
  // 
  for (unsigned int sliceIndex = 0; sliceIndex < numSlices; ++sliceIndex)
    {
      _extractContours(surface, 
		       contourZPositions[sliceIndex], 
		       sliceNumbers[sliceIndex],
		       anastruct.contours);
    }

  //
  // update anastruct min/max points from contour min/max points
  //
  anastruct.updateMinMax();
  anastruct.label = name;
}

//
// assumes surface is in world coordinates: creates an anastruct in
// world coordinates for an image with the given origin and spacing
//
void
AnastructUtils
::surfaceToAnastruct(const Surface& surface,
                     Anastruct& anastruct,
                     const Vector3D<double>& imOrigin,
                     const Vector3D<double>& imSpacing,
                     const std::string& name)
{
  Surface s(surface);
  SurfaceUtils::worldToImageIndex(s, imOrigin, imSpacing);
  AnastructUtils::surfaceToAnastruct(s, anastruct, name);
  AnastructUtils::imageIndexToWorld(anastruct, imOrigin, imSpacing);  
}

//
// assumes surface is in voxel coordinates.  produces an anastruct in
// voxel coordinates.
//
void
AnastructUtils
::surfaceToAnastruct(const Surface& surface, 
		     Anastruct& anastruct,
                     const std::string& name)
{
  //
  // decide contour z positions
  //
  Vector3D<double> surfaceMax, surfaceMin;
  surface.getAxisExtrema(surfaceMin, surfaceMax);
  
  // get slice index for min and max
  int minSliceIndex = static_cast<int>(ceil((surfaceMin.z)));
  int maxSliceIndex = static_cast<int>(floor((surfaceMax.z)));
  
  // include all intervening slices
  unsigned int numContours = maxSliceIndex - minSliceIndex + 1;
  double *contourZPositions = new double[numContours];
  int    *sliceNumbers      = new int[numContours];
  for (int contourSliceIndex = minSliceIndex, positionIndex = 0;
       contourSliceIndex <= maxSliceIndex;
       ++contourSliceIndex, ++positionIndex)
    {
      contourZPositions[positionIndex] = contourSliceIndex;
      sliceNumbers[positionIndex] = contourSliceIndex;
    }
  
  // create the anastruct from the surface
  AnastructUtils::surfaceToAnastruct(surface, anastruct, numContours, 
				     sliceNumbers, contourZPositions, name);

  delete [] contourZPositions;
  delete [] sliceNumbers;
}

//
// assume anastruct is in world coordinates (x,y,and z); convert to
// voxel coordinates.  slice number remains the same.
//
void 
AnastructUtils
::worldToImageIndex(Anastruct& anastruct,
		    const Vector3D<double>& imageOrigin,
		    const Vector3D<double>& imageSpacing)
{
  for (Anastruct::ContourIter contourIter = anastruct.contours.begin();
       contourIter != anastruct.contours.end();
       ++contourIter)
    {
      for (Contour::VertexIter vertexIter = contourIter->vertices.begin();
	   vertexIter != contourIter->vertices.end();
	   ++vertexIter)
	{
	  vertexIter->x = (vertexIter->x - imageOrigin.x) / imageSpacing.x;
	  vertexIter->y = (vertexIter->y - imageOrigin.y) / imageSpacing.y;
	  vertexIter->z = (vertexIter->z - imageOrigin.z) / imageSpacing.z;
	}
    }
  anastruct.updateMinMax();
}

void 
AnastructUtils
::worldToImageIndexXY(Anastruct& anastruct,
		      const double& imageOriginX,
		      const double& imageOriginY,
		      const double& imageSpacingX,
		      const double& imageSpacingY)
{
  for (Anastruct::ContourIter contourIter = anastruct.contours.begin();
       contourIter != anastruct.contours.end();
       ++contourIter)
    {
       for (Contour::VertexIter vertexIter = contourIter->vertices.begin();
	   vertexIter != contourIter->vertices.end();
	   ++vertexIter)
	{
	  vertexIter->x = (vertexIter->x - imageOriginX) / imageSpacingX;
	  vertexIter->y = (vertexIter->y - imageOriginY) / imageSpacingY;
	}
    }
  anastruct.updateMinMax();
}

//
// assume anastruct is in voxel coordinates (x,y,and z); convert to
// world coordinates.  slice number remains the same.
//
void 
AnastructUtils
::imageIndexToWorld(Anastruct& anastruct,
		    const Vector3D<double>& imageOrigin,
		    const Vector3D<double>& imageSpacing)
{
  for (Anastruct::ContourIter contourIter = anastruct.contours.begin();
       contourIter != anastruct.contours.end();
       ++contourIter)
    {
      for (Contour::VertexIter vertexIter = contourIter->vertices.begin();
	   vertexIter != contourIter->vertices.end();
	   ++vertexIter)
	{
	  vertexIter->x = vertexIter->x * imageSpacing.x + imageOrigin.x;
	  vertexIter->y = vertexIter->y * imageSpacing.y + imageOrigin.y;
	  vertexIter->z = vertexIter->z * imageSpacing.z + imageOrigin.z;
	}
    }
  anastruct.updateMinMax();
}

void 
AnastructUtils
::imageIndexToWorldXY(Anastruct& anastruct,
		      const double& imageOriginX,
		      const double& imageOriginY,
		      const double& imageSpacingX,
		      const double& imageSpacingY)
{
  for (Anastruct::ContourIter contourIter = anastruct.contours.begin();
       contourIter != anastruct.contours.end();
       ++contourIter)
    {
      for (Contour::VertexIter vertexIter = contourIter->vertices.begin();
	   vertexIter != contourIter->vertices.end();
	   ++vertexIter)
	{
	  vertexIter->x = vertexIter->x * imageSpacingX + imageOriginX;
	  vertexIter->y = vertexIter->y * imageSpacingY + imageOriginY;
	}
    }
  anastruct.updateMinMax();
}

void AnastructUtils
::scale(Anastruct &anastruct, const double& scale)
{
  for (Anastruct::ContourIter contourIter = anastruct.contours.begin();
       contourIter != anastruct.contours.end();
       ++contourIter)
    {
      for (Contour::VertexIter vertexIter = contourIter->vertices.begin();
	   vertexIter != contourIter->vertices.end();
	   ++vertexIter)
	{
	  vertexIter->x = vertexIter->x * scale;
	  vertexIter->y = vertexIter->y * scale;
	}
    }
  anastruct.updateMinMax();
}

void
AnastructUtils
::_extractContours(const Surface& surface,
		   const double& zPosition,
		   int sliceNumber,
		   std::vector<Contour>& contours,
		   const double& tolerance)
{
  //
  // compute intersection of surface with plane (a LineList)
  //
  LineList lines;
  Vector3D<double> planePoint(0, 0, zPosition);
  Vector3D<double> planeNormal(0, 0, 1);
  SurfaceUtils::intersectWithPlane(surface,
				   planePoint,
				   planeNormal,
				   lines);

  // debug output...
  //std::cerr << "##### slice number " << sliceNumber << " #####" << std::endl;
  //std::copy(lines.begin(), lines.end(),
  //	    std::ostream_iterator<Line3D>(std::cerr, "\n"));

  //
  // if we can't make a contour, just return
  //
  if (lines.size() < 3)
    {
      return;
    }

 //  if (sliceNumber == 33)
//     {
//       std::copy(lines.begin(), lines.end(), std::ostream_iterator<Line3D>(std::cout, "\n"));
//     }

  //
  // convert the LineList to contours and add them to the list
  //
  while (!lines.empty())
    {
      // std::cerr << "\n\n##### creating a new contour " 
      // << sliceNumber << " #####\n\n" << std::endl;

      // create a new contour
      contours.push_back(Contour());
      Contour *currContourPtr = &contours[contours.size() - 1];
      currContourPtr->sliceNumber = sliceNumber;

      // add the next available line
      currContourPtr->vertices.push_back(lines.front().p1);
      currContourPtr->vertices.push_back(lines.front().p2);
      Vector3D<double> currEndpoint = lines.front().p2;

      // std::cerr << lines.front() << std::endl;

      lines.pop_front();

      // add lines while connected
      while (!lines.empty())
	{
	  // find closest point to current endpoint
	  double minDistance = currEndpoint.distance(lines.front().p1);
	  LineIterator closestIter = lines.begin();
	  bool p1IsClosest = true;
	  
	  // loop to find closest point
	  for (LineIterator lineIter = lines.begin();
	       lineIter != lines.end();
	       lineIter++)
	    {
	      double p1Distance = currEndpoint.distance(lineIter->p1);
	      double p2Distance = currEndpoint.distance(lineIter->p2);
	      
	      if (p1Distance < minDistance || p2Distance < minDistance)
		{
		  closestIter = lineIter;
		  p1IsClosest = (p1Distance < p2Distance);
		  minDistance = (p1IsClosest ? p1Distance : p2Distance);
		}
	    }
	  
	  // add the closest line if within tolerence
	  if (minDistance < tolerance)
	    {
	      if (lines.size() > 1)
		{
		  if (p1IsClosest)
		    {
		      currContourPtr->vertices.push_back(closestIter->p2);
		      currEndpoint = closestIter->p2;
		    }
		  else
		    {
		      currContourPtr->vertices.push_back(closestIter->p1);
		      currEndpoint = closestIter->p1;
		    }
		}
	      else
		{
		  // make sure that the contour is closed
		  // 		  std::cerr << "firstPoint" << firstPoint << std::endl;
		  // 		  std::cerr << "p1" << closestIter->p1 
		  // 			    << ", " << firstPoint.distance(closestIter->p1) 
		  // 			    << std::endl;
		  // 		  std::cerr << "p2" << closestIter->p2 
		  // 			    << ", " << firstPoint.distance(closestIter->p2) 
		  // 			    << std::endl;
		  // this fails in certain cases need to look at it
		  // 		  assert(firstPoint.distance(p1IsClosest ? 
		  // 					     closestIter->p2 : 
		  // 					     closestIter->p1)
		  // 			 < tolerance);
		}
	      lines.erase(closestIter);
	    }
	  else
	    {
	      // break to start a new contour
	      break;
	    }
	}
    } 
}

void
AnastructUtils
::readPLUNCAnastruct(Anastruct& anastruct,
		     const char * const fileName)
{
  Anastruct::PLUNCAnastruct pluncAnastruct;
  int fdes;
#ifdef WIN32
  fdes = open(fileName, O_RDONLY|O_BINARY, 0);
#else
  fdes = open(fileName, O_RDONLY, 0);
#endif
  if (fdes < 0 || read_anastruct(fdes, &pluncAnastruct))
    {
      throw std::runtime_error("Can't open anastruct file.");
    }
  close(fdes);
  anastruct = pluncAnastruct;
  deletePLUNCAnastruct(pluncAnastruct);
}

void
AnastructUtils
::writePLUNCAnastruct(Anastruct& anastruct,
		      const char * const fileName)
{
  Anastruct::PLUNCAnastruct pluncAnastruct;
  pluncAnastruct.contour_count = 0;
  pluncAnastruct.contours = 0;
  anastruct.copyToPLUNCAnastruct(pluncAnastruct);

  int fdes;
#ifdef WIN32
  fdes = open(fileName, O_CREAT|O_WRONLY|O_BINARY, 0700);
#else
  fdes = open(fileName, O_CREAT|O_WRONLY,0700);
#endif
  if (fdes < 0 || write_anastruct(fdes, &pluncAnastruct))
    {
      throw std::runtime_error("cant open file");
      //std::cout<<" error write "<<fdes<<std::endl;
      //  std::cout<<" perror "<<strerror(errno)<<std::endl;
      //  return;
    }
  //close(fdes);
  deletePLUNCAnastruct(pluncAnastruct);
}

void
AnastructUtils
::deletePLUNCAnastruct(Anastruct::PLUNCAnastruct &anastruct)
{
  // use free and not delete [] since plunc creates anastructs using
  // malloc not new
  for (int contourIndex = 0; 
       contourIndex < anastruct.contour_count;
       ++contourIndex)
    {
      if (anastruct.contours[contourIndex].x) 
	{
	  free(anastruct.contours[contourIndex].x);
	}
      if (anastruct.contours[contourIndex].y) 
	{
	  free(anastruct.contours[contourIndex].y);
	}
    }
  if (anastruct.contours)
    {
      free(anastruct.contours);
    }
}

void 
AnastructUtils
::capContour(Anastruct& anastruct)
{
  Anastruct tmpanastruct = anastruct;
  Vector3D<double> center;
  Contour::VertexIter vertexIter;
  tmpanastruct.updateMinMax();
  double zthick=0;
  int k=1;
  while (zthick == 0)
    {
      zthick = 
	(tmpanastruct.contours[tmpanastruct.contours.size()-1].vertices[0].z -
	 tmpanastruct.contours[tmpanastruct.contours.size()-1-k].vertices[0].z);
      k++;
    }
  for (unsigned int i=0; i< tmpanastruct.contours.size();i++)
    {
      if ((tmpanastruct.contours[i].max.z) == (tmpanastruct.max.z))
	{
	  center.set(0,0,0);
	  for (vertexIter = (tmpanastruct.contours[i]).vertices.begin();
	       vertexIter != (tmpanastruct.contours[i]).vertices.end();
	       ++vertexIter)
	    {
	      center = *vertexIter + center;
	    }
          
	  center = center/(tmpanastruct.contours[i]).vertices.size();
          
	  center.z += (2*fabs(zthick))/3;
	  Contour *maxCap;
	  if (zthick > 0)
	    {
	      anastruct.contours.push_back(Contour());
	      maxCap = &(anastruct.contours[anastruct.contours.size() - 1]);
	      maxCap->sliceNumber = (tmpanastruct.contours[i]).sliceNumber+1;
	    }
          
	  //anastruct.contours.insert(anastruct.contours.begin()+i,Contour());
	  //Contour *maxCap = &(anastruct.contours[i]);
	  else
	    {
	      anastruct.contours.insert(anastruct.contours.begin(),Contour());
	      maxCap = &(anastruct.contours[0]);
	      maxCap->sliceNumber = (tmpanastruct.contours[i]).sliceNumber-1;
	    }
	  for (vertexIter = (tmpanastruct.contours[i]).vertices.begin();
	       vertexIter != (tmpanastruct.contours[i]).vertices.end();
	       ++vertexIter)
	    {
	      (maxCap->vertices).push_back( *vertexIter/3 + (2*center)/3 );
	    }
	  maxCap->updateMinMax(zthick);
          
	}
      
      
      if ((tmpanastruct.contours[i].min.z) == (tmpanastruct.min.z))
	{
	  center.set(0,0,0);
	  for (vertexIter = (tmpanastruct.contours[i]).vertices.begin();
	       vertexIter != (tmpanastruct.contours[i]).vertices.end();
	       ++vertexIter)
	    {
	      center = *vertexIter + center;
	    }
          
	  center = center/(tmpanastruct.contours[i]).vertices.size();
	  center.z -= (2*fabs(zthick))/3;
	  Contour *minCap;
	  if (zthick > 0)
	    {
	      anastruct.contours.insert(anastruct.contours.begin(),Contour());
	      minCap = &(anastruct.contours[0]);
	      minCap->sliceNumber = (tmpanastruct.contours[i]).sliceNumber-1;
	    }
	  //anastruct.contours.insert(anastruct.contours.begin()+i,Contour());
	  //Contour *minCap = &(anastruct.contours[i]);
	  //after//Contour *minCap = &(anastruct.contours[0]);
	  else
	    {
	      anastruct.contours.push_back(Contour());
	      minCap = &(anastruct.contours[anastruct.contours.size() - 1]);
	      minCap->sliceNumber = (tmpanastruct.contours[i]).sliceNumber+1;
	    }
          
	  for (vertexIter = (tmpanastruct.contours[i]).vertices.begin();
	       vertexIter != (tmpanastruct.contours[i]).vertices.end();
	       ++vertexIter)
	    {
	      (minCap->vertices).push_back( *vertexIter/3 + (2*center)/3 );
	    }
	  minCap->updateMinMax(zthick);
	}
    }
  anastruct.updateMinMax();
}
