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

#include "SurfaceUtilsVtk.h"

#include <vtkBYUReader.h>
#include <vtkBYUWriter.h>
#include <vtkCellArray.h>
#include <vtkCleanPolyData.h>
#include <vtkPoints.h>
#include <vtkPolyDataMapper.h>
#include <vtkPolyDataNormals.h>
#include <vtkTriangleFilter.h>

#include "SurfaceUtils.h"
#include <Vector3D.h>

void SurfaceToVtkPolyData(const Surface& surface, vtkPolyData* polyData)
{
    vtkPoints* points = vtkPoints::New();
    vtkCellArray* verts = vtkCellArray::New();
    for (unsigned int i = 0; i < surface.vertices.size(); ++i) {
        verts->InsertNextCell(1);
        const Vector3D<double>& vert = surface.vertices[i];
        verts->InsertCellPoint(
            points->InsertNextPoint( vert[0], vert[1], vert[2] ));
    }

    polyData->SetPoints(points);
    polyData->SetVerts(verts);

    vtkCellArray* polys = vtkCellArray::New();
    for (unsigned int i = 0; i < surface.facets.size(); ++i) {
        const std::vector<unsigned int>& facet = surface.facets[i];
        polys->InsertNextCell( facet.size() );
        for (unsigned int j = 0; j < facet.size(); ++j )
        {
            polys->InsertCellPoint( facet[j] );
        }
    }
    
    polyData->SetPolys(polys);
    
}

void VtkPolyDataToSurface(vtkPolyData* polyData, Surface& surface)
{
    surface.clear();

    vtkPoints* points = polyData->GetPoints();
    for (int i = 0; i < points->GetNumberOfPoints(); ++i) {
        Vector3D<double> vertex;
        points->GetPoint(i, &vertex[0]);
        surface.addVertex(vertex);
    }

    vtkCellArray* polys = polyData->GetPolys();
    vtkIdType numPointsInCell;
    vtkIdType* pointsInCell;
    std::vector<unsigned int> facet;
    polys->InitTraversal();
    while (polys->GetNextCell(numPointsInCell, pointsInCell)) {
        facet.clear();
        for (vtkIdType i = 0; i < numPointsInCell; ++i) {
            facet.push_back(pointsInCell[i]);
        }
        surface.facets.push_back(facet);
    }

}

void 
SetVtkPolyDataPoints(vtkPolyData* polyData, const Surface& surface)
{
  vtkPoints* points = polyData->GetPoints();
  vtkIdType nPolyDataPts = points->GetNumberOfPoints();
  vtkIdType nSurfPts = surface.numVertices();
  
  if(nPolyDataPts != nSurfPts){
    throw AtlasWerksException(__FILE__, __LINE__, 
			      "Error, cannot set polydata points from "
			      "surface with different number of points");
  }
  
  for (int vIdx = 0; vIdx < nPolyDataPts; ++vIdx) {
    const Vector3D<double> &v = surface.vertices[vIdx];
    points->SetPoint(vIdx, &v.x);
  }
}

// Change surface in place so that all faces are oriented
// counterclockwise as viewed from the outside.  Surface is assumed to
// be closed (i.e., to have no holes and to contain a volume).
void 
FixSurfaceOrientation(Surface& surface)
{

    // Convert surface to vtk
    vtkPolyData* polyData = vtkPolyData::New();
    SurfaceToVtkPolyData(surface, polyData);

    // Merge vertices so that orientation can be handled
    vtkCleanPolyData* cleanPolyData = vtkCleanPolyData::New();
    cleanPolyData->SetInputData(polyData);
    cleanPolyData->SetTolerance(0.0);

    cleanPolyData->Update();

    // Fix normals
    vtkPolyDataNormals* polyDataNormals = vtkPolyDataNormals::New();
    polyDataNormals->SetInputData( cleanPolyData->GetOutput() );
    polyDataNormals->SplittingOff();
    polyDataNormals->ComputePointNormalsOff();
    polyDataNormals->ComputeCellNormalsOn();

    // Do the work
    polyDataNormals->Update();

    // Convert back to Surface data type
    VtkPolyDataToSurface(polyData, surface);

    // Clean up
    polyData->Delete();
    cleanPolyData->Delete();
    polyDataNormals->Delete();
    
}

// Change surface in place so that all faces are oriented
// counterclockwise as viewed from the outside.  Surface is assumed to
// be closed (i.e., to have no holes and to contain a volume).
void 
FixSurfaceOrientationByu(const char* in, const char* out)
{
    // Read file
    vtkBYUReader* reader = vtkBYUReader::New();
    reader->SetFileName(in);

    reader->Update();

    // Merge vertices so that orientation can be handled
    vtkCleanPolyData* cleanPolyData = vtkCleanPolyData::New();
    cleanPolyData->SetInputData( reader->GetOutput() );
    cleanPolyData->SetTolerance(0.0);

    cleanPolyData->Update();

    // Fix normals
    vtkPolyDataNormals* polyDataNormals = vtkPolyDataNormals::New();
    polyDataNormals->SetInputData( cleanPolyData->GetOutput() );
    polyDataNormals->SplittingOff();
    polyDataNormals->ComputePointNormalsOff();
    polyDataNormals->ComputeCellNormalsOn();

    // Do the work
    polyDataNormals->Update();

    //Write file
    vtkBYUWriter* writer = vtkBYUWriter::New();
    writer->SetInputData( polyDataNormals->GetOutput() );
    writer->SetGeometryFileName(out);

    writer->Update();

    // Clean up
    reader->Delete();
    cleanPolyData->Delete();
    polyDataNormals->Delete();
    writer->Delete();
    
}

void
VertexNiceness(vtkPolyData* polyData,
               unsigned int vertexIndex,
               bool& isManifold,
               bool& isOriented,
               bool& isClosed,
               bool& allTriangles)
{
    vtkIdList* allIncidentCells = vtkIdList::New();
    polyData->GetPointCells(vertexIndex, allIncidentCells);
    std::vector<unsigned int> incidentPolys;
    for (int i = 0; i < allIncidentCells->GetNumberOfIds(); ++i) {
        unsigned int id = allIncidentCells->GetId(i);
        int cellType = polyData->GetCellType(id);
        cellType += 1;
    }
}

void
GetSurfaceNiceness( vtkPolyData* polyData,
                    bool& isManifold,
                    bool& isOriented,
                    bool& isClosed,
                    bool& allTriangles)
{
    vtkPolyData* newMesh = vtkPolyData::New();
    newMesh->SetPoints( polyData->GetPoints() );
    vtkCellArray* polys = polyData->GetPolys();
    vtkCellArray* newPolys = vtkCellArray::New();
    newPolys->DeepCopy(polys);
    newMesh->SetPolys(newPolys);
    newMesh->BuildLinks();
}
