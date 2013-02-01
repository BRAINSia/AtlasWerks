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

#ifndef SURFACE_UTILS_VTK_H
#define SURFACE_UTILS_VTK_H

#include <vtkPolyData.h>

#include <Surface.h>

/**
 * Creates a vtkPolyData object by inserting Surface vertices as both
 * points and visible vertices, and creating polys from Surface polys
 */
void SurfaceToVtkPolyData(const Surface& surface, vtkPolyData* polyData);
/**
 * This function creates Surface object from vertices and all 'poly'
 * cells from vtkPolyData
 */
void VtkPolyDataToSurface(vtkPolyData* polyData, Surface& surface);
/**
 * Sets vertices of vtkPolyData to vertex locations of Surface.
 * Number of vertices must be the same in both.  Used to modify vertex
 * locations via a surface, then write back the modification to a
 * vtkPolyData without modifying any other information.
 */
void SetVtkPolyDataPoints(vtkPolyData* polyData, const Surface& surface);
void FixSurfaceOrientation(Surface& surface);
void FixSurfaceOrientationByu(const char* in, const char* out);

#endif // ndef SURFACE_UTILS_VTK_H
