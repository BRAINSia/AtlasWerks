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

#ifndef __VTK_UTILS_H__
#define __VTK_UTILS_H__

#include<vtkImageData.h>
#include<vtkPointData.h>
#include<vtkFloatArray.h>

#include "AtlasWerksTypes.h"

class VtkUtils
{
public:

  static
  vtkImageData* 
  arrayToVtkImageData(const Array3D<Real>& array, 
		      const OriginType &origin = OriginType(0.0, 0.0, 0.0),
		      const SpacingType &spacing = SpacingType(1.0, 1.0, 1.0));

  static
  vtkImageData* 
  ImageToVtkImageData(const RealImage& image) ;

  static
  vtkImageData* 
  VectorFieldToVtkImageData(const VectorField& array, 
			    const OriginType &origin = OriginType(0.0, 0.0, 0.0),
			    const SpacingType &spacing = SpacingType(1.0, 1.0, 1.0));
    
};

#endif // __VTK_UTILS_H__
