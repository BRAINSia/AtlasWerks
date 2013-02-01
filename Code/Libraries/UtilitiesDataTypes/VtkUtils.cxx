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

#include "VtkUtils.h"

vtkImageData* 
VtkUtils::
arrayToVtkImageData(const Array3D<Real>& array, 
		    const OriginType &origin,
		    const SpacingType &spacing)
{
  // allocate the vtk data array
  SizeType size = array.getSize();
  unsigned int nVox = size.productOfElements();
  vtkFloatArray *scalars = vtkFloatArray::New();
  scalars->SetNumberOfComponents(1);
  scalars->SetNumberOfTuples(nVox);

  const Real* dataPtr = array.getDataPointer();
  // copy in the data
  for(unsigned int k=0;k<nVox;k++){
    scalars->SetTupleValue(k, &dataPtr[k]);
  }
    
  vtkImageData *vtkData = vtkImageData::New();
  vtkData->GetPointData()->SetScalars(scalars);
  vtkData->SetDimensions(size.x, size.y, size.z);
  vtkData->SetScalarType(VTK_FLOAT);
  vtkData->SetOrigin(origin.x, origin.y, origin.z);
  vtkData->SetSpacing(spacing.x, spacing.y, spacing.z);

  return vtkData;
}

vtkImageData* 
VtkUtils::
ImageToVtkImageData(const RealImage& image) 
{
  return arrayToVtkImageData(image, 
			     image.getOrigin(),
			     image.getSpacing());
}

vtkImageData* 
VtkUtils::
VectorFieldToVtkImageData(const VectorField& array, 
			  const OriginType &origin,
			  const SpacingType &spacing)
{
  SizeType size = array.getSize();
  unsigned int nVox = size.productOfElements();
  vtkFloatArray *vectors = vtkFloatArray::New();
  vectors->SetNumberOfComponents(3);
  vectors->SetNumberOfTuples(nVox);
    
  // copy in the data
  const Vector3D<Real> *dataPtr = array.getDataPointer();
  for(unsigned int k=0;k<nVox;k++){
    vectors->SetTupleValue(k, &(dataPtr[k].x));
  }
    
  vtkImageData *vtkData = vtkImageData::New();
  vtkData->GetPointData()->SetScalars(vectors);
  vtkData->SetDimensions(size.x, size.y, size.z);
  vtkData->SetScalarType(VTK_FLOAT);
  vtkData->SetOrigin(origin.x, origin.y, origin.z);
  vtkData->SetSpacing(spacing.x, spacing.y, spacing.z);

  return vtkData;
}
