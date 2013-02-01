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


#ifndef _vectorITKConversions_h
#define _vectorITKConversions_h

#include "Image.h"

#include "itkVectorImage.h"

#include "vnl/vnl_vector.h"

// Convert Image to itk::VectorImage
itk::VectorImage<float>::Pointer
convertImageVolume(Image< vnl_vector<float> >* img);

// Convert itk::VectorImage to Image
Image< vnl_vector<float> >*
convertITKVolume(itk::VectorImage<float>* itkimg);

#endif
