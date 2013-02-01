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

#ifndef _vectorDownsample_h
#define _vectorDownsample_h

#include "Image.h"

#include "vnl/vnl_vector.h"

Image< vnl_vector<float> >*
vectorDownsample(
  Image< vnl_vector<float> >* image,
  double factor,
  double sigma,
  double kernelSize
);

#endif
