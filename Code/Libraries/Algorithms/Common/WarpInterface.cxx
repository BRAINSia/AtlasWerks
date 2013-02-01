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


#include "WarpInterface.h"

#ifdef MPI_ENABLED
#include <mpi.h>
#endif

#include "log.h"

WarpInterface::
WarpInterface(const RealImage *I0, const RealImage *I1)
  : mI0Orig(I0),
    mI1Orig(I1)
{
  mImSize = mI0Orig->getSize();
  if(mI1Orig->getSize() != mImSize){
    throw AtlasWerksException(__FILE__, __LINE__, "Error, image sizes not equal");
  }
  mImOrigin = mI0Orig->getOrigin();
  if(mI1Orig->getOrigin() != mImOrigin){
    throw AtlasWerksException(__FILE__, __LINE__, "Error, image origins not equal");
  }
  mImSpacing = mI0Orig->getSpacing();
  if(mI1Orig->getSpacing() != mImSpacing){
    throw AtlasWerksException(__FILE__, __LINE__, "Error, image spacings not equal");
  }
}
  
WarpInterface::
~WarpInterface()
{
}



