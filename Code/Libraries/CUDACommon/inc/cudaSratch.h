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

#ifndef __CUDA_SCRATCH_H
#define __CUDA_SCRATCH_H

#include <Vector3D.h>

class cplVector3DArray;
struct cplScratchI{
    cplScratchI(const Vector3Di& size, int n);
    ~cplScratchI();
    float**      d_Imgs;
    Vector3Di    imSize;
    unsigned int nImgs;
};

struct cplScratchV{
    cplScratchV(const Vector3Di& size, int n);
    ~cplScratchV();
    cplVector3DArray*  d_Fields;
    Vector3Di          imSize;
    unsigned int       nFields;
};

#endif
