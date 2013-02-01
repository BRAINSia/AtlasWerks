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

#include <cudaScratch.h>

cplScratchI::cplScratchI(const Vector3Di& size, int n)
{
    imSize = size;
    nImgs = n;
    d_Imgs = new float* [nImgs];

    for (int i=0; i < n; ++i){
        dmemAlloc(d_Imgs[i], size.productOfElements());
    }
}

cplScratchI::~cplScratchI(){
    for (int i=0; i < n; ++i){
        dmemFree(d_Imgs[i]);
    }
    delete []d_Imgs;
}


cplScratchV::cplScratchV(const Vector3Df& size, int n){
    imSize = size;
    nFields= n;

    d_Fields = new cplVector3DArray [nFields];
    for (int i=0; i < n; ++i){
        allocateDeviceVector3DArray(d_Fields[i], size.productOfElements());
    }
}

cplScratchI::~cplScratchI(){
    for (int i=0; i < n; ++i){
        freeDeviceVector3DArray(d_Fields[i]);
    }
    delete []d_Fields;
}
