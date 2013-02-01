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

#ifndef __CUDA_ACCUMULATE_H
#define __CUDA_ACCUMULATE_H

#include <cuda_runtime.h>
class cplVector3DArray;

template<class T>
void cplAccumulate(T* d_o, T** d_i, int nImg, int nP, void* temp, cudaStream_t stream=NULL);

template<class T>
void cplAccumulate(T* d_odata, T* d_idata, unsigned int nP, unsigned int pitch,
                    unsigned int nImgs, cudaStream_t stream=NULL);

void cplAccumulate(cplVector3DArray& d_o, cplVector3DArray* d_i, int nImg, int nP, void* temP, cudaStream_t stream=NULL);

void testAccumulate(int w, int h, int l, int nImgs);
#endif
