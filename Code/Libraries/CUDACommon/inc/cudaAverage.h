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

#ifndef __CUDA_AVERAGE_H
#define __CUDA_AVERAGE_H

#include <cuda_runtime.h>
class cplVector3DArray;

/**
 * Average a series of arrays (d_o = (d_i[0] + d_i[1] + ... + d_i[nP-1])/nP).
 * 'temp' is a temporary scratch array, will be allocated if NULL is
 * passed in.
 */
template<class T>
void cplAverage(T* d_o, T** d_i, int nImg, int nP, void* temp, cudaStream_t stream=NULL);

template<class T>
void cplAverage(T* d_odata, T* d_idata, unsigned int nP, unsigned int pitch,
                 unsigned int nImgs, cudaStream_t stream=NULL);

void cplAverage(cplVector3DArray& d_o, const cplVector3DArray* d_i, int nImg, int nP, void* temp, cudaStream_t stream=NULL);

void testAverage(int w, int h, int l, int nImgs);
                
    
#endif
