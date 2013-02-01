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

#ifndef __CUDA_ACCUMULATE_DEF_H
#define __CUDA_ACCUMULATE_DEF_H

#include "VectorMath.h"
#include <cplMacro.h>
#include <cutil_comfunc.h>
#include <cudaInterface.h>

/////////////////////////////////////////////////////////////////////////////////
// Adding set of image 
//////////////////////////////////////////////////////////////////////////////////
template<class T>
__global__ void cplAccumulate_kernel(T* d_o, const T* d_i, const T* d_i1, int n){
    uint blockId = get_blockID();
    uint      id = get_threadID(blockId);
    if (id < n){
        d_o[id] = d_i[id] + d_i1[id];
    }
}

template<class T>
void cplAccumulate(T* d_o, const T* d_i, const T* d_i1, int n, cudaStream_t stream){
    dim3 threads(256);
    dim3 grids(iDivUp(n,threads.x));
    checkConfig(grids);
    cplAccumulate_kernel<<<grids, threads,0,stream>>>(d_o, d_i, d_i1, n);
}

template<class T>
__global__ void cplAccumulate_kernel(T* d_o, const T* d_i, const T* d_i1, const T* d_i2, int n){
    uint blockId = get_blockID();
    uint      id = get_threadID(blockId);
    if (id < n){
        d_o[id] = d_i[id] + d_i1[id] + d_i2[id];
    }
}

template<class T>
void cplAccumulate(T* d_o, const T* d_i, const T* d_i1, const T* d_i2, int n, cudaStream_t stream){
    dim3 threads(256);
    dim3 grids(iDivUp(n,threads.x));
    checkConfig(grids);
    cplAccumulate_kernel<<<grids, threads,0,stream>>>(d_o, d_i, d_i1, d_i2, n);
}


template<class T>
__global__ void cplAccumulate_kernel(T* d_o,
                                      const T* d_i, const T* d_i1,
                                      const T* d_i2,const T* d_i3, 
                                      int n){
    uint blockId = get_blockID();
    uint      id = get_threadID(blockId);
    if (id < n){
        d_o[id] = d_i[id] + d_i1[id] + d_i2[id] + d_i3[id];
    }
}

template<class T>
void cplAccumulate(T* d_o, const T* d_i, const T* d_i1, const T* d_i2, const T* d_i3, int n, cudaStream_t stream){
    dim3 threads(256);
    dim3 grids(iDivUp(n,threads.x));
    checkConfig(grids);
    cplAccumulate_kernel<<<grids, threads,0,stream>>>(d_o, d_i, d_i1, d_i2, d_i3, n);
}


template<class T>
__global__ void cplAccumulate_kernel(T* d_o,
                               const T* d_i, const T* d_i1,
                               const T* d_i2,const T* d_i3, const T* d_i4,
                               int n){
    uint blockId = get_blockID();
    uint      id = get_threadID(blockId);
    if (id < n){
        d_o[id] = d_i[id] + d_i1[id] + d_i2[id] + d_i3[id] + d_i4[id];
    }
}

template<class T>
void cplAccumulate(T* d_o, const T* d_i, const T* d_i1, const T* d_i2, const T* d_i3, const T* d_i4, int n, cudaStream_t stream){
    dim3 threads(256);
    dim3 grids(iDivUp(n,threads.x));
    checkConfig(grids);
    cplAccumulate_kernel<<<grids, threads,0,stream>>>(d_o, d_i, d_i1, d_i2, d_i3, d_i4, n);
}


template<class T>
__global__ void cplAccumulate_kernel(T* d_o,
                               const T* d_i, const T* d_i1, const T* d_i2,
                               const T* d_i3,const T* d_i4, const T* d_i5,
                               int n){
    uint blockId = get_blockID();
    uint      id = get_threadID(blockId);
    if (id < n){
        d_o[id] = d_i[id] + d_i1[id] + d_i2[id] + d_i3[id] + d_i4[id] + d_i5[id];
    }
}

template<class T>
void cplAccumulate(T* d_o, const T* d_i, const T* d_i1, const T* d_i2, const T* d_i3, const T* d_i4, const T* d_i5, int n, cudaStream_t stream){
    dim3 threads(256);
    dim3 grids(iDivUp(n,threads.x));
    checkConfig(grids);
    cplAccumulate_kernel<<<grids, threads,0,stream>>>(d_o, d_i, d_i1, d_i2, d_i3, d_i4, d_i5, n);
}

template<class T>
__global__ void cplAccumulate_kernel(T* d_o,
                               const T* d_i, const T* d_i1, const T* d_i2,
                               const T* d_i3,const T* d_i4, const T* d_i5, const T* d_i6,
                               int n){
    uint blockId = get_blockID();
    uint      id = get_threadID(blockId);
    if (id < n){
        d_o[id] = d_i[id] + d_i1[id] + d_i2[id] + d_i3[id] + d_i4[id] + d_i5[id] + d_i6[id];
    }
}

template<class T>
void cplAccumulate(T* d_o, const T* d_i, const T* d_i1, const T* d_i2, const T* d_i3, const T* d_i4, const T* d_i5, const T* d_i6, int n, cudaStream_t stream){
    dim3 threads(256);
    dim3 grids(iDivUp(n,threads.x));
    checkConfig(grids);
    cplAccumulate_kernel<<<grids, threads,0,stream>>>(d_o, d_i, d_i1, d_i2, d_i3, d_i4, d_i5, d_i6, n);
}


template<class T>
__global__ void cplAccumulate_kernel(T* d_o,
                               const T* d_i, const T* d_i1, const T* d_i2, const T* d_i3,
                               const T* d_i4,const T* d_i5, const T* d_i6, const T* d_i7,
                               int n){
    uint blockId = get_blockID();
    uint      id = get_threadID(blockId);
    if (id < n){
        d_o[id] = d_i[id] + d_i1[id] + d_i2[id] + d_i3[id] + d_i4[id]
            + d_i5[id] + d_i6[id] + d_i7[id];
    }
}

template<class T>
void cplAccumulate(T* d_o, const T* d_i, const T* d_i1, const T* d_i2, const T* d_i3, const T* d_i4, const T* d_i5, const T* d_i6, const T* d_i7, int n, cudaStream_t stream){
    dim3 threads(256);
    dim3 grids(iDivUp(n,threads.x));
    checkConfig(grids);
    cplAccumulate_kernel<<<grids, threads,0,stream>>>(d_o, d_i, d_i1, d_i2, d_i3, d_i4, d_i5, d_i6, d_i7, n);
}


template<class T>
void cplAccumulate_small(T* d_o, T** d_i, int nImg, int n, cudaStream_t stream){
    dim3 threads(256);
    dim3 grids(iDivUp(n, threads.x));
    checkConfig(grids);

    if (nImg == 1){
        cudaMemcpy(d_o, d_i[0], sizeof(T) * n, cudaMemcpyDeviceToDevice);
    } else if (nImg == 2){
        cplAccumulate(d_o, d_i[0], d_i[1], n, stream);
    } else if (nImg == 3){
        cplAccumulate(d_o, d_i[0], d_i[1], d_i[2], n, stream);
    } else if (nImg == 4){
        cplAccumulate(d_o, d_i[0], d_i[1], d_i[2], d_i[3], n, stream);
    } else if (nImg == 5){
        cplAccumulate(d_o, d_i[0], d_i[1], d_i[2], d_i[3], d_i[4], n, stream);
    } else if (nImg == 6){
        cplAccumulate(d_o, d_i[0], d_i[1], d_i[2], d_i[3], d_i[4], d_i[5], n, stream);
    } else if (nImg == 7){
        cplAccumulate(d_o, d_i[0], d_i[1], d_i[2], d_i[3], d_i[4], d_i[5], d_i[6], n, stream);
    } else if (nImg == 8){
        cplAccumulate(d_o, d_i[0], d_i[1], d_i[2], d_i[3], d_i[4], d_i[5], d_i[6], d_i[7], n, stream);
    }
}

template<class T>
void cplAccumulate(T* d_o, T** d_i, int nImg, int n, void* temp, cudaStream_t stream){
    if (nImg <= 8)
        cplAccumulate_small(d_o, d_i, nImg, n, stream);
    else {
        int hasTemp = (temp != NULL);

        T* d_temp;
        if (hasTemp)
            d_temp = (T*) temp;
        else
            dmemAlloc(d_temp, n);

        int nA = nImg >> 3;
        int sA = nImg & 0x07;

        int i=0;
        cplAccumulate(d_o, d_i[i*8], d_i[i*8+1], d_i[i*8+2], d_i[i*8+3],
                       d_i[i*8+4], d_i[i*8+5], d_i[i*8+6], d_i[i*8+7], n, stream);
        if  (nA >1) {
            for (i=1; i < nA; ++i){
                cplAccumulate(d_temp, d_i[i*8], d_i[i*8+1], d_i[i*8+2], d_i[i*8+3],
                               d_i[i*8+4], d_i[i*8+5], d_i[i*8+6], d_i[i*8+7], n, stream);
                cplVectorOpers::Add_I(d_o, d_temp, n, stream);
            }
        }

        if (sA > 0) {
            cplAccumulate_small(d_temp, d_i + nImg - sA, sA, n, stream);
            cplVectorOpers::Add_I(d_o, d_temp, n, stream);
        }
        if (!hasTemp) {
            dmemFree(d_temp);
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
///
////////////////////////////////////////////////////////////////////////////////
template<class T>
__global__ void cplAccumulate_kernel(T* d_odata, const T* d_idata, unsigned int nP, unsigned int pitch, unsigned int nImgs)
{
    const uint blockId= get_blockID();
    unsigned int   id = get_threadID(blockId);
    
    if (id < nP){
        T sum(0);
        uint iid = id;
        for (int i=0; i< nImgs; ++i, iid+= pitch)
            sum += d_idata[iid];
        d_odata[id] = sum;
    }
}

template<class T>
void cplAccumulate(T* d_odata, T* d_idata, unsigned int nP, unsigned int pitch,
                    unsigned int nImgs, cudaStream_t stream)
{
    dim3 threads(256);
    dim3 grids(iDivUp(nP, threads.x));
    checkConfig(grids);
    cplAccumulate_kernel<<<grids, threads, 0, stream>>>(d_odata, d_idata, nP, pitch, nImgs);
}

#include <cudaVector3DArray.h>
void cplAccumulate(cplVector3DArray& d_o, cplVector3DArray* d_i, int nImg, int nP, void* temp, cudaStream_t stream)
{
    float** d_fArray = new float* [nImg];

    for (int i=0; i < nImg; ++i) d_fArray[i] = d_i[i].x;
    cplAccumulate(d_o.x, d_fArray, nImg, nP, temp, stream);

    for (int i=0; i < nImg; ++i) d_fArray[i] = d_i[i].y;
    cplAccumulate(d_o.y, d_fArray, nImg, nP, temp, stream);

    for (int i=0; i < nImg; ++i) d_fArray[i] = d_i[i].z;
    cplAccumulate(d_o.z, d_fArray, nImg, nP, temp, stream);

    delete []d_fArray;
};

#endif
