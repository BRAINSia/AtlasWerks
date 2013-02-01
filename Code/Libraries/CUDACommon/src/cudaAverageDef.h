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

#ifndef __CUDA_AVERAGE_DEF_H
#define __CUDA_AVERAGE_DEF_H

#include "VectorMath.h"
#include <cplMacro.h>
#include <cutil_comfunc.h>
#include <cudaInterface.h>

/////////////////////////////////////////////////////////////////////////////////
// Average functions
//////////////////////////////////////////////////////////////////////////////////
template<class T>
__global__ void cplAvg_kernel(T* d_o, const T* d_i, const T* d_i1, int n){
    const uint blockId = get_blockID();
    unsigned int   id = get_threadID(blockId);

    if (id < n){
        d_o[id] = (d_i[id] + d_i1[id])/2 ;
    }
}

template<class T>
void cplAvg(T* d_o, const T* d_i, const T* d_i1, int nP, cudaStream_t stream)
{
    dim3 threads(256);
    dim3 grids(iDivUp(nP, threads.x));
    checkConfig(grids);
    cplAvg_kernel<<<grids, threads, 0, stream >>>(d_o, d_i, d_i1, nP);
}

/////////////////////////////////////////////////////////////////////////////////

template<class T>
__global__ void cplAvg_kernel(T* d_o, const T* d_i, const T* d_i1, const T* d_i2, int n){
    const uint blockId = get_blockID();
    unsigned int   id = get_threadID(blockId);
    
    if (id < n){
        d_o[id] = (d_i[id] + d_i1[id] + d_i2[id])/3;
    }
}

template<class T>
void cplAvg(T* d_o, const T* d_i, const T* d_i1, const T* d_i2, int nP, cudaStream_t stream)
{
    dim3 threads(256);
    dim3 grids(iDivUp(nP, threads.x));
    checkConfig(grids);
    cplAvg_kernel<<<grids, threads, 0, stream >>>(d_o, d_i, d_i1, d_i2, nP);
}

/////////////////////////////////////////////////////////////////////////////////
template<class T>
__global__ void cplAvg_kernel(T* d_o,
                               const T* d_i, const T* d_i1,
                               const T* d_i2,const T* d_i3, 
                               int n){
    const uint blockId = get_blockID();
    unsigned int   id = get_threadID(blockId);
    if (id < n){
        d_o[id] = (d_i[id] + d_i1[id] + d_i2[id] + d_i3[id])/4;
    }
}

template<class T>
void cplAvg(T* d_o, const T* d_i, const T* d_i1, const T* d_i2, const T* d_i3, int nP, cudaStream_t stream)
{
    dim3 threads(256);
    dim3 grids(iDivUp(nP, threads.x));
    checkConfig(grids);
    cplAvg_kernel<<<grids, threads, 0, stream >>>(d_o, d_i, d_i1, d_i2, d_i3, nP);
}
/////////////////////////////////////////////////////////////////////////////////


template<class T>
__global__ void cplAvg_kernel(T* d_o,
                               const T* d_i, const T* d_i1,
                               const T* d_i2,const T* d_i3, const T* d_i4,
                               int n){
    const uint blockId = get_blockID();
    unsigned int   id = get_threadID(blockId);
    if (id < n){
        d_o[id] = (d_i[id] + d_i1[id] + d_i2[id] + d_i3[id] + d_i4[id])/5;
    }
}

template<class T>
void cplAvg(T* d_o, const T* d_i, const T* d_i1, const T* d_i2, const T* d_i3, const T* d_i4, int nP, cudaStream_t stream)
{
    dim3 threads(256);
    dim3 grids(iDivUp(nP, threads.x));
    checkConfig(grids);
    cplAvg_kernel<<<grids, threads, 0, stream >>>(d_o, d_i, d_i1, d_i2, d_i3, d_i4, nP);
}
/////////////////////////////////////////////////////////////////////////////////

template<class T>
__global__ void cplAvg_kernel(T* d_o,
                               const T* d_i, const T* d_i1, const T* d_i2,
                               const T* d_i3,const T* d_i4, const T* d_i5,
                               int n){
    const uint blockId = get_blockID();
    unsigned int   id = get_threadID(blockId);
    if (id < n){
        d_o[id] = (d_i[id] + d_i1[id] + d_i2[id] + d_i3[id] + d_i4[id] + d_i5[id])/6;
    }
}

template<class T>
void cplAvg(T* d_o, const T* d_i, const T* d_i1, const T* d_i2, const T* d_i3, const T* d_i4, const T* d_i5, int nP, cudaStream_t stream)
{
    dim3 threads(256);
    dim3 grids(iDivUp(nP, threads.x));
    checkConfig(grids);
    cplAvg_kernel<<<grids, threads, 0, stream >>>(d_o, d_i, d_i1, d_i2, d_i3, d_i4, d_i5, nP);
}
/////////////////////////////////////////////////////////////////////////////////

template<class T>
__global__ void cplAvg_kernel(T* d_o,
                               const T* d_i, const T* d_i1, const T* d_i2,
                               const T* d_i3,const T* d_i4, const T* d_i5, const T* d_i6,
                               int n){
    const uint blockId = get_blockID();
    unsigned int   id = get_threadID(blockId);
    if (id < n){
        d_o[id] = (d_i[id] + d_i1[id] + d_i2[id] + d_i3[id] + d_i4[id] + d_i5[id] + d_i6[id])/7;
    }
}

template<class T>
void cplAvg(T* d_o, const T* d_i, const T* d_i1, const T* d_i2, const T* d_i3, const T* d_i4, const T* d_i5, const T* d_i6, int nP, cudaStream_t stream)
{
    dim3 threads(256);
    dim3 grids(iDivUp(nP, threads.x));
    checkConfig(grids);
    cplAvg_kernel<<<grids, threads, 0, stream >>>(d_o, d_i, d_i1, d_i2, d_i3, d_i4, d_i5, d_i6, nP);
}

/////////////////////////////////////////////////////////////////////////////////
template<class T>
__global__ void cplAvg_kernel(T* d_o,
                               const T* d_i, const T* d_i1, const T* d_i2, const T* d_i3,
                               const T* d_i4,const T* d_i5, const T* d_i6, const T* d_i7,
                               int n){
    const uint blockId = get_blockID();
    unsigned int   id = get_threadID(blockId);
    if (id < n){
        d_o[id] = (d_i[id] + d_i1[id] + d_i2[id] + d_i3[id] + d_i4[id] + d_i5[id] + d_i6[id] + d_i7[id])/8;
    }
}

template<class T>
void cplAvg(T* d_o, const T* d_i, const T* d_i1, const T* d_i2, const T* d_i3, const T* d_i4, const T* d_i5, const T* d_i6, const T* d_i7,  int nP, cudaStream_t stream)
{
    dim3 threads(256);
    dim3 grids(iDivUp(nP, threads.x));
    checkConfig(grids);
    cplAvg_kernel<<<grids, threads, 0, stream >>>(d_o, d_i, d_i1, d_i2, d_i3, d_i4, d_i5, d_i6, d_i7, nP);
}

/////////////////////////////////////////////////////////////////////////////////
template<class T>
void cplAvg_small(T* d_o, T* d_i[], int nImg, int nP, cudaStream_t stream){

    if (nImg == 1)
        cudaMemcpy(d_o, d_i[0], nP * sizeof(T), cudaMemcpyDeviceToDevice);
    else if (nImg == 2)
        cplAvg(d_o, d_i[0], d_i[1], nP, stream);
    else if (nImg == 3)
        cplAvg(d_o, d_i[0], d_i[1], d_i[2], nP, stream);
    else if (nImg == 4)
        cplAvg(d_o, d_i[0], d_i[1], d_i[2], d_i[3], nP, stream);
    else if (nImg == 5)
        cplAvg(d_o, d_i[0], d_i[1], d_i[2], d_i[3], d_i[4], nP, stream);
    else if (nImg == 6)
        cplAvg(d_o, d_i[0], d_i[1], d_i[2], d_i[3], d_i[4], d_i[5], nP, stream);
    else if (nImg == 7)
        cplAvg(d_o, d_i[0], d_i[1], d_i[2], d_i[3], d_i[4], d_i[5], d_i[6], nP, stream);
    else if (nImg == 8)
        cplAvg(d_o, d_i[0], d_i[1], d_i[2], d_i[3], d_i[4], d_i[5], d_i[6], d_i[7], nP, stream);
}


template <class T>
void cplAverage(T* odata, T** idata, int nImg, int nP, void* temp, cudaStream_t stream){
    if (nImg <= 8){
        cplAvg_small(odata, idata, nImg, nP, stream);
    }
    else {
        int hasTemp = (temp != NULL);
        T* d_temp;
        if (hasTemp)
            d_temp = (T*) temp;
        else
            dmemAlloc(d_temp, nP);
                
        int nS = nImg >> 3;
        int nA = nImg & 0x07;
        cplAvg(odata, idata[0], idata[1], idata[2], idata[3], idata[4], idata[5], idata[6], idata[7], nP, stream);
        
        for (int i=1; i < nS; ++i){
            cplAvg(d_temp, idata[i*8], idata[i*8+1], idata[i*8+2], idata[i*8+3],
                    idata[i*8+4], idata[i*8+5], idata[i*8+6], idata[i*8+7], nP, stream);
            cplVectorOpers::Add_I(odata, d_temp, nP, stream);
        }

        if (nA ==0){
            cplVectorOpers::MulC_I(odata, 1.f / nS, nP, stream);
        }else {
            cplAvg_small(d_temp, idata + (nImg - nA), nA, nP, stream);
            cplVectorOpers::MulC_Add_MulC_I(odata, float(8) / nImg, d_temp, (float)nA / nImg, nP, stream);
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
__global__ void cplAverage_kernel(T* d_odata, const T* d_idata, unsigned int nP, unsigned int pitch, unsigned int nImgs)
{
    const uint blockId= get_blockID();
    unsigned int   id = get_threadID(blockId);
    
    if (id < nP){
        T sum(0);
        uint iid = id;
        for (int i=0; i< nImgs; ++i, iid+= pitch)
            sum += d_idata[iid];
        d_odata[id] = sum / nImgs;
    }
}

template<class T>
void cplAverage(T* d_odata, T* d_idata, unsigned int nP, unsigned int pitch,
                 unsigned int nImgs, cudaStream_t stream)
{
    dim3 threads(256);
    dim3 grids(iDivUp(nP, threads.x));
    checkConfig(grids);
    
    cplAverage_kernel<<<grids, threads, 0, stream>>>(d_odata, d_idata, nP, pitch, nImgs);
}

#include <cudaVector3DArray.h>
void cplAverage(cplVector3DArray& d_o, const cplVector3DArray* d_i, int nImg, int nP, void* temp, cudaStream_t stream)
{
    float** d_fArray = new float* [nImg];

    for (int i=0; i < nImg; ++i) d_fArray[i] = d_i[i].x;
    cplAverage(d_o.x, d_fArray, nImg, nP, temp, stream);

    for (int i=0; i < nImg; ++i) d_fArray[i] = d_i[i].y;
    cplAverage(d_o.y, d_fArray, nImg, nP, temp, stream);

    for (int i=0; i < nImg; ++i) d_fArray[i] = d_i[i].z;
    cplAverage(d_o.z, d_fArray, nImg, nP, temp, stream);
    
    delete []d_fArray;
};

#endif
