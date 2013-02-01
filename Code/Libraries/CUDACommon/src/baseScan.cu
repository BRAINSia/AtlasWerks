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

#ifndef __BASE_CUDA_CU
#define __BASE_CUDA_CU

#include "typeConvert.h"

#define WARP_SIZE 32
#define LOG_WARP_SIZE 5

#ifdef __DEVICE_EMULATION__
#define __SYNC __syncthreads();
#else
#define __SYNC
#endif

#if (CUDART_VERSION  < 2020)
template<class T, int maxlevel>
__device__ T scanwarp(T val, T* sData)
{
    // The following is the same as 2 * RadixSort::WARP_SIZE * warpId + threadInWarp = 
    // 64*(threadIdx.x >> 5) + (threadIdx.x & (RadixSort::WARP_SIZE - 1))
    int idx = 2 * threadIdx.x - (threadIdx.x & (WARP_SIZE - 1));
    sData[idx] = 0;
    idx += WARP_SIZE;
    sData[idx] = val;          __SYNC

#ifdef __DEVICE_EMULATION__
        T t = sData[idx -  1]; __SYNC 
        sData[idx] += t;       __SYNC
        t = sData[idx -  2];   __SYNC 
        sData[idx] += t;       __SYNC
        t = sData[idx -  4];   __SYNC 
        sData[idx] += t;       __SYNC
        t = sData[idx -  8];   __SYNC 
        sData[idx] += t;       __SYNC
        t = sData[idx - 16];   __SYNC 
        sData[idx] += t;       __SYNC
#else
        if (0 <= maxlevel) { sData[idx] += sData[idx - 1]; } __SYNC
        if (1 <= maxlevel) { sData[idx] += sData[idx - 2]; } __SYNC
        if (2 <= maxlevel) { sData[idx] += sData[idx - 4]; } __SYNC
        if (3 <= maxlevel) { sData[idx] += sData[idx - 8]; } __SYNC
        if (4 <= maxlevel) { sData[idx] += sData[idx -16]; } __SYNC
#endif

        return sData[idx] - val;  // convert inclusive -> exclusive
}
#else
template<class T, int maxlevel>
__device__ T scanwarp(T val, volatile T* sData)
{
    // The following is the same as 2 * RadixSort::WARP_SIZE * warpId + threadInWarp = 
    // 64*(threadIdx.x >> 5) + (threadIdx.x & (RadixSort::WARP_SIZE - 1))
    int idx = 2 * threadIdx.x - (threadIdx.x & (WARP_SIZE - 1));
    sData[idx] = 0;
    idx += WARP_SIZE;
    T t = sData[idx] = val;    __SYNC

#ifdef __DEVICE_EMULATION__
        T t = sData[idx -  1]; __SYNC 
        sData[idx] += t;       __SYNC
        t = sData[idx -  2];   __SYNC 
        sData[idx] += t;       __SYNC
        t = sData[idx -  4];   __SYNC 
        sData[idx] += t;       __SYNC
        t = sData[idx -  8];   __SYNC 
        sData[idx] += t;       __SYNC
        t = sData[idx - 16];   __SYNC 
        sData[idx] += t;       __SYNC
#else
        if (0 <= maxlevel) { sData[idx] = t = t + sData[idx - 1]; } __SYNC
        if (1 <= maxlevel) { sData[idx] = t = t + sData[idx - 2]; } __SYNC
        if (2 <= maxlevel) { sData[idx] = t = t + sData[idx - 4]; } __SYNC
        if (3 <= maxlevel) { sData[idx] = t = t + sData[idx - 8]; } __SYNC
        if (4 <= maxlevel) { sData[idx] = t = t + sData[idx -16]; } __SYNC
#endif
       return t - val;  // convert inclusive -> exclusive
}
#endif

#endif
