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

#ifndef __SEGMENTED_SUM__CU
#define __SEGMENTED_SUM__CU

#include <cpl.h>

#include "bitSet.h"
#include "cudaScan.h"
#include "cudaMemoryPlan.h"
#include "cudaReduce.h"
#include "cudaSegmentedScanUtils.h"
#include <cpuImage3D.h>

/**
 * @brief Compute the number of segment for each 1024 data block inside an array
 *        each thread block can handle 1024 * 32 block
 * @param[in]  g_iflags 32bit flags array with 1 indicate the last of the segment  
 * @param[out] g_count  number of segment for each 32 bit block inside an array              
 */
__device__ __host__ uint popc(const uint flag){
    uint cnt;

    cnt  = flag - ((flag >> 1) & 0x55555555);
    cnt  = (cnt & 0x33333333) + ((cnt >> 2) & 0x33333333);
    cnt  = ((cnt + (cnt >> 4) & 0xF0F0F0F) * 0x1010101) >> 24;

    return cnt;
}

__global__ void segCount_kernel(uint* g_count, uint* g_iFlags, int n){
    const int WPAD = 32 + 1;

    uint blockId= blockIdx.x + blockIdx.y * gridDim.x;
    uint tid    = threadIdx.x;
    uint id     = tid + blockId * 1024;
    
    __shared__ uint s_data[32 * WPAD];
    
    for (int i=0; i<32; ++i)
        if (id + i * 32 < n)
            s_data[tid + i * WPAD] = g_iFlags[id + i * 32];
        else
            s_data[tid + i * WPAD] = 0;

    __syncthreads();
    uint cnt;
    for (int i=0; i<32; ++i){
        uint flag = s_data[tid + i*WPAD];
        cnt = popc(flag);
        s_data[tid + i*WPAD]= cnt;
    }
    __syncthreads();
    uint  sum = 0;
    uint* row = s_data + tid * WPAD;
    for (int i=0; i< 32; ++i)
        sum += row[i];
    g_count[blockId * 32 + tid] = sum;
}

void segCount(uint* g_count, uint* g_iFlags, int n){
    dim3 threads(32);
    uint nBlocks = iDivUp(n, 1024);
    dim3 grids(nBlocks);
    segCount_kernel<<<grids, threads>>>(g_count, g_iFlags, n);
}
//-----------------------------------------------------------------------------------------

template<class T>
__global__ void segSum_kernel_sparse(T* g_last, T* g_lSum, T* g_iData, uint* g_iFlags, uint* g_StartW , int n){

    const int WPAD = 32 + 1;
    uint blockId= blockIdx.x + blockIdx.y * gridDim.x;
    uint tid    = threadIdx.x;
    uint id     = tid + blockId * 1024;

    __shared__ uint s_cnt[64];
    __shared__ uint s_flags[32];
    __shared__ T    s_tmp[64];
    __shared__ T    s_data[32*WPAD];
    __shared__ uint s_start;
    
    if (tid == 0) 
        s_start = g_StartW[blockId];
    
    //1.Read data to shared mem
    uint flag= g_iFlags[tid + 32 * blockId];
    if ((tid + 32 * blockId) * 32 > n)
        flag = 0;
    
    uint i1  = 0;
    for (int i=0; i< 32 * 32; i+=32, i1+=WPAD)
        if (id + i < n)
            s_data[tid + i1] = g_iData[id + i];
        else
            s_data[tid + i1] = 0;

    __syncthreads();
    //2. Reduce row using H thread
    {
        T*row = s_data + tid * WPAD;
        T res = row[0];
        
        for (uint i= 1; i<32; ++i){
            res      += row[i];
            row[i]    = res;
            // reset value if needed
            res      *= ((flag & (1 <<i)) == 0);
        }
        
        s_tmp[tid] = res;
    }

    __syncthreads();
    
    // Mark the end of the block
    //1.b count the number of bit 1 = the number of segment inside compressed flag
    uint cnt = popc(flag);
    s_cnt[tid     ] = 0;        
    s_cnt[tid + 32] = cnt;
    s_flags[tid]    = (cnt > 0);

    __syncthreads();
    
    //1.c reduce the number of element counter
    uint idx = tid + 32;
    s_cnt[idx] += s_cnt[idx - 1 ];
    s_cnt[idx] += s_cnt[idx - 2 ];
    s_cnt[idx] += s_cnt[idx - 4 ];
    s_cnt[idx] += s_cnt[idx - 8 ];
    s_cnt[idx] += s_cnt[idx - 16];
        
    __syncthreads();

    
    //3. Segmented scan the sum
    if (tid ==0) {
        T res  = 0;
        for (uint i=0; i<32; ++i){
            T t = s_tmp[i];
            s_tmp[i] = res;
            res    = s_flags[i] ? t : res + t;
        }
        s_tmp[32] = res;
    }
    __syncthreads();
    if (tid == 0)
        g_last[blockId] = s_tmp[32];
    {
        T*row = s_data + tid * WPAD;
        // there's break inside
        if (cnt > 0) {
            int i = 0;
            int s = s_start + s_cnt[idx-1];
            while ((flag & (1 << i)) == 0) ++i;
            g_lSum[s++] = row[i] + s_tmp[tid];
            
            --cnt;
            while (cnt > 0){
                ++i;
                if ((flag & (1 << i)) > 0) {
                    g_lSum[s++] = row[i];
                    --cnt;
                }
            }
            
         }
    }
}


template<class T>
void segSumBlock_sparse(T* g_last, T* g_lSum, T* g_iData, uint* g_iFlags, uint* g_StartW, int n){
    dim3 threads(32);
    uint nBlocks = iDivUp(n, 1024);
    dim3 grids(nBlocks);
    segSum_kernel_sparse<T><<<grids, threads>>>(g_last, g_lSum, g_iData, g_iFlags, g_StartW, n);
}

//------------------------------------------------------------------------------------
template<class T>
__global__ void segSum_kernel_dense(T* g_last, T* g_lSum, T* g_iData, uint* g_iFlags, uint* g_StartW , int n){

    const int WPAD = 32 + 1;
    uint blockId= blockIdx.x + blockIdx.y * gridDim.x;
    uint tid    = threadIdx.x;
    uint id     = tid + blockId * 1024;

    __shared__ uint s_cnt[64];
    __shared__ uint s_flags[32];
    __shared__ T    s_tmp[64];
    __shared__ T    s_data[32*(32+1)];
    __shared__ T    s_buffer[512];
    __shared__ uint s_start;
    
    if (tid == 0) 
        s_start = g_StartW[blockId];
    
    //1.Read data to shared mem
    uint flag= g_iFlags[tid + 32 * blockId];
    if ((tid + 32 * blockId) * 32 > n)
        flag = 0;
    
    uint i1  = 0;
    for (int i=0; i< 32 * 32; i+=32, i1+=WPAD)
        if (id + i < n)
            s_data[tid + i1] = g_iData[id + i];
        else
            s_data[tid + i1] = 0;

    __syncthreads();
    //2. Reduce row using H thread
    {
        T*row = s_data + tid * WPAD;
        T res = row[0];
        
        for (uint i= 1; i<32; ++i){
            res      += row[i];
            row[i]    = res;
            // reset value if needed
            res      *= ((flag & (1 <<i)) == 0);
        }
        
        s_tmp[tid] = res;
    }

    __syncthreads();
    
    // Mark the end of the block
    //1.b count the number of bit 1 = the number of segment inside compressed flag
    uint cnt = popc(flag);

    s_cnt[tid     ] = 0;        
    s_cnt[tid + 32] = cnt;
    s_flags[tid]    = (cnt > 0);

    __syncthreads();
    
    //1.c reduce the number of element counter
    uint idx = tid + 32;
    s_cnt[idx] += s_cnt[idx - 1 ];
    s_cnt[idx] += s_cnt[idx - 2 ];
    s_cnt[idx] += s_cnt[idx - 4 ];
    s_cnt[idx] += s_cnt[idx - 8 ];
    s_cnt[idx] += s_cnt[idx - 16];
        
    __syncthreads();

    
    //3. Segmented scan the sum
    if (tid ==0) {
        T res  = 0;
        for (uint i=0; i<32; ++i){
            T t = s_tmp[i];
            s_tmp[i] = res;
            res    = s_flags[i] ? t : res + t;
        }
        s_tmp[32] = res;
    }
    __syncthreads();
    if (tid == 0)
        g_last[blockId] = s_tmp[32];

    {
        T*row = s_data + tid * WPAD;
        // there's break inside
        if (cnt > 0) {
            int i = 0;
            int s = s_cnt[idx-1];
            while ((flag & (1 << i)) == 0) ++i;
            s_buffer[s++] = row[i] + s_tmp[tid];
            
            --cnt;
            while (cnt > 0){
                ++i;
                if ((flag & (1 << i)) > 0) {
                    s_buffer[s++] = row[i];
                    --cnt;
                }
            }
            
         }
    }

    __syncthreads();
    uint start_align = (s_start >> 4) << 4;
    uint end         = s_start + s_cnt[63];
    uint oId         = start_align + tid;

    if (oId >= s_start && oId < end)
        g_lSum[oId]  = s_buffer[oId - s_start];
    
    oId += 32;            
    while (oId < end){
        g_lSum[oId]  = s_data [oId - s_start];
        oId += 32;
    }
}


template<class T>
void segSumBlock_dense(T* g_last, T* g_lSum, T* g_iData, uint* g_iFlags, uint* g_StartW, int n){
    dim3 threads(32);
    uint nBlocks = iDivUp(n, 1024);
    dim3 grids(nBlocks);
    segSum_kernel_dense<T><<<grids, threads>>>(g_last, g_lSum, g_iData, g_iFlags, g_StartW, n);
}

//------------------------------------------------------------------------------------
template<class T>
__global__ void segSumSmall_kernel(T* g_lSum, T* g_iData, uint* g_iFlags, int n){
    const int WPAD = 32 + 1;
    uint  tid      = threadIdx.x;
    __shared__ uint s_cnt[64];
    __shared__ uint s_flags[32];
    __shared__ T s_tmp[64];
    __shared__ T s_data[32*(32+1)];
    
    //1.Read data to shared mem
    uint flag;
    if (tid * 32 < n)
        flag = g_iFlags[tid];
    else
        flag = 0;

    uint i1 = 0;
    for (int i=0; i< 32 * 32; i+=32, i1+=WPAD)
        if (tid + i < n)
            s_data[tid + i1] = g_iData[tid + i];
        else
            s_data[tid + i1] = 0;
    
    __syncthreads();

    //2. Reduce row using H thread
    {
        uint*row = s_data + tid * WPAD;
        T res = 0;
        for (uint i= 0; i<32; ++i){
            res      += row[i];
            row[i]    = res;
            // reset value if needed
            res      *= ((flag & (1 <<i)) == 0);
        }
        s_tmp[tid] = res;
    }

    
    __syncthreads();
    
    // Mark the end of the block
    //1.b count the number of bit 1 = the number of segment inside compressed flag
    uint cnt = popc(flag);

    s_cnt[tid     ] = 0;        
    s_cnt[tid + 32] = cnt;
    s_flags[tid]    = (cnt > 0);

    __syncthreads();
    
    //1.c reduce the number of element counter
    uint idx = tid + 32;
    s_cnt[idx] += s_cnt[idx - 1 ];
    s_cnt[idx] += s_cnt[idx - 2 ];
    s_cnt[idx] += s_cnt[idx - 4 ];
    s_cnt[idx] += s_cnt[idx - 8 ];
    s_cnt[idx] += s_cnt[idx - 16];
        
    __syncthreads();

    //3. Segmented scan the sum
    if (tid ==0) {
        T res  = 0;
        for (uint i=0; i<32; ++i){
            T t = s_tmp[i];
            s_tmp[i] = res;
            res    = s_flags[i] ? t : res + t;
        }
        s_tmp[32] = res;
    }
    __syncthreads();

    {
        T*row = s_data + tid * WPAD;
        // there's break inside
        if (cnt > 0) {
            int i = 0;
            int s = s_cnt[idx-1];
            while ((flag & (1 << i)) == 0) ++i;
            g_lSum[s++] = row[i] + s_tmp[tid];
            --cnt;

            
            while (cnt > 0){
                ++i;
                if ((flag & (1 << i)) > 0) {
                    g_lSum[s++] = row[i];
                    --cnt;
                }
            }
            
        }
    }
}

template<class T>
void segSumSmall(T* g_lSum, T* g_iData, uint* g_iFlags, int n){
    segSumSmall_kernel<<<1,32>>>(g_lSum, g_iData, g_iFlags, n);
}

//------------------------------------------------------------------------------------

template<class T>
__global__ void segScanSmall_kernel(T* g_lScan, T* g_iData, uint* g_iFlags, int n){
    const int WPAD = 32 + 1;
    //uint blockId = blockIdx.x + blockIdx.y * gridDim.x;
    uint tid     = threadIdx.x;

    __shared__ T    s_tmp[64];
    __shared__ uint s_flags[32];
    __shared__ T    s_data[32*(32+1)];
    
    //1.Read data to shared mem
    uint flag;
    //if ((blockId * 32 + tid) * 32 < n)
    if ( tid * 32 < n)
        flag = g_iFlags[tid];
    else
        flag = 0;

    uint i1 = 0;
    for (int i=0; i< 32 * 32; i+=32, i1+=WPAD)
        if (tid + i < n)
            s_data[tid + i1] = g_iData[tid + i];
        else
            s_data[tid + i1] = 0;
    
    __syncthreads();
    // Mark the end of the block
    //1.b count the number of bit 1 = the number of segment inside compressed flag
    uint cnt      = popc(flag);
    s_flags[tid]  = (cnt > 0);
    __syncthreads();
    //2. Reduce row using H thread
    {
        T*row = s_data + tid * WPAD;
        T res = 0;
        for (uint i= 0; i<32; ++i){
            T t = row[i];
            row[i] = res;
            res   += t;
            res   *= ((flag & (1 <<i)) == 0);
        }
        s_tmp[tid] =res;
    }
    
    __syncthreads();
    //3. Segmented scan the sum
    if (tid ==0) {
        T res  = 0;
        for (uint i=0; i<32; ++i){
            T t      = s_tmp[i];
            s_tmp[i] = res;
            res      = s_flags[i] ? t : res + t;
        }
    }
    __syncthreads();

    T *row  = s_data + tid * WPAD;
    T ad    = s_tmp[tid];
    for (int i=0; i<32; ++i){
        row[i]  += ad;
        if ((flag & (1 <<i)) > 0 )
            ad = 0;
    }
    
    //4. Write out the result
    i1 = 0;
    for (int i=0; i< 32 * 32; i+=32, i1+=WPAD)
        if (tid + i < n)
            g_lScan[tid + i] = s_data[tid + i1];

}

template<class T>
void segScanSmall(T* g_lScan, T* g_iData, uint* g_iFlags, int n){
    segScanSmall_kernel<<<1,32>>>(g_lScan, g_iData, g_iFlags, n);
}

//------------------------------------------------------------------------------------
template<class T>
__global__ void segScanSmall_inclusive_kernel(T* g_lScan, T* g_iData, uint* g_iFlags, int n){
    const int WPAD = 32 + 1;
    //uint blockId = blockIdx.x + blockIdx.y * gridDim.x;
    uint tid     = threadIdx.x;

    __shared__ T    s_tmp[64];
    __shared__ uint s_flags[32];
    __shared__ T    s_data[32*(32+1)];
    
    //1.Read data to shared mem
    uint flag;
    //if ((blockId * 32 + tid) * 32 < n)
    if ( tid * 32 < n)
        flag = g_iFlags[tid];
    else
        flag = 0;

    uint i1 = 0;
    for (int i=0; i< 32 * 32; i+=32, i1+=WPAD)
        if (tid + i < n)
            s_data[tid + i1] = g_iData[tid + i];
        else
            s_data[tid + i1] = 0;
    
    __syncthreads();
    // Mark the end of the block
    //1.b count the number of bit 1 = the number of segment inside compressed flag
    uint cnt = popc(flag);
    s_flags[tid]    = (cnt > 0);
    __syncthreads();
    //2. Reduce row using H thread
    {
        T*row = s_data + tid * WPAD;
        T res = 0;
        for (uint i= 0; i<32; ++i){
            T t    = row[i];
            res   += t;
            row[i] = res;
            res   *= ((flag & (1 <<i)) == 0);
        }
        s_tmp[tid] =res;
    }
    
    __syncthreads();
    //3. Segmented scan the sum
    if (tid ==0) {
        T res  = 0;
        for (uint i=0; i<32; ++i){
            T t      = s_tmp[i];
            s_tmp[i] = res;
            res      = s_flags[i] ? t : res + t;
        }
    }
    __syncthreads();

    T *row  = s_data + tid * WPAD;
    T ad    = s_tmp[tid];
    for (int i=0; i<32; ++i){
        row[i]  += ad;
        if ((flag & (1 <<i)) > 0 )
            ad = 0;
    }
    
    //4. Write out the result
    i1 = 0;
    for (int i=0; i< 32 * 32; i+=32, i1+=WPAD)
        if (tid + i < n)
            g_lScan[tid + i] = s_data[tid + i1];

}

template<class T>
void segScanSmall_inclusive(T* g_lScan, T* g_iData, uint* g_iFlags, int n){
    segScanSmall_inclusive_kernel<<<1,32>>>(g_lScan, g_iData, g_iFlags, n);
}

//------------------------------------------------------------------------------------
template<class T>
__global__ void segScan_block_kernel(T* g_lScan, T* g_iData, uint* g_iFlags, int n, T* g_temp){
    const int WPAD = 32 + 1;
    uint blockId = blockIdx.x + blockIdx.y * gridDim.x;
    uint tid     = threadIdx.x;
    uint id      = tid + blockId * 1024;

    __shared__ T    s_tmp[64];
    __shared__ uint s_flags[32];
    __shared__ T    s_data[32*(32+1)];
    
    //1.Read data to shared mem
    uint flag;
    if ((blockId * 32 + tid) * 32 < n)
        flag = g_iFlags[blockId * 32 + tid];
    else
        flag = 0;

    uint i1 = 0;
    for (int i=0; i< 32 * 32; i+=32, i1+=WPAD)
        if (id + i < n)
            s_data[tid + i1] = g_iData[id + i];
        else
            s_data[tid + i1] = 0;
    
    __syncthreads();
    // Mark the end of the block
    //1.b count the number of bit 1 = the number of segment inside compressed flag
    uint cnt = popc(flag);
    s_flags[tid]    = (cnt > 0);
    __syncthreads();
    //2. Reduce row using H thread
    {
        T*row = s_data + tid * WPAD;
        T res = 0;
        for (uint i= 0; i<32; ++i){
            T t = row[i];
            row[i] = res;
            res   += t;
            res   *= ((flag & (1 <<i)) == 0);
        }
        s_tmp[tid] =res;
    }
    
    __syncthreads();
    //3. Segmented scan the sum
    if (tid ==0) {
        T res     = 0;
        for (uint i=0; i<32; ++i){
            T t      = s_tmp[i];
            s_tmp[i] = res;
            res      = s_flags[i] ? t : res + t;
        }
        g_temp[blockId] = res;
    }
    __syncthreads();

    T *row  = s_data + tid * WPAD;
    T ad    = s_tmp[tid];
    for (int i=0; i<32; ++i){
        row[i]  += ad;
        if ((flag & (1 <<i)) > 0 )
            ad = 0;
    }
    
    //4. Write out the result
    i1 = 0;
    for (int i=0; i< 32 * 32; i+=32, i1+=WPAD)
        if (id + i < n)
            g_lScan[id + i] = s_data[tid + i1];
}

template<class T>
__global__ void segScan_inclusive_block_kernel(T* g_lScan, T* g_iData, uint* g_iFlags, int n, T* g_temp){
    const int WPAD = 32 + 1;
    uint blockId = blockIdx.x + blockIdx.y * gridDim.x;
    uint tid     = threadIdx.x;
    uint id      = tid + blockId * 1024;

    __shared__ T    s_tmp[64];
    __shared__ uint s_flags[32];
    __shared__ T    s_data[32*(32+1)];
    
    //1.Read data to shared mem
    uint flag;
    if ((blockId * 32 + tid) * 32 < n)
        flag = g_iFlags[blockId * 32 + tid];
    else
        flag = 0;

    uint i1 = 0;
    for (int i=0; i< 32 * 32; i+=32, i1+=WPAD)
        if (id + i < n)
            s_data[tid + i1] = g_iData[id + i];
        else
            s_data[tid + i1] = 0;
    
    __syncthreads();
    // Mark the end of the block
    //1.b count the number of bit 1 = the number of segment inside compressed flag
    uint cnt = popc(flag);
    s_flags[tid]    = (cnt > 0);
    __syncthreads();
    //2. Reduce row using H thread
    {
        T*row = s_data + tid * WPAD;
        T res = 0;
        for (uint i= 0; i<32; ++i){
            T t    = row[i];
            res   += t;
            row[i] = res;
            res   *= ((flag & (1 <<i)) == 0);
        }
        s_tmp[tid] =res;
    }
    
    __syncthreads();
    //3. Segmented scan the sum
    if (tid ==0) {
        T res     = 0;
        for (uint i=0; i<32; ++i){
            T t      = s_tmp[i];
            s_tmp[i] = res;
            res      = s_flags[i] ? t : res + t;
        }
        g_temp[blockId] = res;
    }
    __syncthreads();

    T *row  = s_data + tid * WPAD;
    T ad    = s_tmp[tid];
    for (int i=0; i<32; ++i){
        row[i]  += ad;
        if ((flag & (1 <<i)) > 0 )
            ad = 0;
    }
    
    //4. Write out the result
    i1 = 0;
    for (int i=0; i< 32 * 32; i+=32, i1+=WPAD)
        if (id + i < n)
            g_lScan[id + i] = s_data[tid + i1];
}


//------------------------------------------------------------------------------------

__global__ void findFirstPos_kernel(uint* g_pos, uint* g_cnt, uint n){
    const int WPAD = 33;
    uint blockId= blockIdx.x + blockIdx.y * gridDim.x;
    uint tid    = threadIdx.x;
    uint id     = tid + blockId * 1024;
    __shared__ uint s_data[32*WPAD];
    uint i1 = 0;
    for (int i=0; i< 32 * 32; i+=32, i1+=WPAD)
        if (id + i < n)
            s_data[tid + i1] = g_cnt[id + i];
        else
            s_data[tid + i1] = 0;

    __syncthreads();

    uint* row = s_data + tid * WPAD;
    uint  pos = 0;

    uint i=0;
    // find the first non zero flag
    for (; (i < 32) && (row[i] == 0); ++i)
        pos += 32;

    // find the first non zero bit
    if (i!=32){
        int flag = row[i];
        int b = 0;
        while ((flag & (1 << b)) == 0)
            ++b;
        pos +=b;
    }

    // write out the position
    int bl = blockId * 32 + tid;
    if (bl * 32 < n) 
        g_pos[bl] = pos;
}


void findFirstPos(uint* g_pos, uint* g_cnt, uint n){
    uint nBlocks = iDivUp(n, 1024);
    dim3 threads(32);
    dim3 grids(nBlocks);
    checkConfig(grids);
    findFirstPos_kernel<<<grids, threads>>>(g_pos, g_cnt, n);
}


//------------------------------------------------------------------------------------
template<class T>
__global__ void addResult_kernel(T *g_Scan, uint* g_flags, uint* g_comFlags, T* g_addValue, uint nBlock){
    //read the flags
    uint tid = threadIdx.x;

    if (tid < nBlock) {
        uint comflag = g_comFlags[tid];
        if (comflag > 0) {
            T value   = g_addValue[tid];
            uint i = 0;
            while ((comflag & (1 << i))==0) ++i;
            uint flag = g_flags[tid * 32 + i];

            uint i1 = 0;
            while ((flag & (1 << i1))==0) ++i1;
            g_Scan[tid * 1024 + i * 32 + i1] +=value;
        }
    }
}


template<class T>
__global__ void postAddScan_block_kernel(T* g_scan, T* g_value, uint* g_pos, int n){

    __shared__ T     value;
    __shared__ uint  pos;

    uint blockId= blockIdx.x + blockIdx.y * gridDim.x;
    uint tid    = threadIdx.x;
    uint off    = blockId * 1024;

    if (tid  == 0) {
        pos   = g_pos[blockId];
        value = g_value[blockId];
    }
    __syncthreads();
    int lim = min(pos, 1023);
    while (tid <= lim) {
        g_scan[tid + off] += value;
        tid               += blockDim.x;
    }
}
    
template<class T>
void postAddScan_block(T* g_scan, T* g_value, uint* g_pos, int n){
    int  nBlock = iDivUp(n, 1024);
    dim3 threads(256);
    dim3 grids(nBlock);
    checkConfig(grids);
    postAddScan_block_kernel<<<grids, threads>>>(g_scan, g_value, g_pos, n);
}

template<class T, int inclusive>
void segScan(T* g_scan, T* g_iData, uint* g_iFlags, int n){

    uint nBlocks = iDivUp(n,1024);
    dim3 threads(32);
    dim3 grids(nBlocks);
    checkConfig(grids);

    if(nBlocks == 1){
        if (inclusive){
            segScanSmall_inclusive_kernel<T><<<1,32>>>(g_scan, g_iData, g_iFlags, n);
        }
        else {
            segScanSmall_kernel<T><<<1,32>>>(g_scan, g_iData, g_iFlags, n);
        }
    }
    else {
        T* d_temp;

        // Perform scan on the block
        // Output the local scan value
        // Output the last scan value of each block
        
        cudaMalloc((void**)&d_temp, nBlocks * sizeof(T));
        if (inclusive){
            segScan_inclusive_block_kernel<T><<<grids, threads>>>(g_scan, g_iData, g_iFlags, n, d_temp);
        }
        else
            segScan_block_kernel<T><<<grids, threads>>>(g_scan, g_iData, g_iFlags, n, d_temp);

        uint* d_comPos;
        cudaMalloc((void**)&d_comPos, nBlocks * sizeof(uint) );
        findFirstPos(d_comPos, g_iFlags, iDivUp(n, 32));

            
        T*    h_temp  = new T [nBlocks];
        uint* h_comPos= new uint [nBlocks];
        
        cudaMemcpy(h_temp  , d_temp  , sizeof(T) * nBlocks, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_comPos, d_comPos, sizeof(uint) * nBlocks, cudaMemcpyDeviceToHost);

        T res = 0, t  = 0;
        for (unsigned int i=0;i < nBlocks; ++i){
            t = h_temp[i];
            h_temp[i] = res;
            res +=t;
            if (h_comPos[i] != 1024)
                res = t;
        }

        cudaMemcpy(d_temp, h_temp, sizeof(T) * nBlocks, cudaMemcpyHostToDevice);
        // Add the value to each block
        
        postAddScan_block(g_scan, d_temp, d_comPos, n);

        dmemFree(d_temp);
        dmemFree(d_comPos);
        delete [] h_temp;
        delete [] h_comPos;
    }
}

template void segScan<float, 0>(float* g_scan, float* g_iData, uint* g_iFlags, int n);
template void segScan<int, 0>(int* g_scan, int* g_iData, uint* g_iFlags, int n);
template void segScan<uint, 0>(uint* g_scan, uint* g_iData, uint* g_iFlags, int n);

template void segScan<float, 1>(float* g_scan, float* g_iData, uint* g_iFlags, int n);
template void segScan<int, 1>(int* g_scan, int* g_iData, uint* g_iFlags, int n);
template void segScan<uint, 1>(uint* g_scan, uint* g_iData, uint* g_iFlags, int n);

    
template<class T>
void segScan(T* g_scan, T* g_iData, uint* g_iFlags, int n){
    segScan<T, 0>(g_scan, g_iData, g_iFlags, n);
}

template void segScan<float>(float* g_scan, float* g_iData, uint* g_iFlags, int n);
template void segScan<int>(int* g_scan, int* g_iData, uint* g_iFlags, int n);
template void segScan<uint>(uint* g_scan, uint* g_iData, uint* g_iFlags, int n);


template<class T>
void segScan_block_cpu(T* scan, T* idata, uint* iflags, int n, uint* blockSum){
    int nSegs  = iDivUp(n, 32);
    int flag;
    T   sum = 0;
    int id  = 0;

    for (int j=0; j< nSegs; ++j){
        flag = iflags[j];
        for (int i=0; i<32 && id < n; ++i, ++id){
            T old    = idata[id];
            scan[id] = sum;
            sum     += old;
            if (flag & (1<<i))
                sum = 0;
        }
        if ((j & 0x1F) == 0x1F){
            blockSum[j>>5] = sum;
            sum = 0;
        }
    }
}

template<class T>
void segScan_inclusive_block_cpu(T* scan, T* idata, uint* iflags, int n, uint* blockSum){
    
    int nSegs  = iDivUp(n, 32);
    int flag;
    T   sum = 0;
    int id  = 0;

    for (int j=0; j< nSegs; ++j){
        flag = iflags[j];
        for (int i=0; i<32 && id < n; ++i, ++id){
            T old    = idata[id];
            sum     += old;
            scan[id] = sum;
            sum     *= ((flag & (1<<i))==0);
        }
        if ((j & 0x1F) == 0x1F){
            blockSum[j>>5] = sum;
            sum = 0;
        }
    }
}


template<class T>
void segScan_cpu(T* scan, T* idata, uint* iflags, int n){
    int nSegs  = iDivUp(n, 32);
    int flag;
    T   sum = 0;
    int id  = 0;

    for (int j=0; j< nSegs; ++j){
        flag = iflags[j];
        for (int i=0; i<32 && id < n; ++i, ++id){
            T old    = idata[id];
            scan[id] = sum;
            sum     += old;
            if (flag & (1<<i))
                sum = 0;
        }
    }
}

void segScan_cpu(int2* scan, int2* idata, uint* iflags, int n){
    int nSegs  = iDivUp(n, 32);
    int flag;
    int2  sum = make_int2(0,0);
    int id  = 0;

    for (int j=0; j< nSegs; ++j){
        flag = iflags[j];
        for (int i=0; i<32 && id < n; ++i, ++id){
            int2 old    = idata[id];
            scan[id] = sum;
            sum     += old;
            if (flag & (1<<i))
                sum = make_int2(0,0);
        }
    }
}

void segScan_cpu(float2* scan, float2* idata, uint* iflags, int n){
    int nSegs  = iDivUp(n, 32);
    int flag;
    float2  sum = make_float2(0,0);
    int id  = 0;

    for (int j=0; j< nSegs; ++j){
        flag = iflags[j];
        for (int i=0; i<32 && id < n; ++i, ++id){
            float2 old    = idata[id];
            scan[id] = sum;
            sum     += old;
            if (flag & (1<<i))
                sum = make_float2(0,0);
        }
    }
}


template<class T>
void segScan_inclusive_cpu(T* scan, T* idata, uint* iflags, int n){
    int nSegs  = iDivUp(n, 32);
    int flag;
    T   sum = 0;
    int id  = 0;

    for (int j=0; j< nSegs; ++j){
        flag = iflags[j];
        for (int i=0; i<32 && id < n; ++i, ++id){
            T old    = idata[id];
            sum     += old;
            scan[id] = sum;
            if (flag & (1<<i))
                sum = 0;
        }
    }
}

void segScan_inclusive_cpu(int2* scan, int2* idata, uint* iflags, int n){
    int nSegs  = iDivUp(n, 32);
    int flag;
    int2   sum = make_int2(0,0);
    int id  = 0;

    for (int j=0; j< nSegs; ++j){
        flag = iflags[j];
        for (int i=0; i<32 && id < n; ++i, ++id){
            int2 old    = idata[id];
            sum     += old;
            scan[id] = sum;
            if (flag & (1<<i))
                sum = make_int2(0,0);
        }
    }
}


void segScan_inclusive_cpu(float2* scan, float2* idata, uint* iflags, int n){
    int  nSegs  = iDivUp(n, 32);
    int  flag;
    float2 sum = make_float2(0,0);
    int id  = 0;

    for (int j=0; j< nSegs; ++j){
        flag = iflags[j];
        for (int i=0; i<32 && id < n; ++i, ++id){
            float2 old    = idata[id];
            sum     += old;
            scan[id] = sum;
            if (flag & (1<<i))
                sum = make_float2(0,0);
        }
    }
}

void testSegScan(int n, int s, int inclusive){
    fprintf(stderr, "Number of input %d\n", n);
    int nSegs = iDivUp(n, 32);

    //1. Initialize random input
    uint* h_iData = new uint[n];
    uint* h_scan  = new uint[n];
    for (int i=0; i<n; ++i)
        h_iData[i] = 1;

    //2. Initialize the pos
    bitSet bst(n);
    bst.insert(n-1);
    for (uint i=0; i< s; ++i) {
        int pos = rand() % n;
        bst.insert(pos);
    }


    //3. Get the position
    int total = s + 1;
    int* segEnd = new int[total];
    total = bst.getCount();
    bst.ListAll((uint*)segEnd, total);
    //4. Allocate GPU mem and copy the input

    uint* d_iData;
    dmemAlloc(d_iData, n);
    cudaMemcpy(d_iData , h_iData, sizeof(uint) * n, cudaMemcpyHostToDevice);

    //5. Mark the range of each segment 
    int id = 0;
    int* aId = new int[n];
    for (int i=0; i < n; ++i){
        aId[i] = id;
        if (bst.check(i))
            ++id;
    }
    uint* d_aId;
    dmemAlloc(d_aId, n);
    cudaMemcpy(d_aId, &aId[0], n * sizeof(uint), cudaMemcpyHostToDevice);

    //6. Generate the flags, 1 indicate end of the segment 0 is inside segment
    uint* d_iFlags;
    dmemAlloc(d_iFlags, nSegs);
    buildLastPosFlags(d_iFlags, (uint4*) d_aId, n);

    uint* d_scan;
    dmemAlloc(d_scan, n);

    int nIter = 100;
    if (inclusive )
        segScan<uint, 1>(d_scan, d_iData, d_iFlags, n);
    else
        segScan<uint, 0>(d_scan, d_iData, d_iFlags, n);
    
    uint timer;
    cutCreateTimer(&timer);
    cutStartTimer(timer);
    cutResetTimer(timer);
    for (int i=0; i < nIter; ++i)
        if (inclusive)
            segScan<uint, 1>(d_scan, d_iData, d_iFlags, n);
        else
            segScan<uint, 0>(d_scan, d_iData, d_iFlags, n);
    cudaThreadSynchronize();
    cutStopTimer(timer);
    
    printf("GPU Segmentated scan running time: %f ms\n", cutGetTimerValue(timer) / nIter);

    cutStartTimer(timer);
    cutResetTimer(timer);
    if (inclusive)
        segScan_inclusive_cpu(h_scan, h_iData, bst.m_bits, n);
    else 
        segScan_cpu(h_scan, h_iData, bst.m_bits, n);
    cutStopTimer(timer);
    printf("CPU Segmentated scan running time: %f ms\n", cutGetTimerValue(timer));

    testError(h_scan, d_scan, n, "scan result");

    uint* d_iflags;
    uint* h_iflags = new uint [n];

    for (int i=0; i< n; ++i)
        h_iflags[i] = 0;
    h_iflags[0] = 1;

    for (int i=0; i< n-1; ++i)
        if (bst.check(i))
            h_iflags[i+1]=1;

    dmemAlloc(d_iflags, n);
    cudaMemcpy(d_iflags, h_iflags, n * sizeof(int), cudaMemcpyHostToDevice);

    CUDPPConfiguration scanConfig;
    CUDPPHandle        mScanPlan;        // CUDPP plan handle for prefix sum
    scanConfig.algorithm = CUDPP_SEGMENTED_SCAN;
    scanConfig.datatype  = CUDPP_INT;
    scanConfig.op        = CUDPP_ADD;
    if (inclusive)
        scanConfig.options   = CUDPP_OPTION_INCLUSIVE | CUDPP_OPTION_FORWARD;
    else 
        scanConfig.options   = CUDPP_OPTION_EXCLUSIVE | CUDPP_OPTION_FORWARD;
    
    cudppPlan(&mScanPlan, scanConfig, n, 1, 0);

    cutResetTimer(timer);
    for (int i=0; i < nIter; ++i)
        cudppSegmentedScan(mScanPlan, d_scan, d_iData, d_iflags, n);
    cudaThreadSynchronize();
    cutStopTimer(timer);
    printf("CUDPP Segmentated scan running time: %f ms\n", cutGetTimerValue(timer)/nIter);

    testError(h_scan, d_scan, n, "CUDPP scan result");
    
    
    addScanPlan addScan;
    addScan.allocate(n);
    
    cutStartTimer(timer);
    cutResetTimer(timer);
    for (int i=0; i < nIter; ++i)
        addScan.scan(d_scan, d_iData, n);
    
    cudaThreadSynchronize();
    cutStopTimer(timer);
    printf("GPU Normal scan running time: %f ms\n", cutGetTimerValue(timer) / nIter);
    
    dmemFree(d_aId);
    dmemFree(d_iData);
    dmemFree(d_iFlags);
    dmemFree(d_scan);

    delete []aId;
    delete []segEnd;
    delete []h_iData;
    delete []h_scan;
}

inline void compare(double* h_ref, double* d_in, int n, double eps, char* name){
    double* h_in = new double [n];
    cudaMemcpy(h_in, d_in , n * sizeof(double), cudaMemcpyDeviceToHost);

    int fi; 
    int i=0;

    for (i=0; i< n; ++i)
        if (abs(h_in[i] - h_ref[i]) > eps)
            break;
    fi = i;

    if (fi == n) {
        fprintf(stderr, "%s TEST PASSED \n", name);
    }
    else {
        double max = abs(h_in[i] - h_ref[i]);
        int mi = i;
        for (;i< n; ++i)
            if (abs(h_in[i] - h_ref[i]) > max){
                max = abs(h_in[i] - h_ref[i]);
                mi = i;
            }
        fprintf(stderr, "%s TEST FAILED first at %d ref %g rel %g \r max at %d ref %g rel %g \n", name, fi, h_ref[fi], h_in[fi],
                mi, h_ref[mi], h_in[mi]);
    }
    
    delete [] h_in;
}


void testSegScan_block(int n, int s){
    fprintf(stderr, "Number of input %d\n", n);
    int nSegs = iDivUp(n, 32);

//1. Initialize random input
    uint* h_iData = new uint[n];
    uint* h_scan  = new uint[n];
    for (int i=0; i<n; ++i)
        h_iData[i] = rand() % 16;

    //2. Initialize the pos
    bitSet bst(n);
    bst.insert(n-1);
    for (uint i=0; i< s; ++i) {
        int pos = rand() % n;
        bst.insert(pos);
    }

    //3. Get the position
    int total = s + 1;
    int* segEnd = new int[total];
    bst.ListAll((uint*)segEnd, total);
    
    //4. Allocate GPU mem and copy the input 
    uint* d_iData;
    dmemAlloc(d_iData, n);
    cudaMemcpy(d_iData , h_iData, sizeof(uint) * n, cudaMemcpyHostToDevice);

    //5. Mark the range of each segment 
    int id = 0;
    int* aId = new int[n];
    for (int i=0; i < n; ++i){
        aId[i] = id;
        if (bst.check(i))
            ++id;
    }

    uint* d_aId;
    dmemAlloc(d_aId, n);
    cudaMemcpy(d_aId, &aId[0], n * sizeof(uint), cudaMemcpyHostToDevice);

    //6. Generate the flags, 1 indicate end of the segment 0 is inside segment
    uint* d_iFlags;
    dmemAlloc(d_iFlags, nSegs);
    buildLastPosFlags(d_iFlags, (uint4*) d_aId, n);

    uint* d_scan;
    dmemAlloc(d_scan, n);

    uint nBlocks     = iDivUp(n, 1024);
    uint* d_blockSum;
    dmemAlloc(d_blockSum, nBlocks);
    cudaMemset(d_blockSum, 0, nBlocks * sizeof(uint));
        
    if (nBlocks ==0)
        segScanSmall_kernel<uint><<<nBlocks,32>>>(d_scan, d_iData, d_iFlags, n);
    else
        segScan_block_kernel<uint><<<nBlocks,32>>>(d_scan, d_iData, d_iFlags, n, d_blockSum);

    uint* h_blockSum = new uint [nBlocks];
    segScan_block_cpu(h_scan, h_iData, bst.m_bits, n, h_blockSum);

    testError(h_scan, d_scan, n, "scan block");
    testError(h_blockSum, d_blockSum, nBlocks, "block sum");
    
    dmemFree(d_aId);
    dmemFree(d_iData);
    dmemFree(d_iFlags);
    dmemFree(d_scan);
    dmemFree(d_blockSum);

    delete []h_blockSum;
    delete []aId;
    delete []segEnd;
    delete []h_iData;
    delete []h_scan;
}

void testSegScan_inclusive_block(int n, unsigned int s){
    fprintf(stderr, "Number of input %d\n", n);
    int nSegs = iDivUp(n, 32);

//1. Initialize random input
    uint* h_iData = new uint[n];
    uint* h_scan  = new uint[n];
    for (int i=0; i<n; ++i)
        h_iData[i] = rand() % 16;

    //2. Initialize the pos
    bitSet bst(n);
    bst.insert(n-1);
    for (uint i=0; i< s; ++i) {
        int pos = rand() % n;
        bst.insert(pos);
    }

    //3. Get the position
    int total = s + 1;
    int* segEnd = new int[total];
    bst.ListAll((uint*)segEnd, total);
    
    //4. Allocate GPU mem and copy the input 
    uint* d_iData;
    dmemAlloc(d_iData, n);
    cudaMemcpy(d_iData , h_iData, sizeof(uint) * n, cudaMemcpyHostToDevice);

    //5. Mark the range of each segment 
    int id = 0;
    int* aId = new int[n];
    for (int i=0; i < n; ++i){
        aId[i] = id;
        if (bst.check(i))
            ++id;
    }

    uint* d_aId;
    dmemAlloc(d_aId, n);
    cudaMemcpy(d_aId, &aId[0], n * sizeof(uint), cudaMemcpyHostToDevice);

    //6. Generate the flags, 1 indicate end of the segment 0 is inside segment
    uint* d_iFlags;
    dmemAlloc(d_iFlags, nSegs);
    buildLastPosFlags(d_iFlags, (uint4*) d_aId, n);

    uint* d_scan;
    dmemAlloc(d_scan, n);

    uint nBlocks     = iDivUp(n, 1024);
    uint* d_blockSum;

    dmemAlloc(d_blockSum, nBlocks);
    cudaMemset(d_blockSum, 0, nBlocks * sizeof(uint));
            
    if (nBlocks ==1)
        segScanSmall_inclusive_kernel<uint><<<1,32>>>(d_scan, d_iData, d_iFlags, n);
    else
        segScan_inclusive_block_kernel<uint><<<nBlocks,32>>>(d_scan, d_iData, d_iFlags, n, d_blockSum);

    uint* h_blockSum = new uint [nBlocks];
    segScan_inclusive_block_cpu(h_scan, h_iData, bst.m_bits, n, h_blockSum);

    testError(h_scan, d_scan, n, "scan block");
    testError(h_blockSum, d_blockSum, nBlocks, "block sum");
    
    dmemFree(d_aId);
    dmemFree(d_iData);
    dmemFree(d_iFlags);
    dmemFree(d_scan);
    dmemFree(d_blockSum);

    delete []h_blockSum;
    delete []aId;
    delete []segEnd;
    delete []h_iData;
    delete []h_scan;
}


template<class T>
__global__ void addSegmentSum_kernel(T* d_Sum, T* d_offset, uint* d_wpos, uint* d_cnt, uint n){
    uint blockId= blockIdx.x + blockIdx.y * gridDim.x;
    uint tid    = threadIdx.x;
    uint id     = tid + blockId * blockDim.x;
    if (id < n) {
        uint cnt = d_cnt[id];
        if (cnt > 0) {
            uint pos    = d_wpos[id];
            d_Sum[pos] += d_offset[id]; 
        }
    }
}

template<class T>
void addSegmentSum(T* d_Sum, T* d_offset, uint* d_wpos, uint* d_cnt, uint nSubSeg){
    uint nBlocks = iDivUp(nSubSeg,64);
    dim3 threads(64);
    dim3 grids(nBlocks);
    checkConfig(grids);
    addSegmentSum_kernel<T><<<grids, threads>>>(d_Sum, d_offset, d_wpos, d_cnt, nSubSeg);
}

template<class T>
void segmentedSum(T* d_iData, uint* d_iFlags, uint n){
    //1. Count the number of segments = the total number of bit 1 
    uint totalSeg = 0;
    uint nFlags   = iDivUp(n, 32);                   // number of flag
    uint nSubSegs = iDivUp(nFlags, 32);              // number of 1024 segment   

    uint* d_cnt,*d_wpos;

    cudaMemoryPlan cntPlan(sizeof(uint) * nSubSegs);
    d_cnt = cntPlan.buffer<uint>();

    cudaMemoryPlan wposPlan(sizeof(uint) * nSubSegs);
    d_wpos = wposPlan.buffer<uint>();

    //1.a Compute the number of segment per each 1024 elements array
    segCount(d_cnt, d_iFlags, nFlags);

    //1.b Compute total number of segment
    reducePlan    rdPlan;
    totalSeg = rdPlan.Sum_32u_C1(d_cnt, nSubSegs);
    
    //2. Allocate the memory so that it can store result for all segment
    cudaMemoryPlan sumPlan(sizeof(T) * totalSeg);
    T* d_sum = sumPlan.buffer<T>();

    //3. Classify the problem base on input size 
    if (n < 1024) {
        segSumSmall<T>(d_sum, d_iData, d_iFlags, n);
    }
    else {
        // 4. Compute the exact position of each segment in each 1024 boundary in the
        // final result
        scanImpl scan(CUDPP_INT, false);        
        scan.add_scan_init(nSubSegs);
        scan.scan(d_wpos, d_cnt, nSubSegs);

        // 5.Compute the overlap part of each subsegment
        cudaMemoryPlan lastPlan(nSubSegs * sizeof(uint));
        uint* d_last = lastPlan.buffer<uint>();
        
        if (totalSeg * 64 > n) {
            segSumBlock_dense<T>(d_last, d_sum, d_iData, d_iFlags, d_wpos, n);
        }
        else {
            segSumBlock_sparse<T>(d_last, d_sum, d_iData, d_iFlags, d_wpos, n);
        }

        // 6.Compress the flags
        uint nFlagsL1 = iDivUp(nSubSegs, 32);
        cudaMemoryPlan flagL1Plan(sizeof(uint) * nFlagsL1);
        uint* d_FlagL1 = flagL1Plan.buffer<uint>();
        compressFlags(d_FlagL1, d_cnt, nSubSegs);
        
        // 7.Perform small segmented scan
        uint* d_sumSeg;
        cudaMemoryPlan sumSegPlan(nSubSegs * sizeof(uint));
        d_sumSeg = sumSegPlan.buffer<uint>();
        segScan<T>(d_sumSeg, d_last, d_FlagL1, nSubSegs);
        
        // 8.Accumulate the result
        addSegmentSum<T>(d_sum, d_sumSeg, d_wpos, d_cnt, nSubSegs);
    }
    printDeviceArray1D((int*)d_sum, totalSeg, "Segemented scan");
}

/**
 * @brief Scan the array and find the last position of each segment
 *        then write result on to a 32 bit flags array 
 * @param[in]  g_index4 : array with the index as input
 *             n        : number of elememnts   
 * @param[out] g_flags  : flags array               
 */


__global__ void buildLastPosFlags_kernel(uint* g_flags, uint4* g_index4, int n){

    const uint nThread   = 128;
    __shared__ uint s_flags[nThread + 64];
    __shared__ uint s_first[nThread + 64];

    uint blockId = blockIdx.x + blockIdx.y * gridDim.x;
    uint tid     = threadIdx.x;
    uint id      = blockId * nThread + tid;

    uint4 data4;
    data4 = g_index4[id];
    uint ai = tid + (tid >> 4);
    
    s_flags[ai]  = 0;
    
    id*=4;
    if (id + 1 < n)
        if (data4.x != data4.y) 
            s_flags[ai] = 1;
    
    if (id + 2 < n)
        if (data4.y != data4.z) 
            s_flags[ai] += 2;
    
    if (id + 3 < n)
        if (data4.z != data4.w) 
            s_flags[ai] += 4;
    
    if (tid == 0)
        s_first[nThread-1] = (id + nThread * 4 < n) ? (((uint*)g_index4)[id + nThread *4]) : (0xFFFFFFFF);
    else
        s_first[tid-1] = data4.x;
    
    __syncthreads();

    if (id + 4 < n) {
        if (data4.w != s_first[tid])
            s_flags[ai] += 8;
    }else {
        if (id < n) { // add the last element
            int k = (n-1) - id;
            s_flags[ai] |= (1<<k);
        }
    }

    __syncthreads();
    if (tid < 16){
        int comFlag =0;
        uint sId    = tid * 8;
        sId        += sId >> 4;
        for (int i=0; i< 8; ++i, ++sId)
            comFlag += s_flags[sId] << (4*i);

        id = blockId * 16 + tid;
        if (id * 32 < n)  
            g_flags[id] = comFlag;
    }
}

/* 
 * In : - g_iData: input data array in the form xx..xy..yz...z
 *      - n      : number of element in the input array
 *      Input constrait: g_iData is 256 bit aligned 
 * Out:
 *      g_flags : 32 bit flags array that indicate the position of the end of each segment
 */

void buildLastPosFlags(uint* g_flags, uint4* g_iData, uint n){
    dim3 threads(128);
    uint nBlocks = iDivUp(n, 512);
    dim3 grids(nBlocks);
    checkConfig(grids);
    buildLastPosFlags_kernel<<<grids, threads>>>(g_flags, g_iData, n);
}


// Kernel for float2, int2
template<int inclusive>
__global__ void segScan_block_kernel(int2* g_lScan, int2* g_iData, uint* g_iFlags, uint n, int2* g_temp){
    const int WPAD = 32 + 1;
    uint blockId = blockIdx.x + blockIdx.y * gridDim.x;
    uint tid     = threadIdx.x;
    uint id      = tid + blockId * 1024;

    __shared__ int  s_tmp[64];
    __shared__ int  s_tmp1[64];
    __shared__ uint s_flags[32];
    __shared__ int  s_data0[32*(32+1)];
    __shared__ int  s_data1[32*(32+1)];
    
    //1.Read data to shared mem
    uint flag;

    if (tid < 32){
        if ((blockId * 32 + tid) * 32 < n)
            flag = g_iFlags[blockId * 32 + tid];
        else
            flag = 0;
    }
    
    {
        uint i2 = tid & 31;
        uint i1 = tid >> 5;
    
        for (uint i=0; i< 32 * 32; i+=256, i1+=8){
            uint w = i2 + i1 * 33;
            if (id + i < n) {
                int2 data = g_iData[id + i];
                s_data0[w] = data.x;
                s_data1[w] = data.y;
            }
            else {
                s_data0[w] = 0;
                s_data1[w] = 0;
            }
        }
    }
    
    __syncthreads();
    // Mark the end of the block
    //1.b count the number of bit 1 = the number of segment inside compressed flag
    if (tid < 32){
        uint cnt = popc(flag);
        s_flags[tid]    = (cnt > 0);
    }
    __syncthreads();
    //2. Reduce row using H thread
    if (tid < 32)
    {
        int*row  = s_data0 + tid * WPAD;
        int*row1 = s_data1 + tid * WPAD;
        
        int2 res;
        res.x = 0;
        res.y = 0;
        
        for (uint i= 0; i<32; ++i){
            int t  = row[i];
            int t1 = row1[i];
            if (inclusive == 0){
                row[i] = res.x;
                row1[i]= res.y;
                
                if (flag & (1 <<i)){
                    res.x = 0;
                    res.y = 0;
                }
                else {
                    res.x += t;
                    res.y += t1;
                }
            }
            else {
                res.x += t;
                res.y += t1;

                row[i] = res.x;
                row1[i]= res.y;

                res.x  *= ((flag & (1 <<i)) == 0);
                res.y  *= ((flag & (1 <<i)) == 0);
            }
        }
        s_tmp [tid] = res.x;
        s_tmp1[tid] = res.y;
    }
    
    __syncthreads();
    //3. Segmented scan the sum
    if (tid ==0) {
        int2 res     ;

        res.x = 0;
        res.y = 0;

        for (uint i=0; i<32; ++i){
            int t    = s_tmp[i];
            int t1   = s_tmp1[i];
            
            s_tmp [i] = res.x;
            s_tmp1[i] = res.y;

            if (s_flags[i]){
                res.x = t;
                res.y = t1;
            }
            else {
                res.x +=t;
                res.y +=t1;
            }
        }
        g_temp[blockId] = res;
    }
    __syncthreads();

    if (tid < 32){
        
        int *row  = s_data0 + tid * WPAD;
        int *row1 = s_data1 + tid * WPAD;
    
        int ad    = s_tmp[tid];
        int ad1   = s_tmp1[tid];
    
        for (int i=0; i<32; ++i){
            row[i]  += ad;
            row1[i] += ad1;
        
            if ((flag & (1 <<i)) > 0 ){
                ad = 0;
                ad1= 0;
            }
        }
    }

    __syncthreads();

    //4. Write out the result
    {
        uint i2 = tid & 31;
        uint i1 = tid >> 5;
        for (uint i=0; i< 32 * 32; i+=256, i1+=8){
            uint w = i2 + i1 * 33;
            if (id + i < n) {
                int2 odata;
                odata.x = s_data0[w];
                odata.y = s_data1[w];
                g_lScan[id + i] = odata;
            }
        }
    }
}

// Kernel for float2, in2
template<int inclusive>
__global__ void segScanSmall_kernel(int2* g_lScan, int2* g_iData, uint* g_iFlags, int n){
    const int WPAD = 32 + 1;
    uint tid     = threadIdx.x;
    
    __shared__ int  s_tmp[64];
    __shared__ int  s_tmp1[64];
    __shared__ uint s_flags[32];
    __shared__ int  s_data0[32*(32+1)];
    __shared__ int  s_data1[32*(32+1)];
    
    //1.Read data to shared mem
    uint flag;

    if ( tid * 32 < n)
        flag = g_iFlags[tid];
    else
        flag = 0;

    uint i1 = 0;
    for (int i=0; i< 32 * 32; i+=32, i1+=WPAD)
        if (tid + i < n) {
            int2 data = g_iData[tid + i];
            s_data0[tid + i1] = data.x;
            s_data1[tid + i1] = data.y;
        }
        else {
            s_data0[tid + i1] = 0;
            s_data1[tid + i1] = 0;
        }
    
    __syncthreads();
    // Mark the end of the block
    //1.b count the number of bit 1 = the number of segment inside compressed flag
    uint cnt = popc(flag);
    s_flags[tid]    = (cnt > 0);
    __syncthreads();
    //2. Reduce row using H thread
    {
        int*row  = s_data0 + tid * WPAD;
        int*row1 = s_data1 + tid * WPAD;
        
        int2 res;
        res.x = 0;
        res.y = 0;
        
        for (uint i= 0; i<32; ++i){
            int t  = row[i];
            int t1 = row1[i];
            if (inclusive == 0){
                row[i] = res.x;
                row1[i]= res.y;
                
                if (flag & (1 <<i)){
                    res.x = 0;
                    res.y = 0;
                }
                else {
                    res.x += t;
                    res.y += t1;
                }
            }
            else {
                res.x += t;
                res.y += t1;

                row[i] = res.x;
                row1[i]= res.y;

                res.x  *= ((flag & (1 <<i)) == 0);
                res.y  *= ((flag & (1 <<i)) == 0);
            }
        }
        s_tmp [tid] = res.x;
        s_tmp1[tid] = res.y;
    }
    
    __syncthreads();
    //3. Segmented scan the sum
    if (tid ==0) {
        int2 res     ;

        res.x = 0;
        res.y = 0;

        for (uint i=0; i<32; ++i){
            int t    = s_tmp[i];
            int t1   = s_tmp1[i];
            
            s_tmp [i] = res.x;
            s_tmp1[i] = res.y;

            if (s_flags[i]){
                res.x = t;
                res.y = t1;
            }
            else {
                res.x +=t;
                res.y +=t1;
            }
        }
    }
    __syncthreads();

    int *row  = s_data0 + tid * WPAD;
    int *row1 = s_data1 + tid * WPAD;
    
    int ad    = s_tmp[tid];
    int ad1   = s_tmp1[tid];
    
    for (int i=0; i<32; ++i){
        row[i]  += ad;
        row1[i] += ad1;
        
        if ((flag & (1 <<i)) > 0 ){
            ad = 0;
            ad1= 0;
        }
    }
    
    //4. Write out the result
    i1 = 0;
    for (int i=0; i< 32 * 32; i+=32, i1+=WPAD)
        if (tid + i < n){
            int2 odata;
            odata.x = s_data0[tid + i1];
            odata.y = s_data1[tid + i1];
            
            g_lScan[tid + i] = odata;
        }
}

template<int inclusive>
void segScan(int2* g_scan, int2* g_iData, uint* g_iFlags, int n){

    uint nBlocks = iDivUp(n,1024);
    dim3 threads(32);
    dim3 grids(nBlocks);
    checkConfig(grids);

    if(nBlocks == 1){
        if (inclusive)
            segScanSmall_kernel<1><<<1,32>>>(g_scan, g_iData, g_iFlags, n);
        else
            segScanSmall_kernel<0><<<1,32>>>(g_scan, g_iData, g_iFlags, n);
    }
    else {
        int2* d_temp;
        dmemAlloc(d_temp, nBlocks);
        if (inclusive)
            segScan_block_kernel<1><<<grids, 256>>>(g_scan, g_iData, g_iFlags, n, d_temp);
        else
            segScan_block_kernel<0><<<grids, 256>>>(g_scan, g_iData, g_iFlags, n, d_temp);

        uint* d_comPos;
        dmemAlloc(d_comPos, nBlocks);
        findFirstPos(d_comPos, g_iFlags, iDivUp(n, 32));

        int2* h_temp  = new int2 [nBlocks];
        uint* h_comPos= new uint [nBlocks];

        cudaMemcpy(h_temp  , d_temp  , sizeof(int2) * nBlocks, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_comPos, d_comPos, sizeof(uint) * nBlocks, cudaMemcpyDeviceToHost);

        int2 res , t;
        res.x = res.y = t.x = t.y = 0;
        for (unsigned int i=0;i < nBlocks; ++i){
            t = h_temp[i];
            h_temp[i] = res;
            res +=t;
            if (h_comPos[i] != 1024)
                res = t;
        }

        cudaMemcpy(d_temp, h_temp, sizeof(int2) * nBlocks, cudaMemcpyHostToDevice);

        // Add the value to each block
        postAddScan_block(g_scan, d_temp, d_comPos, n);

        dmemFree(d_temp);
        dmemFree(d_comPos);
        delete [] h_temp;
        delete [] h_comPos;
    }
}

void testSegScan_int2(int n, unsigned int s, int inclusive){
    fprintf(stderr, "Number of input %d\n", n);
    int nSegs = iDivUp(n, 32);

    //1. Initialize random input
    int2* h_iData = new int2[n];
    int2* h_scan  = new int2[n];
    
    for (int i=0; i<n; ++i){
        h_iData[i].x = 1;
        h_iData[i].y = 1;
    }

    //2. Initialize the pos
    bitSet bst(n);
    bst.insert(n-1);
    for (uint i=0; i< s; ++i) {
        int pos = rand() % n;
        bst.insert(pos);
    }


    //3. Get the position
    int total = s + 1;
    int* segEnd = new int[total];
    total = bst.getCount();
    bst.ListAll((uint*)segEnd, total);

    //4. Allocate GPU mem and copy the input

    int2* d_iData;
    dmemAlloc(d_iData, n);
    cudaMemcpy(d_iData , h_iData, sizeof(int2) * n, cudaMemcpyHostToDevice);

    //5. Mark the range of each segment 
    int id = 0;
    int* aId = new int[n];
    for (int i=0; i < n; ++i){
        aId[i] = id;
        if (bst.check(i))
            ++id;
    }

    uint* d_aId;
    dmemAlloc(d_aId, n);
    cudaMemcpy(d_aId, &aId[0], n * sizeof(uint), cudaMemcpyHostToDevice);

    //6. Generate the flags, 1 indicate end of the segment 0 is inside segment
    uint* d_iFlags;
    dmemAlloc(d_iFlags, nSegs);
    buildLastPosFlags(d_iFlags, (uint4*) d_aId, n);

    int2* d_scan;
    dmemAlloc(d_scan, n);

    int nIter = 1000;
    if (inclusive )
        segScan<1>(d_scan, d_iData, d_iFlags, n);
    else
        segScan<0>(d_scan, d_iData, d_iFlags, n);

    uint timer;
    cutCreateTimer(&timer);
    cutStartTimer(timer);

    cutResetTimer(timer);
    for (int i=0; i < nIter; ++i)
        if (inclusive)
            segScan<1>(d_scan, d_iData, d_iFlags, n);
        else
            segScan<0>(d_scan, d_iData, d_iFlags, n);

    cudaThreadSynchronize();
    cutStopTimer(timer);
    printf("Segmentated scan ave execution time: %f ms\n", cutGetTimerValue(timer) / nIter);
    
    if (inclusive)
        segScan_inclusive_cpu(h_scan, h_iData, bst.m_bits, n);
    else 
        segScan_cpu(h_scan, h_iData, bst.m_bits, n);
    testError((int*) h_scan, (int*) d_scan, 2 * n, "scan result");
    
    dmemFree(d_aId);
    dmemFree(d_iData);
    dmemFree(d_iFlags);
    dmemFree(d_scan);

    delete []aId;
    delete []segEnd;
    delete []h_iData;
    delete []h_scan;
}



// function with float2
template<int inclusive>
__global__ void segScan_block_kernel(float2* g_lScan, float2* g_iData, uint* g_iFlags, uint n, float2* g_temp){
    const int WPAD = 32 + 1;
    uint blockId = blockIdx.x + blockIdx.y * gridDim.x;
    uint tid     = threadIdx.x;
    uint id      = tid + blockId * 1024;

    __shared__ float  s_tmp[64];
    __shared__ float  s_tmp1[64];
    __shared__ uint   s_flags[32];
    __shared__ float  s_data0[32*(32+1)];
    __shared__ float  s_data1[32*(32+1)];
    
    //1.Read data to shared mem
    uint flag;

    if (tid < 32){
        if ((blockId * 32 + tid) * 32 < n)
            flag = g_iFlags[blockId * 32 + tid];
        else
            flag = 0;
    }
    
    {
        uint i2 = tid & 31;
        uint i1 = tid >> 5;
    
        for (uint i=0; i< 32 * 32; i+=256, i1+=8){
            uint w = i2 + i1 * 33;
            if (id + i < n) {
                float2 data = g_iData[id + i];
                s_data0[w] = data.x;
                s_data1[w] = data.y;
            }
            else {
                s_data0[w] = 0;
                s_data1[w] = 0;
            }
        }
    }
    
    __syncthreads();
    // Mark the end of the block
    //1.b count the number of bit 1 = the number of segment inside compressed flag
    if (tid < 32){
        uint cnt = popc(flag);
        s_flags[tid]    = (cnt > 0);
    }
    __syncthreads();
    //2. Reduce row using H thread
    if (tid < 32)
    {
        float*row  = s_data0 + tid * WPAD;
        float*row1 = s_data1 + tid * WPAD;
        
        float2 res;
        res.x = 0;
        res.y = 0;
        
        for (uint i= 0; i<32; ++i){
            float t  = row[i];
            float t1 = row1[i];
            if (inclusive == 0){
                row[i] = res.x;
                row1[i]= res.y;
                
                if (flag & (1 <<i)){
                    res.x = 0;
                    res.y = 0;
                }
                else {
                    res.x += t;
                    res.y += t1;
                }
            }
            else {
                res.x += t;
                res.y += t1;

                row[i] = res.x;
                row1[i]= res.y;

                res.x  *= ((flag & (1 <<i)) == 0);
                res.y  *= ((flag & (1 <<i)) == 0);
            }
        }
        s_tmp [tid] = res.x;
        s_tmp1[tid] = res.y;
    }
    
    __syncthreads();
    //3. Segmented scan the sum
    if (tid ==0) {
        float2 res     ;

        res.x = 0;
        res.y = 0;

        for (uint i=0; i<32; ++i){
            float t    = s_tmp[i];
            float t1   = s_tmp1[i];
            
            s_tmp [i] = res.x;
            s_tmp1[i] = res.y;

            if (s_flags[i]){
                res.x = t;
                res.y = t1;
            }
            else {
                res.x +=t;
                res.y +=t1;
            }
        }
        g_temp[blockId] = res;
    }
    __syncthreads();

    if (tid < 32){
        
        float *row  = s_data0 + tid * WPAD;
        float *row1 = s_data1 + tid * WPAD;
    
        float ad    = s_tmp[tid];
        float ad1   = s_tmp1[tid];
    
        for (int i=0; i<32; ++i){
            row[i]  += ad;
            row1[i] += ad1;
        
            if ((flag & (1 <<i)) > 0 ){
                ad = 0;
                ad1= 0;
            }
        }
    }

    __syncthreads();

    //4. Write out the result
    {
        uint i2 = tid & 31;
        uint i1 = tid >> 5;
        for (uint i=0; i< 32 * 32; i+=256, i1+=8){
            uint w = i2 + i1 * 33;
            if (id + i < n) {
                float2 odata;
                odata.x = s_data0[w];
                odata.y = s_data1[w];
                g_lScan[id + i] = odata;
            }
        }
    }
}

template<int inclusive>
__global__ void segScanSmall_kernel(float2* g_lScan, float2* g_iData, uint* g_iFlags, int n){
    const int WPAD = 32 + 1;
    uint tid     = threadIdx.x;
    
    __shared__ float  s_tmp[64];
    __shared__ float  s_tmp1[64];
    __shared__ uint s_flags[32];
    __shared__ float  s_data0[32*(32+1)];
    __shared__ float  s_data1[32*(32+1)];
    
    //1.Read data to shared mem
    uint flag;

    if ( tid * 32 < n)
        flag = g_iFlags[tid];
    else
        flag = 0;

    uint i1 = 0;
    for (int i=0; i< 32 * 32; i+=32, i1+=WPAD)
        if (tid + i < n) {
            float2 data = g_iData[tid + i];
            s_data0[tid + i1] = data.x;
            s_data1[tid + i1] = data.y;
        }
        else {
            s_data0[tid + i1] = 0;
            s_data1[tid + i1] = 0;
        }
    
    __syncthreads();
    // Mark the end of the block
    //1.b count the number of bit 1 = the number of segment inside compressed flag
    uint cnt = popc(flag);
    s_flags[tid]    = (cnt > 0);
    __syncthreads();
    //2. Reduce row using H thread
    {
        float*row  = s_data0 + tid * WPAD;
        float*row1 = s_data1 + tid * WPAD;
        
        float2 res;
        res.x = 0;
        res.y = 0;
        
        for (uint i= 0; i<32; ++i){
            float t  = row[i];
            float t1 = row1[i];
            if (inclusive == 0){
                row[i] = res.x;
                row1[i]= res.y;
                
                if (flag & (1 <<i)){
                    res.x = 0;
                    res.y = 0;
                }
                else {
                    res.x += t;
                    res.y += t1;
                }
            }
            else {
                res.x += t;
                res.y += t1;

                row[i] = res.x;
                row1[i]= res.y;

                res.x  *= ((flag & (1 <<i)) == 0);
                res.y  *= ((flag & (1 <<i)) == 0);
            }
        }
        s_tmp [tid] = res.x;
        s_tmp1[tid] = res.y;
    }
    
    __syncthreads();
    //3. Segmented scan the sum
    if (tid ==0) {
        float2 res     ;

        res.x = 0;
        res.y = 0;

        for (uint i=0; i<32; ++i){
            float t    = s_tmp[i];
            float t1   = s_tmp1[i];
            
            s_tmp [i] = res.x;
            s_tmp1[i] = res.y;

            if (s_flags[i]){
                res.x = t;
                res.y = t1;
            }
            else {
                res.x +=t;
                res.y +=t1;
            }
        }
    }
    __syncthreads();

    float *row  = s_data0 + tid * WPAD;
    float *row1 = s_data1 + tid * WPAD;
    
    float ad    = s_tmp[tid];
    float ad1   = s_tmp1[tid];
    
    for (int i=0; i<32; ++i){
        row[i]  += ad;
        row1[i] += ad1;
        
        if ((flag & (1 <<i)) > 0 ){
            ad = 0;
            ad1= 0;
        }
    }
    
    //4. Write out the result
    i1 = 0;
    for (int i=0; i< 32 * 32; i+=32, i1+=WPAD)
        if (tid + i < n){
            float2 odata;
            odata.x = s_data0[tid + i1];
            odata.y = s_data1[tid + i1];
            
            g_lScan[tid + i] = odata;
        }
}

template<int inclusive>
void segScan(float2* g_scan, float2* g_iData, uint* g_iFlags, int n){

    uint nBlocks = iDivUp(n,1024);
    dim3 threads(32);
    dim3 grids(nBlocks);
    checkConfig(grids);

    if(nBlocks == 1){
        if (inclusive)
            segScanSmall_kernel<1><<<1,32>>>(g_scan, g_iData, g_iFlags, n);
        else
            segScanSmall_kernel<0><<<1,32>>>(g_scan, g_iData, g_iFlags, n);
    }
    else {
        float2* d_temp;

        // Perform scan on the block
        // Output the local scan value
        // Output the last scan value of each block
        dmemAlloc(d_temp, nBlocks);

        if (inclusive)
            segScan_block_kernel<1><<<grids, 256>>>(g_scan, g_iData, g_iFlags, n, d_temp);
        else
            segScan_block_kernel<0><<<grids, 256>>>(g_scan, g_iData, g_iFlags, n, d_temp);

        uint* d_comPos;
        dmemAlloc(d_comPos, nBlocks);
        findFirstPos(d_comPos, g_iFlags, iDivUp(n, 32));

        
        float2* h_temp  = new float2 [nBlocks];
        uint* h_comPos= new uint [nBlocks];

        cudaMemcpy(h_temp  , d_temp  , sizeof(float2) * nBlocks, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_comPos, d_comPos, sizeof(uint) * nBlocks, cudaMemcpyDeviceToHost);

        float2 res , t;
        res.x = res.y = t.x = t.y = 0;
        for (unsigned int i=0;i < nBlocks; ++i){
            t = h_temp[i];
            h_temp[i] = res;
            res +=t;
            if (h_comPos[i] != 1024)
                res = t;
        }

        cudaMemcpy(d_temp, h_temp, sizeof(float2) * nBlocks, cudaMemcpyHostToDevice);

        // Add the value to each block
        postAddScan_block(g_scan, d_temp, d_comPos, n);

        dmemFree(d_temp);
        dmemFree(d_comPos);
        delete [] h_temp;
        delete [] h_comPos;
    }
}

void testSegScan_float2(int n, unsigned int s, int inclusive){
    fprintf(stderr, "Number of input %d\n", n);
    int nSegs = iDivUp(n, 32);

    //1. Initialize random input
    float2* h_iData = new float2[n];
    float2* h_scan  = new float2[n];
    
    for (int i=0; i<n; ++i){
        h_iData[i].x = 1;
        h_iData[i].y = 1;
    }

    //2. Initialize the pos
    bitSet bst(n);
    bst.insert(n-1);
    for (uint i=0; i< s; ++i) {
        int pos = rand() % n;
        bst.insert(pos);
    }


    //3. Get the position
    int total = s + 1;
    int* segEnd = new int[total];
    total = bst.getCount();
    bst.ListAll((uint*)segEnd, total);

    //4. Allocate GPU mem and copy the input

    float2* d_iData;
    dmemAlloc(d_iData, n);
    cudaMemcpy(d_iData , h_iData, sizeof(float2) * n, cudaMemcpyHostToDevice);

    //5. Mark the range of each segment 
    int id = 0;
    int* aId = new int[n];
    for (int i=0; i < n; ++i){
        aId[i] = id;
        if (bst.check(i))
            ++id;
    }

    uint* d_aId;
    dmemAlloc(d_aId, n);
    cudaMemcpy(d_aId, &aId[0], n * sizeof(uint), cudaMemcpyHostToDevice);

    //6. Generate the flags, 1 indicate end of the segment 0 is inside segment
    uint* d_iFlags;
    dmemAlloc(d_iFlags, nSegs);
    buildLastPosFlags(d_iFlags, (uint4*) d_aId, n);

    float2* d_scan;
    dmemAlloc(d_scan, n);

    int nIter = 1000;
    if (inclusive )
        segScan<1>(d_scan, d_iData, d_iFlags, n);
    else
        segScan<0>(d_scan, d_iData, d_iFlags, n);

    uint timer;
    cutCreateTimer(&timer);
    cutStartTimer(timer);

    cutResetTimer(timer);
    for (int i=0; i < nIter; ++i)
        if (inclusive)
            segScan<1>(d_scan, d_iData, d_iFlags, n);
        else
            segScan<0>(d_scan, d_iData, d_iFlags, n);

    cudaThreadSynchronize();
    cutStopTimer(timer);
    printf("Segmentated scan ave execution time: %f ms\n", cutGetTimerValue(timer) / nIter);
    
    if (inclusive)
        segScan_inclusive_cpu(h_scan, h_iData, bst.m_bits, n);
    else 
        segScan_cpu(h_scan, h_iData, bst.m_bits, n);

    testError((float*) h_scan, (float*) d_scan, 1e-4, 2 * n, "scan result");
    
    dmemFree(d_aId);
    dmemFree(d_iData);
    dmemFree(d_iFlags);
    dmemFree(d_scan);

    delete []aId;
    delete []segEnd;
    delete []h_iData;
    delete []h_scan;
}


/*----------------------------------------------------------------------
 Function for double
 *--------------------------------------------------------------------------*/

#ifdef DOUBLE_SUPPORT

#include <double_math.h>

void segScan_inclusive_cpu(double2* scan, double2* idata, uint* iflags, int n){
    int nSegs  = iDivUp(n, 32);
    int flag;
    double2   sum = make_double2(0,0);
    int id  = 0;

    for (int j=0; j< nSegs; ++j){
        flag = iflags[j];
        for (int i=0; i<32 && id < n; ++i, ++id){
            double2 old    = idata[id];
            sum     += old;
            scan[id] = sum;
            if (flag & (1<<i))
                sum = make_double2(0,0);
        }
    }
}

void segScan_cpu(double2* scan, double2* idata, uint* iflags, int n){
    int nSegs  = iDivUp(n, 32);
    int flag;
    double2  sum = make_double2(0,0);
    int id  = 0;

    for (int j=0; j< nSegs; ++j){
        flag = iflags[j];
        for (int i=0; i<32 && id < n; ++i, ++id){
            double2 old    = idata[id];
            scan[id] = sum;
            sum     += old;
            if (flag & (1<<i))
                sum = make_double2(0,0);
        }
    }
}

template<int inclusive>
__global__ void segScan_block_kernel(double* g_lScan, double* g_iData, uint* g_iFlags, uint n, double* g_temp){
    const int WPAD = 32 + 1;
    uint blockId = blockIdx.x + blockIdx.y * gridDim.x;
    uint tid     = threadIdx.x;
    uint id      = tid + blockId * 1024;

    __shared__ int  s_tmp[64];
    __shared__ int  s_tmp1[64];
    __shared__ int  s_data0[32*WPAD];
    __shared__ int  s_data1[32*WPAD];
    __shared__ uint s_flags[32];
    
    //1.Read data to shared mem
    uint flag;
    if (tid < 32){
        if ((blockId * 32 + tid) * 32 < n)
            flag = g_iFlags[blockId * 32 + tid];
        else
            flag = 0;
    }
    
    {
        uint i2 = tid & 31;
        uint i1 = tid >> 5;
    
        for (uint i=0; i< 32 * 32; i+=256, i1+=8){
            uint w = i2 + i1 * WPAD;
            if (id + i < n) {
                double data = g_iData[id + i];
                s_data0[w] = __double2hiint(data);
                s_data1[w] = __double2loint(data);
            }
            else {
                s_data0[w] = 0;
                s_data1[w] = 0;
            }
        }
    }
    
    __syncthreads();
    // Mark the end of the block
    //1.b count the number of bit 1 = the number of segment inside compressed flag
    if (tid < 32){
        s_flags[tid] = __popc(flag);
    }

    __syncthreads();
    //2. Reduce row using H thread
    if (tid < 32)
    {
        int*row  = s_data0 + tid * WPAD;
        int*row1 = s_data1 + tid * WPAD;
        
        double res = 0;

        for (uint i= 0; i<32; ++i){
            double t = __hiloint2double(row[i], row1[i]);
            
            if (inclusive == 0){
                row[i] = __double2hiint(res);
                row1[i]= __double2loint(res);

                res = (flag & (1 <<i)) ? 0 : res + t;
            }
            else {
                res += t;

                row[i] = __double2hiint(res);
                row1[i]= __double2loint(res);

                res  *= ((flag & (1 <<i)) == 0);
            }
        }
        
        s_tmp [tid] = __double2hiint(res);
        s_tmp1[tid] = __double2loint(res);
    }
    
    __syncthreads();
    
    //3. Segmented scan the sum
    if (tid ==0) {
        double res = 0;
        
        for (uint i=0; i<32; ++i){
            double t = __hiloint2double(s_tmp[i], s_tmp1[i]);

            s_tmp [i] = __double2hiint(res);
            s_tmp1[i] = __double2loint(res);

            res = (s_flags[i]) ? t : res + t;
        }
        g_temp[blockId] = res;
    }

    __syncthreads();

    if (tid < 32){
        
        int *row  = s_data0 + tid * WPAD;
        int *row1 = s_data1 + tid * WPAD;

        double ad = __hiloint2double(s_tmp[tid] ,s_tmp1[tid]);
        
        for (int i=0; i<32; ++i){

            double res = __hiloint2double(row[i],row1[i]);

            res = res + ad;

            row[i] = __double2hiint(res);
            row1[i]= __double2loint(res);
            
            if ((flag & (1 <<i)) > 0 )
                ad = 0;
        }
    }
    
    __syncthreads();

    //4. Write out the result
    {
        uint i2 = tid & 31;
        uint i1 = tid >> 5;

        for (uint i=0; i< 32 * 32; i+=256, i1+=8){
            uint w = i2 + i1 * WPAD;
            if (id + i < n) {
                double odata;
                odata = __hiloint2double(s_data0[w], s_data1[w]);
                g_lScan[id + i] = odata;
            }
        }
    }
}

template<int inclusive>
__global__ void segScanSmall_kernel(double* g_lScan, double* g_iData, uint* g_iFlags, int n){
    const int WPAD = 32 + 1;
    uint tid     = threadIdx.x;
    
    __shared__ int  s_tmp[64];
    __shared__ int  s_tmp1[64];
    __shared__ uint s_flags[32];
    
    __shared__ int  s_data0[32*(32+1)];
    __shared__ int  s_data1[32*(32+1)];
    
    //1.Read data to shared mem
    uint flag;
    if ( tid * 32 < n)
        flag = g_iFlags[tid];
    else
        flag = 0;

    uint i1 = 0;
    for (int i=0; i< 32 * 32; i+=32, i1+=WPAD)
        if (tid + i < n) {
            double data = g_iData[tid + i];
            s_data0[tid + i1] = __double2hiint(data);
            s_data1[tid + i1] = __double2loint(data);
        }
        else {
            s_data0[tid + i1] = 0;
            s_data1[tid + i1] = 0;
        }
    
    __syncthreads();
    // Mark the end of the block
    //1.b count the number of bit 1 = the number of segment inside compressed flag
    uint cnt = popc(flag);
    s_flags[tid]    = (cnt > 0);
    __syncthreads();
    //2. Reduce row using H thread
    {
        int*row  = s_data0 + tid * WPAD;
        int*row1 = s_data1 + tid * WPAD;
        
        double res = 0;
        
        for (uint i= 0; i<32; ++i){
            double t = __hiloint2double(row[i], row1[i]);
            
            if (inclusive == 0){
                row[i] = __double2hiint(res);
                row1[i]= __double2loint(res);
                
                res = (flag & (1 <<i)) ? 0 : res + t;
            }
            else {
                res +=t;

                row[i] = __double2hiint(res);
                row1[i]= __double2loint(res);

                if ((flag & (1 <<i)) == 1)
                    res = 0;
            }
        }
        s_tmp [tid] = __double2hiint(res);
        s_tmp1[tid] = __double2loint(res);
    }
    
    __syncthreads();
    //3. Segmented scan the sum
    if (tid ==0) {
        double res = 0;
        
        for (uint i=0; i<32; ++i){
            double t = __hiloint2double(s_tmp[i], s_tmp1[i]);

            s_tmp[i] = __double2hiint(res);
            s_tmp1[i]= __double2loint(res);

            res = (s_flags[i]) ? t : res + t;
        }
    }
    __syncthreads();

    int *row  = s_data0 + tid * WPAD;
    int *row1 = s_data1 + tid * WPAD;
    
    double ad = __hiloint2double(s_tmp[tid],  s_tmp1[tid]);
    
    for (int i=0; i<32; ++i){
        double res = __hiloint2double(row[i],  row1[i]);
        res += ad;
        row[i]  = __double2hiint(res);
        row1[i] = __double2loint(res);
        if ((flag & (1 <<i)) > 0 )
            ad = 0;
    }
    
    //4. Write out the result
    i1 = 0;
    for (int i=0; i< 32 * 32; i+=32, i1+=WPAD)
        if (tid + i < n){
            double odata;
            odata = __hiloint2double(s_data0[tid + i1], s_data1[tid + i1]);
            g_lScan[tid + i] = odata;
        }
}

template<int inclusive>
void segScanDouble(double* g_scan, double* g_iData, uint* g_iFlags, int n){

    uint nBlocks = iDivUp(n,1024);
    dim3 threads(32);
    dim3 grids(nBlocks);
    checkConfig(grids);

    if(nBlocks == 1){
        if (inclusive)
            segScanSmall_kernel<1><<<1,32>>>(g_scan, g_iData, g_iFlags, n);
        else
            segScanSmall_kernel<0><<<1,32>>>(g_scan, g_iData, g_iFlags, n);

        
    }
    else {
        double* d_temp;

        // Perform scan on the block
        // Output the local scan value
        // Output the last scan value of each block
        dmemAlloc(d_temp, nBlocks);

        if (inclusive)
            segScan_block_kernel<1><<<grids, 256>>>(g_scan, g_iData, g_iFlags, n, d_temp);
        else
            segScan_block_kernel<0><<<grids, 256>>>(g_scan, g_iData, g_iFlags, n, d_temp);

        uint* d_comPos;
        dmemAlloc(d_comPos, nBlocks);
        findFirstPos(d_comPos, g_iFlags, iDivUp(n, 32));
        
        double* h_temp= new double [nBlocks];
        uint* h_comPos= new uint [nBlocks];

        cudaMemcpy(h_temp  , d_temp  , sizeof(double) * nBlocks, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_comPos, d_comPos, sizeof(uint) * nBlocks, cudaMemcpyDeviceToHost);

        double res =0, t =0;
        for (int i=0;i < nBlocks; ++i){
            t = h_temp[i];
            h_temp[i] = res;
            res +=t;
            if (h_comPos[i] != 1024)
                res = t;
        }

        cudaMemcpy(d_temp, h_temp, sizeof(double) * nBlocks, cudaMemcpyHostToDevice);

        // Add the value to each block
        postAddScan_block(g_scan, d_temp, d_comPos, n);

        dmemFree(d_temp);
        dmemFree(d_comPos);
        delete [] h_temp;
        delete [] h_comPos;
    }
}

void testSegScanDouble(int n, int s, int inclusive){
    fprintf(stderr, "Test scan double function Number of input %d\n", n);
    int nSegs = iDivUp(n, 32);

//1. Initialize random input
    double* h_iData = new double[n];
    double* h_scan  = new double[n];

    for (int i=0; i<n; ++i)
        h_iData[i] = 1;

    //2. Initialize the pos
    bitSet bst(n);
    bst.insert(n-1);
    for (double i=0; i< s; ++i) {
        int pos = rand() % n;
        bst.insert(pos);
    }


    //3. Get the position
    int total = s + 1;
    int* segEnd = new int[total];
    total = bst.getCount();

    bst.ListAll((uint*)segEnd, total);

    //4. Allocate GPU mem and copy the input

    double* d_iData;
    allocateDeviceArray((void**) &d_iData, sizeof(double) * n);
    cudaMemcpy(d_iData , h_iData, sizeof(double) * n, cudaMemcpyHostToDevice);

    //5. Mark the range of each segment 
    int id = 0;
    int* aId = new int[n];
    for (int i=0; i < n; ++i){
        aId[i] = id;
        if (bst.check(i))
            ++id;
    }

    uint* d_aId;
    dmemAlloc(d_aId, n);
    cudaMemcpy(d_aId, &aId[0], n * sizeof(uint), cudaMemcpyHostToDevice);

    //6. Generate the flags, 1 indicate end of the segment 0 is inside segment
    uint* d_iFlags;
    dmemAlloc(d_iFlags, nSegs);
    buildLastPosFlags(d_iFlags, (uint4*) d_aId, n);

    double* d_scan;
    dmemAlloc(d_scan, n);

    int nIter = 100;
    if (inclusive )
        segScanDouble<1>(d_scan, d_iData, d_iFlags, n);
    else
        segScanDouble<0>(d_scan, d_iData, d_iFlags, n);
    
    /*
    uint timer;
    cutCreateTimer(&timer);
    cutStartTimer(timer);

    cutResetTimer(timer);
    for (int i=0; i < nIter; ++i)
        if (inclusive)
            segScan<uint, 1>(d_scan, d_iData, d_iFlags, n);
        else
            segScan<uint, 0>(d_scan, d_iData, d_iFlags, n);

    cudaThreadSynchronize();
    cutStopTimer(timer);
    printf("Segmentated scan ave execution time: %f ms\n", cutGetTimerValue(timer) / nIter);
    */
    
    if (inclusive)
        segScan_inclusive_cpu(h_scan, h_iData, bst.m_bits, n);
    else 
        segScan_cpu(h_scan, h_iData, bst.m_bits, n);

    compare(h_scan, d_scan, n, 1e-8, "Scan result");

    dmemFree(d_aId);
    dmemFree(d_iData);
    dmemFree(d_iFlags);
    dmemFree(d_scan);

    delete []aId;
    delete []segEnd;
    delete []h_iData;
    delete []h_scan;
}


/*
template<int inclusive>
__global__ void segScan_block_kernel(double2* g_lScan, double2* g_iData, uint* g_iFlags, uint n, double2* g_temp){
    const int WPAD = 32 + 1;
    uint blockId = blockIdx.x + blockIdx.y * gridDim.x;
    uint tid     = threadIdx.x;
    uint id      = tid + blockId * 1024;

    __shared__ double  s_tmp[64];
    __shared__ double  s_tmp1[64];
    __shared__ uint    s_flags[32];
    __shared__ double  s_data0[32*(32+1)];
    __shared__ double  s_data1[32*(32+1)];
    
    //1.Read data to shared mem
    uint flag;

    if (tid < 32){
        if ((blockId * 32 + tid) * 32 < n)
            flag = g_iFlags[blockId * 32 + tid];
        else
            flag = 0;
    }
    
    {
        uint i2 = tid & 31;
        uint i1 = tid >> 5;
    
        for (uint i=0; i< 32 * 32; i+=256, i1+=8){
            uint w = i2 + i1 * 33;
            if (id + i < n) {
                double2 data = g_iData[id + i];
                s_data0[w] = data.x;
                s_data1[w] = data.y;
            }
            else {
                s_data0[w] = 0;
                s_data1[w] = 0;
            }
        }
    }
    
    __syncthreads();
    // Mark the end of the block
    //1.b count the number of bit 1 = the number of segment inside compressed flag
    if (tid < 32){
        uint cnt = popc(flag);
        s_flags[tid]    = (cnt > 0);
    }
    __syncthreads();
    //2. Reduce row using H thread
    if (tid < 32)
    {
        double*row  = s_data0 + tid * WPAD;
        double*row1 = s_data1 + tid * WPAD;
        
        double2 res;
        res.x = 0;
        res.y = 0;
        
        for (uint i= 0; i<32; ++i){
            double t  = row[i];
            double t1 = row1[i];
            if (inclusive == 0){
                row[i] = res.x;
                row1[i]= res.y;
                
                if (flag & (1 <<i)){
                    res.x = 0;
                    res.y = 0;
                }
                else {
                    res.x += t;
                    res.y += t1;
                }
            }
            else {
                res.x += t;
                res.y += t1;

                row[i] = res.x;
                row1[i]= res.y;

                res.x  *= ((flag & (1 <<i)) == 0);
                res.y  *= ((flag & (1 <<i)) == 0);
            }
        }
        s_tmp [tid] = res.x;
        s_tmp1[tid] = res.y;
    }
    
    __syncthreads();
    //3. Segmented scan the sum
    if (tid ==0) {
        double2 res     ;

        res.x = 0;
        res.y = 0;

        for (uint i=0; i<32; ++i){
            double t    = s_tmp[i];
            double t1   = s_tmp1[i];
            
            s_tmp [i] = res.x;
            s_tmp1[i] = res.y;

            if (s_flags[i]){
                res.x = t;
                res.y = t1;
            }
            else {
                res.x +=t;
                res.y +=t1;
            }
        }
        g_temp[blockId] = res;
    }
    __syncthreads();

    if (tid < 32){
        
        double *row  = s_data0 + tid * WPAD;
        double *row1 = s_data1 + tid * WPAD;
    
        double ad    = s_tmp[tid];
        double ad1   = s_tmp1[tid];
    
        for (int i=0; i<32; ++i){
            row[i]  += ad;
            row1[i] += ad1;
        
            if ((flag & (1 <<i)) > 0 ){
                ad = 0;
                ad1= 0;
            }
        }
    }

    __syncthreads();

    //4. Write out the result
    {
        uint i2 = tid & 31;
        uint i1 = tid >> 5;
        for (uint i=0; i< 32 * 32; i+=256, i1+=8){
            uint w = i2 + i1 * 33;
            if (id + i < n) {
                double2 odata;
                odata.x = s_data0[w];
                odata.y = s_data1[w];
                g_lScan[id + i] = odata;
            }
        }
    }
}

template<int inclusive>
__global__ void segScanSmall_kernel(double2* g_lScan, double2* g_iData, uint* g_iFlags, int n){
    const int WPAD = 32 + 1;
    uint tid     = threadIdx.x;
    __shared__ double  s_tmp[64];
    __shared__ double  s_tmp1[64];
    __shared__ uint s_flags[32];
    __shared__ double  s_data0[32*(32+1)];
    __shared__ double  s_data1[32*(32+1)];
    
    //1.Read data to shared mem
    uint flag;

    if ( tid * 32 < n)
        flag = g_iFlags[tid];
    else
        flag = 0;

    uint i1 = 0;
    for (int i=0; i< 32 * 32; i+=32, i1+=WPAD)
        if (tid + i < n) {
            double2 data = g_iData[tid + i];
            s_data0[tid + i1] = data.x;
            s_data1[tid + i1] = data.y;
        }
        else {
            s_data0[tid + i1] = 0;
            s_data1[tid + i1] = 0;
        }
    
    __syncthreads();
    // Mark the end of the block
    //1.b count the number of bit 1 = the number of segment inside compressed flag
    uint cnt = popc(flag);
    s_flags[tid]    = (cnt > 0);
    
    __syncthreads();
    //2. Reduce row using H thread
    {
        double*row  = s_data0 + tid * WPAD;
        double*row1 = s_data1 + tid * WPAD;
        
        double2 res;
        res.x = 0;
        res.y = 0;
        
        for (uint i= 0; i<32; ++i){
            double t  = row[i];
            double t1 = row1[i];
            if (inclusive == 0){
                row[i] = res.x;
                row1[i]= res.y;
                
                if (flag & (1 <<i)){
                    res.x = 0;
                    res.y = 0;
                }
                else {
                    res.x += t;
                    res.y += t1;
                }
            }
            else {
                res.x += t;
                res.y += t1;

                row[i] = res.x;
                row1[i]= res.y;

                res.x  *= ((flag & (1 <<i)) == 0);
                res.y  *= ((flag & (1 <<i)) == 0);
            }
        }
        s_tmp [tid] = res.x;
        s_tmp1[tid] = res.y;
    }
    
    __syncthreads();
    //3. Segmented scan the sum
    if (tid ==0) {
        double2 res     ;

        res.x = 0;
        res.y = 0;

        for (uint i=0; i<32; ++i){
            double t    = s_tmp[i];
            double t1   = s_tmp1[i];
            
            s_tmp [i] = res.x;
            s_tmp1[i] = res.y;

            if (s_flags[i]){
                res.x = t;
                res.y = t1;
            }
            else {
                res.x +=t;
                res.y +=t1;
            }
        }
    }
    __syncthreads();

    //4. carry the result from previous segment
    double *row  = s_data0 + tid * WPAD;
    double *row1 = s_data1 + tid * WPAD;
    
    double ad    = s_tmp[tid];
    double ad1   = s_tmp1[tid];
    
    for (int i=0; i<32; ++i){
        row[i]  += ad;
        row1[i] += ad1;
        
        if ((flag & (1 <<i)) > 0 ){
            ad = 0;
            ad1= 0;
        }
    }
    
    //4. Write out the result
    i1 = 0;
    for (int i=0; i< 32 * 32; i+=32, i1+=WPAD)
        if (tid + i < n){
            double2 odata;
            odata.x = s_data0[tid + i1];
            odata.y = s_data1[tid + i1];
            
            g_lScan[tid + i] = odata;
        }
}


template<int inclusive>
void segScan(double2* g_scan, double2* g_iData, uint* g_iFlags, int n){

    uint nBlocks = iDivUp(n,1024);
    dim3 threads(32);
    dim3 grids(nBlocks);
    checkConfig(grids);

    if(nBlocks == 1){
        if (inclusive)
            segScanSmall_kernel<1><<<1,32>>>(g_scan, g_iData, g_iFlags, n);
        else
            segScanSmall_kernel<0><<<1,32>>>(g_scan, g_iData, g_iFlags, n);
    }
    else {
        double2* d_temp;

        // Perform scan on the block
        // Output the local scan value
        // Output the last scan value of each block
        
        dmemAlloc(d_temp, nBlocks);

        
        if (inclusive)
            segScan_block_kernel<1><<<grids, 256>>>(g_scan, g_iData, g_iFlags, n, d_temp);
        else
            segScan_block_kernel<0><<<grids, 256>>>(g_scan, g_iData, g_iFlags, n, d_temp);

        uint* d_comPos;
        dmemAlloc(d_comPos, nBlocks);
        findFirstPos(d_comPos, g_iFlags, iDivUp(n, 32));

        
        double2* h_temp  = new double2 [nBlocks];
        uint* h_comPos= new uint [nBlocks];

        cudaMemcpy(h_temp  , d_temp  , sizeof(double2) * nBlocks, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_comPos, d_comPos, sizeof(uint) * nBlocks, cudaMemcpyDeviceToHost);

        double2 res , t;
        res.x = res.y = t.x = t.y = 0;
        for (int i=0;i < nBlocks; ++i){
            t = h_temp[i];
            h_temp[i] = res;
            res +=t;
            if (h_comPos[i] != 1024)
                res = t;
        }

        cudaMemcpy(d_temp, h_temp, sizeof(double2) * nBlocks, cudaMemcpyHostToDevice);

        // Add the value to each block
        postAddScan_block(g_scan, d_temp, d_comPos, n);

        dmemFree(d_temp);
        dmemFree(d_comPos);
        delete [] h_temp;
        delete [] h_comPos;
    }
}

void testSegScan_double2(int n, int s, int inclusive){
    fprintf(stderr, "Number of input %d\n", n);
    int nSegs = iDivUp(n, 32);

    //1. Initialize random input
    double2* h_iData = new double2[n];
    double2* h_scan  = new double2[n];
    
    for (int i=0; i<n; ++i){
        h_iData[i].x = 1;
        h_iData[i].y = 1;
    }

    //2. Initialize the pos
    bitSet bst(n);
    bst.insert(n-1);
    for (uint i=0; i< s; ++i) {
        int pos = rand() % n;
        bst.insert(pos);
    }


    //3. Get the position
    int total = s + 1;
    int* segEnd = new int[total];
    total = bst.getCount();
    bst.ListAll((uint*)segEnd, total);

    //4. Allocate GPU mem and copy the input

    double2* d_iData;
    allocateDeviceArray((void**) &d_iData, sizeof(double2) * n);
    cudaMemcpy(d_iData , h_iData, sizeof(double2) * n, cudaMemcpyHostToDevice);

    //5. Mark the range of each segment 
    int id = 0;
    int* aId = new int[n];
    for (int i=0; i < n; ++i){
        aId[i] = id;
        if (bst.check(i))
            ++id;
    }

    uint* d_aId;
    dmemAlloc(d_aId, n);
    cudaMemcpy(d_aId, &aId[0], n * sizeof(uint), cudaMemcpyHostToDevice);

    //6. Generate the flags, 1 indicate end of the segment 0 is inside segment
    uint* d_iFlags;
    dmemAlloc(d_iFlags,nSegs);
    buildLastPosFlags(d_iFlags, (uint4*) d_aId, n);

    double2* d_scan;
    dmemAlloc(d_scan, n);

    int nIter = 1000;
    if (inclusive )
        segScan<1>(d_scan, d_iData, d_iFlags, n);
    else
        segScan<0>(d_scan, d_iData, d_iFlags, n);

    uint timer;
    cutCreateTimer(&timer);
    cutStartTimer(timer);

    cutResetTimer(timer);
    for (int i=0; i < nIter; ++i)
        if (inclusive)
            segScan<1>(d_scan, d_iData, d_iFlags, n);
        else
            segScan<0>(d_scan, d_iData, d_iFlags, n);

    cudaThreadSynchronize();
    cutStopTimer(timer);
    printf("Segmentated scan ave execution time: %f ms\n", cutGetTimerValue(timer) / nIter);
    
    if (inclusive)
        segScan_inclusive_cpu(h_scan, h_iData, bst.m_bits, n);
    else 
        segScan_cpu(h_scan, h_iData, bst.m_bits, n);
    
    //testError((double*) h_scan, (double*) d_scan, 1e-4, 2 * n, "scan result");
    compare((double*)h_scan, (double*) d_scan, 2 * n, 1e-8, "Scan result");
        
    dmemFree(d_aId);
    dmemFree(d_iData);
    dmemFree(d_iFlags);
    dmemFree(d_scan);

    delete []aId;
    delete []segEnd;
    delete []h_iData;
    delete []h_scan;
}
*/
#endif
#endif
