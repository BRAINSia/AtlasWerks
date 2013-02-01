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

#include <cutil_inline.h>
#include <cudaSortConfig.h>
#include "cutil_comfunc.h"
#include <cudaSortUtils.h>
#include <cudaSort.h>
#include <assert.h>
#include <stdio.h>
#include <cudaInterface.h>
#include <cudaEventTimer.h>

bool checkHostOrderBits4(uint* d_a, unsigned int n, int startBit, const char* name);
bool checkDeviceOrderBits4(uint* d_a, unsigned int n, int startBit, const char* name);
bool checkHostPairOrderBits4(uint2* d_a, unsigned int n, int startBit, const char* name);
bool checkDevicePairOrderBits4(uint2* d_a, unsigned int n, int startBit, const char* name);

inline bool checkHostOrderBits4(unsigned int* h_a, unsigned int n, int startBit, const char *name){
    unsigned int i=0;
    for (i=0; i< n-1; ++i){
        if (((h_a[i] >> startBit) & 0x0F) > ((h_a[i+1] >> startBit) & 0x0F)){
            std::cerr << "Test " << name << " at bits " << startBit <<" FAILED:  Wrong order at "<< i <<" "<< h_a[i] <<" "<< h_a[i+1] << std::endl;
            return false;
        }
    }
    std::cerr << "Test " << name << " at bits " << startBit << " PASSED " << std::endl;
    return true;
}


inline bool checkDeviceOrderBits4(uint* d_a, unsigned int n, int startBit, const char* name){
    uint* h_a = new uint[n];
    cudaMemcpy((void*)h_a, (void*)d_a, n * sizeof(uint), cudaMemcpyDeviceToHost);
    bool r = checkHostOrderBits4(h_a, n, startBit, name);
    delete []h_a;
    return r;
}

inline bool checkHostPairOrderBits4(uint2* h_a, unsigned int n, int startBit, const char* name){
    unsigned int i=0;
    std::cerr << "Check bit " << startBit << " ";
    for (i=0; i< n-1; ++i){
        if (((h_a[i].x >> startBit) & 0x0F) > ((h_a[i+1].x >> startBit) & 0x0F)){
            std::cerr << "Test " << name << " FAILED: Wrong order at "<< i <<" "<< h_a[i].x <<" "<< h_a[i+1].x << std::endl;
            return false;
        }
    }
    if (i == n-1)
        std::cerr << "Test " << name << "PASSED " << std::endl;
    return true;
}

inline bool checkDevicePairOrderBits4(uint2* d_a, unsigned int n, int startBit, const char* name){
    uint2* h_a = new uint2[n];
    cudaMemcpy((void*)h_a, (void*)d_a, n * sizeof(uint2), cudaMemcpyDeviceToHost);
    bool r = checkHostPairOrderBits4(h_a, n, startBit, name);
    delete []h_a;
    return r;
}



__device__ uint get_blockID(){
    return blockIdx.x + blockIdx.y * gridDim.x;
}

__device__ uint get_threadID(uint blockId){
    return blockId * blockDim.x + threadIdx.x;
}

__device__ uint intFlip(uint f){
    return (f  ^ ((unsigned int)1 << 31));
}

//////////////////////////////////////////////////////////////////////////////////
// Flip a float for sorting
//  finds SIGN of fp number.
//  if it's 1 (negative float), it flips all bits
//  if it's 0 (positive float), it flips the sign only
//////////////////////////////////////////////////////////////////////////////////

template <bool doFlip>
__device__ uint floatFlip(uint f)
{
    if (doFlip)
    {
        uint mask = -int(f >> 31) | 0x80000000;
	return f ^ mask;
    }
    else
        return f;
}

////////////////////////////////////////////////////////////////////////////////
// flip a float back (invert FloatFlip)
//  signed was flipped from above, so:
//  if sign is 1 (negative), it flips the sign bit back
//  if sign is 0 (positive), it flips all bits back
////////////////////////////////////////////////////////////////////////////////
template <bool doFlip>
__device__ uint floatUnflip(uint f)
{
    if (doFlip)
    {
        uint mask = ((f >> 31) - 1) | 0x80000000;
	    return f ^ mask;
    }
    else
        return f;
}

//////////////////////////////////////////////////////////////////////////////////
// Supplement funtion
//////////////////////////////////////////////////////////////////////////////////
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
// return the number of the same types before the last component of the current data
// of the implicit counting
__device__ uint4 count4(uint4 data){
#if (CUDART_VERSION  < 2020)
    extern __shared__ uint ptr[];
#else
    extern volatile __shared__ uint ptr[];
#endif
    uint idx = threadIdx.x;

    uint4 bmask  = data;

    uint  sum[3];
    sum[0] = ((bmask.x<3) << (bmask.x * 10));
    sum[1] = ((bmask.y<3) << (bmask.y * 10)) + sum[0];
    sum[2] = ((bmask.z<3) << (bmask.z * 10)) + sum[1];
    
    uint mycount =  ((bmask.w<3) << (bmask.w * 10)) + sum[2];

    // scan on the warp 
    mycount = scanwarp<uint, WARP_SIZE_LOG - 1>(mycount, ptr);

    __syncthreads();
    
    if ((idx & (WARP_SIZE-1)) == (WARP_SIZE-1)){
        ptr[idx >> 5] = mycount + ((bmask.w<3) << (bmask.w * 10)) + sum[2];
    }

    __syncthreads();

    // scan on the warp result
#ifndef __DEVICE_EMULATION__
    if (idx < WARP_SIZE)
#endif
    {
        ptr[idx] = scanwarp<uint, WARP_PER_BLOCK_LOG - 1>(ptr[idx], ptr);
    }

    __syncthreads();
    mycount +=ptr[idx>>5];

    bmask.x = mycount;
    bmask.y = mycount + sum[0];
    bmask.z = mycount + sum[1];
    bmask.w = mycount + sum[2];

    return bmask;
}

template<int ctasize>
__device__ uint4 irank4(uint4 data){
    uint4 address = count4(data);
    
    // the total contain the number of 0, 1, 2
    __shared__ uint more[32];

    uint& total = more[0];
    
    
    if (threadIdx.x == ctasize -1){
        total = address.w + ((data.w < 3) << (data.w * 10));
    }

    __syncthreads();
    uint4 rank;
    uint idx   = threadIdx.x << 2;

#if 1
    uint sumn3 = address.x + ((total  + (total << 10)) << 10) ;
    uint sum   = total - address.x ;

    idx += ((sum + (sum >> 10) + (sum >> 20))&0x3FF);

    rank.x = (data.x != 3) ? ((sumn3 >> (data.x*10)) & 0x3FF)                      : idx;
    rank.y = (data.y != 3) ? ((sumn3 >> (data.y*10)) & 0x3FF) + (data.y == data.x) : idx + (data.y == data.x);
    rank.z = (data.z != 3) ? ((sumn3 >> (data.z*10)) & 0x3FF) + (data.z == data.x) + (data.z == data.y): idx + (data.z == data.x) + (data.z == data.y);
    rank.w = (data.w != 3) ? ((sumn3 >> (data.w*10)) & 0x3FF) + (data.w == data.x) + (data.w == data.y)  + (data.w == data.z): idx + (data.w == data.x) + (data.w == data.y)  + (data.w == data.z);
    
#else
    uint shift = ((total  + (total << 10)) << 10);
    
    uint sumn3 = address.x + shift;
    uint sum   = total - address.x;
    rank.x = (data.x != 3) ? ((sumn3 >> (data.x*10)) & 0x3FF) : idx + ((sum + (sum >> 10) + (sum >> 20))&0x3FF);

    sumn3 = address.y + shift;
    sum   = total - address.y;
    rank.y = (data.y != 3) ? ((sumn3 >> (data.y*10)) & 0x3FF) : idx + 1 + ((sum + (sum >> 10) + (sum >> 20))&0x3FF);

    sumn3 = address.z + shift;
    sum   = total - address.z;
    rank.z = (data.z != 3) ? ((sumn3 >> (data.z*10)) & 0x3FF) : idx + 1 + ((sum + (sum >> 10) + (sum >> 20))&0x3FF);

    sumn3 = address.w + shift;
    sum   = total - address.w;
    rank.w = (data.w != 3) ? ((sumn3 >> (data.w*10)) & 0x3FF) : idx + 1 + ((sum + (sum >> 10) + (sum >> 20))&0x3FF);
#endif
    return rank;
}


template<uint startbit, uint endbit>
__device__ void sortbit4(uint4& data, uint4& index) {
    extern __shared__ uint shuffmem[];
    
    for (uint shift = startbit; shift < endbit; shift+=2) {
        uint4 bmask;
        
        bmask.x = ((data.x >> shift) & 0x03);
        bmask.y = ((data.y >> shift) & 0x03);
        bmask.z = ((data.z >> shift) & 0x03);
        bmask.w = ((data.w >> shift) & 0x03);

        uint4 r = irank4<CTA_SIZE>(bmask);

        shuffmem[(r.x & 3) * CTA_SIZE + (r.x >> 2)] = index.x;
        shuffmem[(r.y & 3) * CTA_SIZE + (r.y >> 2)] = index.y;
        shuffmem[(r.z & 3) * CTA_SIZE + (r.z >> 2)] = index.z;
        shuffmem[(r.w & 3) * CTA_SIZE + (r.w >> 2)] = index.w;
        __syncthreads();

        index.x = shuffmem[threadIdx.x];
        index.y = shuffmem[threadIdx.x +     CTA_SIZE];
        index.z = shuffmem[threadIdx.x + 2 * CTA_SIZE];
        index.w = shuffmem[threadIdx.x + 3 * CTA_SIZE];

        __syncthreads();
         
        shuffmem[(r.x & 3) * CTA_SIZE + (r.x >> 2)] = data.x;
        shuffmem[(r.y & 3) * CTA_SIZE + (r.y >> 2)] = data.y;
        shuffmem[(r.z & 3) * CTA_SIZE + (r.z >> 2)] = data.z;
        shuffmem[(r.w & 3) * CTA_SIZE + (r.w >> 2)] = data.w;

        __syncthreads();

        data.x = shuffmem[threadIdx.x];
        data.y = shuffmem[threadIdx.x +     CTA_SIZE];
        data.z = shuffmem[threadIdx.x + 2 * CTA_SIZE];
        data.w = shuffmem[threadIdx.x + 3 * CTA_SIZE];
        
        __syncthreads();

    }
}

texture<uint4, 1, cudaReadModeElementType> com_tex_uint4;

template<uint startbit, uint endbit>
__global__ void radixSortBlocks(uint2* odata, 
                                int* doffset, int* dcount, uint nBlocks)
{
#if (CUDART_VERSION  < 2020)
    extern __shared__ uint4 sMem4[];
#else
    extern volatile __shared__ uint4 sMem4[];
#endif    
    uint4 data, index;
    //uint blockId = blockIdx.x;
    uint blockId = get_blockID();
    
    if (blockId < nBlocks){
        uint id = blockId* blockDim.x + threadIdx.x;
        data    = tex1Dfetch(com_tex_uint4, id);
        index   = make_uint4(id * 4, id * 4 + 1, id * 4 + 2, id * 4 + 3);
        
        __syncthreads();
        
        sortbit4<startbit, endbit>(data, index);

        uint4* od = (uint4*) odata;
        uint4* oi = (uint4*) (odata + nBlocks * 512);

        od[id] = data;
        oi[id] = index;

        // read the next number
#if (CUDART_VERSION  < 2020)
        int* smem = (int*)sMem4;
#else        
        volatile int* smem = (int*)sMem4;
#endif
        uint dprev;

        dprev = (threadIdx.x == 0) ? (KWAY-1) : (((smem[threadIdx.x - 1 + CTA_SIZE *3] >> startbit) & KMASK) + KWAY); 
    
        __syncthreads();

        data.x = ((data.x >> startbit) & KMASK) + KWAY;
        data.y = ((data.y >> startbit) & KMASK) + KWAY;
        data.z = ((data.z >> startbit) & KMASK) + KWAY;
        data.w = ((data.w >> startbit) & KMASK) + KWAY;
        
        if (threadIdx.x <WARP_SIZE) 
            smem[threadIdx.x] = -1;

        __syncthreads();

        int i = threadIdx.x << 2;
        if (data.x != dprev)  smem[dprev ] = i - 1;
        if (data.y != data.x) smem[data.x] = i;
        if (data.z != data.y) smem[data.y] = i + 1;
        if (data.w != data.z) smem[data.z] = i + 2;

        if (threadIdx.x == CTA_SIZE - 1)
            smem[data.w] = 1023;
        
        __syncthreads();


        if (threadIdx.x<KWAY) {
            uint idx  = threadIdx.x + KWAY;
            smem[idx] = MAX(smem[idx], smem[idx-1]);
            smem[idx] = MAX(smem[idx], smem[idx-2]);
            smem[idx] = MAX(smem[idx], smem[idx-4]);
            smem[idx] = MAX(smem[idx], smem[idx-8]);
        }

        __syncthreads();
        
        if (threadIdx.x<KWAY) {
            uint idx = threadIdx.x + KWAY;
            doffset[blockId * KWAY+threadIdx.x]  = smem[idx -1] + 1; // stored in row-major instead
            //dcount [threadIdx.x*gridDim.x+blockId] = smem[idx   ] - smem[idx-1];
            dcount [threadIdx.x*nBlocks+blockId] = smem[idx   ] - smem[idx-1];
        }
    }
}


template<uint startbit, uint endbit>
__global__ void radixSortBlocks(uint2* odata, uint2* idata,
                                int* doffset, int* dcount, uint nBlocks)
{
#if (CUDART_VERSION  < 2020)
    extern __shared__ uint4 sMem4[];
#else
    extern volatile __shared__ uint4 sMem4[];
#endif    
    uint4 data, index;
    //uint blockId = blockIdx.x;
    uint blockId = get_blockID();
    
    if (blockId < nBlocks){
        uint4 a    = tex1Dfetch(com_tex_uint4, blockId * 512 + threadIdx.x * 2);
        uint4 b    = tex1Dfetch(com_tex_uint4, blockId * 512 + threadIdx.x * 2+1);
        
        data   = make_uint4(a.x, a.z, b.x, b.z);
        index  = make_uint4(a.y, a.w, b.y, b.w);

        __syncthreads();
        
        sortbit4<startbit, endbit>(data, index);

        uint4* od = (uint4*) odata;
        uint4* oi = (uint4*) (odata + nBlocks * 512);

        int i = blockId* blockDim.x + threadIdx.x;

        od[i] = data;
        oi[i] = index;

        // read the next number
#if (CUDART_VERSION  < 2020)
        int* smem = (int*)sMem4;
#else
        volatile int* smem = (int*)sMem4;
#endif
        uint dprev;

        dprev = (threadIdx.x == 0) ? (KWAY-1) : (((smem[threadIdx.x - 1 + CTA_SIZE *3] >> startbit) & KMASK) + KWAY); 
        __syncthreads();

        data.x = ((data.x >> startbit) & KMASK) + KWAY;
        data.y = ((data.y >> startbit) & KMASK) + KWAY;
        data.z = ((data.z >> startbit) & KMASK) + KWAY;
        data.w = ((data.w >> startbit) & KMASK) + KWAY;
        
        if (threadIdx.x <WARP_SIZE) 
            smem[threadIdx.x] = -1;

        __syncthreads();

        i = threadIdx.x << 2;
        if (data.x != dprev)  smem[dprev ] = i - 1;
        if (data.y != data.x) smem[data.x] = i;
        if (data.z != data.y) smem[data.y] = i + 1;
        if (data.w != data.z) smem[data.z] = i + 2;

        if (threadIdx.x == CTA_SIZE - 1)
            smem[data.w] = 1023;
        
        __syncthreads();


        if (threadIdx.x<KWAY) {
            uint idx = threadIdx.x + KWAY;
            smem[idx] = MAX(smem[idx], smem[idx-1]);
            smem[idx] = MAX(smem[idx], smem[idx-2]);
            smem[idx] = MAX(smem[idx], smem[idx-4]);
            smem[idx] = MAX(smem[idx], smem[idx-8]);
        }

        __syncthreads();
        
        if (threadIdx.x<KWAY) {
            uint idx = threadIdx.x + KWAY;
            doffset[blockId * KWAY+threadIdx.x]  = smem[idx -1] + 1; // stored in row-major instead
            dcount [threadIdx.x*nBlocks+blockId] = smem[idx   ] - smem[idx-1];
        }
    }
}

__global__ void pre_scatter(int* d_sum, int* d_off, uint nblocks){
    uint id = blockIdx.x * blockDim.x + threadIdx.x;
    uint n  = nblocks * 16;
    if (id  < n){
        uint bid = id >> 4;
        uint tid = id & 0x0F;

        d_off[id] = d_sum[bid + tid * nblocks] - d_off[id];
    }
}

__global__ void scatter_gpu(uint2 *odata,
                            uint  *idata, uint * iindex,
                            int* doffset, int shift) {

    __shared__ unsigned lsum[KWAY];
    uint blockId = get_blockID();
    if (threadIdx.x<KWAY)
        lsum[threadIdx.x] = doffset[blockId * KWAY + threadIdx.x];

    __syncthreads();


    uint tid = threadIdx.x;
    uint off = blockId*1024;

    for (int i=0; i< 32; ++i){
        unsigned value = idata [tid+off];
        unsigned index = iindex[tid+off];

        unsigned key   =  (value>>shift) & KMASK;
        unsigned oid   =  tid+lsum[key];

        odata [oid] = make_uint2(value, index);

        tid  += 32;
    }
}

__global__ void scatter_gpu(uint2 *odata,
                            uint  *idata, uint * iindex,
                            int   *dsum, int* doffset, int nBlocks, int shift) {

    __shared__ unsigned lsum[KWAY];
    uint blockId = get_blockID();
    if (threadIdx.x<KWAY)
        lsum[threadIdx.x] = dsum[threadIdx.x*nBlocks+blockId] - doffset[blockId * KWAY + threadIdx.x];
    
    __syncthreads();
  

    uint tid = threadIdx.x;
    uint off = blockId*1024;
    
    for (int i=0; i< 32; ++i){
        unsigned value = idata [tid+off];
        unsigned index = iindex[tid+off];

        unsigned key   =  (value>>shift) & KMASK;
        unsigned oid   =  tid+lsum[key];
        
        odata [oid] = make_uint2(value, index);
        
        tid  += 32;
    }
}

////////////////////////////////////////////////////////////////////////////////
// Sorting plan 
////////////////////////////////////////////////////////////////////////////////

cplSort::cplSort(int maxSize, cplReduce* rd):
    m_rd(rd), d_ping(NULL), d_pong(NULL), d_off(NULL), d_cnt(NULL), d_sum(NULL), nc(0), flip(0){
    nc = 0;
    setCapacity(maxSize);
}

void cplSort::setCapacity(int size){
    if (size > nc){
        nc = iAlignUp(size,1024);
        clean();
        init();
    }
}

void cplSort::init(){
    int  nBlocks = nc / 1024;
    uint nWays   = nBlocks * KWAY;

    // init sorting memroy
    cudaMalloc((void**)&d_ping, nc * sizeof(uint2));
    cudaMalloc((void**)&d_pong, nc * sizeof(uint2));

    // init supplement memory
    cudaMalloc((void**)&d_off, nWays * sizeof(uint)); 
    cudaMalloc((void**)&d_cnt, nWays * sizeof(uint)); 
    cudaMalloc((void**)&d_sum, nWays * sizeof(uint));

    // init cudaPlan
    CUDPPConfiguration scanConfig;
    scanConfig.algorithm = CUDPP_SCAN;
    scanConfig.datatype  = CUDPP_UINT;
    scanConfig.op        = CUDPP_ADD;
    scanConfig.options   = CUDPP_OPTION_EXCLUSIVE | CUDPP_OPTION_FORWARD;

    cudppPlan(&m_scanPlan, scanConfig, nWays, 1, 0);
    cutilCheckMsg("Sorting init");
}

void cplSort::clean(){
    cudaSafeDelete(d_ping);
    cudaSafeDelete(d_pong);
    cudaSafeDelete(d_off);
    cudaSafeDelete(d_cnt);
    cudaSafeDelete(d_sum);
    cudppDestroyPlan(m_scanPlan);

    
}

////////////////////////////////////////////////////////////////////////////////
// Load and padding only a single channel. Flip the input if it is the float input 
////////////////////////////////////////////////////////////////////////////////

template<int flip>
__global__ void load_C1_kernel(uint2* d_o, uint* d_i, int n, int nAlign){
    uint blockId = get_blockID();
    uint   id    = get_threadID(blockId);
    
    if (id < nAlign ){
        d_o[id].x = (id < n) ? floatFlip<flip>(d_i[id]) : UINT_MAX;
        d_o[id].y = id;
    }
}

void cplSort::load_C1(uint2* d_o, uint* d_i, int n, int nAlign, int flip){
    assert(nAlign <= nc);
    assert(n      <= nAlign);

    dim3 threads(256);
    dim3 grids(iDivUp(nAlign, threads.x));
    checkConfig(grids);
    
    if (flip)
        load_C1_kernel<1><<<grids, threads>>>(d_ping, d_i, n, nAlign);
    else
        load_C1_kernel<0><<<grids, threads>>>(d_ping, d_i, n, nAlign);
}

////////////////////////////////////////////////////////////////////////////////
// Load and padding AoS input. Flip the input if it is the float input 
////////////////////////////////////////////////////////////////////////////////

template<int flip>
__global__ void loadAOS_C2_kernel(uint2* d_o, uint2* d_i, int n, int nAlign){
    uint blockId = get_blockID();
    uint   id    = get_threadID(blockId);
    
    if (id < nAlign ){
        if (flip)
            d_o[id] = (id <n) ? make_uint2(floatFlip<1>(d_i[id].x), d_i[id].y) : make_uint2(UINT_MAX, UINT_MAX);
        else
            d_o[id] = (id <n) ? d_i[id]: make_uint2(UINT_MAX, UINT_MAX);

    }
}

void cplSort::loadAOS_C2(uint2* d_o, uint2* d_i, int n, int nAlign, int flip){
    assert(nAlign <=nc);
    assert(n      <=nAlign);
    
    dim3 threads(256);
    dim3 grids(iDivUp(nAlign, threads.x));
    checkConfig(grids);
    
    if (flip)
        loadAOS_C2_kernel<1><<<grids, threads>>>(d_ping, d_i, n, nAlign);
    else
        loadAOS_C2_kernel<0><<<grids, threads>>>(d_ping, d_i, n, nAlign);
}

////////////////////////////////////////////////////////////////////////////////
//
////////////////////////////////////////////////////////////////////////////////
template<int flip>
__global__ void loadSOA_C2_kernel(uint2* d_o, uint* d_i, uint* d_ii, int n, int nAlign){
    uint blockId = get_blockID();
    uint   id    = get_threadID(blockId);
    
    
    if (id < nAlign ){
        uint data = d_i[id];
        uint index= d_ii[id];

        if (flip)
            d_o[id] = (id <n) ? make_uint2(floatFlip<1>(data), index) : make_uint2(UINT_MAX, UINT_MAX);
        else 
            d_o[id] = (id <n) ? make_uint2(data, index) : make_uint2(UINT_MAX, UINT_MAX);
    }
}

void cplSort::loadSOA_C2(uint2* d_o, uint* d_i, uint* d_ii, int n, int nAlign, int flip){
    dim3 threads(256);
    dim3 grids(iDivUp(nAlign, threads.x));
    checkConfig(grids);
    
    if (flip)
        loadSOA_C2_kernel<1><<<grids, threads>>>(d_o, d_i, d_ii, n, nAlign);
    else
        loadSOA_C2_kernel<0><<<grids, threads>>>(d_o, d_i, d_ii, n, nAlign);
}

////////////////////////////////////////////////////////////////////////////////
//
////////////////////////////////////////////////////////////////////////////////

template<int flip>
__global__ void sortReturn_kernel(uint2* d_o, uint2* d_i, int n){
    uint blockId = get_blockID();
    uint   id    = get_threadID(blockId);

    if (id < n){
        if (!flip) d_o[id] = d_i[id];
        else d_o[id] = make_uint2(floatUnflip<1>(d_i[id].x), d_i[id].y);
    }
}

void cplSort::sortReturn(uint2* d_o, uint2* d_i, int n, int flip){
    dim3 threads(256);
    dim3 grids(iDivUp(n, threads.x));
    checkConfig(grids);
    
    if (flip)
        sortReturn_kernel<1><<<grids, threads>>>(d_o, d_i, n);
    else
        sortReturn_kernel<0><<<grids, threads>>>(d_o, d_i, n);
}


template<int flip>
__global__ void sortReturn_kernel(uint* d_o, uint* d_oi, uint2* d_i, int n){
    uint blockId = blockIdx.x  + blockIdx.y * gridDim.x;
    uint   id    = threadIdx.x + blockId * blockDim.x;
    if (id < n){
        uint2 data = d_i[id];
        if (flip)
            d_o[id] = floatUnflip<1>(data.x);
        else 
            d_o[id] = data.x;
        d_oi[id] = data.y;
    }
}

void cplSort::sortReturn(uint* d_o, uint* d_oi, uint2* d_i, int n, int flip){
    dim3 threads(256);
    dim3 grids(iDivUp(n, threads.x));
    checkConfig(grids);
    
    if (flip)
        sortReturn_kernel<1><<<grids, threads>>>(d_o, d_oi, d_i, n);
    else
        sortReturn_kernel<0><<<grids, threads>>>(d_o, d_oi, d_i, n);
}

////////////////////////////////////////////////////////////////////////////////
//
////////////////////////////////////////////////////////////////////////////////

template<uint startbit, uint endbit>
__global__ void radixSortSingleBlock_kernel(uint2* odata)
{
#if (CUDART_VERSION  < 2020)
    extern __shared__ uint4 sMem4[];
#else
    extern volatile __shared__ uint4 sMem4[];
#endif
    uint4 data, index;
    uint  id = threadIdx.x << 1;

    uint4 a    = tex1Dfetch(com_tex_uint4, id);
    uint4 b    = tex1Dfetch(com_tex_uint4, id+1);
        
    data   = make_uint4(a.x, a.z, b.x, b.z);
    index  = make_uint4(a.y, a.w, b.y, b.w);

    __syncthreads();
    sortbit4<startbit, endbit>(data, index);
    
    *((uint4*)odata + id    ) = make_uint4(data.x, index.x, data.y, index.y);
    *((uint4*)odata + id +1 ) = make_uint4(data.z, index.z, data.w, index.w);
}

void cplSort::radixSortSingleBlock(uint2* d_o, uint2* d_i, int n, int nBits){
    int bits = (nBits == 0) ? 32 : nBits;
    assert(n <= 1024);
    cudaBindTexture(0, com_tex_uint4, d_i, 1024 * sizeof(uint2));
    if (bits <= 4)
        radixSortSingleBlock_kernel<0,4><<<1, CTA_SIZE,  (CTA_SIZE * 4 + 64)* sizeof(uint) >>>(d_o);
    else if (bits <= 8)
        radixSortSingleBlock_kernel<0,8><<<1, CTA_SIZE,  (CTA_SIZE * 4 + 64)* sizeof(uint) >>>(d_o);
    else if (bits <= 12)
        radixSortSingleBlock_kernel<0,12><<<1, CTA_SIZE,  (CTA_SIZE * 4 + 64)* sizeof(uint) >>>(d_o);
    else if (bits <= 16)
        radixSortSingleBlock_kernel<0,16><<<1, CTA_SIZE,  (CTA_SIZE * 4 + 64)* sizeof(uint) >>>(d_o);
    else if (bits <= 20)
        radixSortSingleBlock_kernel<0,20><<<1, CTA_SIZE,  (CTA_SIZE * 4 + 64)* sizeof(uint) >>>(d_o);
    else if (bits <= 24)
        radixSortSingleBlock_kernel<0,24><<<1, CTA_SIZE,  (CTA_SIZE * 4 + 64)* sizeof(uint) >>>(d_o);
    else if (bits <= 28)
        radixSortSingleBlock_kernel<0,28><<<1, CTA_SIZE,  (CTA_SIZE * 4 + 64)* sizeof(uint) >>>(d_o);
    else if (bits <= 32)
        radixSortSingleBlock_kernel<0,32><<<1, CTA_SIZE,  (CTA_SIZE * 4 + 64)* sizeof(uint) >>>(d_o);
}

template<uint startbit, uint endbit, int flip>
__global__ void radixSortSingleBlock_kernel(uint4* d_o, uint4* d_oi, uint4* d_i, uint4* d_ii, int n)
{
#if (CUDART_VERSION  < 2020)
    extern __shared__ uint4 sMem4[];
#else
    extern volatile __shared__ uint4 sMem4[];
#endif
    uint4 data, index;
    uint  id = threadIdx.x;
    uint idx = id << 2;

    if (idx + 3 >= n){
        if (idx >= n)
        {
            data  = make_uint4(UINT_MAX, UINT_MAX, UINT_MAX, UINT_MAX);
            index = make_uint4(UINT_MAX, UINT_MAX, UINT_MAX, UINT_MAX);
        }
        else
        {
            // for non-full block, we handle uint1 values instead of uint4
            uint *datas1    = (uint*)d_i;
            uint *indexs1  = (uint*)d_ii;
        
            data.x = (idx   < n) ? floatFlip<flip>(datas1[idx])   : UINT_MAX;
            data.y = (idx+1 < n) ? floatFlip<flip>(datas1[idx+1]) : UINT_MAX;
            data.z = (idx+2 < n) ? floatFlip<flip>(datas1[idx+2]) : UINT_MAX;
            data.w = UINT_MAX;
        
            index.x = (idx   < n) ? indexs1[idx]   : UINT_MAX;
            index.y = (idx+1 < n) ? indexs1[idx+1] : UINT_MAX;
            index.z = (idx+2 < n) ? indexs1[idx+2] : UINT_MAX;
            index.w = UINT_MAX;
        }
    }else{
        data   = d_i[id];
        index  = d_ii[id];
        if (flip)
        {
            data.x = floatFlip<1>(data.x);
            data.y = floatFlip<1>(data.y);
            data.z = floatFlip<1>(data.z);
            data.w = floatFlip<1>(data.w);
        }
    }
    
    __syncthreads();
    
    sortbit4<startbit, endbit>(data, index);

    if(idx+3 >= n){
        if (idx < n) 
        {
            // for non-full block, we handle uint1 values instead of uint4
            uint *datas1  = (uint*)d_o;
            uint *indexs1 = (uint*)d_oi;
            
            datas1[idx]  = floatUnflip<flip>(data.x);
            indexs1[idx] = index.x;
            
            if (idx + 1 < n){
                datas1[idx + 1]  = floatUnflip<flip>(data.y);
                indexs1[idx + 1] = index.y;
                
                if (idx + 2 < n)
                {
                    datas1[idx + 2]  = floatUnflip<flip>(data.z);
                    indexs1[idx + 2] = index.z;
                }
            }
        }
    } else
    {
        if (flip){
            data.x = floatUnflip<1>(data.x);
            data.y = floatUnflip<1>(data.y);
            data.z = floatUnflip<1>(data.z);
            data.w = floatUnflip<1>(data.w);
        }
        d_o[id] = data;
        d_oi[id] = index;
    }
}

template<int  flip>
void cplSort::radixSortSingleBlock(uint* d_o, uint* d_oi, uint* d_i, uint* d_ii, int n, int nBits)
{
    int bits = (nBits == 0) ? 32 : nBits;
    assert(n <= 1024);
    if (flip){
        if (bits <= 4)
            radixSortSingleBlock_kernel<0,4, 1><<<1, CTA_SIZE,  (CTA_SIZE * 4 + 64)* sizeof(uint) >>>((uint4*)d_o, (uint4*)d_oi, (uint4*)d_i, (uint4*)d_ii, n);
        else if (bits <= 8)
            radixSortSingleBlock_kernel<0,8, 1><<<1, CTA_SIZE,  (CTA_SIZE * 4 + 64)* sizeof(uint) >>>((uint4*)d_o, (uint4*)d_oi, (uint4*)d_i, (uint4*)d_ii, n);
        else if (bits <= 12)
            radixSortSingleBlock_kernel<0,12, 1><<<1, CTA_SIZE,  (CTA_SIZE * 4 + 64)* sizeof(uint) >>>((uint4*)d_o, (uint4*)d_oi, (uint4*)d_i, (uint4*)d_ii, n);
        else if (bits <= 16)
            radixSortSingleBlock_kernel<0,16, 1><<<1, CTA_SIZE,  (CTA_SIZE * 4 + 64)* sizeof(uint) >>>((uint4*)d_o, (uint4*)d_oi, (uint4*)d_i, (uint4*)d_ii, n);
        else if (bits <= 20)
            radixSortSingleBlock_kernel<0,20, 1><<<1, CTA_SIZE,  (CTA_SIZE * 4 + 64)* sizeof(uint) >>>((uint4*)d_o, (uint4*)d_oi, (uint4*)d_i, (uint4*)d_ii, n);
        else if (bits <= 24)
            radixSortSingleBlock_kernel<0,24, 1><<<1, CTA_SIZE,  (CTA_SIZE * 4 + 64)* sizeof(uint) >>>((uint4*)d_o, (uint4*)d_oi, (uint4*)d_i, (uint4*)d_ii, n);
        else if (bits <= 28)
            radixSortSingleBlock_kernel<0,28, 1><<<1, CTA_SIZE,  (CTA_SIZE * 4 + 64)* sizeof(uint) >>>((uint4*)d_o, (uint4*)d_oi, (uint4*)d_i, (uint4*)d_ii, n);
        else if (bits <= 32)
            radixSortSingleBlock_kernel<0,32, 1><<<1, CTA_SIZE,  (CTA_SIZE * 4 + 64)* sizeof(uint) >>>((uint4*)d_o, (uint4*)d_oi, (uint4*)d_i, (uint4*)d_ii, n);
    }
    else {
        if (bits <= 4)
            radixSortSingleBlock_kernel<0,4, 0><<<1, CTA_SIZE,  (CTA_SIZE * 4 + 64)* sizeof(uint) >>>((uint4*)d_o, (uint4*)d_oi, (uint4*)d_i, (uint4*)d_ii, n);
        else if (bits <= 8)
            radixSortSingleBlock_kernel<0,8, 0><<<1, CTA_SIZE,  (CTA_SIZE * 4 + 64)* sizeof(uint) >>>((uint4*)d_o, (uint4*)d_oi, (uint4*)d_i, (uint4*)d_ii, n);
        else if (bits <= 12)
            radixSortSingleBlock_kernel<0,12, 0><<<1, CTA_SIZE,  (CTA_SIZE * 4 + 64)* sizeof(uint) >>>((uint4*)d_o, (uint4*)d_oi, (uint4*)d_i, (uint4*)d_ii, n);
        else if (bits <= 16)
            radixSortSingleBlock_kernel<0,16, 0><<<1, CTA_SIZE,  (CTA_SIZE * 4 + 64)* sizeof(uint) >>>((uint4*)d_o, (uint4*)d_oi, (uint4*)d_i, (uint4*)d_ii, n);
        else if (bits <= 20)
            radixSortSingleBlock_kernel<0,20, 0><<<1, CTA_SIZE,  (CTA_SIZE * 4 + 64)* sizeof(uint) >>>((uint4*)d_o, (uint4*)d_oi, (uint4*)d_i, (uint4*)d_ii, n);
        else if (bits <= 24)
            radixSortSingleBlock_kernel<0,24, 0><<<1, CTA_SIZE,  (CTA_SIZE * 4 + 64)* sizeof(uint) >>>((uint4*)d_o, (uint4*)d_oi, (uint4*)d_i, (uint4*)d_ii, n);
        else if (bits <= 28)
            radixSortSingleBlock_kernel<0,28, 0><<<1, CTA_SIZE,  (CTA_SIZE * 4 + 64)* sizeof(uint) >>>((uint4*)d_o, (uint4*)d_oi, (uint4*)d_i, (uint4*)d_ii, n);
        else if (bits <= 32)
            radixSortSingleBlock_kernel<0,32, 0><<<1, CTA_SIZE,  (CTA_SIZE * 4 + 64)* sizeof(uint) >>>((uint4*)d_o, (uint4*)d_oi, (uint4*)d_i, (uint4*)d_ii, n);

    }
}    

template<class T>
void cplSort::getInputRange(T& maxValue, T& minValue, T* d_i, unsigned int n){
    m_rd->MaxMin(maxValue, minValue, d_i, n);
}
////////////////////////////////////////////////////////////////////////////////
// Convert the input to unsigned input base on the minimumvalue of the input 
////////////////////////////////////////////////////////////////////////////////

__global__ void loadSOA_offset_kernel(uint2* d_o, int* d_i, uint* d_ii, int minV, int n, int nAlign){
    uint blockId = get_blockID();
    uint   id    = get_threadID(blockId);
    
    if (id < nAlign ){
        uint data = d_i[id] - minV;
        uint index= d_ii[id];
        d_o[id] = (id <n) ? make_uint2(data, index) : make_uint2(UINT_MAX, UINT_MAX);
    }
}

__global__ void loadSOA_offset_kernel(uint2* d_o, uint* d_i, uint* d_ii, uint minV, int n, int nAlign){
    uint blockId = get_blockID();
    uint   id    = get_threadID(blockId);
    
    if (id < nAlign ){
        uint data = d_i[id] - minV;
        uint index= d_ii[id];
        d_o[id] = (id <n) ? make_uint2(data, index) : make_uint2(UINT_MAX, UINT_MAX);
    }
}

void cplSort::loadSOA_offset(uint2* d_o, uint* d_i, uint* d_ii, uint minV, int n, int nAlign){
    dim3 threads(256);
    dim3 grids(iDivUp(n, threads.x));
    checkConfig(grids);
    loadSOA_offset_kernel<<<grids,threads>>>(d_o,d_i, d_ii, minV, n, nAlign);
}

void cplSort::loadSOA_offset(uint2* d_o, int* d_i, uint* d_ii, int minV, int n, int nAlign){
    dim3 threads(256);
    dim3 grids(iDivUp(n, threads.x));
    checkConfig(grids);
    loadSOA_offset_kernel<<<grids,threads>>>(d_o,d_i, d_ii, minV, n, nAlign);
}


////////////////////////////////////////////////////////////////////////////////
//
////////////////////////////////////////////////////////////////////////////////

__global__ void sortReturnSOA_offset_kernel(int* d_o, uint* d_oi, uint2* d_i, int minV, int n){
    uint blockId = get_blockID();
    uint   id    = get_threadID(blockId);

    if (id < n){
        d_o[id]  = d_i[id].x + minV;
        d_oi[id] = d_i[id].y;
    }
}

__global__ void sortReturnSOA_offset_kernel(uint* d_o, uint* d_oi, uint2* d_i, uint minV, int n){
    uint blockId = get_blockID();
    uint   id    = get_threadID(blockId);

    if (id < n){
        d_o[id]  = d_i[id].x + minV;
        d_oi[id] = d_i[id].y;
    }
}

void cplSort::sortReturnSOA_offset(int* d_o, uint* d_oi, uint2* d_i, int minV, int n){
    dim3 threads(256);
    dim3 grids(iDivUp(n, threads.x));
    checkConfig(grids);
    sortReturnSOA_offset_kernel<<<grids, threads>>>(d_o, d_oi, d_i, minV, n);
}

void cplSort::sortReturnSOA_offset(uint* d_o, uint* d_oi, uint2* d_i, uint minV, int n){
    dim3 threads(256);
    dim3 grids(iDivUp(n, threads.x));
    checkConfig(grids);
    sortReturnSOA_offset_kernel<<<grids, threads>>>(d_o, d_oi, d_i, minV, n);
}

#define __MEASURE__DETAILED__ 1

void cplSort::implSort(uint2* d_o, uint2* d_i, int n, int nBits){
    assert(n <= nc);
    assert((n & 1023) == 0);
    
    
    int nBlocks = n >> 10;
    int nWays   = nBlocks * KWAY;

    dim3 threads(CTA_SIZE);
    dim3 grids(nBlocks);
    checkConfig(grids);

    dim3 grids2(iDivUp(nBlocks * 16, 64));
    checkConfig(grids2);

    cudaEventTimer timer;
    timer.start();
    timer.reset();
#if __MEASURE__DETAILED__
    float l0 = 0, l1 = 0, l2 = 0;
#endif
    int bits = (nBits == 0) ? 32 : nBits;
    
    for (uint shift=0; shift < bits; shift+=4){
        // 1. Local presorting with the bits
        cudaBindTexture(0, com_tex_uint4, d_i, n * sizeof(uint2));
        
        if (shift ==0)
            radixSortBlocks<0,4><<<grids, threads, (CTA_SIZE * 4 + 64)* sizeof(uint)>>>(d_o, d_i, d_off, d_cnt, nBlocks);
        else if (shift ==4)
            radixSortBlocks<4,8><<<grids, threads, (CTA_SIZE * 4 + 64)* sizeof(uint)>>>(d_o, d_i, d_off, d_cnt, nBlocks);
        else if (shift ==8)
            radixSortBlocks<8,12><<<grids, threads, (CTA_SIZE * 4 + 64)* sizeof(uint)>>>(d_o, d_i, d_off, d_cnt, nBlocks);
        else if (shift ==12)
            radixSortBlocks<12,16><<<grids, threads, (CTA_SIZE * 4 + 64)* sizeof(uint)>>>(d_o, d_i, d_off, d_cnt, nBlocks);
        else if (shift ==16)
            radixSortBlocks<16,20><<<grids, threads, (CTA_SIZE * 4 + 64)* sizeof(uint)>>>(d_o, d_i, d_off, d_cnt, nBlocks);
        else if (shift ==20)
            radixSortBlocks<20,24><<<grids, threads, (CTA_SIZE * 4 + 64)* sizeof(uint)>>>(d_o, d_i, d_off, d_cnt, nBlocks);
        else if (shift ==24)
            radixSortBlocks<24,28><<<grids, threads, (CTA_SIZE * 4 + 64)* sizeof(uint)>>>(d_o, d_i, d_off, d_cnt, nBlocks);
        else if (shift ==28)
            radixSortBlocks<28,32><<<grids, threads, (CTA_SIZE * 4 + 64)* sizeof(uint)>>>(d_o, d_i, d_off, d_cnt, nBlocks);
        // 2. Global bit counting
#if __MEASURE__DETAILED__
        timer.stop();
        l0 += timer.getTime();
        timer.reset();
#endif
        
        cudppScan(m_scanPlan, d_sum, d_cnt, nWays);

#if __MEASURE__DETAILED__
        timer.stop();
        l1 += timer.getTime();
        timer.reset();
#endif
        
#if 0
        // 3. Global mapping 
        scatter_gpu<<<grids, 32>>>(d_i, (uint*)d_o, (uint*)(d_o + n/2), d_sum, d_off, nBlocks, shift);
#else
        pre_scatter<<<grids2, 64>>>(d_sum, d_off, nBlocks);
        scatter_gpu<<<grids, 32>>>(d_i, (uint*)d_o, (uint*)(d_o + n/2), d_off, shift);
#endif
#if __MEASURE__DETAILED__
        timer.stop();
        l2 += timer.getTime();
        timer.reset();
#endif
        //checkDevicePairOrderBits4(d_i, n, shift, "Test bits");
    }

#if __MEASURE__DETAILED__
    timer.stop();
    float totalTime = l0 + l1 + l2;
    fprintf(stderr, "Total time: %f Contribution %f %f %f \n", totalTime, l0, l1, l2);

#else
    timer.stop();
    float totalTime = timer.getTime();
    fprintf(stderr, "Sorting rate: %f Mpair\n", n / totalTime / 1000);
#endif
    //timer.printTime("radix sorting", 1);
}

////////////////////////////////////////////////////////////////////////////////
// Sorting with SOA structure 
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// flip = 1 : float input
//        0 : integer input 
////////////////////////////////////////////////////////////////////////////////

void cplSort::sortSupport(uint* d_o, uint* d_oi, uint* d_i, uint* d_ii, int n, int flip, int nBits){
    assert(n <= nc);
    if (n <= 1024){
        if (flip)
            radixSortSingleBlock<1>(d_o, d_oi, d_i, d_ii, n, nBits);
        else
            radixSortSingleBlock<0>(d_o, d_oi, d_i, d_ii, n, nBits);
    }
    else {
        uint nAlign = iAlignUp(n, 1024);
        loadSOA_C2(d_ping, d_i, d_ii, n, nAlign, flip);
        implSort(d_pong, d_ping, nAlign, nBits);
        sortReturn(d_o, d_oi, d_ping, n, flip);
    }
}


__global__ void loadSOA_appf_kernel(float2* d_o, float* d_i, float* d_index,
                                    float a, float b, int n, int nAlign)
{
    uint blockId = get_blockID();
    uint   id    = get_threadID(blockId);
    
    if (id < nAlign ){
        d_o[id] = (id <n) ? make_float2(d_i[id] * a + b, d_index[id]) : make_float2(FLT_MAX, FLT_MAX);
    }
}

__global__ void returnSOA_appf_kernel(float* d_o, float* d_oi, float2* d_i,
                                      float a, float b, int n)
{
    uint blockId = get_blockID();
    uint   id    = get_threadID(blockId);
    
    if (id < n){
        float2 data = d_i[id];
        d_o[id]  = data.x * a + b;
        d_oi[id] = data.y;
    }
}

__global__ void returnSOA_appf_kernel(float2* d_o, float2* d_i,
                                      float a, float b, int n)
{
    uint blockId = get_blockID();
    uint   id    = get_threadID(blockId);
    
    if (id < n){
        float2 data = d_i[id];
        d_o[id]  = make_float2(data.x * a + b, data.y);
    }
}


void cplSort::sort(float* d_o, uint* d_oi, float* d_i, uint* d_ii, int n){
    if (m_appSort == false) {
        sortSupport((uint*)d_o, d_oi, (uint*)d_i, d_ii, n, 1, 0);
    }
    else
    {
        // compute the range of the input
        float maxV, minV;
        getInputRange(maxV, minV, d_i, n);
        
        // map the value of the input to the range of [0.5, 1)
        float2* d_pingf = (float2*) d_ping;
        
        dim3 threads(256);
        unsigned int nAlign = iAlignUp(n, 1024);
        dim3 grids(iDivUp(nAlign, threads.x));
        checkConfig(grids);

        double nMin = 0.5;
        double nMax = 0.999997;
        
        double a = (nMax - nMin) / ((double)maxV - (double)minV);
        double b =  nMin - a * minV;
        loadSOA_appf_kernel<<<grids, threads>>>(d_pingf, d_i, (float*)d_ii, (float)a, (float)b, n, nAlign);

        implSort(d_pong, d_ping, nAlign, 24);

        a = ((double)maxV - (double)minV) / (nMax - nMin);
        b =  minV - a * nMin;
        returnSOA_appf_kernel<<<grids, threads>>>(d_o, (float*)d_oi, (float2*)d_ping, (float)a, (float)b, n);
        
    }
}

void cplSort::sort(uint* d_o, uint* d_oi, uint* d_i, uint* d_ii, int n, int nBits){
    sortSupport((uint*)d_o, d_oi, (uint*)d_i, d_ii, n, 0, nBits);
}

void cplSort::sort(int* d_o, uint* d_oi, int* d_i, uint* d_ii, int n, int nBits){
    assert(n <= nc);
    int nAlign = iAlignUp(n, 1024);
    int minV, maxV;
    // Compute the maxximum and minimum value
    getInputRange(maxV, minV, d_i, n);
    
    // Convert input with offset
    loadSOA_offset(d_ping, d_i, d_ii, minV, n, nAlign);
    
    // compute the range of the input
    // Count the number of bits
    unsigned int range = maxV - minV + 1;
    nBits = 4;
    while (((unsigned long)1 << nBits) < range)
        nBits +=4;
    assert(nBits <= 32);
    
    // Sorting step
    if (n <=1024){
        radixSortSingleBlock(d_pong, d_ping, n, nBits);
        // return input with offset
        sortReturnSOA_offset(d_o, d_oi, d_pong, minV, n);
    }
    else{
        implSort(d_pong, d_ping, nAlign, nBits);
        // return input with offset
        sortReturnSOA_offset(d_o, d_oi, d_ping, minV, n);
    }
}

void cplSort::sort(float* d_data, uint* d_index, int n)
{
    sort(d_data, d_index, d_data, d_index, n);
}

void cplSort::sort(int* d_data,   uint* d_index, int n, int nBits)
{
    sort(d_data, d_index, d_data, d_index, n, nBits);
}
    
void cplSort::sort(uint* d_data,  uint* d_index, int n, int nBits)
{
    sort(d_data, d_index, d_data, d_index, n, nBits);
}

////////////////////////////////////////////////////////////////////////////////
// Sorting with A0S structure 
////////////////////////////////////////////////////////////////////////////////

void cplSort::sortSupport(uint2* d_o, uint2* d_i, int n, int flip, int nBits){
    assert(n <= nc);
    int nAlign = iAlignUp(n, 1024);

    loadAOS_C2(d_ping, d_i, n, nAlign , flip);
    if (n <= 1024){
        radixSortSingleBlock(d_pong, d_ping, n, nBits);
        sortReturn(d_o, d_pong, n, flip);
    }
    else {
        implSort(d_pong, d_ping, nAlign, nBits);
        sortReturn(d_o, d_ping, n, flip);
    }
}


__global__ void loadAOSInt_kernel(uint2* d_o, uint2* d_i, int n, int nAlign){
    uint blockId = blockIdx.x  + blockIdx.y * gridDim.x;
    uint   id    = threadIdx.x + blockId * blockDim.x;

    if (id < nAlign ){
        uint2 t = d_i[id];
        d_o[id] = (id <n) ? make_uint2(intFlip(t.x), t.y) : make_uint2(UINT_MAX, UINT_MAX);
    }
}

__global__ void sortReturnSOAInt_kernel(uint* d_o, uint* d_oi, uint2* d_i, int n)
{
    uint blockId = blockIdx.x  + blockIdx.y * gridDim.x;
    uint   id    = threadIdx.x + blockId * blockDim.x;
    if (id < n){
        uint2 t = d_i[id];
        d_o[id]  = intFlip(t.x);
        d_oi[id] = t.y;
    }
}

__global__ void sortReturnAOSInt_kernel(uint2* d_o, uint2* d_i, int n)
{
    uint blockId = blockIdx.x  + blockIdx.y * gridDim.x;
    uint   id    = threadIdx.x + blockId * blockDim.x;
    if (id < n){
        uint2 t = d_i[id];
        d_o[id]  = make_uint2(intFlip(t.x), t.y);
    }
}

void cplSort::loadAOSInt(uint2* d_o, uint2* d_i, int n, int nAlign){
    dim3 threads(256);
    dim3 grids(iDivUp(n, threads.x));
    checkConfig(grids);
    loadAOSInt_kernel<<<grids, threads>>>(d_o, d_i, n, nAlign);
}

void cplSort::sortReturnSOAInt(uint* d_o, uint* d_oi, uint2* d_i, int n)
{
    dim3 threads(256);
    dim3 grids(iDivUp(n, threads.x));
    checkConfig(grids);
    sortReturnSOAInt_kernel<<<grids, threads>>>(d_o, d_oi, d_i, n);
}

void cplSort::sortReturnAOSInt(uint2* d_o, uint2* d_i, int n)
{
    dim3 threads(256);
    dim3 grids(iDivUp(n, threads.x));
    checkConfig(grids);
    sortReturnAOSInt_kernel<<<grids, threads>>>(d_o, d_i, n);
}

void cplSort::sort(int2* d_o, int2* d_i, int n, int nBits){
    assert(n <= nc);
    int nAlign = iAlignUp(n, 1024);
    loadAOSInt(d_ping, (uint2*)d_i, n, nAlign);
    if (n <= 1024){
        radixSortSingleBlock(d_pong, d_ping, n, nBits);
        sortReturnAOSInt((uint2*)d_o, d_pong, n);
    }
    else {
        implSort(d_pong, d_ping, nAlign, nBits);
        sortReturnAOSInt((uint2*)d_o, d_ping, n);
    }
}

void cplSort::sort(uint2* d_o, uint2* d_i, int n, int nBits){
    sortSupport(d_o, d_i, n , 0, nBits);
}

void cplSort::sort(float2* d_o, float2* d_i, int n){
    sortSupport((uint2*)d_o, (uint2*)d_i, n , 1, 0);
}

void cplSort::sort(float2* d_data, int n){
    sort(d_data, d_data, n);
}

void cplSort::sort(uint2* d_data,  int n, int nBits){
    sort(d_data, d_data, n, nBits);
}

void cplSort::sort(int2* d_data, int n, int nBits){
    sort(d_data, d_data, n, nBits);
}


void testSortSOA(int n, int nBits){
    fprintf(stderr," Test SOA size %d \n", n);
    int nBlocks = iDivUp(n, 1024);

    uint2 * h_ui = new uint2[n] ;
    float2* h_fi = new float2[n];
    int2*   h_ii = new int2[n];
    
    makeRandomUintVector(h_ui, n, nBits);

    unsigned int mean = (unsigned long) 1 << (nBits - 1);
    
    for (int i=0; i< n; ++i){
        h_fi[i].x = (h_ui[i].x - (float) mean) / (mean / 2);
        h_fi[i].y = h_ui[i].y;
    }

    for (int i=0; i< n; ++i){
        if (rand() & 0x01)
            h_ii[i].x = h_ui[i].x;
        else
            h_ii[i].x = -h_ui[i].x;
        h_ii[i].y = h_ui[i].y;
    }

    
    uint2* d_i;
    uint2  *d_uo;
    int2 *d_io;
    float2* d_fo;

    cudaMalloc((void**)&d_i, n * sizeof(uint2));

    d_uo = (uint2*) d_i;
    d_io = (int2*) d_i;
    d_fo = (float2*) d_i;
    
    cplReduce rd;
    cplSort plan(n, &rd);
    
    cudaMemcpy(d_i, h_ui, n * sizeof(uint2), cudaMemcpyHostToDevice);
    plan.sort(d_uo, (uint2*)d_i, n);
    checkDevicePairOrder(d_uo, n, "unsigned sort");
    
    
    cudaMemcpy(d_i, h_fi, n * sizeof(float2), cudaMemcpyHostToDevice);
    plan.sort(d_fo, (float2*)d_i, n);
    checkDevicePairOrder(d_fo, n, "float sort");
    
    cudaMemcpy(d_i, h_ii, n * sizeof(int2), cudaMemcpyHostToDevice);
    plan.sort(d_io, (int2*)d_i, n);
    checkDevicePairOrder(d_io, n, "int sort");

    cudaFree(d_i);
            
    delete []h_ui;
    delete []h_fi;
    delete []h_ii;
}


void testSortAOS(int n, int nBits){
    fprintf(stderr," Test AOS size %d \n", n);
    uint * h_u = new uint[n] ;
    float* h_f = new float[n] ;
    int  * h_i = new int[n] ;
    
    // create the input
    makeRandomUintVector(h_u, n, nBits);

    for (int i=0; i < n; ++i){
        h_i[i] = (int) h_u[i] - 128;
        h_f[i] = h_i[i];
    }

    uint* d_i, *d_o;
    dmemAlloc(d_i, n);
    dmemAlloc(d_o, n);
    
    uint * d_ui = (uint*)d_i  , *d_uo = (uint*)  d_o;
    float* d_fi = (float*)d_i, *d_fo = (float*) d_o;
    int  * d_ii = (int*)d_i   , *d_io = (int*)   d_o;
    
    uint* d_iidx, *d_oidx;
    dmemAlloc(d_iidx, n);
    dmemAlloc(d_oidx, n);
    
    cplReduce rd;
    cplSort plan(n, &rd);

    cudaMemcpy(d_ui, h_u, n * sizeof(uint), cudaMemcpyHostToDevice);

    cudaEventTimer timer;
    timer.start();
    cudaMemcpy(d_o, d_ui, n * sizeof(uint), cudaMemcpyDeviceToDevice);
    timer.stop();
    timer.printTime("Memcpy time ");

    plan.sort(d_uo, d_oidx, d_ui, d_iidx, n);
    checkDeviceOrder(d_uo, n, "Unsigned sort");

    cudaMemcpy(d_fi, h_f, n * sizeof(float), cudaMemcpyHostToDevice);
    plan.EnableAppSorting();
    plan.sort(d_fo, d_oidx, d_fi, d_iidx, n);
    checkDeviceOrder(d_fo, n, "Float sort app");
    
    cudaMemcpy(d_fi, h_f, n * sizeof(float), cudaMemcpyHostToDevice);
    plan.DisableAppSorting();
    plan.sort(d_fo, d_oidx, d_fi, d_iidx, n);
    checkDeviceOrder(d_fo, n, "Float sort");

    cudaMemcpy(d_ii, h_i, n * sizeof(uint), cudaMemcpyHostToDevice);
    plan.sort(d_io, d_oidx, d_ii, d_iidx, n);
    checkDeviceOrder(d_io, n, "Int sort");

    dmemFree(d_i);
    dmemFree(d_o);
    dmemFree(d_iidx);
    dmemFree(d_oidx);

    delete []h_i;
    delete []h_u;
    delete []h_f;
}

