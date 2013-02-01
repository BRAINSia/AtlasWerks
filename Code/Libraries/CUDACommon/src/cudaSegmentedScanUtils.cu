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

#include <cudaSegmentedScanUtils.h>
#include <cutil_comfunc.h>
#include <bitSet.h>
#include <libDefine.h>
#include <cplMacro.h>
#include <VectorMath.h>


////////////////////////////////////////////////////////////////////////////////
// The buildSegmentedFlags functions 
// to create flags array to used with CUDPP segmented scan
// with the value of the flags = 1 indicate the start of the segment
// and 0 otherwise
////////////////////////////////////////////////////////////////////////////////
__global__ void buildSegmentedFlagsFromPos_kernel(uint* d_flags, uint* d_pos, int nSegs)
{
    uint blockId = get_blockID();
    uint id      = get_threadID(blockId);
    if (id < nSegs) {
        d_flags[d_pos[id]] = 1;
    }
}


void buildSegmentedFlagsFromPos(uint* d_flags, int n,
                                uint* d_pos, int nSegs, cudaStream_t stream)
{
    dim3 threads(CTA_SIZE);
    dim3 grids(iDivUp(n, CTA_SIZE));
    checkConfig(grids);
    
    cplVectorOpers::SetMem(d_flags, (uint)0, n);
    buildSegmentedFlagsFromPos_kernel<<<grids, threads, 0, stream>>>(d_flags, d_pos, nSegs);
}


////////////////////////////////////////////////////////////////////////////////
// The buildSegmentedFlags functions 
// using shared memory that is fast on the new card
////////////////////////////////////////////////////////////////////////////////


__global__ void buildSegmentedFlags_share_kernel(uint* d_f, uint* d_i, int n){
    __shared__ int s_data[CTA_SIZE+1];
    uint blockId = get_blockID();
    uint id      = get_threadID(blockId);

    if (id < n){
        s_data[threadIdx.x+1] = d_i[id];
        if (threadIdx.x == 0 && id > 0)
            s_data[0] = d_i[id-1];
    }

    __syncthreads();

    if (id == 0)
        d_f[0] = 1;
    else if (id < n){
        d_f[id] = (s_data[threadIdx.x+1] != s_data[threadIdx.x]);
    }
}

void buildSegmentedFlags_share(uint* d_f, uint* d_i, int n, cudaStream_t stream){
    dim3 threads(CTA_SIZE);
    dim3 grids(iDivUp(n, CTA_SIZE));
    checkConfig(grids);
    
    buildSegmentedFlags_share_kernel<<<grids, threads,0,stream>>>(d_f, d_i, n);
}


////////////////////////////////////////////////////////////////////////////////
// The buildSegmentedFlags functions 
// slowest version, used as reference
////////////////////////////////////////////////////////////////////////////////

__global__ void buildSegmentedFlags_kernel(uint* d_f, uint* d_i, int n){
    uint blockId = get_blockID();
    uint id      = get_threadID(blockId);
    
    if (id == 0)
        d_f[0] = 1;
    else if (id < n){
        d_f[id] = (d_i[id] != d_i[id-1]);
    }
}

void buildSegmentedFlags(uint* d_f, uint* d_i, int n, cudaStream_t stream){
    dim3 threads(CTA_SIZE);
    dim3 grids(iDivUp(n, CTA_SIZE));
    checkConfig(grids);
    
    buildSegmentedFlags_kernel<<<grids, threads,0,stream>>>(d_f, d_i, n);
}

////////////////////////////////////////////////////////////////////////////////
// The buildSegmentedFlags functions 
// texture version, exploi the texture cache, is the fastest implementation
////////////////////////////////////////////////////////////////////////////////

texture<uint , 1, cudaReadModeElementType> com_tex_uint;
__global__ void buildSegmentedFlags_tex_kernel(uint* d_f, int n){
    uint blockId = get_blockID();
    uint id      = get_threadID(blockId);

    if (id == 0)
        d_f[0] = 1;
    else if (id < n){
        d_f[id] = (tex1Dfetch(com_tex_uint,id) != tex1Dfetch(com_tex_uint, id-1));
    }
}


void buildSegmentedFlags_tex(uint* d_f, uint* d_i, int n, cudaStream_t stream){
    dim3 threads(CTA_SIZE);
    dim3 grids(iDivUp(n, CTA_SIZE));
    checkConfig(grids);
    
    cudaBindTexture(0, com_tex_uint, d_i, n * sizeof(uint));
    buildSegmentedFlags_tex_kernel<<<grids, threads,0,stream>>>(d_f, n);
}

////////////////////////////////////////////////////////////////////////////////
// Find the last position of the segment
// with the input in the form xxxxyyyy..yzz..z
// ATTENION : this function only reset the value apprear in the input
//            the right way to used this function is to call a cplVectorOpers::SetMem to 
//            initiate a proper value first before calling this function 
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// Find the last position of the segment
// New version using texture 
////////////////////////////////////////////////////////////////////////////////

__global__ void findLastPos_kernel_tex(uint* d_o, uint* d_i, int n){
    uint blockId = get_blockID();
    uint id      = get_threadID(blockId);

    if (id < n){
        uint data = tex1Dfetch(com_tex_uint,id);
        uint ndata = tex1Dfetch(com_tex_uint,id+1);
        if (data != ndata){
            d_o[data] = id;
        }
    }
}

void findLastPos(uint* d_o, uint* d_i, uint n, cudaStream_t stream){
    dim3 threads(128);
    dim3 grids(iDivUp(n, threads.x));
    checkConfig(grids);
    cudaBindTexture(0, com_tex_uint, d_i, n * sizeof(uint));
    findLastPos_kernel_tex<<<grids, threads,0,stream>>>(d_o, d_i, n);
}

////////////////////////////////////////////////////////////////////////////////
// Find the last position of the segment
// Old version using share memory and T4 format
////////////////////////////////////////////////////////////////////////////////
__global__ void findLastPos_kernel(uint* g_last, uint4* g_index4, int n){

    const uint nThread   = 128;
    __shared__ uint s_first[nThread + 128];

    uint blockId = blockIdx.x + blockIdx.y * gridDim.x;
    uint tid     = threadIdx.x;
    uint id      = blockId * nThread + tid;

    uint4 data4 = g_index4[id];

    
    id*=4;
    if (id + 1 < n)
        if (data4.x != data4.y)
            g_last[data4.x] = id;
    
    if (id + 2 < n)
        if (data4.y != data4.z)
            g_last[data4.y] = id + 1;

    if (id + 3 < n)
        if (data4.z != data4.w)
            g_last[data4.z] = id + 2;
    
    if (tid == 0)
        s_first[nThread-1] = (id + nThread * 4 < n) ? (((uint*)g_index4)[id + nThread *4]) : (0xFFFFFFFF);
    else
        s_first[tid-1] = data4.x;
    
    __syncthreads();

    if (id + 4 < n) {
        if (data4.w != s_first[tid])
            g_last[data4.w] = id + 3;
    }else {
        if (id < n) { // add the last element
            int k = (n-1) - id;
            int lv;
            
            if (k == 0) lv = data4.x;
            if (k == 1) lv = data4.y;
            if (k == 2) lv = data4.z;
            if (k == 3) lv = data4.w;
            
            g_last[lv] = n-1;
        }
    }
}

void findLastPos(uint* g_pos, uint4* g_iData, uint n){
    dim3 threads(128);
    uint nBlocks = iDivUp(n, threads.x * 4);
    dim3 grids(nBlocks);
    checkConfig(grids);
    findLastPos_kernel<<<grids, threads>>>(g_pos, g_iData, n);
}

////////////////////////////////////////////////////////////////////////////////
// Compress flags function
// that convert the flags array to the array of binary flags
////////////////////////////////////////////////////////////////////////////////
__global__ void compressFlags_kernel(uint* g_flags, uint* g_cnt, uint n){
    
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
    uint flag = 0;
    uint* row = s_data + tid * WPAD;
    for (uint i=0; i < 32; ++i)
        if (row[i] > 0) flag |= (1 << i);

    if ((blockId * 32 + tid) * 32 < n) 
        g_flags[blockId * 32 + tid] = flag;
}

void compressFlags(uint* g_flags, uint* g_cnt, uint n, cudaStream_t stream){
    uint nBlocks = iDivUp(n, 1024);
    dim3 threads(32);
    dim3 grids(nBlocks);
    checkConfig(grids);
    
    compressFlags_kernel<<<grids, threads,0,stream>>>(g_flags, g_cnt, n);
}

__global__ void cplCompressFlags_kernel(uint* d_of, uint* d_if, int n){
    __shared__ uint s_flags[32][32+1];

    uint blockId = get_blockID();
    uint off     = blockId * 1024;
    uint tid     = threadIdx.x;

    for (uint i=tid; i < 1024; i+= blockDim.x)
    {
        uint sIdx = i & 31;
        uint sIdy = i >> 5;

        s_flags[sIdy][sIdx] = (off + i < n) ? d_if[off + i] : 0;
    }
        
    __syncthreads();

    if (tid < 32 ){
        uint flags = 0;
        for (uint i=0; i< 32; ++i)
            if (s_flags[tid][i])
                flags |= (1 << i);

        if ((((blockId << 5) + threadIdx.x) << 5) < n)
            d_of[(blockId << 5) + threadIdx.x] = flags;
    }
}
    
void cplCompressFlags(uint* d_of, uint* d_if, int n, cudaStream_t stream){
    dim3 threads(256);
    dim3 grids(iDivUp(n, 1024));
    checkConfig(grids);

    cplCompressFlags_kernel<<<grids, threads, 0, stream>>>(d_of, d_if, n);
}

__global__ void cplDecompressFlags_kernel(uint* d_of, uint* d_if, int n){
    __shared__ uint s_flags[32][32 + 1];

    uint blockId = get_blockID();
    uint tid     = threadIdx.x;

    
    if (tid < 32){
        uint flags = (blockId * 32 + tid < n) ? d_if[blockId * 32 + tid] : 0;
        for (int i=0; i < 32; ++i)
            s_flags[tid][i] = ((flags & (1 << i)) > 0);
    }
    
    __syncthreads();

    uint off = blockId * 1024;
    for (uint i=tid; i < 1024; i+= blockDim.x)
    {
        uint sIdx = i & 31;
        uint sIdy = i >> 5;
        
        if (off + i < n * 32)
            d_of[off + i] = s_flags[sIdy][sIdx];
    }
}

void cplDecompressFlags(uint* d_of, uint* d_if, int n, cudaStream_t stream){
    dim3 threads(256);
    dim3 grids(iDivUp(n, 32));
    checkConfig(grids);

    cplDecompressFlags_kernel<<<grids, threads, 0, stream>>>(d_of, d_if, n);
}



void cudaSegmentedScanUtilsTest(int n, int s){
    bitSet bst(n);
    bst.insert(n-1);
    uint* h_f = new uint[n];
    for (int i=0; i< n; ++i)
        h_f[i] = 0;
    h_f[0] = 1;
    
    for (uint i=0; i< s; ++i) {
        int pos = rand() % n;
        bst.insert(pos);
        if (pos < n- 1)
            h_f[pos + 1] = 1;
    }

    int id = 0;
    int* aId = new int[n];
    uint* pos = new uint [s+1];
        
    for (int i=0; i < n; ++i){
        aId[i] = id;
        if (bst.check(i))
            ++id;
    }
    uint* d_i;
    uint* d_f;

    
    cudaMalloc((void**)&d_i, n * sizeof(uint));
    cudaMalloc((void**)&d_f, n * sizeof(uint));

    cudaMemcpy(d_i, aId, n * sizeof(uint), cudaMemcpyHostToDevice);

    int nIters = 100;

    float elapsedTime = 0.f;
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start,0);
    for (int i=0; i< nIters; ++i)
        buildSegmentedFlags(d_f, d_i, n);
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf( "\nRegular build: %f (ms)\n\n", elapsedTime/nIters);
    fprintf(stderr, " >>> C >>>");
    testError(h_f, d_f, n,"Regular scan ");
    
    //printDeviceArray1D(d_i, n, "Input ");
    cudaEventRecord(start,0);
    for (int i=0; i< nIters; ++i)
        buildSegmentedFlags_tex(d_f, d_i, n);
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf( "\nTex build: %f (ms)\n\n", elapsedTime/nIters);
    testError(h_f, d_f, n,"Tex scan ");
        
    cudaEventRecord(start,0);
    for (int i=0; i< nIters; ++i)
        buildSegmentedFlags_share(d_f, d_i, n);
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf( "\nShare build: %f (ms)\n\n", elapsedTime/nIters);
    testError(h_f, d_f, n,"Share scan ");

    uint* d_pos;
    cudaMalloc((void**)&d_pos, (s+1) * sizeof(uint));
    cudaEventRecord(start,0);
    for (int i=0; i< nIters; ++i)
        findLastPos(d_pos, (uint4*)d_i, n);
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf( "\nFind last pos old: %f (ms)\n\n", elapsedTime/nIters);
    
    cudaMemcpy(pos, d_pos, (s + 1) * sizeof (uint), cudaMemcpyDeviceToHost);
    cudaEventRecord(start,0);
    for (int i=0; i< nIters; ++i)
        findLastPos(d_pos, d_i, n);
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf( "\nFind last with texture: %f (ms)\n\n", elapsedTime/nIters);
    testError(pos, d_pos, s+1,"Find last pos");
}
