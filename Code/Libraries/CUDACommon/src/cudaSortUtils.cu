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

#include <cpl.h>
#include <cudaSortUtils.h>
#include <typeConvert.h>
#include <cudaTexFetch.h>

#ifdef __DEVICE_EMULATION__
#define __SYNC __syncthreads();
#else
#define __SYNC
#endif


#if (CUDART_VERSION  < 2020)
template<int blockSize>
__device__ void sum_CTA(int val, int* sdata){
    // do reduction in shared mem
    int tid = threadIdx.x;
    sdata[tid] = val;
    __syncthreads();
    
    if (blockSize >= 512) { if (tid < 256) { sdata[tid] +=  sdata[tid + 256]; } __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
    if (blockSize >= 128) { if (tid <  64) { sdata[tid] += sdata[tid +  64]; } __syncthreads(); }

    if (tid < 32)
    {
        if (blockSize >=  64) sdata[tid] += sdata[tid + 32]; __SYNC;
        if (blockSize >=  32) sdata[tid] += sdata[tid + 16]; __SYNC;
        if (blockSize >=  16) sdata[tid] += sdata[tid +  8]; __SYNC;
        if (blockSize >=   8) sdata[tid] += sdata[tid +  4]; __SYNC;
        if (blockSize >=   4) sdata[tid] += sdata[tid +  2]; __SYNC;
        if (blockSize >=   2) sdata[tid] += sdata[tid +  1]; __SYNC;
    }
    __syncthreads();
}
#else
template<int blockSize>
__device__ void sum_CTA(int val, volatile int* sdata){
    // do reduction in shared mem
    int tid = threadIdx.x;
    sdata[tid] = val;
    __syncthreads();
    
    if (blockSize >= 512) { if (tid < 256) { sdata[tid] = val = val +  sdata[tid + 256]; } __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) { sdata[tid] = val = val +  sdata[tid + 128]; } __syncthreads(); }
    if (blockSize >= 128) { if (tid <  64) { sdata[tid] = val = val +  sdata[tid +  64]; } __syncthreads(); }

    if (tid < 32)
    {
        if (blockSize >=  64) sdata[tid] = val = val +  sdata[tid + 32]; __SYNC;
        if (blockSize >=  32) sdata[tid] = val = val +  sdata[tid + 16]; __SYNC;
        if (blockSize >=  16) sdata[tid] = val = val +  sdata[tid +  8]; __SYNC;
        if (blockSize >=   8) sdata[tid] = val = val +  sdata[tid +  4]; __SYNC;
        if (blockSize >=   4) sdata[tid] = val = val +  sdata[tid +  2]; __SYNC;
        if (blockSize >=   2) sdata[tid] = val = val +  sdata[tid +  1]; __SYNC;
    }
    __syncthreads();
}
#endif

template<typename T, int rep, int blockSize, int dir>
__global__ void checkOrder_kernel_int(int* d_o, int n){
    uint blockId = blockIdx.x + blockIdx.y * gridDim.x;
    uint id      = blockId * blockSize * rep + threadIdx.x;
#if (CUDART_VERSION  >= 2020)
    volatile __shared__ int s_flags[blockSize];
#else
    __shared__ int s_flags[blockSize];
#endif
    int ord = 0;
    for (int i=0; i < rep; ++i){
        if (id < n)
            ord += (dir == 0) ? (fetch(id, (T*)NULL) > fetch(id+1, (T*)NULL)) :
                (fetch(id, (T*)NULL) < fetch(id+1, (T*)NULL));
        id +=blockSize;
    }
    sum_CTA<blockSize>(ord, s_flags);
    if (threadIdx.x ==0)
        d_o[blockId] = s_flags[0];
}

template<typename T, int rep>
int checkOrderTexture(cplReduce* p_Rd, T* d_i, int n, int* d_temp, int dir){
    if (n == 1)
        return 1;
    const int blockSize = 128;
    
    dim3 threads(blockSize);
    dim3 grids(iDivUp(n-1, blockSize * rep));

    cache_bind(d_i);
        
    if (dir==0)
        checkOrder_kernel_int<T, rep, blockSize, 0><<<grids, threads>>>(d_temp, n-1);
    else
        checkOrder_kernel_int<T, rep, blockSize, 1><<<grids, threads>>>(d_temp, n-1);
    
    int sum = p_Rd->Sum(d_temp, grids.x);
    return (sum ==0);
}

/* Brief: this function will perform paralle reduction sum on the block
 * Block size is the number of elements per block
 * The number of thread per block is half the block size
 */
template<class T, int blockSize, int dir>
__global__ void order_kernel(int* g_odata, T* g_idata, unsigned int n){
#define T4 typename typeToVector<T,4>::Result
    
    __shared__ T   sdata[blockSize * 2];
    __shared__ int cond [blockSize];

    T4 * const g_idata4 = (T4*) g_idata;
            
    int blockId= blockIdx.x + blockIdx.y * gridDim.x;
    int offset = blockId * (blockSize * 2);

    const T lim= (dir == 0) ? getMax<T>() : getMin<T>();
    
    int ai     = threadIdx.x;
    int bi     = ai + blockSize;

    T4 data0, data1;

    int i = 4 * (ai + offset);

    data0.x = lim;
    if (i < n)  data0   = g_idata4[ai + offset];
    if (i+1>=n) data0.y = lim;
    if (i+2>=n) data0.z = lim;
    if (i+3>=n) data0.w = lim;
    

    i = 4 * (bi + offset);
    data1.x = lim;
    if (i < n)  data1   = g_idata4[bi + offset];
    if (i+1>=n) data1.y = lim;
    if (i+2>=n) data1.z = lim;
    if (i+3>=n) data1.w = lim;

    if (ai >=1)
        sdata[ai-1] = data0.x;
    else
        sdata[(2*blockSize-1)] = ((blockId + 1) * (8 * blockSize) < n) ? g_idata[(blockId + 1) *  (8 * blockSize)] : lim;
    
    sdata[bi-1] = data1.x;
    
    __syncthreads();

    int s0, s1;
    if (dir == 0){
        s0 = (data0.x > data0.y) + (data0.y > data0.z) + (data0.z > data0.w) + (data0.w > sdata[ai]);
        s1 = (data1.x > data1.y) + (data1.y > data1.z) + (data1.z > data1.w) + (data1.w > sdata[bi]);
    }
    else {
        s0 = (data0.x < data0.y) + (data0.y < data0.z) + (data0.z < data0.w) + (data0.w < sdata[ai]);
        s1 = (data1.x < data1.y) + (data1.y < data1.z) + (data1.z < data1.w) + (data1.w < sdata[bi]);
    }

    sum_CTA<blockSize>(s0 + s1, cond);

    if (threadIdx.x ==0)
        g_odata[blockId] = cond[0];
}

/*
 * Brief the function check the input array if it is in order or out of order 
 * @in : d_i : data array   
 *       n   : number of elements
 *       dir : 0: assending order  1: decending order
 */

template<class T>
int checkOrderPlan::checkOrder(T* d_i, int n, int* d_block, int dir)
{
    // 1.  Compute the number of block
    const int nThreads = 128;
    int     nBlocks    = iDivUp(n, nThreads * 8);
    
    // 2 . Check order, write result of each block to the block
    dim3 threads(nThreads);
    dim3 grids(nBlocks);
    
    if (dir == 0)
        order_kernel<T, nThreads,0><<<grids, threads>>>(d_block, d_i, n);
    else 
        order_kernel<T, nThreads,1><<<grids, threads>>>(d_block, d_i, n);

    // 3 . Final check order on the block
    uint sum = m_rd->Sum(d_block, nBlocks);
    return (sum == 0);
};


template int checkOrderPlan::checkOrder<float>(float* d_i, int n, int* d_block, int dir);
template int checkOrderPlan::checkOrder<uint>(uint* d_i, int n, int* d_block, int dir);
template int checkOrderPlan::checkOrder<int>(int* d_i, int n,  int* d_block, int dir);

/*
 * Brief the function check the input array if it is in order or out of order 
 * @in : d_i : data array   
 *       n   : number of elements
 *       dir : 0: assending order  1: decending order
 */

template<class T>
int checkOrderPlan::fastCheckOrder(T* d_i, int n, int * d_block, int dir){
    // Check with 64K element first
    const int seg = 1 << 16;
    dim3 threads(128);
    int  nBlocks = iDivUp(n, threads.x * 8);

    int  h_temp[64];
    int nSubSize  = min(seg, n);
    int nSubBlock = min(64, nBlocks);

    // Check the order with the 64 segment fist
    if (dir == 0)
        order_kernel<T, 128,0><<<nSubBlock, 128>>>(d_block, d_i, nSubSize);
    else 
        order_kernel<T, 128,1><<<nSubBlock, 128>>>(d_block, d_i, nSubSize);

    cudaMemcpy(h_temp, d_block, sizeof(int) * nSubBlock, cudaMemcpyDeviceToHost);

    for (int i=0; i< nSubBlock; ++i)
        if (h_temp[i] != 0) return 0;

    // Check order with the rest
    if ( n <= seg)
        return 1;

    int nLeft = n - (seg - 1024);
    int result = checkOrder<T>(d_i + seg - 1024, nLeft, d_block, dir);
    return result;
}


template int checkOrderPlan::fastCheckOrder<float>(float* d_i, int n, int* d_block, int dir);
template int checkOrderPlan::fastCheckOrder<int>(int* d_i, int n, int* d_block, int dir);
template int checkOrderPlan::fastCheckOrder<uint>(uint* d_i, int n, int* d_block, int dir);

//------------------------------------------------------------------------------------------
void testCheckOrderPerformance(int n){
    fprintf(stderr, "Size of the problem %d \n", n);
    // Check right order test
    int nMaxBlocks = iDivUp(n, 256);
        
    int* d_i;
    int* d_block;
    
    dmemAlloc(d_i, n);
    dmemAlloc(d_block, nMaxBlocks);

    cplReduce     rd;
    rd.init();

    checkOrderPlan plan(&rd);

    int nIters = 100;
    float elapsedTime = 0.f;
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start,0);
    for (int i=0; i< nIters; ++i)
        plan.checkOrder(d_i, n, d_block, 0);
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    fprintf(stderr, "\nCheck order time  : %f (ms)\n\n", elapsedTime);

    cudaEventRecord(start,0);
    for (int i=0; i< nIters; ++i)
        checkOrderTexture<int, 2>(&rd, d_i, n, d_block, 0);
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    fprintf(stderr, "\nCheck order time with texture rep 2: %f (ms)\n\n", elapsedTime);

    cudaEventRecord(start,0);
    for (int i=0; i< nIters; ++i)
        checkOrderTexture<int, 4>(&rd, d_i, n, d_block, 0);
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    fprintf(stderr, "\nCheck order time with texture rep 4: %f (ms)\n\n", elapsedTime);

    cudaEventRecord(start,0);
    for (int i=0; i< nIters; ++i)
        checkOrderTexture<int, 6>(&rd, d_i, n, d_block, 0);
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    fprintf(stderr, "\nCheck order time with texture rep 6: %f (ms)\n\n", elapsedTime);

    dmemFree(d_block);
    dmemFree(d_i);
}
void testCheckOrder(int n){
    fprintf(stderr, "Size of the problem %d \n", n);
    int nMaxBlocks = iDivUp(n, 256);
    
    // Check right order test
    uint* d_i;
    int* d_block;
    
    dmemAlloc(d_i, n);
    dmemAlloc(d_block, nMaxBlocks);

    cplReduce    rd;
    rd.init();
    
    checkOrderPlan plan(&rd);
    /*
      Test set with testing regular functions
    */
    //1. constant test
    cplVectorOpers::SetMem(d_i,(uint) 5, n);
    int r = 0, i;
    for (i=1; i< n; ++i){
        r = checkOrderTexture<uint, 6>(&rd, d_i, i, d_block, 0);
        if (r == 0)
            break;
    }
    if (r==0){
        fprintf(stderr, "Constant test FAILED at %d \n", i);
    }
    else
        fprintf(stderr, "Constant test PASSED \n");

    //2. Ramp test
    cplVectorOpers::SetLinear(d_i, n);
    for (i=1; i< n; ++i){
        r = checkOrderTexture<uint, 6>(&rd, d_i, i, d_block, 0);
            //plan.checkOrder(d_i, i, 0);
        if (r == 0)
            break;
    }
    if (r==0){
        fprintf(stderr, "Ramp test FAILED at %d \n", i);
    }
    else
        fprintf(stderr, "Ramp test PASSED \n");

    // Check wrong order test

    //3. Ramp down test
    cplVectorOpers::SetLinearDown(d_i, n);
    for (i=2; i< n; ++i){
        r = plan.checkOrder(d_i, i, d_block, 0);
        if (r == 1)
            break;
    }
    if (r==1){
        fprintf(stderr, "Ramp down test FAILED at %d \n", i);
    }
    else
        fprintf(stderr, "Ramp down test PASSED \n");

    //4. Spike Ramp test
    for (i=3; i< n; ++i){
        cplVectorOpers::SetLinear(d_i, i-1);
        cplVectorOpers::SetMem(d_i + i-1, (uint) 0, 1);
        r = checkOrderTexture<uint, 6>(&rd, d_i, i, d_block, 0);
        //plan.checkOrder(d_i, i, 0);
        if (r == 1)
            break;
    }
    if (r==1){
        fprintf(stderr, "Spike ramp test FAILED at %d \n", i);
    }
    else
        fprintf(stderr, "Spike ramp test PASSED \n");

   
    /*
      Test with fast testing
    */
    fprintf(stderr, "Testing with fast checking function\n");
    //1. constant test
    cplVectorOpers::SetMem(d_i,(uint) 5, n);
    
    for (i=1; i< n; ++i){
        r = plan.fastCheckOrder(d_i, i, d_block, 0);
        if (r == 0)
            break;
    }
    if (r==0){
        fprintf(stderr, "Constant test FAILED at %d \n", i);
    }
    else
        fprintf(stderr, "Constant test PASSED \n");

    //2. Ramp test
    cplVectorOpers::SetLinear(d_i, n);
    for (i=1; i< n; ++i){
        r = plan.fastCheckOrder(d_i, i, d_block, 0);
        if (r == 0)
            break;
    }
    if (r==0){
        fprintf(stderr, "Ramp test FAILED at %d \n", i);
    }
    else
        fprintf(stderr, "Ramp test PASSED \n");

    // Check wrong order test

    //3. Ramp down test
    cplVectorOpers::SetLinearDown(d_i, n);
    for (i=2; i< n; ++i){
        r = plan.fastCheckOrder(d_i, i, d_block, 0);
        if (r == 1)
            break;
    }
    if (r==1){
        fprintf(stderr, "Ramp down test FAILED at %d \n", i);
    }
    else
        fprintf(stderr, "Ramp down test PASSED \n");

    //4. Spike Ramp test
    for (i=3; i< n; ++i){
        cplVectorOpers::SetLinear(d_i, i-1);
        cplVectorOpers::SetMem(d_i + i-1, (uint) 0, 1);
        r = plan.fastCheckOrder(d_i, i, d_block, 0);
        if (r == 1)
            break;
    }
    if (r==1){
        fprintf(stderr, "Spike ramp test FAILED at %d \n", i);
    }
    else
        fprintf(stderr, "Spike ramp test PASSED \n");

    // performance check
    dmemFree(d_i);
    dmemFree(d_block);
}

////////////////////////////////////////////////////////////////////////////////
//
////////////////////////////////////////////////////////////////////////////////
template<class T2>
inline bool checkHostPairOrder(T2* h_a, unsigned int n){
    unsigned int i=0;
    for (i=0; i< n-1; ++i){
        if (h_a[i].x > h_a[i+1].x){
            std::cerr << "Test FAILED: Wrong order at "<< i <<" "<< h_a[i].x <<" "<< h_a[i+1].x << std::endl;
            return false;
        }
    }
    if (i == n-1)
        std::cerr << "Test PASSED " << std::endl;
    return true;
}

template bool checkHostPairOrder(float2* h_a, unsigned int n);
template bool checkHostPairOrder(int2* h_a, unsigned int n);
template bool checkHostPairOrder(uint2* h_a, unsigned int n);

template<class T2>
inline bool checkHostPairOrder(T2* h_a, unsigned int n, const char* name){
    unsigned int i=0;
    for (i=0; i< n-1; ++i){
        if (h_a[i].x > h_a[i+1].x){
            std::cerr << "Test " << name << " FAILED: Wrong order at "<< i <<" "<< h_a[i].x <<" "<< h_a[i+1].x << std::endl;
            return false;
        }
    }
    if (i == n-1)
        std::cerr << "Test " << name << " PASSED " << std::endl;
    return true;
}

template bool checkHostPairOrder(float2* h_a, unsigned int n, const char* name);
template bool checkHostPairOrder(int2* h_a, unsigned int n, const char* name);
template bool checkHostPairOrder(uint2* h_a, unsigned int n, const char* name);

////////////////////////////////////////////////////////////////////////////////
//
////////////////////////////////////////////////////////////////////////////////

template<class T2>
inline bool checkDevicePairOrder(T2* d_a, unsigned int n){
    T2* h_a = new T2[n];
    cudaMemcpy(h_a, d_a, n * sizeof(uint2), cudaMemcpyDeviceToHost);
    bool result = checkHostPairOrder(h_a, n);
    delete []h_a;
    return result;
}

template bool checkDevicePairOrder(float2* h_a, unsigned int n);
template bool checkDevicePairOrder(int2* h_a, unsigned int n);
template bool checkDevicePairOrder(uint2* h_a, unsigned int n);

template<class T2>
inline bool checkDevicePairOrder(T2* d_a, unsigned int n, const char* name){
    T2* h_a = new T2[n];
    cudaMemcpy(h_a, d_a, n * sizeof(uint2), cudaMemcpyDeviceToHost);
    bool result = checkHostPairOrder(h_a, n, name);
    delete []h_a;
    return result;
}

template bool checkDevicePairOrder(float2* h_a, unsigned int n, const char* name);
template bool checkDevicePairOrder(int2* h_a, unsigned int n, const char* name);
template bool checkDevicePairOrder(uint2* h_a, unsigned int n, const char* name);

////////////////////////////////////////////////////////////////////////////////
//
////////////////////////////////////////////////////////////////////////////////

template<class T>
inline bool checkHostOrder(T* h_a, unsigned int n){
    unsigned int i=0;
    for (i=0; i< n-1; ++i){
        if (h_a[i] > h_a[i+1]){
            std::cerr << "Test FAILED: Wrong order at "<< i <<" "<< h_a[i] <<" "<< h_a[i+1] << std::endl;
            return false;
        }
    }
    std::cerr << "Test PASSED " << std::endl;
    return true;
}

template bool checkHostOrder(float* d_a, unsigned int a);
template bool checkHostOrder(int* d_a, unsigned int a);
template bool checkHostOrder(uint* d_a, unsigned int a);

template<class T>
inline bool checkHostOrder(T* h_a, unsigned int n, const char *name){
    unsigned int i=0;
    for (i=0; i< n-1; ++i){
        if (h_a[i] > h_a[i+1]){
            std::cerr << "Test " << name << " FAILED:  Wrong order at "<< i <<" "<< h_a[i] <<" "<< h_a[i+1] << std::endl;
            return false;
        }
    }
    std::cerr << "Test " << name << " PASSED " << std::endl;
    return true;
}

template bool checkHostOrder(float* d_a, unsigned int a, const char* name);
template bool checkHostOrder(int* d_a, unsigned int a, const char* name);
template bool checkHostOrder(uint* d_a, unsigned int a, const char* name);

////////////////////////////////////////////////////////////////////////////////
//
////////////////////////////////////////////////////////////////////////////////
template<class T>
inline bool checkDeviceOrder(T* d_a, unsigned int n){
    T* h_a = new T[n];
    cudaMemcpy((void*)h_a, (void*)d_a, n * sizeof(T), cudaMemcpyDeviceToHost);
    bool r = checkHostOrder<T>(h_a, n);
    delete []h_a;
    return r;
}

template bool checkDeviceOrder(float* d_a, unsigned int a);
template bool checkDeviceOrder(int* d_a, unsigned int a);
template bool checkDeviceOrder(uint* d_a, unsigned int a);


template<class T>
inline bool checkDeviceOrder(T* d_a, unsigned int n, const char* name){
    T* h_a = new T[n];
    cudaMemcpy((void*)h_a, (void*)d_a, n * sizeof(T), cudaMemcpyDeviceToHost);
    bool r = checkHostOrder<T>(h_a, n, name);
    delete []h_a;
    return r;
}

template bool checkDeviceOrder(float* d_a, unsigned int a, const char* name);
template bool checkDeviceOrder(int* d_a, unsigned int a, const char* name);
template bool checkDeviceOrder(uint* d_a, unsigned int a, const char* name);


