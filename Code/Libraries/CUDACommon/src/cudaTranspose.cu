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
#include <cudaTranspose.h>
#include <cudaTexFetch.h>

#define BLOCK_DIM 16
// This kernel is optimized to ensure all global reads and writes are coalesced,
// and to avoid bank conflicts in shared memory.  This kernel is up to 11x faster
// than the naive kernel below.  Note that the shared memory array is sized to 
// (BLOCK_DIM+1)*BLOCK_DIM.  This pads each row of the 2D block in shared memory 
// so that bank conflicts do not occur when threads address the array column-wise.
template<typename T>
__global__ void transpose_x16y16_kernel(T *odata, T *idata, uint width, uint height)
{
	__shared__ T block[BLOCK_DIM][BLOCK_DIM+1];
	
	// read the matrix tile into shared memory
	uint xIndex = (blockIdx.x << 4) + threadIdx.x;
	uint yIndex = (blockIdx.y << 4) + threadIdx.y;
    
	if((xIndex < width) && (yIndex < height))
	{
		uint index_in = yIndex * width + xIndex;
		block[threadIdx.y][threadIdx.x] = idata[index_in];
	}

	__syncthreads();

	// write the transposed matrix tile to global memory
	xIndex = (blockIdx.y << 4) + threadIdx.x;
	yIndex = (blockIdx.x << 4) + threadIdx.y;
	if((xIndex < height) && (yIndex < width))
	{
		unsigned int index_out = yIndex * height + xIndex;
		odata[index_out] = block[threadIdx.x][threadIdx.y];
	}
}

template<typename T>
void transpose_x16y16(T* odata, T* idata, int width, int height, cudaStream_t stream){
    dim3 threads(BLOCK_DIM, BLOCK_DIM);
    dim3 grids(iDivUp(width,BLOCK_DIM), iDivUp(height,BLOCK_DIM));
    
    transpose_x16y16_kernel<<<grids, threads, 0, stream>>>(odata,idata, width, height);
}

/*--------------------------------------------------------------------------------------*/
__global__ void transpose_x16y16_C2_kernel(uint2 *odata, uint2 *idata, uint width, uint height)
{
    __shared__ uint block_x[BLOCK_DIM][BLOCK_DIM+1];
    __shared__ uint block_y[BLOCK_DIM][BLOCK_DIM+1];

	
    // read the matrix tile into shared memory
    uint xIndex = (blockIdx.x << 4) + threadIdx.x;
    uint yIndex = (blockIdx.y << 4) + threadIdx.y;
  
    if((xIndex < width) && (yIndex < height))
    {
        uint index_in = yIndex * width + xIndex;
        uint2 data = idata[index_in];;
        block_x[threadIdx.y][threadIdx.x] = data.x;
        block_y[threadIdx.y][threadIdx.x] = data.y;
    }

    __syncthreads();

    // write he transposed matrix tile to global memory
    xIndex = (blockIdx.y << 4) + threadIdx.x;
    yIndex = (blockIdx.x << 4) + threadIdx.y;
    if((xIndex < height) && (yIndex < width))
    {
        unsigned int index_out = yIndex * height + xIndex;
        odata[index_out] = make_uint2(block_x[threadIdx.x][threadIdx.y], block_y[threadIdx.x][threadIdx.y]);
    }
}

void transpose_x16y16_C2(uint2* odata, uint2* idata, int width, int height, cudaStream_t stream){
    dim3 threads(BLOCK_DIM, BLOCK_DIM);
    dim3 grids(iDivUp(width,BLOCK_DIM), iDivUp(height,BLOCK_DIM));
    transpose_x16y16_C2_kernel<<<grids, threads, 0, stream>>>(odata,idata, width, height);
}

/*-----------------------------------------------------------------------------------------------------*/
// This naive transpose kernel suffers from completely non-coalesced writes.
// It can be up to 10x slower than the kernel above for large matrices.
template<typename T>
__global__ void transpose_naive_kernel(T *odata, T* idata, int width, int height)
{
    unsigned int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int yIndex = blockDim.y * blockIdx.y + threadIdx.y;
   
    if (xIndex < width && yIndex < height)
    {
        unsigned int index_in  = xIndex + width * yIndex;
        unsigned int index_out = yIndex + height * xIndex;
        odata[index_out] = idata[index_in]; 
    }
}

template<typename T>
void transpose_naive(T* odata, T* idata, int width, int height, cudaStream_t stream){
    dim3 threads(BLOCK_DIM, BLOCK_DIM);
    dim3 grids(iDivUp(width,BLOCK_DIM), iDivUp(height,BLOCK_DIM));
    transpose_naive_kernel<<<grids, threads, 0, stream>>>(odata,idata, width, height);
}

/*-----------------------------------------------------------------------------------------------------*/
/* Transpose with the 32x8 kernel size , for each kernel read 4 element instead of one */

template<typename T>
__global__ void transpose_x32y8_kernel(T *odata, T *idata, int dimx, int dimy)
{
    __shared__ T block[32][32+1];

    unsigned int xBlock = blockIdx.x << 5;
    unsigned int yBlock = blockIdx.y << 5;

    unsigned int yIndex = yBlock + threadIdx.y;
    unsigned int xIndex = xBlock + threadIdx.x;
  
    unsigned int index_out, index_in;

    // load block and transpose into shared memory
    for (int i=0; i < 32; i += 8) {
        // turn 2d index into 1d index
        index_in = yIndex *  dimx + xIndex;
        // load data into shared memory
        if (yIndex < dimy && xIndex < dimx)
            block[threadIdx.x][threadIdx.y + i] = idata[index_in];
        yIndex  += 8;
    }

    __syncthreads();

    yIndex = xBlock + threadIdx.y;
    xIndex = yBlock + threadIdx.x;
        
    // store transposed block back to device memory
    for (int i=0; i < 32; i += 8) {
        // turn 2d index into 1d index
        index_out = yIndex * dimy + xIndex;
        // store transposed block back to device memory
        if (yIndex < dimx && xIndex < dimy)
            odata[index_out] = block[threadIdx.y + i][threadIdx.x];
        yIndex   += 8;
    }
}


template<typename T>
void transpose_x32y8(T *odata, T *idata, int width, int height, cudaStream_t stream){
    dim3 threads(32, 8);
    dim3 grids(iDivUp(width, threads.x), iDivUp(height, threads.y * 4));
    transpose_x32y8_kernel<<<grids, threads, 0, stream>>>(odata,idata, width, height);
}

/*-----------------------------------------------------------------------------------------------------*/
template<typename T>
__global__ void transpose_x32y8_dev(T *odata, T *idata, int lg_dimx, int lg_dimy)
{
    __shared__ T block[32][32+1];
    
    unsigned int xBlock = blockIdx.x << 5; // blockIdx.x * 32
    unsigned int yBlock = blockIdx.y << 5; // blockIdx.y * 32

    unsigned int yIndex = yBlock + threadIdx.y;
    unsigned int xIndex = xBlock + threadIdx.x;
  
    unsigned int index_out, index_in;

    int dimx = 1 << lg_dimx;
    int dimy = 1 << lg_dimy;
    
    
    // load block and transpose into shared memory
    for (int i=0; i < 32; i += 8) {
        // turn 2d index into 1d index
        index_in = (yIndex << lg_dimx) + xIndex;
        // load data into shared memory
        if (yIndex < dimy && xIndex < dimx)
            block[threadIdx.x][threadIdx.y + i] = idata[index_in];
        yIndex  += 8;
    }

    __syncthreads();

    yIndex = xBlock + threadIdx.y;
    xIndex = yBlock + threadIdx.x;
        
    // store transposed block back to device memory
    for (int i=0; i < 32; i += 8) {
        // turn 2d index into 1d index
        index_out = (yIndex << lg_dimy) + xIndex;
        // store transposed block back to device memory
        if (yIndex < dimx && xIndex < dimy)
            odata[index_out] = block[threadIdx.y + i][threadIdx.x];
        yIndex   += 8;
    }

}

template<typename T>
void transpose_x32y8_v2(T *odata, T *idata, int width, int height, cudaStream_t stream){
    dim3 threads(32, 8);
    dim3 grids(iDivUp(width, 32), iDivUp(height, 32));
    transpose_x32y8_dev<<<grids, threads, 0, stream>>>(odata,idata, 11, 11);
}

/*-----------------------------------------------------------------------------------------------------*/
template<typename T>
void transpose(T *odata, T *idata, int width, int height, cudaStream_t stream){
    dim3 threads(32, 8);
    dim3 grids(iDivUp(width, 32), iDivUp(height, 32));
    
    if (isPow2(width) && isPow2(height)){
        int logw = log2_Pow2(width);
        int logh = log2_Pow2(height);
        transpose_x32y8_dev<<<grids, threads, 0, stream>>>(odata,idata, logw, logh);
    }
    else {
        transpose_x32y8_kernel<<<grids, threads, 0, stream>>>(odata,idata, width, height);
    }
}

template void transpose(float* odata, float* idata, int width, int height, cudaStream_t stream);
template void transpose(int* odata, int* idata, int width, int height, cudaStream_t stream);
template void transpose(uint* odata, uint* idata, int width, int height, cudaStream_t stream);

void testTranspose(int w, int h){
    
    fprintf(stderr, "Input size %d %d \n",w, h);

    uint* d_idata, *d_odata;
    uint* h_idata, *h_odata;
    
    uint n = w * h;
    h_idata = new uint [n];
    h_odata = new uint [n];

    dmemAlloc(d_idata, n);
    dmemAlloc(d_odata, n);
    
    int id = 0;
    for (int j=0; j < h; ++j)
        for (int i=0; i< w; ++i, ++id){
            uint r = rand();
            h_idata[id] = r;
        }
    uint timer;
    CUT_SAFE_CALL( cutCreateTimer( &timer));
    CUT_SAFE_CALL( cutStartTimer( timer));
    id = 0;
    for (int j=0; j < h; ++j)
        for (int i=0; i< w; ++i, ++id){
            h_odata[i * h + j] = h_idata[id];
        }
    cudaStream_t stream = 0;

    CUT_SAFE_CALL( cutStopTimer( timer));
    printf( "CPU Processing time: %f (ms)\n", cutGetTimerValue(timer));
    
    cudaMemcpy(d_idata, h_idata, n * sizeof(uint), cudaMemcpyHostToDevice);
    int nIter = 100;

    transpose_naive(d_odata, d_idata, w, h, stream);      
    cplVectorOpers::SetMem(d_odata, (unsigned )0, n );
    CUT_SAFE_CALL( cutResetTimer( timer));
    CUT_SAFE_CALL( cutStartTimer( timer));
    for (int i=0; i < nIter; ++i)
        transpose_naive(d_odata, d_idata, w, h, stream);
    cudaThreadSynchronize();
    CUT_SAFE_CALL( cutStopTimer( timer));
    printf( "Processing time naive version: %f (ms)\n", cutGetTimerValue(timer)/nIter);
    testError(h_odata, d_odata, n, "Transpose naive");
    
    transpose(d_odata, d_idata, w, h, stream);
    cplVectorOpers::SetMem(d_odata, (unsigned )0, n );
    CUT_SAFE_CALL( cutResetTimer( timer));
    CUT_SAFE_CALL( cutStartTimer( timer));
    for (int i=0; i < nIter; ++i)
        transpose_x16y16(d_odata, d_idata, w, h, stream);
    cudaThreadSynchronize();
    CUT_SAFE_CALL( cutStopTimer( timer));
    printf( "Processing time 16x16: %f (ms)\n", cutGetTimerValue(timer)/nIter);
    testError(h_odata, d_odata, n, "transpose x16y16");

    cplVectorOpers::SetMem(d_odata, (unsigned )0, n );
    transpose_x32y8(d_odata, d_idata, w, h, stream);
    CUT_SAFE_CALL( cutResetTimer( timer));
    CUT_SAFE_CALL( cutStartTimer( timer));
    for (int i=0; i < nIter; ++i)
        transpose_x32y8(d_odata, d_idata, w, h, stream);
    cudaThreadSynchronize();
    CUT_SAFE_CALL( cutStopTimer( timer));
    printf( "Processing time 32x8: %f (ms)\n", cutGetTimerValue(timer)/nIter);
    testError(h_odata, d_odata, n, "Transpose v1");

    cplVectorOpers::SetMem(d_odata, (unsigned )0, n, stream );
    transpose(d_odata, d_idata, w, h, stream);
    CUT_SAFE_CALL( cutResetTimer( timer));
    CUT_SAFE_CALL( cutStartTimer( timer));
    for (int i=0; i < nIter; ++i)
        transpose(d_odata, d_idata, w, h, stream);
    cudaThreadSynchronize();
    CUT_SAFE_CALL( cutStopTimer( timer));
    printf( "Processing time: %f (ms)\n", cutGetTimerValue(timer)/nIter);
    testError(h_odata, d_odata, n, "Transpose");

    dmemFree(d_idata);
    dmemFree(d_odata);
    delete []h_idata;
    delete []h_odata;
    CUT_SAFE_CALL( cutDeleteTimer( timer));
}


void testTranspose_C2(int w, int h){
    fprintf(stderr, "Input size %d %d \n",w, h);

    uint2* d_idata, *d_odata;
    uint2* h_idata, *h_odata;
    
    uint n = w * h;
    
    h_idata = new uint2 [n];
    h_odata = new uint2 [n];
    
    cudaMalloc((void**)&d_idata, n * sizeof(uint2));
    cudaMalloc((void**)&d_odata, n * sizeof(uint2));
    
    int id = 0;
    for (int j=0; j < h; ++j)
        for (int i=0; i< w; ++i, ++id){
            uint2 r = make_uint2(rand(), rand());
            h_idata[id] = r;
        }
    
    uint timer;
    CUT_SAFE_CALL( cutCreateTimer( &timer));
    CUT_SAFE_CALL( cutStartTimer( timer));
    id = 0;
    for (int j=0; j < h; ++j)
        for (int i=0; i< w; ++i, ++id){
            h_odata[i * h + j] = h_idata[id];
        }
    CUT_SAFE_CALL( cutStopTimer( timer));
    printf( "CPU Processing time: %f (ms)\n", cutGetTimerValue(timer));
    
    cudaMemcpy(d_idata, h_idata, n * sizeof(uint2), cudaMemcpyHostToDevice);
    int nIter = 100;

    transpose_naive(d_odata, d_idata, w, h);      
    cplVectorOpers::SetMem(d_odata, make_uint2(0,0), n );
    CUT_SAFE_CALL( cutResetTimer( timer));
    CUT_SAFE_CALL( cutStartTimer( timer));
    for (int i=0; i < nIter; ++i)
        transpose_naive(d_odata, d_idata, w, h);
    cudaThreadSynchronize();
    CUT_SAFE_CALL( cutStopTimer( timer));
    printf( "Processing time naive version: %f (ms)\n", cutGetTimerValue(timer)/nIter);
    testError((uint*)h_odata, (uint*)d_odata, 2 * n, "Transpose naive");
    
    transpose_x16y16_C2(d_odata, d_idata, w, h, NULL);
    cplVectorOpers::SetMem(d_odata, make_uint2(0,0), n );
    CUT_SAFE_CALL( cutResetTimer( timer));
    CUT_SAFE_CALL( cutStartTimer( timer));
    for (int i=0; i < nIter; ++i)
        transpose_x16y16_C2(d_odata, d_idata, w, h, NULL);
    cudaThreadSynchronize();
    CUT_SAFE_CALL( cutStopTimer( timer));
    printf( "Processing time 16x16: %f (ms)\n", cutGetTimerValue(timer)/nIter);
    testError((uint*)h_odata, (uint*)d_odata, 2*n, "transpose x16y16");

    transpose_x16y16(d_odata, d_idata, w, h, NULL);
    cplVectorOpers::SetMem(d_odata, make_uint2(0,0), n );
    CUT_SAFE_CALL( cutResetTimer( timer));
    CUT_SAFE_CALL( cutStartTimer( timer));
    for (int i=0; i < nIter; ++i)
        transpose_x16y16(d_odata, d_idata, w, h, NULL);
    cudaThreadSynchronize();
    CUT_SAFE_CALL( cutStopTimer( timer));
    printf( "Processing time 16x16 regular: %f (ms)\n", cutGetTimerValue(timer)/nIter);
    testError((uint*)h_odata, (uint*)d_odata, 2*n, "transpose x16y16 regular");

    cudaFree(d_idata);
    cudaFree(d_odata);
    
    delete []h_idata;
    delete []h_odata;
    CUT_SAFE_CALL( cutDeleteTimer( timer));
}


/*---------------------------------------------------------------------------------------------------*/
template<typename T>
__global__ void transpose_batch_kernel(T *odata, T *idata, int width, int height, int n, int pitch)
{
    __shared__ T block[BLOCK_DIM][BLOCK_DIM+1];
  
    unsigned int xIn  = (blockIdx.x << 4) + threadIdx.x;
    unsigned int yIn  = (blockIdx.y << 4) + threadIdx.y;
  
    unsigned int xOut = (blockIdx.y << 4) + threadIdx.x;
    unsigned int yOut = (blockIdx.x << 4) + threadIdx.y;
  
    uint index_in  = yIn  * width  + xIn;
    uint index_out = yOut * height + xOut;
  
    for (unsigned int i=0; i< n;++i){
        // read the matrix tile into shared memory
        if((xIn < width) && (yIn < height)) {
            T data = idata[index_in];;
            block[threadIdx.y][threadIdx.x] = data;
        }
        __syncthreads();
        // write the transposed matrix tile to global memory
        if((xOut < height) && (yOut < width)) {
            odata[index_out] = block[threadIdx.x][threadIdx.y];
        }
        index_in  += pitch;
        index_out += pitch;
    }
}

void cudaTranspose_batch(float2* d_odata, float2* d_idata, int width, int height, int n, int pitch, cudaStream_t stream){
    dim3 threads(BLOCK_DIM,BLOCK_DIM);
    dim3 grids(iDivUp(width,BLOCK_DIM), iDivUp(height,BLOCK_DIM));
    transpose_batch_kernel<<<grids, threads, 0, stream>>>(d_odata, d_idata, width, height, n, pitch);
}

// Still have bug with --w=64 --h=128 --n=2000
__global__ void transpose_batch_C2_kernel(float2 *odata, float2 *idata, int width, int height, int n, int pitch)
{
    __shared__ float block_x[BLOCK_DIM][BLOCK_DIM+1];
    __shared__ float block_y[BLOCK_DIM][BLOCK_DIM+1];
  
    unsigned int xIn  = (blockIdx.x << 4) + threadIdx.x;
    unsigned int yIn  = (blockIdx.y << 4) + threadIdx.y;
  
    unsigned int xOut = (blockIdx.y << 4) + threadIdx.x;
    unsigned int yOut = (blockIdx.x << 4) + threadIdx.y;
  
    uint index_in  = yIn  * width  + xIn;
    uint index_out = yOut * height + xOut;
  
    for (unsigned int i=0; i< n;++i){
        // read the matrix tile into shared memory
        if((xIn < width) && (yIn < height)) {
            float2 data = idata[index_in];;
            block_x[threadIdx.y][threadIdx.x] = data.x;
            block_y[threadIdx.y][threadIdx.x] = data.y;
        }
        __syncthreads();
        // write the transposed matrix tile to global memory
        if((xOut < height) && (yOut < width)) {
            float2 data = make_float2(block_x[threadIdx.x][threadIdx.y], block_y[threadIdx.x][threadIdx.y]);		
            odata[index_out] = data;
        }
        index_in  += pitch;
        index_out += pitch;
    }
}

void cudaTranspose_batch_C2(float2* d_odata, float2* d_idata, int width, int height, int n, int pitch, cudaStream_t stream){
    dim3 threads(BLOCK_DIM,BLOCK_DIM);
    dim3 grids(iDivUp(width,BLOCK_DIM), iDivUp(height,BLOCK_DIM));
    transpose_batch_C2_kernel<<<grids, threads, 0, stream>>>(d_odata, d_idata, width, height, n, pitch);
}

__global__ void transpose_batch_simple_C2_kernel(float2 *odata, float2 *idata, int width, int height, int n, int pitch)
{
    // read the matrix tile into shared memory
    unsigned int xIndex = blockIdx.x * BLOCK_DIM + threadIdx.x;
    unsigned int yIndex = blockIdx.y * BLOCK_DIM + threadIdx.y;
  
    unsigned int bl = yIndex / height;
    yIndex  = yIndex - bl * height;
  
    if ((xIndex < width) && (bl < n)){
        unsigned int offset = pitch * bl;
        float2 data = idata[offset + xIndex +  yIndex * width];
        odata[offset + yIndex + xIndex * height] = data;
    }
}

void cudaTranspose_batch_simple_C2(float2* d_odata, float2* d_idata, int width, int height, int n, int pitch, cudaStream_t stream){
    dim3 threads(BLOCK_DIM,BLOCK_DIM);
    dim3 grids(iDivUp(width,BLOCK_DIM), iDivUp(height * n, BLOCK_DIM));
    transpose_batch_simple_C2_kernel<<<grids, threads, 0, stream>>>(d_odata, d_idata, width, height, n, pitch);
}

void cpuTranspose_batch(float2* h_odata, float2* h_idata, int w, int h, int n, int pitch){
  
    for (int i=0; i< n * pitch; ++i)
        h_odata[i]=make_float2(0,0);

    for (int k=0; k< n; ++k){
        uint offset = k * pitch;
        for (int j=0; j< h; ++j)
            for (int i=0; i< w; ++i)
                h_odata[offset + i * h + j] = h_idata[offset + j * w + i];
    }
}

void testTranspose_batch(int width, int height, int n) 
{
    long int s, e;
    fprintf(stderr, "Size of the problem w=%d h=%d n=%d \n", width, height, n);
    
    int planeSize  = width * height;
    int planeSizeA = iAlignUp(planeSize, 64);
    int nElemsA    = planeSizeA * n;
    
    int size = sizeof(float2) * nElemsA;
    
    float2* h_signal = new float2 [nElemsA] ;
    float2* h_odata  = new float2 [nElemsA] ;
    
    // Initalize the memory for the signal
    for (int b=0; b< n; ++b){
        for (int i = 0; i < planeSize; ++i) {
            h_signal[i + b * planeSizeA].x = (float) rand() /RAND_MAX;
            h_signal[i + b * planeSizeA].y = 0;
        }
    }
    
    QueryHPCTimer(&s);
    cpuTranspose_batch(h_odata, h_signal, width , height, n, planeSizeA);
    QueryHPCTimer(&e);
    fprintf(stderr, "Running time for CPU transpose %f us\n", (float)(e -  s));

        
    float2* d_data;
    float2* d_odata;

    cudaMalloc((void**)&d_data, size);
    cudaMalloc((void**)&d_odata, size);
    
    cudaMemcpy(d_data, h_signal, size, cudaMemcpyHostToDevice);
    cplVectorOpers::SetMem(d_odata, make_float2(0,0), nElemsA);

    // run the first version 
    QueryHPCTimer(&s);
    for (int i=0; i< n; ++i){
        int offset = planeSizeA * i;
        transpose(d_odata + offset, d_data + offset, width, height);
    }
    cudaThreadSynchronize();
    QueryHPCTimer(&e);
    fprintf(stderr, "Running time for the naive transpose %f us\n", (float)(e -  s));
    testError((float*)h_odata, (float*)d_odata, 1e-5, nElemsA * 2, "Transpose sperate");

    // run the third version 
    // limited with n * height / 16 < 64k
    cplVectorOpers::SetMem(d_odata, make_float2(0,0), nElemsA);
    QueryHPCTimer(&s);
    cudaTranspose_batch_simple_C2(d_odata,d_data, width , height, n, planeSizeA, NULL);
    cudaThreadSynchronize();
    QueryHPCTimer(&e);
    fprintf(stderr, "Running time for neive batch %f us\n ", (float)(e -  s));
    testError((float*)h_odata, (float*)d_odata, 1e-5, nElemsA * 2, "Naive batch transpose");
    
    // run the second version
    cudaStream_t stream = 0;
    cplVectorOpers::SetMem(d_odata, make_float2(0,0), nElemsA);
    QueryHPCTimer(&s);
    cudaTranspose_batch_C2(d_odata, d_data, width , height, n, planeSizeA, stream);
    cudaThreadSynchronize();
    QueryHPCTimer(&e);
    fprintf(stderr, "Running time for the transpose with loop %f us\n", (float)(e -  s));
    testError((float*)h_odata, (float*)d_odata, 1e-5, nElemsA * 2, "Transpose");

    delete []h_signal;
    delete []h_odata;

}


template<typename T, bool rightDir>
__global__ void cplShiftCoordinateGlobal_kernel(T* d_o, T* d_i, int sizeX, int sizeY, int sizeZ)
{
    uint xid = blockIdx.x * blockDim.x + threadIdx.x;
    uint yid = blockIdx.y * blockDim.y + threadIdx.y;

    if (xid >= sizeX || yid >= sizeY)
        return;

    for (uint zid=0; zid < sizeZ; ++zid){
        uint iid = xid + (yid + zid * sizeY) * sizeX;
        uint oid;
        if (rightDir)
            oid = zid + (xid + yid * sizeX) * sizeZ;
        else
            oid = yid + (zid + xid * sizeZ) * sizeY;
        
        d_o[oid] = d_i[iid];
    }
}

template<typename T>
void cplShiftCoordinatGlobal(T* d_o, T* d_i, int sizeX, int sizeY, int sizeZ, bool dir, cudaStream_t stream)
{
    dim3 threads(16, 16);
    dim3 grids(iDivUp(sizeX, threads.x),iDivUp(sizeY, threads.y));

    if (dir == true)
        cplShiftCoordinateGlobal_kernel<T, true><<<grids, threads, 0, stream>>>(d_o, d_i, sizeX, sizeY, sizeZ);
    else
        cplShiftCoordinateGlobal_kernel<T, false><<<grids, threads, 0, stream>>>(d_o, d_i,sizeX, sizeY, sizeZ);
}


// This one still failed with
// dir = 0 odd size
// strange have no idea what happen, should use the shared mem operation 
template<typename T, bool rightDir>
__global__ void cplShiftCoordinate_tex_kernel(T* d_o, int sizeX, int sizeY, int sizeZ)
{
    uint zid = blockIdx.x * blockDim.x + threadIdx.x;
    uint xid = blockIdx.y * blockDim.y + threadIdx.y;

    if (zid >= sizeZ || xid >= sizeX)
        return;

    uint oid = zid + xid * sizeZ;

    for (uint yid=0; yid < sizeY; ++yid, oid += sizeX * sizeZ){
        uint iid;
        if (!rightDir)
            iid = yid + (zid + xid * sizeZ) * sizeY;
        else 
            iid = xid + (yid + zid * sizeY) * sizeX;
        
//      d_o[oid] = tex1Dfetch(comtex_float, iid);
        d_o[oid] = fetch(iid, (T*)NULL);
    }
}

template<typename T>
void cplShiftCoordinate_tex(T* d_o, T* d_i, int sizeX, int sizeY, int sizeZ, bool dir, cudaStream_t stream)
{
    dim3 threads(16, 16);
    dim3 grids(iDivUp(sizeZ, threads.x),iDivUp(sizeX, threads.y));

    cache_bind(d_i);
    //cudaBindTexture(0, comtex_float, d_i, sizeX * sizeY * sizeZ * sizeof(float));
    
    if (dir == true)
        cplShiftCoordinate_tex_kernel<T, true><<<grids, threads, 0, stream>>>(d_o, sizeX, sizeY, sizeZ);
    else
        cplShiftCoordinate_tex_kernel<T, false><<<grids, threads, 0, stream>>>(d_o,sizeX, sizeY, sizeZ);
}

template<typename T>
void cplShiftCoordinate_tex(T* d_o, T* d_i, const Vector3Di& size, bool dir, cudaStream_t stream){
    cplShiftCoordinate_tex(d_o, d_i, size.x, size.y, size.z, dir, stream);
}

//TODO
// check in the case the size is not a multiply of 8 or write a saparate function

#define TILE_DIM 8
template<typename T>
__global__ void cplShiftRightCoordinate_shared_kernel(T* d_o, T* d_i, uint sizeX, uint sizeY, uint sizeZ)
{
    __shared__ T sdata[TILE_DIM][TILE_DIM+1][TILE_DIM+1];

    const uint iPlaneSize = sizeX * sizeY;
    const uint oPlaneSize = sizeX * sizeZ;

    uint bx = blockIdx.x * TILE_DIM;
    uint by = blockIdx.y * TILE_DIM;
    
    uint tx = threadIdx.x;
    uint ty = threadIdx.y;
    
    uint bz = 0;
    uint iid = bx + tx + (by + ty) * sizeX;
    while (bz < sizeZ){
        for (uint tz = 0; tz < TILE_DIM; ++tz, iid +=iPlaneSize)
            sdata[tz][ty][tx] = ((bx + tx < sizeX) && (by + ty < sizeY) && (bz + tz < sizeZ)) ? d_i[iid] : 0;

        __syncthreads();
        uint oid = bz + bx * sizeZ + by * sizeX * sizeZ + (tx + ty * sizeZ);
        if ((bz + tx < sizeZ) && (bx + ty < sizeX))
            for (uint tz = 0; tz < TILE_DIM && (by + tz < sizeY); ++tz, oid += oPlaneSize)
                d_o[oid] = sdata[tx][tz][ty];
        bz += TILE_DIM;
    }
    
}

template<typename T>
__global__ void cplShiftLeftCoordinate_shared_kernel(T* d_o, T* d_i, uint sizeX, uint sizeY, uint sizeZ)
{
    __shared__ T sdata[TILE_DIM][TILE_DIM+1][TILE_DIM+1];
    
    const uint iPlaneSize = sizeX * sizeY;
    const uint oPlaneSize = sizeY * sizeZ;

    uint bx = blockIdx.x * TILE_DIM;
    uint by = blockIdx.y * TILE_DIM;
    
    uint tx = threadIdx.x;
    uint ty = threadIdx.y;
    
    uint bz = 0;
    uint iid = bx + tx + (by + ty) * sizeX;
    
    while (bz < sizeZ){
        for (uint tz = 0; tz < TILE_DIM; ++tz, iid +=iPlaneSize)
            sdata[tz][ty][tx] = ((bx + tx < sizeX) && (by + ty < sizeY) && (bz + tz < sizeZ)) ? d_i[iid] : 0;

        __syncthreads();

        uint oid = by + bz * sizeY + bx * sizeY * sizeZ + (tx + ty * sizeY);

        if ((by + tx < sizeY) && (bz + ty < sizeZ))
            for (uint tz = 0; tz < TILE_DIM && (bx + tz < sizeX); ++tz, oid += oPlaneSize)
                d_o[oid] = sdata[ty][tx][tz];
        bz += TILE_DIM;
    }
    
}

template<typename T>
void cplShiftCoordinate_shared(T* d_o, T* d_i, int sizeX, int sizeY, int sizeZ, bool dir, cudaStream_t stream){
    dim3 threads(8, 8);
    dim3 grids(iDivUp(sizeX, threads.x),iDivUp(sizeY, threads.y));
    if (dir)
        cplShiftRightCoordinate_shared_kernel<<<grids, threads, 0, stream>>>(d_o, d_i, sizeX, sizeY, sizeZ);
    else
        cplShiftLeftCoordinate_shared_kernel<<<grids, threads, 0, stream>>>(d_o, d_i, sizeX, sizeY, sizeZ);
}

template<typename T>
inline void cplShiftCoordinate(T* d_o, T* d_i, int sizeX, int sizeY, int sizeZ, bool dir, cudaStream_t stream){
    cplShiftCoordinate_shared(d_o, d_i, sizeX, sizeY, sizeZ, dir, stream);
}

template<typename T>
inline void cplShiftCoordinate(T* d_o, T* d_i, const Vector3Di& size, bool dir, cudaStream_t stream){
    cplShiftCoordinate(d_o, d_i, size.x, size.y, size.z, dir, stream);
}


template void cplShiftCoordinate(float* d_o, float* d_i, int sizeX, int sizeY, int sizeZ, bool dir, cudaStream_t stream);
template void cplShiftCoordinate(uint* d_o, uint* d_i, int sizeX, int sizeY, int sizeZ, bool dir, cudaStream_t stream);
template void cplShiftCoordinate(int* d_o, int* d_i, int sizeX, int sizeY, int sizeZ, bool dir, cudaStream_t stream);

void  cpuShift3DMem(float* h_o, float* h_i,
                    int sizeX, int sizeY, int sizeZ, bool dir)
{
    int id = 0;
    if (dir)
        for (int z=0; z< sizeZ; ++z)
            for (int y=0; y< sizeY; ++y)
                for (int x=0; x< sizeX; ++x, ++id)
                    h_o[ z + x * sizeZ + y * sizeX * sizeZ] = h_i[id];
    else
        for (int z=0; z< sizeZ; ++z)
            for (int y=0; y< sizeY; ++y)
                for (int x=0; x< sizeX; ++x, ++id)
                    h_o[ y + z * sizeY + x * sizeY * sizeZ] = h_i[id];

}


/*----------------------------------------------------------------------------------------------------*/
// the funtion perform the 3D circulation shift between the dimention that
// move the dimention in the direction
// z y x > x z y > y x z > z y x

/*----------------------------------------------------------------------------------------------------*/
template<typename T>
void cpuSegShiftMem(T* d_odata, T* d_idata, int w, int h, int l,
                    int nSeg, int size){
    for (int seg = 0; seg < nSeg; ++seg){
        int offset = seg * size;
        for (int k=0; k < l; ++k)
            for (int j=0; j < h; ++j)
                for (int i=0; i < w; ++i){
                    T data = d_idata[i + (j + k * h) * w + offset];
                    d_odata[j + (k + i * l) * h + offset] = data;
                }
    }
}
                                                      
template<typename T>
__global__ void cudaSegShiftMem_kernel(T* d_odata, T* d_idata, int w, int h, int l,
                                       int nSeg, int size){

    uint tid     = threadIdx.x;
    uint blockId = blockIdx.x + blockIdx.y * gridDim.x;
    
    const uint n = w * h * l;

    if (blockId < nSeg){
        uint offset = blockId * size;
        uint id = tid;
        while (id < n){
            // read from the position
            T data = d_idata[id + offset];
            uint k = id / (h * w);
            uint j = (id - k * h * w) / w;
            uint i = (id - k * h * w) - j * w;
            uint oid = j + (k + i * l) * h;
            
            d_odata[oid + offset] = data;
            id += blockDim.x;
        }
    }
}

template<typename T>
void cudaSegShiftMem(T* d_odata, T* d_idata, int w, int h, int l, int nSeg, int size, cudaStream_t stream)
{
    dim3 threads(64);
    dim3 grids(nSeg);
    checkConfig(grids);
    
    cudaSegShiftMem_kernel<<<grids, threads, 0, stream>>>(d_odata, d_idata, w, h, l, nSeg, size);
}

/*----------------------------------------------------------------------------------------------------*/
template<typename T>
__global__ void cudaSegShiftMem_kernel_tex(T* d_odata, T* d_idata, uint w, uint h, uint l,
                                           uint nSeg, uint size)
{

    uint tid     = threadIdx.x;
    uint blockId = blockIdx.x + blockIdx.y * gridDim.x;
    
    const uint n = w * h * l;
    
    if (blockId < nSeg){
        uint offset = blockId * size;
        uint odd = offset & 63;
        uint sid = (tid + 64 - odd) & 63;
        uint id = sid;
                
        uint i = 0, j = id, k =0;
        
        while (id < n){
            // read from the position
            i = id / (l * h);
            k = (id - i * l * h) / h;
            j = id - i * l * h - k * h;

            uint iid = i + (j + k * h) * w;
            //d_odata[id + offset] = tex1Dfetch(comtex_float2,offset + iid);
            d_odata[id + offset] = fetch(offset + iid, (T*)NULL);
            id += blockDim.x;
        }
    }
    
}

template<typename T>
void cudaSegShiftMem_tex(T* d_odata, T* d_idata, int w, int h, int l, int nSeg, int size, cudaStream_t stream)
{
    dim3 threads(64);
    dim3 grids(nSeg);
    checkConfig(grids);
    cache_bind(d_idata);
    //cudaBindTexture(0, comtex_float2, d_idata, size * nSeg * sizeof(float2));
    cudaSegShiftMem_kernel_tex<<<grids, threads, 0, stream>>>(d_odata, d_idata, w, h, l, nSeg, size);
}

/*----------------------------------------------------------------------------------------------------*/
template<int buffer_size>
__global__ void cudaSegShiftMem_kernel_shared(float* d_odata, float* d_idata, int w, int h, int l,
                                              int nSeg, int size){

    __shared__ float sdata[buffer_size];
    
    uint tid     = threadIdx.x;
    uint blockId = blockIdx.x + blockIdx.y * gridDim.x;
    
    const uint n = w * h * l;

    if (blockId < nSeg){
        
        uint offset = blockId * size;
        // coalesced read to the share memory
        uint odd = offset & 63;
        uint sid = (tid + 64 - odd) & 63;
        
        uint rid = sid;
        while (rid < n){
            float data = d_idata[rid + offset];;

            uint k = rid / (h * w);
            uint j = (rid - k * h * w) / w;
            uint i = rid % w;
            uint oid = j + (k + i * l) * h;

            sdata[oid] = data;
            rid       += 64;
        }
        
        __syncthreads();

        // coalseced write
        uint oid = sid;
        while (oid < n){
            // read from the position
            d_odata[oid + offset] = sdata[oid];
            oid += 64;
        }
    }
}

void cudaSegShiftMem_shared(float* d_odata, float* d_idata, int w, int h, int l, int nSeg, int size, cudaStream_t stream)
{
    dim3 threads(64);
    dim3 grids(nSeg);
    
    checkConfig(grids);
    if (w * h * l <= 2000)
        cudaSegShiftMem_kernel_shared<2000><<<grids, threads, 0, stream>>>(d_odata, d_idata, w, h, l, nSeg, size);
    else
        fprintf(stderr, " Does not apply");
}

void testSegShiftMem(int width, int height, int length, int nSeg)
{
    long int s, e;
    fprintf(stderr, "Size of the problem w=%d h=%d l=%d n=%d \n", width, height, length, nSeg);
    int size  = width * height * length;
    int sizeA = size;
    //int sizeA = iAlignUp128(size);

    float* h_signal = new float [nSeg * sizeA] ;
    float* h_odata  = new float [nSeg * sizeA] ;
    
    // Initalize the memory for the signal
    for (int b=0; b< nSeg; ++b){
        for (int i = 0; i < size; ++i) {
            h_signal[i + b * sizeA] = (float) rand() /RAND_MAX;
        }
    }

    QueryHPCTimer(&s);
    cpuSegShiftMem(h_odata, h_signal, width, height, length, nSeg, sizeA);
    QueryHPCTimer(&e);
    fprintf(stderr, "Running time for the CPU version %f us\n", (float)(e -  s));
    
    float* d_data;
    float* d_odata;

    cudaMalloc((void**)&d_data, nSeg * sizeA * sizeof(float));
    cudaMalloc((void**)&d_odata,nSeg * sizeA * sizeof(float)) ;
    
    cudaMemcpy(d_data, h_signal, nSeg * sizeA * sizeof(float), cudaMemcpyHostToDevice);
    cplVectorOpers::SetMem(d_odata, 0.f, nSeg * sizeA);

    int nIter = 100;
    // run the first version
    cplVectorOpers::SetMem(d_odata, 0.f, nSeg * sizeA);
    QueryHPCTimer(&s);
    for (int i=0; i< nIter; ++i)
        cudaSegShiftMem(d_odata, d_data, width, height, length, nSeg, sizeA);
    cudaThreadSynchronize();
    QueryHPCTimer(&e);
    fprintf(stderr, "Running time for the naive transpose %f us\n", (float)(e -  s) / nIter);
    testError((float*)h_odata, (float*)d_odata, 1e-5, sizeA * nSeg, "Transpose first");

    /*
    // run the second version
    cplVectorOpers::SetMem(d_odata, 0.f, nSeg * sizeA);
    QueryHPCTimer(&s);
    for (int i=0; i< nIter; ++i)
    cudaSegShiftMem_shared(d_odata, d_data, width, height, length, nSeg, sizeA);
    cudaThreadSynchronize();
    QueryHPCTimer(&e);
    fprintf(stderr, "Running time for coalesced read %f us\n", (float)(e -  s) / nIter);
    testError((float*)h_odata, (float*)d_odata, 1e-5, sizeA * nSeg);
    */
    
    // run the third version
    cplVectorOpers::SetMem(d_odata, 0.f, nSeg * sizeA);
    QueryHPCTimer(&s);
    for (int i=0; i< nIter; ++i)
        cudaSegShiftMem_tex(d_odata, d_data, width, height, length, nSeg, sizeA);
    cudaThreadSynchronize();
    QueryHPCTimer(&e);
    fprintf(stderr, "Running time for the texture %f us\n", (float)(e -  s) / nIter);
    testError((float*)h_odata, (float*)d_odata, 1e-5, sizeA * nSeg, "Transpose third");

    cudaFree(d_odata);
    cudaFree(d_data);
    delete []h_signal;
    delete []h_odata;

}

void testSegShiftMemComplex(int width, int height, int length, int nSeg)
{
    long int s, e;
    fprintf(stderr, "Size of the problem w=%d h=%d l=%d n=%d \n", width, height, length, nSeg);
    uint size  = width * height * length;
    uint sizeA = size;
    //int sizeA = iAlignUp128(size);

    float2* h_signal = new float2 [nSeg * sizeA] ;
    float2* h_odata  = new float2 [nSeg * sizeA] ;
    
    // Initalize the memory for the signal
    int id=0;
    for (unsigned int b=0; b< nSeg; ++b){
        for (unsigned int i = 0; i < size; ++i, ++id) {
            h_signal[i + b * sizeA].x = id;
            h_signal[i + b * sizeA].y = 0;
        }
    }

    QueryHPCTimer(&s);
    cpuSegShiftMem(h_odata, h_signal, width, height, length, nSeg, sizeA);
    QueryHPCTimer(&e);
    fprintf(stderr, "Running time for the CPU version %f us\n", (float)(e -  s));
    
    float2* d_data;
    float2* d_odata;

    cudaMalloc((void**)&d_data, nSeg * sizeA * sizeof(float2));
    cudaMalloc((void**)&d_odata,nSeg * sizeA * sizeof(float2)) ;
    
    cudaMemcpy(d_data, h_signal, nSeg * sizeA * sizeof(float2), cudaMemcpyHostToDevice);
    int nIter = 100;

// run the first version
    cplVectorOpers::SetMem(d_odata, make_float2(0.f, 0.f), nSeg * sizeA);
    QueryHPCTimer(&s);
    for (int i=0; i< nIter; ++i)
        cudaSegShiftMem(d_odata, d_data, width, height, length, nSeg, sizeA);
    cudaThreadSynchronize();
    QueryHPCTimer(&e);
    fprintf(stderr, "Running time for the naive transpose %f us\n", (float)(e -  s) / nIter);
    testError((float*)h_odata, (float*)d_odata, 1e-5, sizeA * nSeg * 2, "Naive trasnpose");
    
    // run the third version
    cplVectorOpers::SetMem(d_odata, make_float2(0.f, 0.f), nSeg * sizeA);
    QueryHPCTimer(&s);
    for (int i=0; i< nIter; ++i)
        cudaSegShiftMem_tex(d_odata, d_data, width, height, length, nSeg, sizeA);
    cudaThreadSynchronize();
    QueryHPCTimer(&e);
    fprintf(stderr, "Running time for the texture %f us\n", (float)(e -  s) / nIter);
    testError((float*)h_odata, (float*)d_odata, 1e-5, sizeA * nSeg * 2, "Texture cache transpose");

    cudaFree(d_odata);
    cudaFree(d_data);
    delete []h_signal;
    delete []h_odata;

}


/*

  void testBatch3DFFT(int argc, char** argv) 
  {
  int width  = 2;
  int height = 2;
  int length = 2;

  int nSeg   = 2;
  long int s, e;

  cutGetCmdLineArgumenti( argc, (const char**) argv, "s", &nSeg);
    
  cutGetCmdLineArgumenti( argc, (const char**) argv, "w", &width);
  cutGetCmdLineArgumenti( argc, (const char**) argv, "h", &height);
  cutGetCmdLineArgumenti( argc, (const char**) argv, "l", &length);

  fprintf(stderr, "Size of the problem w=%d h=%d l=%d n=%d \n", width, height, length, nSeg);

  int size  = width * height * length;
  int sizeA = size;
    
  float2* h_signal = new float2 [nSeg * sizeA] ;
  float2* h_odata  = new float2 [nSeg * sizeA] ;

  fftwf_plan p;
    
  // Initalize the memory for the signal
  for (unsigned int b=0; b< nSeg; ++b){
  for (unsigned int i = 0; i < size; ++i) {
  h_signal[i + b * sizeA].x = i;
  h_signal[i + b * sizeA].y = 0;
  }
  }

  p = fftwf_plan_dft_3d(length, height, width, (fftwf_complex*)h_odata, (fftwf_complex*)h_signal, FFTW_FORWARD, 0 );

  float* h_rSignal = new float [sizeA * nSeg];
  for (int i=0;i < sizeA * nSeg; ++i){
  h_rSignal[i] = h_signal[i].x;
  }
            
  float2* d_data;
  float2* d_odata;
    
  cudaMalloc((void**)&d_data, sizeA * nSeg * sizeof(float2));
  cudaMalloc((void**)&d_odata, sizeA * nSeg * sizeof(float2));
    
  cudaMemcpy(d_data, h_signal,sizeA * nSeg * sizeof(float2), cudaMemcpyHostToDevice);
  //cudaMemcpy(d_data, h_rSignal, sizeA * nSeg * sizeof(float), cudaMemcpyHostToDevice);
  //printDeviceArray1Df((float*)d_data, sizeA, "GPU input ");
    
    
  QueryHPCTimer(&s);
  //cudaFFT3D_batch(d_odata, d_data, width, height, length, nSeg);
  cudaFFT3D_batch_in(d_data, width, height, length, nSeg, d_odata);

  //cudaFFT3D_batch_in_real(d_data, width, height, length, nSeg, d_odata);
    
  cudaThreadSynchronize();
  QueryHPCTimer(&e);
  fprintf(stderr, "Running time for the batch FFT3D %f us\n", (float)(e -  s));
  QueryHPCTimer(&s);
  for (unsigned int seg=0; seg < nSeg; ++seg){
  p = fftwf_plan_dft_3d(length, height, width,
  (fftwf_complex*)h_signal + sizeA * seg,
  (fftwf_complex*)h_odata  + sizeA * seg, FFTW_FORWARD, 0 );
  fftwf_execute(p);
  }

  QueryHPCTimer(&e);
  fprintf(stderr, "Running time for cpu FFT3D %f us\n", (float)(e -  s));

  testError((float*)h_odata, (float*) d_data, 1e-4, sizeA * 2);

    
  }
    
*/

