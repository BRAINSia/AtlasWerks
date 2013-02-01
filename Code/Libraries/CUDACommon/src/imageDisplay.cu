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

#ifndef __IMAGEDISPLAY_CU
#define __IMAGEDISPLAY_CU

#include <cpl.h>

// Horizontal concate the images in the form
// I = I0 I1 I2 I3
__global__ void horizontalConcateImages_kernel(float* result,
                                               float* img0, float* img1,
                                               int w, int h)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;
    int tid = idy * w * 2 + idx;
    int id  = idy * w + idx;

    if (idx < w && idx < h) {
        result[tid    ] = img0[id];
        result[tid + w] = img1[id];
    }
}

void horizontalConcateImages(float* result,
                             float* img0, float* img1, 
                             int w, int h){
    dim3 threads(16, 16);
    dim3 grids(iDivUp(w, threads.x), iDivUp(h, threads.y));
    horizontalConcateImages_kernel<<<grids, threads>>>(result, img0, img1, w, h);
}


__global__ void horizontalConcateImages_kernel(float* result,
                                               float* img0, float* img1, float* img2,
                                               int w, int h)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;
    int tid = idy * w * 3 + idx;
    int id  = idy * w + idx;

    if (idx < w && idx < h) {
        result[tid        ] = img0[id];
        result[tid + w    ] = img1[id];
        result[tid + 2 * w] = img2[id];
    }
}

void horizontalConcateImages(float* result,
                             float* img0, float* img1, float* img2, 
                             int w, int h){
    dim3 threads(16, 16);
    dim3 grids(iDivUp(w, threads.x), iDivUp(h, threads.y));
    horizontalConcateImages_kernel<<<grids, threads>>>(result, img0, img1, img2, w, h);
}


__global__ void horizontalConcateImages_kernel(float* result,
                                               float* img0, float* img1, float* img2, float* img3,
                                               int w, int h)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;
    int tid = idy * w * 4 + idx;
    int id  = idy * w + idx;

    if (idx < w && idx < h) {
        result[tid        ] = img0[id];
        result[tid + w    ] = img1[id];
        result[tid + 2 * w] = img2[id];
        result[tid + 3 * w] = img3[id];
    }
}

void horizontalConcateImages(float* result,
                             float* img0, float* img1, float* img2, float* img3,
                             int w, int h){
    dim3 threads(16, 16);
    dim3 grids(iDivUp(w, threads.x), iDivUp(h, threads.y));
    horizontalConcateImages_kernel<<<grids, threads>>>(result, img0, img1, img2, img3, w, h);
}

void cpliImageCompose4x4(float* result,
                         float* imgArray[4][4],
                         int w, int h){
    int planeSize = w * h;
    for (int i=0; i<4; ++i){
        float* start = result + i * planeSize * 4;
        horizontalConcateImages(start, imgArray[i][0],imgArray[i][1], imgArray[i][2], imgArray[i][3], w, h);
    }
}
                      
// Square concate the images in the form
// I = I0 I1
//     I2 I3
__global__ void rectange2x2ConcateImages_kernel(float* result,
                                                float * img0, float* img1, float* img2, float* img3,
                                                int w, int h)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;

    int tid = idy * w * 2 + idx;
    int id  = idy * w + idx;

    if (idx < w && idx < h) {
        result[tid] = img0[id];
        result[tid + w] = img1[id];
        result[tid + 2 * w * h] = img2[id];
        result[tid + 2 * w * h + w] = img3[id];
    }
}

void rectange2x2ConcateImages(float* result, float * img0, float* img1, float* img2, float* img3, int w, int h){
    dim3 threads(16, 16);
    dim3 grids(iDivUp(w, threads.x), iDivUp(h, threads.y));
    rectange2x2ConcateImages_kernel<<<grids, threads>>>(result, img0, img1, img2, img3, w, h);
}


__global__ void reshapeImages_kernel(float* result,
                                     float* img, int w, int h,
                                     int nImgs, int nCols)
{
    uint idx = threadIdx.x + blockIdx.x * blockDim.x;
    uint idy = threadIdx.y + blockIdx.y * blockDim.y;

    uint imgSize = w * h;
    uint rid = idy * w + idx;
    
    for (int i=0; i< nImgs; ++i)
    {
        uint col = i % nCols;
        uint row = i / nCols;
        
        uint oid = ((col * w) + idx) + ((row * h) + idy) * (nCols * w);
        
        float idata = img[rid + i * imgSize];
        result[oid] = idata;
    }
}

void reshapeImages(float* result,
                   float* img, int w, int h,
                   int nImgs, int nCols)
{
    dim3 threads(16, 16);
    dim3 grids(iDivUp(w, threads.x), iDivUp(h, threads.y));
    reshapeImages_kernel<<<grids, threads>>>(result, img,  w, h, nImgs, nCols);
}

void concateVolumeSlices(float* d_oImg, float** d_iImgs, int nImgs, int slice,
                         unsigned int sizeX, unsigned int sizeY, unsigned int sizeZ)
{
    for (int i=0; i< nImgs; ++i)
        copyArrayDeviceToDevice(d_oImg + i * sizeX * sizeY,
                                d_iImgs[i] + slice * sizeX * sizeY, sizeX * sizeY);
}

#endif
