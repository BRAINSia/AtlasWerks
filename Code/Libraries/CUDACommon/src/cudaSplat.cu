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
#include <cudaSort.h>
#include <cudaMap.h>
#include <cudaScan.h>

#include <cudaSegmentedScanUtils.h>
#include <cpuImage3D.h>
#include <cudaSortUtils.h>
#include <cudaImage3D.h>
/*--------------------------------------------------------------------------------------------
  Splating 
  Inputs : d_src                 : source image
           d_vx, dvy, dvz        : velocity fields
           w , h, l              : size of the volume
  Output :
           d_dst                 : output intergrate from the input
 --------------------------------------------------------------------------------------------*/
__global__ void createIndex_kernel(uint4 * g_cell, float4* g_dmass,
                                   float* g_data,
                                   float* vx, float* vy, float* vz,
                                   int w, int h, int l)
{
    const uint nElems = w * h * l;
    const uint wh     = w * h;

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i < w && j < h){
        uint id       = i + j * w;
        for (int k=0; k < l; ++k, id+=wh) {
            float mass = g_data[id];
            float x = i + vx[id];
            float y = j + vy[id];
            float z = k + vz[id];
            
            int xInt = int(x);
            int yInt = int(y);
            int zInt = int(z);
            
            float dx = 1.f - (x - xInt);
            float dy = 1.f - (y - yInt);
            float dz = 1.f - (z - zInt);

            int new_id = xInt + yInt * w + zInt * w * h;

            uint4  cell = make_uint4(nElems, nElems, nElems, nElems);
            float4 dist = make_float4(0.f, 0.f, 0.f, 0.f);

            if (isInside3D(xInt, yInt, zInt, w, h, l)){
                cell.x = new_id;
                dist.x = mass * dx * dy * dz;
            }
            
            if (isInside3D(xInt + 1, yInt, zInt, w, h, l)){
                cell.y = new_id + 1;
                dist.y = mass * (1.f-dx) * dy * dz;
            }


            if (isInside3D(xInt, yInt+1, zInt, w, h, l)){
                cell.z = new_id + w;
                dist.z = mass * dx * (1.f - dy) * dz;
            }


            if (isInside3D(xInt+1, yInt+1, zInt, w, h, l)){
                cell.w = new_id + w + 1;
                dist.w = mass * (1.f -dx) * (1.f - dy) * dz;
            } 
            
            g_cell[id] = cell;
            g_dmass[id]= dist;

            new_id += w*h;

            cell = make_uint4(nElems, nElems, nElems, nElems);
            dist = make_float4(0.f, 0.f, 0.f, 0.f);

            if (isInside3D(xInt, yInt, zInt+1, w, h, l)){
                cell.x = new_id;
                dist.x = mass * dx * dy * (1.f-dz);
            }
                
            if (isInside3D(xInt + 1, yInt, zInt+1, w, h, l)){
                cell.y = new_id+1;
                dist.y= mass * (1.f-dx) * dy * (1.f-dz);
            }

            if (isInside3D(xInt, yInt+1, zInt+1, w, h, l)){
                cell.z = new_id+w;
                dist.z = mass * dx * (1.f - dy) * (1.f-dz);
            }

            if (isInside3D(xInt+1, yInt+1, zInt+1, w, h, l)){
                cell.w = new_id+w+1;
                dist.w = mass * (1.f -dx) * (1.f - dy) * (1.f-dz);
            }

            g_cell[id + nElems] = cell;
            g_dmass[id + nElems]= dist;
        }
    }
}

void cplSplatingOld(float* d_dst, float* d_src,
                  float* d_vx , float* d_vy, float* d_vz,
                  uint w, uint h, uint l){

    uint size   = w * h * l;
    //uint memSize= size * sizeof(float);
    uint n      = size * 8;       // each volume point affect 8 neighbors
    uint nFlags = iDivUp(n, 32);    // number of 32 bit flags needed 

    uint4 * d_cell;
    float4* d_dist;
    uint  * d_lastPos;
    uint  * d_flags;
    float * d_temp_f;
    
    dmemAlloc(d_cell  , size * 2);
    dmemAlloc(d_dist  , size * 2);
    dmemAlloc(d_temp_f, n );
    dmemAlloc(d_flags , nFlags);
    dmemAlloc(d_lastPos, (size + 16));

    // Initialize destination zero
    cplVectorOpers::SetMem(d_dst, 0.f, size);
    cplVectorOpers::SetMem((uint*)d_cell, size, n);
    
    dim3 threads(16,16);
    dim3 grids(iDivUp(w, threads.x), iDivUp(h, threads.y));
    // Compute the index and the mass contribution of each mapping point
    createIndex_kernel<<<grids, threads>>>(d_cell, d_dist, d_src,
                                           d_vx, d_vy, d_vz,
                                           w, h, l);

    // Sort the index along with the distribution
    cplReduce  rdPlan;
    cplSort sorter(n, &rdPlan);

    sorter.sort((uint*)d_cell, (uint*) d_dist, n);

    //3. Build the segmented flags
    buildLastPosFlags(d_flags, d_cell, n);
    cplVectorOpers::SetMem(d_lastPos, n, size);
    findLastPos(d_lastPos, d_cell, n);
    
    // Intergrate the result to the new grids
    segScan<float, 1>((float*) d_temp_f, (float*)d_dist, d_flags, n);

    //5 Extract the result
    extract_tex(d_dst, d_temp_f, d_lastPos, size, n);

    dmemFree(d_lastPos);
    dmemFree(d_cell);
    dmemFree(d_dist);
    dmemFree(d_flags);
    dmemFree(d_temp_f);
}

void cplSplating(float* d_dst, float* d_src,
                  float* d_vx , float* d_vy, float* d_vz,
                  uint w, uint h, uint l){
    uint size   = w * h * l;
    uint n      = size * 8;       // each volume point affect 8 neighbors

    uint4 * d_cell;
    float4* d_dist;
    uint  * d_lastPos;
    uint  * d_flags;

    dmemAlloc(d_cell  , size * 2);
    dmemAlloc(d_dist  , size * 2);
    dmemAlloc(d_flags , iDivUp(n, 32));
    dmemAlloc(d_lastPos, (size + 16));

    float * d_temp_f = (float*) d_cell;

    // Initialize destination zero
    cplVectorOpers::SetMem(d_dst, 0.f, size);
    cplVectorOpers::SetMem((uint*)d_cell, size, n);

    dim3 threads(16,16);
    dim3 grids(iDivUp(w, threads.x), iDivUp(h, threads.y));
    
    // Compute the index and the mass contribution of each mapping point
    createIndex_kernel<<<grids, threads>>>(d_cell, d_dist, d_src,
                                           d_vx, d_vy, d_vz,
                                           w, h, l);
    
    // Sort the index along with the distribution
    cplReduce  rdPlan;
    cplSort sorter(n, &rdPlan);
    int nAlign = nextPowerOf2(size);
    int bit = 0;
    while ((1 << bit) < nAlign) ++bit;
    sorter.sort((uint*)d_cell, (uint*) d_dist, n, bit);

    checkDeviceOrder((uint*)d_cell, n);
    
    //3. Build the segmented flags
    buildSegmentedFlags_tex(d_flags, (uint*)d_cell, n);
    cplVectorOpers::SetMem(d_lastPos, n, size);
    findLastPos(d_lastPos, (uint*)d_cell, n);
    // Intergrate the result to the new grids
    
    CUDPPConfiguration scanConfig;
    CUDPPHandle        mScanPlan;        // CUDPP plan handle for prefix sum
    scanConfig.algorithm = CUDPP_SEGMENTED_SCAN;
    scanConfig.datatype  = CUDPP_FLOAT;
    scanConfig.op        = CUDPP_ADD;
    scanConfig.options   = CUDPP_OPTION_INCLUSIVE | CUDPP_OPTION_FORWARD;
    cudppPlan(&mScanPlan, scanConfig, n, 1, 0);
    cudppSegmentedScan(mScanPlan, d_temp_f, (float*)d_dist, d_flags, n);
    
    //5 Extract the result
    extract_tex(d_dst, d_temp_f, d_lastPos, size, n);
    cudaThreadSynchronize();
    
    dmemFree(d_lastPos);
    dmemFree(d_cell);
    dmemFree(d_dist);
    dmemFree(d_flags);
}

//#define FIX_SCALE 4194304.f
//#define S2p20(x,t) ((t)((x)* FIX_SCALE + 0.5f))
//#define S2n20(x)   ((float)(x)/FIX_SCALE)

__global__ void atomicSplat_kernel(int* g_udst, float* g_data,
                                   float* vx, float* vy, float* vz,
                                   int w, int h, int l)
{
    const uint wh     = w * h;

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i < w && j < h){
        uint id       = i + j * w;
        for (int k=0; k < l; ++k, id+=wh) {
            float mass = g_data[id];
            
            float x = i + vx[id];
            float y = j + vy[id];
            float z = k + vz[id];
            
            int xInt = int(x);
            int yInt = int(y);
            int zInt = int(z);
            
            float dx = 1.f - (x - xInt);
            float dy = 1.f - (y - yInt);
            float dz = 1.f - (z - zInt);

            uint new_id = (zInt * h + yInt) * w + xInt;
            int dist;
            
            if (isInside3D(xInt, yInt, zInt, w, h, l)){
                dist = S2p20(mass * dx * dy * dz);
                atomicAdd(&g_udst[new_id],dist);
            }
            
            if (isInside3D(xInt + 1, yInt, zInt, w, h, l)){
                dist = S2p20(mass * (1.f-dx) * dy * dz);
                atomicAdd(&g_udst[new_id + 1], dist);
            }

            if (isInside3D(xInt, yInt+1, zInt, w, h, l)){
                dist = S2p20(mass * dx * (1.f - dy) * dz);
                atomicAdd(&g_udst[new_id + w], dist);
            }

            if (isInside3D(xInt+1, yInt+1, zInt, w, h, l)){
                dist = S2p20(mass * (1.f -dx) * (1.f - dy) * dz);
                atomicAdd(&g_udst[new_id + w + 1], dist);
            } 
            
            new_id += w*h;

            if (isInside3D(xInt, yInt, zInt + 1, w, h, l)){
                dist = S2p20(mass * dx * dy * (1.f - dz));
                atomicAdd(&g_udst[new_id],dist);
            }
            
            if (isInside3D(xInt + 1, yInt, zInt+1, w, h, l)){
                dist = S2p20(mass * (1.f-dx) * dy * (1.f -dz));
                atomicAdd(&g_udst[new_id + 1], dist);
            }

            if (isInside3D(xInt, yInt+1, zInt+1, w, h, l)){
                dist = S2p20(mass * dx * (1.f - dy) * (1.f -dz));
                atomicAdd(&g_udst[new_id + w], dist);
            }

            if (isInside3D(xInt+1, yInt+1, zInt+1, w, h, l)){
                dist = S2p20(mass * (1.f -dx) * (1.f - dy) * (1.f -dz));
                atomicAdd(&g_udst[new_id + w + 1], dist);
            } 
        }
    }
}

__global__ void atomicSplat_kernel_shared(int* g_udst, float* g_data,
                                          float* vx, float* vy, float* vz,
                                          int w, int h, int l)
{
    __shared__ int s_0[256];
    __shared__ int s_1[256];
    __shared__ int s_2[256];

    const uint wh     = w * h;

    int xc = blockIdx.x * blockDim.x;
    int yc = blockIdx.y * blockDim.y;
    
    int i = xc + threadIdx.x;
    int j = yc + threadIdx.y;
    
    if (i < w && j < h){
        uint id       = i + j * w;
        s_0[threadIdx.y * blockDim.x + threadIdx.x] = 0;
        s_1[threadIdx.y * blockDim.x + threadIdx.x] = 0;
        int* s_p = s_0, *s_c = s_1, *s_n = s_2;

        for (int k=0; k < l; ++k, id+=wh) {
            // Initialize the new buffer with zero 
            s_n[threadIdx.y * blockDim.x + threadIdx.x] = 0;

            //__syncthreads();

            float mass = g_data[id];
            
            float x = i + vx[id];
            float y = j + vy[id];
            float z = k + vz[id];
            
            int xInt = int(x);
            int yInt = int(y);
            int zInt = int(z);
            
            float dx = 1.f - (x - xInt);
            float dy = 1.f - (y - yInt);
            float dz = 1.f - (z - zInt);

            uint new_id = (zInt * h + yInt) * w + xInt;
            int dist;
            
            if (isInside3D(xInt - xc, yInt - yc, zInt + 1 - k, blockDim.x-1,  blockDim.y-1, 2)){
                int* s_l0, *s_l1;
                
                if (zInt == k){
                    s_l0 = s_c;
                    s_l1 = s_n;
                }
                else {
                    s_l0 = s_p;
                    s_l1 = s_c;
                }

                uint sid = (xInt - xc) + (yInt-yc) * 16;
                dist = S2p20(mass * dx * dy * dz);
                atomicAdd(s_l0 + sid, dist);

                dist = S2p20(mass * (1.f-dx) * dy * dz);
                atomicAdd(s_l0 + sid + 1, dist);
                
                dist = S2p20(mass * dx * (1.f - dy) * dz);
                atomicAdd(s_l0 + sid + 16, dist);

                dist = S2p20(mass * (1.f -dx) * (1.f - dy) * dz);
                atomicAdd(s_l0 + sid + 16 +1, dist);
                
                dist = S2p20(mass * dx * dy * (1-dz));
                atomicAdd(s_l1 + sid, dist);
                
                dist = S2p20(mass * (1.f-dx) * dy * (1-dz));
                atomicAdd(s_l1 + sid + 1, dist);
                
                dist = S2p20(mass * dx * (1.f - dy) * (1-dz));
                atomicAdd(s_l1 + sid + 16, dist);
                    
                dist = S2p20(mass * (1.f -dx) * (1.f - dy) * (1-dz));
                atomicAdd(s_l1 + sid + 16 +1, dist);
            }else
#if 1
                if (isInside3D(xInt, yInt, zInt, w-1, h-1, l-1)){
                    dist = S2p20(mass * dx * dy * dz);
                    atomicAdd(&g_udst[new_id],dist);

                    dist = S2p20(mass * (1.f-dx) * dy * dz);
                    atomicAdd(&g_udst[new_id + 1], dist);

                    dist = S2p20(mass * dx * (1.f - dy) * dz);
                    atomicAdd(&g_udst[new_id + w], dist);

                    dist = S2p20(mass * (1.f -dx) * (1.f - dy) * dz);
                    atomicAdd(&g_udst[new_id + w + 1], dist);

                    new_id += w*h;

                    dist = S2p20(mass * dx * dy * (1.f - dz));
                    atomicAdd(&g_udst[new_id],dist);

                    dist = S2p20(mass * (1.f-dx) * dy * (1.f -dz));
                    atomicAdd(&g_udst[new_id + 1], dist);

                    dist = S2p20(mass * dx * (1.f - dy) * (1.f -dz));
                    atomicAdd(&g_udst[new_id + w], dist);

                    dist = S2p20(mass * (1.f -dx) * (1.f - dy) * (1.f -dz));
                    atomicAdd(&g_udst[new_id + w + 1], dist);
                }
#else
            {
                if (isInside3D(xInt, yInt, zInt, w, h, l)){
                    dist = S2p20(mass * dx * dy * dz);
                    atomicAdd(&g_udst[new_id],dist);
                }
            
                if (isInside3D(xInt + 1, yInt, zInt, w, h, l)){
                    dist = S2p20(mass * (1.f-dx) * dy * dz);
                    atomicAdd(&g_udst[new_id + 1], dist);
                }
                if (isInside3D(xInt, yInt+1, zInt, w, h, l)){
                    dist = S2p20(mass * dx * (1.f - dy) * dz);
                    atomicAdd(&g_udst[new_id + w], dist);
                }
                if (isInside3D(xInt+1, yInt+1, zInt, w, h, l)){
                    dist = S2p20(mass * (1.f -dx) * (1.f - dy) * dz);
                    atomicAdd(&g_udst[new_id + w + 1], dist);
                } 
                new_id += w*h;
                if (isInside3D(xInt, yInt, zInt + 1, w, h, l)){
                    dist = S2p20(mass * dx * dy * (1.f - dz));
                    atomicAdd(&g_udst[new_id],dist);
                }
                if (isInside3D(xInt + 1, yInt, zInt+1, w, h, l)){
                    dist = S2p20(mass * (1.f-dx) * dy * (1.f -dz));
                    atomicAdd(&g_udst[new_id + 1], dist);
                }

                if (isInside3D(xInt, yInt+1, zInt+1, w, h, l)){
                    dist = S2p20(mass * dx * (1.f - dy) * (1.f -dz));
                    atomicAdd(&g_udst[new_id + w], dist);
                }
                if (isInside3D(xInt+1, yInt+1, zInt+1, w, h, l)){
                    dist = S2p20(mass * (1.f -dx) * (1.f - dy) * (1.f -dz));
                    atomicAdd(&g_udst[new_id + w + 1], dist);
                }
            }
#endif
            __syncthreads();
            
            //write out the previous layer 
            if( k > 0){
                atomicAdd(&g_udst[id - wh], s_p[threadIdx.x + threadIdx.y * 16]);
            }

            //write out the current layer if it is the last 
            if ( k == l - 1){
                atomicAdd(&g_udst[id], s_c[threadIdx.x + threadIdx.y * 16]);
            }
            
            int* temp = s_p;
            s_p = s_c;
            s_c = s_n;
            s_n = temp;
        }
    }
}



void cplSplatingAtomicUnsigned(float* d_dst, float* d_src,
                                float* d_vx , float* d_vy, float* d_vz,
                                uint sizeX, uint sizeY, uint sizeZ){
    
    int* d_idst = (int*) d_dst; // unsigned version of
    uint nElems  = sizeX * sizeY * sizeZ;
    
    // Initialize the result with Zero
    cplVectorOpers::SetMem(d_idst, 0, nElems);

    dim3 threads(16,16);
    dim3 grids(iDivUp(sizeX, threads.x), iDivUp(sizeY, threads.y));
    
    // Compute the index and the mass contribution of each mapping point
    atomicSplat_kernel<<<grids, threads>>>(d_idst, d_src,
                                           d_vx, d_vy, d_vz,
                                           sizeX, sizeY, sizeZ);

    cplVectorOpers::FixedToFloating(d_dst, d_idst, nElems, 0);
}

void cplSplatingAtomicUnsigned_shared(float* d_dst, float* d_src,
                                       float* d_vx , float* d_vy, float* d_vz,
                                       uint sizeX, uint sizeY, uint sizeZ){
    
    int* d_idst = (int*) d_dst; // unsigned version of
    uint nElems  = sizeX * sizeY * sizeZ;
    
    // Initialize the result with Zero
    cplVectorOpers::SetMem(d_idst, 0, nElems);

    dim3 threads(16,16);
    dim3 grids(iDivUp(sizeX, threads.x), iDivUp(sizeY, threads.y));
    
    // Compute the index and the mass contribution of each mapping point
    atomicSplat_kernel_shared<<<grids, threads>>>(d_idst, d_src,
                                                  d_vx, d_vy, d_vz,
                                                  sizeX, sizeY, sizeZ);

    cplVectorOpers::FixedToFloating(d_dst, d_idst, nElems, 0);
}


/*--------------------------------------------------------------------------------------------
  Splating version with large input 
  Inputs : d_src                 : source image
           d_vx, dvy, dvz        : velocity fields
           w , h, l              : size of the volume
  Output :
           d_dst                 : output intergrate from the input
 --------------------------------------------------------------------------------------------*/

__global__ void createIndex_kernel2(uint4* g_cell, float4* g_dmass,
                                    float* g_data,
                                    float* vx, float* vy, float* vz,
                                    int w, int h, int l,
                                    int l1, int l2){

    const uint nElems = w * h * l;
    const uint wh     = w * h;

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i < w && j < h){
        uint id       = i + j * w;
        for (int k    = l1; k < l2; ++k, id+=wh) {
            float mass= g_data[id];
            float x   = i + vx[id];
            float y   = j + vy[id];
            float z   = k + vz[id];
            
            int xInt = int(x);
            int yInt = int(y);
            int zInt = int(z);
            
            float dx = 1.f - (x - xInt);
            float dy = 1.f - (y - yInt);
            float dz = 1.f - (z - zInt);

            int new_id = xInt + yInt * w + zInt * wh;

            uint4  cell = make_uint4(nElems, nElems, nElems, nElems);
            float4 dist = make_float4(0.f, 0.f, 0.f, 0.f);
            
            if (isInside3D(xInt, yInt, zInt, w, h, l)){
                cell.x = new_id;
                dist.x = mass * dx * dy * dz;
            }
                
            if (isInside3D(xInt + 1, yInt, zInt, w, h, l)){
                cell.y = new_id + 1;
                dist.y = mass * (1.f-dx) * dy * dz;
            }

            if (isInside3D(xInt, yInt+1, zInt, w, h, l)){
                cell.z = new_id+w;
                dist.z = mass * dx * (1.f - dy) * dz;
            }

            if (isInside3D(xInt+1, yInt+1, zInt, w, h, l)){
                cell.w = new_id + w + 1;
                dist.w = mass * (1.f -dx) * (1.f - dy) * dz;
            }

            g_cell[id] = cell;
            g_dmass[id]= dist;

            cell = make_uint4(nElems, nElems, nElems, nElems);
            dist = make_float4(0.f, 0.f, 0.f, 0.f);

            if (isInside3D(xInt, yInt, zInt+1, w, h, l)){
                cell.x = new_id + wh;
                dist.x = mass * dx * dy * (1.f-dz);
            }
                
            if (isInside3D(xInt + 1, yInt, zInt+1, w, h, l)){
                cell.y = new_id+1 + wh;
                dist.y= mass * (1.f-dx) * dy * (1.f-dz);
            }

            if (isInside3D(xInt, yInt+1, zInt+1, w, h, l)){
                cell.z = new_id+w + wh;
                dist.z = mass * dx * (1.f - dy) * (1.f-dz);
            }

            if (isInside3D(xInt+1, yInt+1, zInt+1, w, h, l)){
                cell.w = new_id+w+1 + wh;
                dist.w = mass * (1.f -dx) * (1.f - dy) * (1.f-dz);
            }

            int offset = (l2 - l1) * wh;

            g_cell [id + offset]= cell;
            g_dmass[id + offset]= dist;
            
        }
    }
}


void cplSplating2(float* d_dst, float* d_src,
                   float* d_vx , float* d_vy, float* d_vz,
                   uint w, uint h, uint l){

    uint size   = w * h * l;
    uint memSize= size * sizeof(float);

    const int nFrac = 8;

    uint  l8     = iDivUp(l,nFrac);
    uint  n      = w * h * l8 * 8;      // we only process 1/8 of the number of point each time
    uint  nFlags = iDivUp(n, 32);       // number of 32 bit flags needed 

    uint4 * d_cell;
    float4* d_dist;
    float*  d_temp_f;

    uint *  d_lastPos;
    uint *  d_flags;
    
    dmemAlloc(d_cell,  n/4);
    dmemAlloc(d_dist,  n/4);
    
    dmemAlloc(d_temp_f,n);
    dmemAlloc(d_flags, nFlags);
    dmemAlloc(d_lastPos, (size + 64));
        
    // Initialize destination zero
    cudaMemset(d_dst, 0, memSize);

    dim3 threads(16,16);
    dim3 grids(iDivUp(w, threads.x), iDivUp(h, threads.y));
    // Compute the index and the mass contribution of each mapping point

    cplReduce      rdPlan;
    cplSort     sorter(n, &rdPlan);
    
    for (int i=0; i < nFrac; ++i)
    {
        int l1      = i * l8;
        int l2      = min((i + 1) *  l8, l);
        uint offset = l1 * w * h;
        n           = (l2 - l1) * w * h * 8;
        
        createIndex_kernel2<<<grids, threads>>>(d_cell, d_dist,
                                                d_src  + offset,
                                                d_vx   + offset, d_vy + offset, d_vz + offset,
                                                w, h, l,
                                                l1, l2);

        // Sort the index along with the distribution
        sorter.sort((uint*)d_cell, (uint*) d_dist, n);

        //3. Build the segmented flags
        buildLastPosFlags(d_flags, d_cell, n);
        cplVectorOpers::SetMem(d_lastPos, n, size);
        findLastPos(d_lastPos, d_cell, n);

        // Intergrate the result to the new grids
        segScan<float, 1>((float*) d_temp_f, (float*)d_dist, d_flags, n);
        addExtract_tex(d_dst, d_temp_f, d_lastPos, size, n);
    }
    dmemFree(d_lastPos);
    dmemFree(d_cell);
    dmemFree(d_dist);
    dmemFree(d_flags);
    dmemFree(d_temp_f);
}


void splatingCPU(float* dst, float* src,
                 float* vx, float* vy, float* vz,
                 int w, int h, int l){

    int size = w * h * l;
    for (int i=0; i<size; ++i)
        dst[i] = 0;

    int id = 0;
    for (int k=0; k<l; ++k)
        for (int j=0; j < h; ++j)
            for (int i=0; i<w; ++i, ++id){
                float mass = src[id];
                
                float x = i + vx[id];
                float y = j + vy[id];
                float z = k + vz[id];

                int xint = int(x);
                int yint = int(y);
                int zint = int(z);

                float dx = 1.f - (x - xint);
                float dy = 1.f - (y - yint);
                float dz = 1.f - (z - zint);
                
                int new_id = xint + yint * w + zint * w * h;

                if (isInside3D(xint, yint, zint, w, h, l))
                    dst[new_id] += mass * dx * dy * dz;
                
                if (isInside3D(xint + 1, yint, zint, w, h, l))
                    dst[new_id+1] += mass * (1.f-dx) * dy * dz;

                if (isInside3D(xint, yint+1, zint, w, h, l))
                    dst[new_id+w] += mass * dx * (1.f - dy) * dz;

                if (isInside3D(xint+1, yint+1, zint, w, h, l))
                    dst[new_id+w+1] += mass * (1.f -dx) * (1.f - dy) * dz;

                if (isInside3D(xint, yint, zint+1, w, h, l))
                    dst[new_id + w * h] += mass * dx * dy * (1.f-dz);
                
                if (isInside3D(xint + 1, yint, zint+1, w, h, l))
                    dst[new_id+1 + w * h] += mass * (1.f-dx) * dy * (1.f-dz);

                if (isInside3D(xint, yint+1, zint+1, w, h, l))
                    dst[new_id+w + w * h] += mass * dx * (1.f - dy) * (1.f-dz);

                if (isInside3D(xint+1, yint+1, zint+1, w, h, l))
                    dst[new_id+w+1 + w * h] += mass * (1.f -dx) * (1.f - dy) * (1.f-dz);
            }
                
}

void splatingCPU_test(uint4* cell, float4* dist, float* src,
                      float* vx, float* vy, float* vz,
                      int w, int h, int l){

    int size = w * h * l;

    int id = 0;
    for (int k=0; k<l; ++k)
        for (int j=0; j < h; ++j)
            for (int i=0; i<w; ++i, ++id){
                float mass = src[id];
                
                float x = i + vx[id];
                float y = j + vy[id];
                float z = k + vz[id];

                int xint = int(x);
                int yint = int(y);
                int zint = int(z);

                float dx = 1.f - (x - xint);
                float dy = 1.f - (y - yint);
                float dz = 1.f - (z - zint);
                
                int new_id = xint + yint * w + zint * w * h;

                cell[id        ] = make_uint4(size, size, size, size);
                cell[id + size ] = make_uint4(size, size, size, size);
                dist[id ]          = make_float4(0.f, 0.f, 0.f, 0.f);
                dist[id + size ] = make_float4(0.f, 0.f, 0.f, 0.f);
                
                if (isInside3D(xint, yint, zint, w, h, l)){
                    cell[id].x = new_id;
                    dist[id].x = mass * dx * dy * dz;
                }
                
                if (isInside3D(xint + 1, yint, zint, w, h, l)){
                    cell[id].y = new_id + 1;
                    dist[id].y = mass * (1.f-dx) * dy * dz;
                }

                if (isInside3D(xint, yint+1, zint, w, h, l)){
                    cell[id].z = new_id+w;
                    dist[id].z = mass * dx * (1.f - dy) * dz;
                }

                if (isInside3D(xint+1, yint+1, zint, w, h, l)){
                    cell[id].w = new_id+w+1;
                    dist[id].w = mass * (1.f -dx) * (1.f - dy) * dz;
                }

                if (isInside3D(xint, yint, zint+1, w, h, l)){
                    cell[id + size].x = new_id + w * h;
                    dist[id + size].x = mass * dx * dy * (1.f-dz);
                }
                
                if (isInside3D(xint + 1, yint, zint+1, w, h, l)){
                    cell[id + size].y = new_id+1 + w * h;
                    dist[id + size].y= mass * (1.f-dx) * dy * (1.f-dz);
                }

                if (isInside3D(xint, yint+1, zint+1, w, h, l)){
                    cell[id + size].z = new_id+w + w * h;
                    dist[id + size].z = mass * dx * (1.f - dy) * (1.f-dz);
                }

                if (isInside3D(xint+1, yint+1, zint+1, w, h, l)){
                    cell[id + size].w = new_id+w+1 + w * h;
                    dist[id + size].w = mass * (1.f -dx) * (1.f - dy) * (1.f-dz);
                }
            }
                
}

void testSplating(uint w, uint h, int l){
    int size = w * h * l;
    float *h_iImg, *h_oImg, *h_vx, *h_vy, *h_vz;

    h_iImg = new float [size];
    h_oImg = new float [size];
    h_vx   = new float [size];
    h_vy   = new float [size];
    h_vz   = new float [size];

    /*
    for (int i=0; i<size; ++i)
        h_iImg[i] = rand() % 256;
    cpuMakeZeroVolumeBoundary(h_iImg, w, h, l);
    */
    genRandomImageWithZeroBound(h_iImg, w, h, l, 2);
    
    
    for (int i=0; i< size; ++i)
        h_vx[i] = (float(rand()) / RAND_MAX - 0.5f) * 4.f;
    
    for (int i=0; i< size; ++i)
        h_vy[i] = (float(rand()) / RAND_MAX - 0.5f) * 4.f;

    for (int i=0; i< size; ++i)
        h_vz[i] = (float(rand()) / RAND_MAX - 0.5f) * 4.f;

    splatingCPU(h_oImg, h_iImg, h_vx, h_vy, h_vz, w, h, l);

    float* d_oImg, *d_iImg, *d_vx, *d_vy, *d_vz;
    dmemAlloc(d_iImg, size);
    dmemAlloc(d_oImg, size);
    dmemAlloc(d_vx, size);
    dmemAlloc(d_vy, size);
    dmemAlloc(d_vz, size);

    cudaMemcpy(d_iImg, h_iImg, sizeof(float) * size, cudaMemcpyHostToDevice);
    
    cudaMemcpy(d_vx, h_vx, sizeof(float) * size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vy, h_vy, sizeof(float) * size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vz, h_vz, sizeof(float) * size, cudaMemcpyHostToDevice);

    unsigned int timer;
    
    CUT_SAFE_CALL( cutCreateTimer( &timer));

    CUT_SAFE_CALL( cutResetTimer( timer));
    CUT_SAFE_CALL( cutStartTimer( timer));
    int nIter = 1;
    for (int i=0; i<nIter; ++i)
        cplSplatingAtomicUnsigned(d_oImg, d_iImg, d_vx, d_vy, d_vz, w, h, l);
    //cplSplating(d_oImg, d_iImg, d_vx, d_vy, d_vz, w, h, l);
        //cplSplatingOld(d_oImg, d_iImg, d_vx, d_vy, d_vz, w, h, l);
                 
    cudaThreadSynchronize();
    CUT_SAFE_CALL( cutStopTimer( timer));
    printf( "Processing time: %f (ms)\n", cutGetTimerValue(timer)/nIter);

    testError(h_oImg, d_oImg, 0.5 * 1e-4, size, "Splating image");
    
    dmemFree(d_oImg);
    dmemFree(d_iImg);
    dmemFree(d_vy);
    dmemFree(d_vx);
    dmemFree(d_vz);

    delete []h_iImg;
    delete []h_oImg;
    delete []h_vx;
    delete []h_vy;
    delete []h_vz;
}


void testSplating2(uint w, uint h, int l){
    int size = w * h * l;
    float *h_iImg1, *h_iImg2;
    float *h_vx, *h_vy, *h_vz;

    h_iImg1 = new float [size];
    h_iImg2 = new float [size];

    h_vx    = new float [size];
    h_vy    = new float [size];
    h_vz    = new float [size];

    fprintf(stderr, "Input image size %d %d %d \n", w, h, l);
    
    genRandomImageWithZeroBound(h_iImg1, w, h, l, 2);
    genRandomImageWithZeroBound(h_iImg2, w, h, l, 2);
    
    genRandomVelocityWithZeroBound(h_vx, w, h, l, 2);
    genRandomVelocityWithZeroBound(h_vy, w, h, l, 2);
    genRandomVelocityWithZeroBound(h_vz, w, h, l, 2);
    
    float* d_iImg1, *d_iImg2;
    float* d_oImg1, *d_oImg2;
    
    float* d_vx, *d_vy, *d_vz;

    dmemAlloc(d_iImg1, size);
    dmemAlloc(d_iImg2, size);
    dmemAlloc(d_oImg1, size);
    dmemAlloc(d_oImg2, size);

    dmemAlloc(d_vx, size);
    dmemAlloc(d_vy, size);
    dmemAlloc(d_vz, size);

    cplReduce rd;

    copyArrayToDevice(d_vx, h_vx, size);
    copyArrayToDevice(d_vy, h_vy, size);
    copyArrayToDevice(d_vz, h_vz, size);

    copyArrayToDevice(d_iImg1, h_iImg1, size);
    copyArrayToDevice(d_iImg2, h_iImg2, size);
    
    float max1 = rd.Max(d_iImg1, size);
    float max2 = rd.Max(d_iImg1, size);

    fprintf(stderr, "Max of image 1 %f Max of image 2 %f \n", max1, max2);
    
    max1 = rd.Max(d_vx, size);
    max2 = rd.Max(d_vy, size);
    float max3 = rd.Max(d_vz, size);

    fprintf(stderr, "Max of vx %f Max of vy %f  Max of vz %f \n", max1, max2, max3);
    
    float min1 = rd.Min(d_vx, size);
    float min2 = rd.Min(d_vy, size);
    float min3 = rd.Min(d_vz, size);

    fprintf(stderr, "Min of vx %f Min of vy %f  Min of vz %f \n", min1, min2, min3);
        
    cudaMemcpy(d_vx, h_vx, sizeof(float) * size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vy, h_vy, sizeof(float) * size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vz, h_vz, sizeof(float) * size, cudaMemcpyHostToDevice);

    unsigned int timer;
    
    CUT_SAFE_CALL( cutCreateTimer( &timer));

    CUT_SAFE_CALL( cutResetTimer( timer));
    CUT_SAFE_CALL( cutStartTimer( timer));

    int nIter = 1;

    cplBackwardMapping3D(d_oImg1, d_iImg1, d_vx, d_vy, d_vz, w, h, l);
    
    cplSplating(d_oImg2, d_iImg2, d_vx, d_vy, d_vz, w, h, l);
    //cplSplating(d_oImg2, d_iImg2, d_vx, d_vy, d_vz, w, h, l);
    
    float prod1 = rd.Dot(d_oImg1, d_iImg2, size);
    float prod2 = rd.Dot(d_iImg1, d_oImg2, size);

    fprintf(stderr, "First product %f Second product %f Different %f %%", prod1, prod2, fabs(prod2 - prod1) / prod1 * 100);
    
    cudaThreadSynchronize();
    CUT_SAFE_CALL( cutStopTimer( timer));
    printf( "Processing time: %f (ms)\n", cutGetTimerValue(timer)/nIter);
    
    dmemFree(d_oImg1);
    dmemFree(d_iImg1);
    dmemFree(d_oImg2);
    dmemFree(d_iImg2);

    
    dmemFree(d_vy);
    dmemFree(d_vx);
    dmemFree(d_vz);

    delete []h_iImg1;
    delete []h_iImg2;
    delete []h_vx;
    delete []h_vy;
    delete []h_vz;
}

__global__ void createIndex3D_kernel(uint4*  d_pc   , float4* d_pm,
                                     float2* d_px_xy, float* d_px_z,
                                     float*  d_pu,
                                     int w, int h, int l,
                                     int n, int nA)
{
    int nElems   = w * h * l;

    uint blockId = get_blockID();
    uint id      = get_threadID(blockId);

    if (id < n){
        // Load the point mass and position 
        float  mass        = d_pu[id];
        volatile float2 xy = d_px_xy[id];
        float  z           = d_px_z[id];
        float  x    = xy.x;
        float  y    = xy.y;

        int    xInt = int(x);
        int    yInt = int(y);
        int    zInt = int(z);

        int nid =   xInt + (yInt + zInt * h) * w;
                                    
        float  dx   = 1.f - (x - xInt);
        float  dy   = 1.f - (y - yInt);
        float  dz   = 1.f - (z - zInt);

        // Compute the first layer zInt
        // Initialize 
        uint4  cell = make_uint4(nElems, nElems, nElems, nElems);
        float4 dist = make_float4(0.f, 0.f, 0.f, 0.f);
        // Compute the contribution
        if (isInside3D(xInt, yInt, zInt, w, h, l)){
            cell.x = nid;
            dist.x = mass * dx * dy * dz;
        }
                
        if (isInside3D(xInt+1, yInt,zInt, w, h, l)){
            cell.y = nid + 1;
            dist.y = mass * (1.f-dx) * dy  * dz;
        }

        if (isInside3D(xInt, yInt+1,zInt, w, h, l)){
            cell.z = nid + w;
            dist.z = mass * dx * (1.f - dy)  * dz;
        }
        
        if (isInside3D(xInt+1, yInt+1,zInt, w, h, l)){
            cell.w = nid + w + 1;
            dist.w = mass * (1.f -dx) * (1.f - dy) * dz;
        }

        // Write out the result
        d_pc[id] = cell;
        d_pm[id] = dist;


        // Compute the second layer zInt+1
        cell = make_uint4(nElems, nElems, nElems, nElems);
        dist = make_float4(0.f, 0.f, 0.f, 0.f);
        nid += w*h;
        
        // Compute the contribution
        if (isInside3D(xInt, yInt, zInt+1, w, h, l)){
            cell.x = nid;
            dist.x = mass * dx * dy * (1.f-dz);
        }
                
        if (isInside3D(xInt+1, yInt,zInt+1, w, h, l)){
            cell.y = nid + 1;
            dist.y = mass * (1.f-dx) * dy  * (1.f-dz);
        }

        if (isInside3D(xInt, yInt+1,zInt+1, w, h, l)){
            cell.z = nid + w;
            dist.z = mass * dx * (1.f - dy)  * (1.f-dz);
        }
        
        if (isInside3D(xInt+1, yInt+1,zInt+1, w, h, l)){
            cell.w = nid + w + 1;
            dist.w = mass * (1.f -dx) * (1.f - dy) * (1.f-dz);
        }

        // Write out the result
        d_pc[id+nA] = cell;
        d_pm[id+nA] = dist;
    }
    else if (id <nA){
        uint4   cell = make_uint4(nElems, nElems, nElems, nElems);
        float4 dist = make_float4(0.f, 0.f, 0.f, 0.f);

        d_pc[id] = cell;
        d_pm[id] = dist;
        d_pc[id+nA] = cell;
        d_pm[id+nA] = dist;
    }
}

void createIndex3D(uint4 * d_pc, float4* d_pm,
                   Vector3D_XY_Z_Array& d_px,
                   float * d_pu,
                   int w, int h, int l,
                   int n, int nA){

    dim3 threads(256);
    dim3 grids(iDivUp(nA, threads.x));
    checkConfig(grids);
        
    createIndex3D_kernel<<<grids, threads>>>(d_pc, d_pm, d_px.xy, d_px.z,
                                             d_pu,
                                             w, h, l,
                                             n, nA);
}


void splat3D(float* d_gu, float* d_pu, Vector3D_XY_Z_Array& d_px,
             int w, int h, int l,
             int nP){

    const uint nNodes = w * h * l;
    const uint nParts = nP;

    uint nPartA = (nParts & 0x1f) ? nParts - (nParts & 0x1f) + 32 : nParts;
    
    uint nTotal = nParts * 4 * 2;
    
    uint4  * d_pc;
    float4* d_pm;
    float * d_temp;

    uint* d_flags, *d_lastPos;
        
    dmemAlloc(d_pc     , 2 * max(nPartA, 64));
    dmemAlloc(d_pm     , 2 * max(nPartA, 64));
    dmemAlloc(d_temp   , 2 * 4 * nPartA);
    
    dmemAlloc(d_flags  , iDivUp(nTotal, 32));
    dmemAlloc(d_lastPos, (nNodes + 32));
        
    //1. Compute the cell id and weight distribution to each cell of each input points
    createIndex3D(d_pc, d_pm, d_px, d_pu, w, h, l, nParts, nPartA);

    //2. Sorting the index with the distribution
    cplReduce      rdPlan;
    cplSort     sorter(nTotal, &rdPlan);
    

    sorter.sort((uint*)d_pc, (uint*)d_pm, nPartA * 8);

    //3. Build the segmented flags
    buildLastPosFlags(d_flags, d_pc, nTotal);
    cplVectorOpers::SetMem(d_lastPos, nTotal, nNodes);
    findLastPos(d_lastPos, d_pc, nTotal);
    
    //4. Perform segmented scan on the distrbution
    segScan<float, 1>((float*) d_temp, (float*)d_pm, d_flags, nTotal);

    //5 Extract the result
    cplVectorOpers::SetMem(d_gu, 0.f, nNodes);
        
    extract_tex(d_gu, d_temp, d_lastPos, nNodes, nTotal);

    dmemFree(d_pc);
    dmemFree(d_pm);
    dmemFree(d_temp);
    dmemFree(d_flags);
    dmemFree(d_lastPos);
    
}

/*------------------------------------------------------------------------------------------*/
__global__ void createIndex3D_kernel(uint4*  d_pc,
                                     float4* d_pmx, float4* d_pmy, float4* d_pmz, float4* d_pmw,
                                     float2* d_px_xy, float* d_px_z,
                                     float4*  d_pu,
                                     int w, int h, int l,
                                     int n, int nA)
{
    int nElems   = w * h * l;

    uint blockId = get_blockID();
    uint id      = get_threadID(blockId);

    if (id < n){
        // Load the point mass and position 
        float4  mass        = d_pu[id];
        volatile float2 xy = d_px_xy[id];
        float  z           = d_px_z[id];
        float  x    = xy.x;
        float  y    = xy.y;

        int    xInt = int(x);
        int    yInt = int(y);
        int    zInt = int(z);

        int nid =   xInt + (yInt + zInt * h) * w;
                                    
        float  dx   = 1.f - (x - xInt);
        float  dy   = 1.f - (y - yInt);
        float  dz   = 1.f - (z - zInt);

        // Compute the first layer zInt
        // Initialize 
        uint4  cell  = make_uint4(nElems, nElems, nElems, nElems);
        float4 distx = make_float4(0.f, 0.f, 0.f, 0.f);
        float4 disty = make_float4(0.f, 0.f, 0.f, 0.f);
        float4 distz = make_float4(0.f, 0.f, 0.f, 0.f);
        float4 distw = make_float4(0.f, 0.f, 0.f, 0.f);
        
        // Compute the contribution
        if (isInside3D(xInt, yInt, zInt, w, h, l)){
            cell.x = nid;
            float f= dx * dy * dz;
            distx.x  = mass.x * f;
            disty.x  = mass.y * f;
            distz.x  = mass.z * f;
            distw.x  = mass.w * f;
        }
                
        if (isInside3D(xInt+1, yInt,zInt, w, h, l)){
            cell.y = nid + 1;
            float f= (1.f-dx) * dy  * dz;

            distx.y  = mass.x * f;
            disty.y  = mass.y * f;
            distz.y  = mass.z * f;
            distw.y  = mass.w * f;
            
        }
        
        if (isInside3D(xInt, yInt+1,zInt, w, h, l)){
            cell.z = nid + w;
            float f= dx * (1.f - dy)  * dz;
            distx.z  = mass.x * f;
            disty.z  = mass.y * f;
            distz.z  = mass.z * f;
            distw.z  = mass.w * f;
        }
        
        if (isInside3D(xInt+1, yInt+1,zInt, w, h, l)){
            cell.w = nid + w + 1;
            float f= (1.f -dx) * (1.f - dy) * dz;
            distx.w  = mass.x * f;
            disty.w  = mass.y * f;
            distz.w  = mass.z * f;
            distw.w  = mass.w * f;
        }

        // Write out the result
        d_pc[id] = cell;
        
        d_pmx[id] = distx;
        d_pmy[id] = disty;
        d_pmz[id] = distz;
        d_pmw[id] = distw;
        


        // Compute the second layer zInt+1
        cell  = make_uint4(nElems, nElems, nElems, nElems);
        distx = make_float4(0.f, 0.f, 0.f, 0.f);
        disty = make_float4(0.f, 0.f, 0.f, 0.f);
        distz = make_float4(0.f, 0.f, 0.f, 0.f);
        distw = make_float4(0.f, 0.f, 0.f, 0.f);
        nid += w*h;
                
        // Compute the contribution
        if (isInside3D(xInt, yInt, zInt+1, w, h, l)){
            cell.x = nid;
            float f= dx * dy * (1-dz);
            distx.x  = mass.x * f;
            disty.x  = mass.y * f;
            distz.x  = mass.z * f;
            distw.x  = mass.w * f;
        }
                
        if (isInside3D(xInt+1, yInt,zInt+1, w, h, l)){
            cell.y = nid + 1;
            float f= (1.f-dx) * dy  * (1-dz);

            distx.y  = mass.x * f;
            disty.y  = mass.y * f;
            distz.y  = mass.z * f;
            distw.y  = mass.w * f;
            
        }
        
        if (isInside3D(xInt, yInt+1,zInt+1, w, h, l)){
            cell.z = nid + w;
            float f= dx * (1.f - dy)  * (1-dz);
            distx.z  = mass.x * f;
            disty.z  = mass.y * f;
            distz.z  = mass.z * f;
            distw.z  = mass.w * f;
        }
        
        if (isInside3D(xInt+1, yInt+1,zInt+1, w, h, l)){
            cell.w = nid + w + 1;
            float f= (1.f -dx) * (1.f - dy) * (1-dz);
            distx.w  = mass.x * f;
            disty.w  = mass.y * f;
            distz.w  = mass.z * f;
            distw.w  = mass.w * f;
        }

        
        // Write out the result
        d_pc[id] = cell;
        
        d_pmx[id+nA] = distx;
        d_pmy[id+nA] = disty;
        d_pmz[id+nA] = distz;
        d_pmw[id+nA] = distw;
    }
    else if (id <nA){
        uint4  cell = make_uint4(nElems, nElems, nElems, nElems);
        float4 dist = make_float4(0.f, 0.f, 0.f, 0.f);

        d_pc[id] = cell;

        d_pmx[id] = dist;
        d_pmy[id] = dist;
        d_pmz[id] = dist;
        d_pmw[id] = dist;
        
        d_pc[id+nA] = cell;
        
        d_pc[id+nA] = cell;
        d_pmx[id+nA] = dist;
        d_pmy[id+nA] = dist;
        d_pmz[id+nA] = dist;
        d_pmw[id+nA] = dist;
    }
}

void createIndex3D(uint4 * d_pc,
                   float4* d_pmx,float4* d_pmy,float4* d_pmz,float4* d_pmw,
                   Vector3D_XY_Z_Array& d_px,
                   float4 * d_pu,
                   int w, int h, int l,
                   int n, int nA){

    dim3 threads(256);
    dim3 grids(iDivUp(nA,threads.x));
    checkConfig(grids);
    
    createIndex3D_kernel<<<grids, threads>>>(d_pc,
                                             d_pmx, d_pmy, d_pmz, d_pmw,
                                             d_px.xy, d_px.z,
                                             d_pu,
                                             w, h, l,
                                             n, nA);
}


/*------------------------------------------------------------------------------------------*/
void splat3D(float4* d_gu, float4* d_pu, Vector3D_XY_Z_Array& d_px,
             int w, int h, int l,
             int nP){

    const uint nNodes = w * h * l;
    const uint nParts = nP;

    uint nPartA = (nParts & 0x1f) ? nParts - (nParts & 0x1f) + 32 : nParts;
    uint nTotal = nParts * 4 * 2;
    
    uint4  * d_pc;
    float4* d_pm;
    float * d_temp;
    uint * d_index;

    uint* d_flags, *d_lastPos;
    dmemAlloc(d_pc     , 2 * max(nPartA, 64));
    dmemAlloc(d_pm     , 4 * 2 * max(nPartA, 64));

    float4* d_pmx = d_pm;
    float4* d_pmy = d_pm + 2 * nPartA;
    float4* d_pmz = d_pm + 4 * nPartA;
    float4* d_pmw = d_pm + 6 * nPartA;
    
    
    dmemAlloc(d_temp   , 2 * 4 * nPartA);
    dmemAlloc(d_index  , 2 * 4 * nPartA);
    dmemAlloc(d_flags  , iDivUp(nTotal, 32));
    dmemAlloc(d_lastPos, (nNodes + 32));
        
    //1. Compute the cell id and weight distribution to each cell of each input points
    createIndex3D(d_pc, d_pmx, d_pmy, d_pmz, d_pmw, d_px, d_pu, w, h, l, nParts, nPartA);

    cplVectorOpers::SetLinear(d_index, nPartA * 8);
    
    //2. Sorting the index with the distribution
    cplReduce      rdPlan;
    cplSort     sorter(nTotal, &rdPlan);
    
    sorter.sort((uint*)d_pc, (uint*)d_index, nPartA * 8);
    
    //3. Build the segmented flags
    buildLastPosFlags(d_flags, d_pc, nTotal);
    cplVectorOpers::SetMem(d_lastPos, nTotal, nNodes);
    findLastPos(d_lastPos, d_pc, nTotal);
    
    //4. Perform segmented scan on the distrbution
    cplMap(d_temp, (float*) d_pmx, d_index, nTotal);
    segScan<float, 1>((float*)d_pmx,(float*) d_temp, d_flags, nTotal);

    cplMap(d_temp, (float*)d_pmy, d_index, nTotal);
    segScan<float, 1>((float*)d_pmy,(float*) d_temp, d_flags, nTotal);

    cplMap(d_temp, (float*)d_pmz, d_index, nTotal);
    segScan<float, 1>((float*)d_pmz,(float*) d_temp, d_flags, nTotal);

    cplMap(d_temp, (float*)d_pmw, d_index, nTotal);
    segScan<float, 1>((float*)d_pmw,(float*) d_temp, d_flags, nTotal);

    //5 Extract the result
    cplVectorOpers::SetMem(d_gu, make_float4(0.f,0.f, 0.f, 0.f), nNodes);
    extract4fTof4(d_gu, (float*)d_pmx, (float*)d_pmy, (float*)d_pmz, (float*)d_pmw, d_lastPos, nNodes, nTotal);

    dmemFree(d_pc);
    dmemFree(d_pm);
    dmemFree(d_temp);
    dmemFree(d_flags);
    dmemFree(d_lastPos);
}


