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

#include <cplMacro.h>
#include <VectorMath.h>
#include <cudaVector3DArray.h>
#include <cudaReduce.h>
#include <cutil_comfunc.h>
#include <cudaSplat.h>
#include <cutil_inline.h>
#include <fstream>
////////////////////////////////////////////////////////////////////////////////
/// Splat both the function p_u and the distance to the neighbor point to the grid
///  d_wd = sum dist * d_w weighted distance
///  d_d  = sum dist       total distance
////////////////////////////////////////////////////////////////////////////////
#define ADJUST_INT 1

__global__ void atomicSplatWeightPos_kernel(int* d_wd, int* d_d, int w, int h, int l,
                                            float* d_w,
                                            float* d_px, float* d_py, float* d_pz, int nP)
{
    uint blockId = get_blockID();
    uint id      = get_threadID(blockId);

    if (id >= nP) return;

    float mass = d_w[id];
    float x = d_px[id];
    float y = d_py[id];
    float z = d_pz[id];
        
    int xInt = int(x);
    int yInt = int(y);
    int zInt = int(z);

#if ADJUST_INT
    if (x < 0 && x != xInt) --xInt;
    if (y < 0 && y != yInt) --yInt;
    if (z < 0 && z != zInt) --zInt;
#endif

    float dx = 1.f - (x - xInt);
    float dy = 1.f - (y - yInt);
    float dz = 1.f - (z - zInt);

    uint nid = (zInt * h + yInt) * w + xInt;
    float  dist;

    if (isInside3D(xInt, yInt, zInt, w, h, l)){
        dist = dx * dy * dz;
        atomicAdd(&d_wd[nid],S2p20(mass * dist));
        atomicAdd(&d_d[nid],S2p20(dist));
    }
    
    if (isInside3D(xInt + 1, yInt, zInt, w, h, l)){
        dist = (1.f-dx) * dy * dz;
        atomicAdd(&d_wd[nid + 1], S2p20(mass * dist));
        atomicAdd(&d_d[nid + 1], S2p20(dist));
    }

    if (isInside3D(xInt, yInt+1, zInt, w, h, l)){
        dist = dx * (1.f - dy) * dz;
        atomicAdd(&d_wd[nid + w], S2p20(mass * dist));
        atomicAdd(&d_d[nid + w], S2p20(dist));
    }

    if (isInside3D(xInt+1, yInt+1, zInt, w, h, l)){
        dist = (1.f -dx) * (1.f - dy) * dz;
        atomicAdd(&d_wd[nid + w + 1], S2p20(mass * dist));
        atomicAdd(&d_d[nid + w + 1], S2p20(dist));
    } 
            
    nid += w*h;
    
    if (isInside3D(xInt, yInt, zInt + 1, w, h, l)){
        dist =  dx * dy * (1.f - dz);
        atomicAdd(&d_wd[nid],S2p20(mass * dist));
        atomicAdd(&d_d[nid],S2p20(dist));
    }
            
    if (isInside3D(xInt + 1, yInt, zInt+1, w, h, l)){
        dist = (1.f-dx) * dy * (1.f -dz);
        atomicAdd(&d_wd[nid + 1], S2p20(mass * dist));
        atomicAdd(&d_d[nid + 1], S2p20(dist));
    }
    
    if (isInside3D(xInt, yInt+1, zInt+1, w, h, l)){
        dist = dx * (1.f - dy) * (1.f -dz);
        atomicAdd(&d_wd[nid + w], S2p20(mass * dist));
        atomicAdd(&d_d[nid + w], S2p20(dist));
    }

    if (isInside3D(xInt+1, yInt+1, zInt+1, w, h, l)){
        dist = (1.f -dx) * (1.f - dy) * (1.f -dz);
        atomicAdd(&d_wd[nid + w + 1], S2p20(mass * dist));
        atomicAdd(&d_d[nid + w + 1], S2p20(dist));
    } 
}

void cplSplatWeightPos_fixed(int* d_wd, int* d_d, int w, int h, int l,
                              float* d_w,
                              float* d_px, float* d_py, float* d_pz,
                              int nP, cudaStream_t stream)
{
    uint gridSize = w * h * l;
    cplVectorOpers::SetMem(d_wd, 0, gridSize, stream);
    cplVectorOpers::SetMem(d_d, 0, gridSize, stream);

    dim3 threads(256);
    dim3 grids(iDivUp(nP, threads.x));
    checkConfig(grids);

    // Compute the index and the mass contribution of each mapping point
    atomicSplatWeightPos_kernel<<<grids, threads, 0, stream>>>(d_wd, d_d, w, h, l,
                                                               d_w, d_px, d_py, d_pz, nP);

}
////////////////////////////////////////////////////////////////////////////////
// Fixed point division
// 
////////////////////////////////////////////////////////////////////////////////
__global__ void cplConvertWeightedDistance_kernel(float* d_fwd, float* d_fd,
                                                   int* d_iwd, int* d_id, uint n)
{
    uint blockId = get_blockID();
    uint id      = get_threadID(blockId);

    if (id < n){
        if (d_id[id] == 0) {
            d_fwd[id] = 0.f;
            d_fd[id] = 0.f;
        }
        else {
            d_fwd[id] = float(d_iwd[id]) / float(d_id[id]);
            d_fd[id]  = S2n20(d_id[id]);
        }
    }
}

void cplConvertWeightedDistance_fixed(float* d_fwd, float* d_fd,
                                       int* d_iwd, int* d_id, uint n, cudaStream_t stream)
{
    dim3 threads(256);
    dim3 grids(iDivUp(n, threads.x));
    checkConfig(grids);
    cplConvertWeightedDistance_kernel<<<grids, threads, 0, stream>>>(d_fwd, d_fd, d_iwd, d_id, n);
}

void cplSplat3D(float* d_wd, float* d_d, uint sizeX, uint sizeY, uint sizeZ,
                 float* d_w, float* d_px , float* d_py, float* d_pz, uint nP, cudaStream_t stream)
{
    unsigned int gridSize = sizeX * sizeY * sizeZ;

    int* d_iwd = (int*) d_wd; // unsigned version of
    int* d_id  = (int*) d_d; // unsigned version of

    // Splat the weight and the distance
    cplSplatWeightPos_fixed(d_iwd, d_id, sizeX, sizeY, sizeZ,
                             d_w, d_px, d_py, d_pz, nP, stream);

    // Divide weight to distance
    cplConvertWeightedDistance_fixed(d_wd, d_d, d_iwd, d_id, gridSize, stream);
}

void cplSplat3D(float* d_wd, float* d_d, const Vector3Di& size,
                 float* d_w, cplVector3DArray& d_pos, uint nP, cudaStream_t stream)
{
    cplSplat3D(d_wd, d_d, size.x, size.y, size.z,
                d_w,
                d_pos.x, d_pos.y, d_pos.z, nP, stream);
}


__device__ void atomicSplat(int* d_wd,
                            float mass, float x, float y, float z,
                            int w, int h, int l)
{
    int xInt = int(x);
    int yInt = int(y);
    int zInt = int(z);

#if ADJUST_INT
    if (x < 0 && x != xInt) --xInt;
    if (y < 0 && y != yInt) --yInt;
    if (z < 0 && z != zInt) --zInt;
#endif
    
    float dx = 1.f - (x - xInt);
    float dy = 1.f - (y - yInt);
    float dz = 1.f - (z - zInt);

    uint nid = (zInt * h + yInt) * w + xInt;
    int dist;

    if (isInside3D(xInt, yInt, zInt, w, h, l)){
        dist = S2p20(mass * dx * dy * dz);
        atomicAdd(&d_wd[nid],dist);
    }
            
    if (isInside3D(xInt + 1, yInt, zInt, w, h, l)){
        dist = S2p20(mass * (1.f-dx) * dy * dz);
        atomicAdd(&d_wd[nid + 1], dist);
    }

    if (isInside3D(xInt, yInt+1, zInt, w, h, l)){
        dist = S2p20(mass * dx * (1.f - dy) * dz);
        atomicAdd(&d_wd[nid + w], dist);
    }
    
    if (isInside3D(xInt+1, yInt+1, zInt, w, h, l)){
        dist = S2p20(mass * (1.f -dx) * (1.f - dy) * dz);
        atomicAdd(&d_wd[nid + w + 1], dist);
    } 
    
    nid += w*h;

    if (isInside3D(xInt, yInt, zInt + 1, w, h, l)){
        dist = S2p20(mass * dx * dy * (1.f - dz));
        atomicAdd(&d_wd[nid],dist);
    }
            
    if (isInside3D(xInt + 1, yInt, zInt+1, w, h, l)){
        dist = S2p20(mass * (1.f-dx) * dy * (1.f -dz));
        atomicAdd(&d_wd[nid + 1], dist);
    }
    
    if (isInside3D(xInt, yInt+1, zInt+1, w, h, l)){
        dist = S2p20(mass * dx * (1.f - dy) * (1.f -dz));
        atomicAdd(&d_wd[nid + w], dist);
    }
    
    if (isInside3D(xInt+1, yInt+1, zInt+1, w, h, l)){
        dist = S2p20(mass * (1.f -dx) * (1.f - dy) * (1.f -dz));
        atomicAdd(&d_wd[nid + w + 1], dist);
    } 
}

__device__ void atomicSplat(int* d_wd, int* d_wd1, int* d_wd2,
                            float mass, float mass1, float mass2,
                            float x, float y, float z,
                            int w, int h, int l)
{
    int xInt = int(x);
    int yInt = int(y);
    int zInt = int(z);

#if ADJUST_INT
    if (x < 0 && x != xInt) --xInt;
    if (y < 0 && y != yInt) --yInt;
    if (z < 0 && z != zInt) --zInt;
#endif

    float dx = 1.f - (x - xInt);
    float dy = 1.f - (y - yInt);
    float dz = 1.f - (z - zInt);

    uint nid = (zInt * h + yInt) * w + xInt;
    int dist;
    float weight;
    
    if (isInside3D(xInt, yInt, zInt, w, h, l)){
        weight = dx * dy * dz;
        
        dist = S2p20(mass * weight);
        atomicAdd(&d_wd[nid],dist);

        dist = S2p20(mass1 * weight);
        atomicAdd(&d_wd1[nid],dist);

        dist = S2p20(mass2 * weight);
        atomicAdd(&d_wd2[nid],dist);
    }
            
    if (isInside3D(xInt + 1, yInt, zInt, w, h, l)){
        weight = (1.f-dx) * dy * dz;
        
        dist = S2p20(mass * weight);
        atomicAdd(&d_wd[nid + 1], dist);

        dist = S2p20(mass1 * weight);
        atomicAdd(&d_wd1[nid + 1], dist);
        
        dist = S2p20(mass2 * weight);
        atomicAdd(&d_wd2[nid + 1], dist);
    }

    if (isInside3D(xInt, yInt+1, zInt, w, h, l)){
        weight = dx * (1.f - dy) * dz;
        
        dist = S2p20(mass * weight);
        atomicAdd(&d_wd[nid + w], dist);

        dist = S2p20(mass1 * weight);
        atomicAdd(&d_wd1[nid + w], dist);

        dist = S2p20(mass2 * weight);
        atomicAdd(&d_wd2[nid + w], dist);
    }
    
    if (isInside3D(xInt+1, yInt+1, zInt, w, h, l)){
        weight = (1.f -dx) * (1.f - dy) * dz;
        
        dist = S2p20(mass * weight);
        atomicAdd(&d_wd[nid + w + 1], dist);

        dist = S2p20(mass1 * weight);
        atomicAdd(&d_wd1[nid + w + 1], dist);

        dist = S2p20(mass2 * weight);
        atomicAdd(&d_wd2[nid + w + 1], dist);
    } 
    
    nid += w*h;

    if (isInside3D(xInt, yInt, zInt + 1, w, h, l)){
        weight = dx * dy * (1.f - dz);
        
        dist = S2p20(mass * weight);
        atomicAdd(&d_wd[nid],dist);

        dist = S2p20(mass1 * weight);
        atomicAdd(&d_wd1[nid],dist);

        dist = S2p20(mass2 * weight);
        atomicAdd(&d_wd2[nid],dist);
    }
            
    if (isInside3D(xInt + 1, yInt, zInt+1, w, h, l)){
        weight =  (1.f-dx) * dy * (1.f -dz);
        
        dist = S2p20(mass * weight);
        atomicAdd(&d_wd[nid + 1], dist);

        dist = S2p20(mass1 * weight);
        atomicAdd(&d_wd1[nid + 1], dist);
        
        dist = S2p20(mass2 * weight);
        atomicAdd(&d_wd2[nid + 1], dist);
    }
    
    if (isInside3D(xInt, yInt+1, zInt+1, w, h, l)){
        weight =  dx * (1.f - dy) * (1.f -dz);

        dist = S2p20(mass * weight);
        atomicAdd(&d_wd[nid + w], dist);

        dist = S2p20(mass1 * weight);
        atomicAdd(&d_wd1[nid + w], dist);

        dist = S2p20(mass2 * weight);
        atomicAdd(&d_wd2[nid + w], dist);
    }
    
    if (isInside3D(xInt+1, yInt+1, zInt+1, w, h, l)){
        weight = (1.f -dx) * (1.f - dy) * (1.f -dz);

        dist = S2p20(mass * weight);
        atomicAdd(&d_wd[nid + w + 1], dist);

        dist = S2p20(mass1 * weight);
        atomicAdd(&d_wd1[nid + w + 1], dist);

        dist = S2p20(mass2 * weight);
        atomicAdd(&d_wd2[nid + w + 1], dist);
    } 
}

__global__ void atomicSplatPos_kernel(int* d_wd , int w, int h, int l, 
                                      float* d_w,
                                      float* d_px, float* d_py, float* d_pz, int nP)
{
    uint blockId = get_blockID();
    uint id      = get_threadID(blockId);

    if (id < nP){
        float mass = d_w[id];
        float x = d_px[id];
        float y = d_py[id];
        float z = d_pz[id];

        atomicSplat(d_wd, mass, x, y, z, w, h, l);
    }
}




__global__ void atomicSplatPos_kernel(int* d_wd , int* d_wd1, int* d_wd2,
                                      int w, int h, int l, 
                                      float* d_w, float* d_w1,float* d_w2,
                                      float* d_px, float* d_py, float* d_pz, int nP)
{
    uint blockId = get_blockID();
    uint id      = get_threadID(blockId);

    if (id < nP){
        float mass = d_w[id], mass1 = d_w1[id], mass2 = d_w2[id];
                
        float x = d_px[id];
        float y = d_py[id];
        float z = d_pz[id];

        atomicSplat(d_wd, d_wd1, d_wd2,
                    mass, mass1, mass2,
                    x, y, z, w, h, l);
    }
}

////////////////////////////////////////////////////////////////////////////////
// The safe version do the normalization on the data first
// so that the input of the data in the range of [0,1]
// and scale the data back to the original range
////////////////////////////////////////////////////////////////////////////////

template<bool inverse>
__global__ void atomicSplatPos_kernel(int* d_wd , int w, int h, int l, 
                                      float* d_w, float max,
                                      float* d_px, float* d_py, float* d_pz, int nP)
{
    uint blockId = get_blockID();
    uint id      = get_threadID(blockId);

    if (id < nP){
        float mass;
        if (inverse){
            mass = d_w[id] * max; // normalized the mass
        } else {
            mass = d_w[id] / max; // normalized the mass
        }
        float x = d_px[id];
        float y = d_py[id];
        float z = d_pz[id];

        atomicSplat(d_wd, mass, x, y, z, w, h, l);
    }
}

/*
void saveToAfile(uint& sizeX, uint& sizeY, uint& sizeZ, uint& nP,
                 float* h_i, float* h_px, float* h_py, float* h_pz)
{
    ofstream fo("save.dat",ofstream::binary);

    fo.write((char*)&sizeX, sizeof(int));
    fo.write((char*)&sizeY, sizeof(int));
    fo.write((char*)&sizeZ, sizeof(int));
    fo.write((char*)&nP, sizeof(int));
    
    fo.write((char*)h_i, sizeX * sizeY * sizeZ * sizeof(float));
    fo.write((char*)h_px, nP * sizeof(float));
    fo.write((char*)h_py, nP * sizeof(float));
    fo.write((char*)h_pz, nP * sizeof(float));

    fo.close();
}

void readFromAfile(int& sizeX, int& sizeY, int& sizeZ, int& nP,
                   float*& h_i, float*& h_px, float*& h_py, float*& h_pz)
{
    ifstream fi("save.dat",ifstream::binary);

    fi.read((char*)&sizeX, sizeof(int));
    fi.read((char*)&sizeY, sizeof(int));
    fi.read((char*)&sizeZ, sizeof(int));
    fi.read((char*)&nP, sizeof(int));

    std::cout << "Size of the image [" << sizeX << "," << sizeY << "," << sizeZ << "]  " << std::endl;
    std::cout << "Number of point " << nP << std::endl;
    
    h_i = new float [sizeX * sizeY * sizeZ];
    h_px= new float [nP];
    h_py= new float [nP];
    h_pz= new float [nP];
    
    fi.read((char*)h_i, sizeX * sizeY * sizeZ * sizeof(float));
    fi.read((char*)h_px, nP * sizeof(float));
    fi.read((char*)h_py, nP * sizeof(float));
    fi.read((char*)h_pz, nP * sizeof(float));

    fi.close();
}
*/

#include <cudaInterface.h>
void cplSplat3D(float* d_wd, uint sizeX, uint sizeY, uint sizeZ,
                float* d_w,  float* d_px , float* d_py, float* d_pz, uint nP, cudaStream_t stream)
{
    // init accumulate array 0
    uint gridSize = sizeX * sizeY * sizeZ;
    int* d_iwd   = (int*) d_wd;   // unsigned version of output

    cplVectorOpers::SetMem(d_iwd, 0, gridSize, stream);

    // splating
    dim3 threads(256);
    dim3 grids(iDivUp(nP, threads.x));
    checkConfig(grids);
    
    atomicSplatPos_kernel<<<grids, threads, 0, stream>>>(d_iwd, sizeX, sizeY, sizeZ,
                                                         d_w, d_px, d_py, d_pz, nP);

    // convert to float buffer
    cplVectorOpers::FixedToFloating(d_wd, d_iwd, gridSize, stream);
}

void cplSplat3D(cplVector3DArray& d_o, const Vector3Di& size,
                 cplVector3DArray& d_i, cplVector3DArray& d_pos, uint nP, cudaStream_t stream)
{
    cplSplat3D(d_o.x, size.x, size.y, size.z, 
                d_i.x, d_pos.x, d_pos.y, d_pos.z, nP, stream);

    cplSplat3D(d_o.y, size.x, size.y, size.z, 
                d_i.y, d_pos.x, d_pos.y, d_pos.z, nP, stream);

    cplSplat3D(d_o.z, size.x, size.y, size.z, 
                d_i.z, d_pos.x, d_pos.y, d_pos.z, nP, stream);
}


void cplSplat3D(cplReduce& rd,
                 float* d_wd, uint sizeX, uint sizeY, uint sizeZ,
                 float* d_w,
                 float* d_px , float* d_py, float* d_pz, uint nP, cudaStream_t stream)
{
    uint gridSize = sizeX * sizeY * sizeZ;
    // compute the range
    float maxV = rd.Max(d_w, nP);
    int* d_iwd = (int*) d_wd;

    // Initialize the result with Zero

    cplVectorOpers::SetMem(d_iwd, 0, gridSize, stream);

    // splating
    dim3 threads(256);
    dim3 grids(iDivUp(nP, threads.x));
    checkConfig(grids);
    // Compute the index and the mass contribution of each mapping point
    atomicSplatPos_kernel<true><<<grids, threads, 0, stream>>>(d_iwd, sizeX, sizeY, sizeZ,
                                                               d_w, 1.f / maxV,
                                                               d_px, d_py, d_pz, nP);

    
    // convert to float buffer
    cplVectorOpers::FixedToFloatingUnnomalized(d_wd, d_iwd, maxV, gridSize);

}

void cplSplat3D(cplReduce& rd,
                float* d_wd, const Vector3Di& size,
                float* d_w, cplVector3DArray& d_pos, uint nP, cudaStream_t stream)
{
    cplSplat3D(rd, d_wd, size.x, size.y , size.z,
                d_w, d_pos.x, d_pos.y, d_pos.z, nP, stream);
}


void cplSplat3DH(float* d_wd, float* d_w,
                  float* d_px , float* d_py, float* d_pz,
                  uint sizeX, uint sizeY, uint sizeZ, cudaStream_t stream )
{
    cplSplat3D(d_wd, sizeX, sizeY, sizeZ,
               d_w, d_px, d_py, d_pz, sizeX * sizeY * sizeZ, stream);
}

void cplSplat3DH(cplReduce& rd, 
                  float* d_wd, float* d_w,
                  float* d_px , float* d_py, float* d_pz,
                  uint sizeX, uint sizeY, uint sizeZ, cudaStream_t stream )
{
    cplSplat3D(rd, d_wd, sizeX, sizeY, sizeZ,
               d_w, d_px, d_py, d_pz, sizeX * sizeY * sizeZ, stream);
}

void cplSplat3DH(float* d_dst, float* d_src,
                 cplVector3DArray& d_p, const Vector3Di& size, cudaStream_t stream)
{
    cplSplat3D(d_dst, size.x, size.y, size.z,
               d_src, d_p.x, d_p.y, d_p.z, size.productOfElements(), stream);
    
}

void cplSplat3DH(cplReduce& rd, float* d_dst, float* d_src,
                 cplVector3DArray& d_p, const Vector3Di& size, cudaStream_t stream)
{
    cplSplat3D(rd, d_dst, size.x, size.y, size.z,
               d_src, d_p.x, d_p.y, d_p.z, size.productOfElements(), stream);
}


void cpuSplat3D(float* h_dst, uint w, uint h, uint l,
                float* h_src, float* h_px , float* h_py, float* h_pz, uint nP)
{
    uint size = w * h * l;
    double* h_dst_d = new double [size];
    for (int i=0; i< size; ++i)
        h_dst_d[i] = 0.0;

    for (int i=0; i< nP; ++i)
    {
        float x = h_px[i];
        float y = h_py[i];
        float z = h_pz[i];
        
        int xint = int(x);
        int yint = int(y);
        int zint = int(z);

#if ADJUST_INT        
        if (x < 0 && x != xint) --xint;
        if (y < 0 && y != yint) --yint;
        if (z < 0 && z != zint) --zint;
#endif

        float dx = 1.f - (x - xint);
        float dy = 1.f - (y - yint);
        float dz = 1.f - (z - zint);
 
        float mass = h_src[i];
        int new_id = xint + yint * w + zint * w * h;

        if (isInside3D(xint, yint, zint, w, h, l))
            h_dst_d[new_id] += mass * dx * dy * dz;
        
        if (isInside3D(xint + 1, yint, zint, w, h, l))
            h_dst_d[new_id+1] += mass * (1.f-dx) * dy * dz;

        if (isInside3D(xint, yint+1, zint, w, h, l))
            h_dst_d[new_id+w] += mass * dx * (1.f - dy) * dz;
        
        if (isInside3D(xint+1, yint+1, zint, w, h, l))
            h_dst_d[new_id+w+1] += mass * (1.f -dx) * (1.f - dy) * dz;
        
        if (isInside3D(xint, yint, zint+1, w, h, l))
            h_dst_d[new_id + w * h] += mass * dx * dy * (1.f-dz);
        
        if (isInside3D(xint + 1, yint, zint+1, w, h, l))
            h_dst_d[new_id+1 + w * h] += mass * (1.f-dx) * dy * (1.f-dz);
        
        if (isInside3D(xint, yint+1, zint+1, w, h, l))
            h_dst_d[new_id+w + w * h] += mass * dx * (1.f - dy) * (1.f-dz);
        
        if (isInside3D(xint+1, yint+1, zint+1, w, h, l))
            h_dst_d[new_id+w+1 + w * h] += mass * (1.f -dx) * (1.f - dy) * (1.f-dz);
        
    }

    for (int i=0; i< size; ++i)
        h_dst[i] = h_dst_d[i];

    delete []h_dst_d;
}


__global__ void atomicVelocitySplat_kernel(int* d_wd, float* d_w,
                                           float* vx, float* vy, float* vz,
                                           int w, int h, int l)
{
    const uint wh     = w * h;

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i < w && j < h){
        uint id       = i + j * w;
        for (int k=0; k < l; ++k, id+=wh) {
            float mass = d_w[id];
            
            float x = i + vx[id];
            float y = j + vy[id];
            float z = k + vz[id];

            atomicSplat(d_wd, mass, x, y, z, w, h, l);
        }
    }
}

__global__ void atomicVelocitySplat_kernel(int  * d_wd, float* d_w,
                                           float* d_ux, float* d_uy, float* d_uz,
                                           int w, int h, int l,
                                           float sx, float sy, float sz)
{
    const uint wh     = w * h;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i < w && j < h){
        uint id       = i + j * w;
        for (int k=0; k < l; ++k, id+=wh) {
            float mass = d_w[id];
            float x = d_ux[id] / sx + i;
            float y = d_uy[id] / sy + j;
            float z = d_uz[id] / sz + k;

            atomicSplat(d_wd, mass, x, y, z, w, h, l);
        }
    }
}

__global__ void atomicVelocitySplat_kernel_shared(int* d_wd, float* d_w,
                                                  float* vx, float* vy, float* vz,
                                                  int w, int h, int l)
{
    __shared__ int s_0[16*16];
    __shared__ int s_1[16*16];
    __shared__ int s_2[16*16];

    const uint wh     = w * h;

    int xc = blockIdx.x * blockDim.x;
    int yc = blockIdx.y * blockDim.y;
    
    int i  = xc + threadIdx.x;
    int j  = yc + threadIdx.y;
    
    if (i < w && j < h){
        uint id       = i + j * w;
        
        s_0[threadIdx.y * blockDim.x + threadIdx.x] = 0;
        s_1[threadIdx.y * blockDim.x + threadIdx.x] = 0;
        int* s_p = s_0, *s_c = s_1, *s_n = s_2;

        for (int k=0; k < l; ++k, id+=wh) {
            // Initialize the new buffer with zero 
            s_n[threadIdx.y * blockDim.x + threadIdx.x] = 0;

            //__syncthreads();

            float mass = d_w[id];
            
            float x = i + vx[id];
            float y = j + vy[id];
            float z = k + vz[id];

            int xInt = int(x);
            int yInt = int(y);
            int zInt = int(z);

#if ADJUST_INT
            if (x < 0 && x != xInt) --xInt;
            if (y < 0 && y != yInt) --yInt;
            if (z < 0 && z != zInt) --zInt;
#endif
            
            float dx = 1.f - (x - xInt);
            float dy = 1.f - (y - yInt);
            float dz = 1.f - (z - zInt);

            uint new_id = (zInt * h + yInt) * w + xInt;
            int dist;
            
            if (isInside3D(xInt - xc, yInt - yc, zInt + 1 - k,
                           blockDim.x-1,  blockDim.y-1, 2))
            {
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
                    atomicAdd(&d_wd[new_id],dist);

                    dist = S2p20(mass * (1.f-dx) * dy * dz);
                    atomicAdd(&d_wd[new_id + 1], dist);

                    dist = S2p20(mass * dx * (1.f - dy) * dz);
                    atomicAdd(&d_wd[new_id + w], dist);

                    dist = S2p20(mass * (1.f -dx) * (1.f - dy) * dz);
                    atomicAdd(&d_wd[new_id + w + 1], dist);

                    new_id += w*h;

                    dist = S2p20(mass * dx * dy * (1.f - dz));
                    atomicAdd(&d_wd[new_id],dist);

                    dist = S2p20(mass * (1.f-dx) * dy * (1.f -dz));
                    atomicAdd(&d_wd[new_id + 1], dist);

                    dist = S2p20(mass * dx * (1.f - dy) * (1.f -dz));
                    atomicAdd(&d_wd[new_id + w], dist);

                    dist = S2p20(mass * (1.f -dx) * (1.f - dy) * (1.f -dz));
                    atomicAdd(&d_wd[new_id + w + 1], dist);
                }
#else
            {
                if (isInside3D(xInt, yInt, zInt, w, h, l)){
                    dist = S2p20(mass * dx * dy * dz);
                    atomicAdd(&d_wd[new_id],dist);
                }
            
                if (isInside3D(xInt + 1, yInt, zInt, w, h, l)){
                    dist = S2p20(mass * (1.f-dx) * dy * dz);
                    atomicAdd(&d_wd[new_id + 1], dist);
                }
                if (isInside3D(xInt, yInt+1, zInt, w, h, l)){
                    dist = S2p20(mass * dx * (1.f - dy) * dz);
                    atomicAdd(&d_wd[new_id + w], dist);
                }
                if (isInside3D(xInt+1, yInt+1, zInt, w, h, l)){
                    dist = S2p20(mass * (1.f -dx) * (1.f - dy) * dz);
                    atomicAdd(&d_wd[new_id + w + 1], dist);
                } 
                new_id += w*h;
                if (isInside3D(xInt, yInt, zInt + 1, w, h, l)){
                    dist = S2p20(mass * dx * dy * (1.f - dz));
                    atomicAdd(&d_wd[new_id],dist);
                }
                if (isInside3D(xInt + 1, yInt, zInt+1, w, h, l)){
                    dist = S2p20(mass * (1.f-dx) * dy * (1.f -dz));
                    atomicAdd(&d_wd[new_id + 1], dist);
                }

                if (isInside3D(xInt, yInt+1, zInt+1, w, h, l)){
                    dist = S2p20(mass * dx * (1.f - dy) * (1.f -dz));
                    atomicAdd(&d_wd[new_id + w], dist);
                }
                if (isInside3D(xInt+1, yInt+1, zInt+1, w, h, l)){
                    dist = S2p20(mass * (1.f -dx) * (1.f - dy) * (1.f -dz));
                    atomicAdd(&d_wd[new_id + w + 1], dist);
                }
            }
#endif
            __syncthreads();
            
            //write out the previous layer 
            if( k > 0){
                atomicAdd(&d_wd[id - wh], s_p[threadIdx.x + threadIdx.y * 16]);
            }

            //write out the current layer if it is the last 
            if ( k == l - 1){
                atomicAdd(&d_wd[id], s_c[threadIdx.x + threadIdx.y * 16]);
            }
            
            int* temp = s_p;
            s_p = s_c;
            s_c = s_n;
            s_n = temp;
        }
    }
}



void cplSplat3DV(float* d_wd, float* d_w,
                 float* d_vx, float* d_vy, float* d_vz,
                 uint sizeX, uint sizeY, uint sizeZ, cudaStream_t stream)
{
    int* d_iwd = (int*) d_wd; // unsigned version of
    uint nElems  = sizeX * sizeY * sizeZ;
    
    // Initialize the result with Zero
    cplVectorOpers::SetMem(d_iwd, 0, nElems, stream);

    dim3 threads(16,16);
    dim3 grids(iDivUp(sizeX, threads.x), iDivUp(sizeY, threads.y));

    // Compute the index and the mass contribution of each mapping point
    atomicVelocitySplat_kernel<<<grids, threads, 0, stream>>>(d_iwd, d_w,
                                                              d_vx, d_vy, d_vz,
                                                              sizeX, sizeY, sizeZ);

    // Convert back from fix point to floating point
    cplVectorOpers::FixedToFloating(d_wd, d_iwd, nElems, stream);
}


void cplSplat3DV(float* d_dst, float* d_src,
                 cplVector3DArray& d_v, const Vector3Di& size, cudaStream_t stream)
{
    cplSplat3DV(d_dst, d_src,
                d_v.x, d_v.y, d_v.z,
                size.x, size.y, size.z, stream);
}


void cplSplat3DV(float* d_wd, float* d_w,
                 float* d_vx, float* d_vy, float* d_vz,
                 uint sizeX, uint sizeY, uint sizeZ,
                 float spX, float spY, float spZ, 
                 cudaStream_t stream)
{
    int* d_iwd = (int*) d_wd; // unsigned version of
    uint nElems  = sizeX * sizeY * sizeZ;
    
    // Initialize the result with Zero
    cplVectorOpers::SetMem(d_iwd, 0, nElems, stream);

    dim3 threads(16,16);
    dim3 grids(iDivUp(sizeX, threads.x), iDivUp(sizeY, threads.y));

    // Compute the index and the mass contribution of each mapping point
    atomicVelocitySplat_kernel<<<grids, threads, 0, stream>>>(d_iwd, d_w,
                                                              d_vx, d_vy, d_vz,
                                                              sizeX, sizeY, sizeZ,
                                                              spX, spY, spZ);
    // Convert back from fix point to floating point
    cplVectorOpers::FixedToFloating(d_wd, d_iwd, nElems, stream);
}

void cplSplat3DV(float* d_dst, float* d_src,
                 cplVector3DArray& d_v, const Vector3Di& size, const Vector3Df& sp, cudaStream_t stream)
    
{
    cplSplat3DV(d_dst, d_src,
                d_v.x, d_v.y, d_v.z,
                size.x, size.y, size.z,
                sp.x, sp.y, sp.z, stream);
}

void cplSplat3DV_shared(float* d_wd, float* d_w,
                         float* d_vx , float* d_vy, float* d_vz,
                         uint sizeX, uint sizeY, uint sizeZ, cudaStream_t stream)
{
    
    int* d_iwd = (int*) d_wd; // unsigned version of
    uint nElems  = sizeX * sizeY * sizeZ;
    
    // Initialize the result with Zero
    cplVectorOpers::SetMem(d_iwd, 0, nElems, stream);

    dim3 threads(16,16);
    dim3 grids(iDivUp(sizeX, threads.x), iDivUp(sizeY, threads.y));
    
    // Compute the index and the mass contribution of each mapping point
    atomicVelocitySplat_kernel_shared<<<grids, threads, 0, stream>>>(d_iwd, d_w,
                                                                     d_vx, d_vy, d_vz,
                                                                     sizeX, sizeY, sizeZ);

    cplVectorOpers::FixedToFloating(d_wd, d_iwd, nElems, stream);
}


