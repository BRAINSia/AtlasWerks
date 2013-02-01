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

#include "cudaDownsizeFilter3D.h"
#include <cutil_comfunc.h>
#include <cudaTexFetch.h>
#include <cudaVector3DArray.h>

__device__ int getId3D(int x, int y, int z,
                       int sizeX, int sizeY, int sizeZ)
{
    return x + (y + z * sizeY) * sizeX;
}

template<bool cache>
__global__ void cplDownsizeFilter3D_kernel(float* d_o , float* d_i,
                                            int  osizeX, int osizeY , int osizeZ,
                                            int  isizeX, int  isizeY, int  isizeZ,
                                            int  fX    , int  fY    , int  fZ)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x < osizeX && y < osizeY){
        int oId    = x + y * osizeX;

        int oPlane = osizeX * osizeY;
        int iPlane = isizeX * isizeY;

        int iX     = x * fX;
        int iY     = y * fY;
        int iZ     = 0;

        int iId    = getId3D(iX, iY, iZ, isizeX, isizeY, isizeZ);

        for (int z=0; z< osizeZ; ++z){
            if (cache)
                d_o[oId] = fetch(iId, (float*)NULL);
            else
                d_o[oId] = d_i[iId];
            
            oId += oPlane;         // advance output
            iId += iPlane * fZ;    // advance input by a factor of fZ
        }
    }
}

void cplDownsizeFilter3D::computeOutputSize()
{
    m_osize.x = m_isize.x / m_f.x;
    m_osize.y = m_isize.y / m_f.y;
    m_osize.z = m_isize.z / m_f.z;
}

void cplDownsizeFilter3D::filter(float* d_o, float* d_i, bool cache, cudaStream_t stream)
{
    dim3 threads(16,16);
    dim3 grids(iDivUp(m_osize.x, threads.x), iDivUp(m_osize.y, threads.y));

    if (cache) {
        cache_bind(d_i);
        cplDownsizeFilter3D_kernel<true><<<grids, threads, 0, stream>>>(d_o, d_i,
                                                                        m_osize.x, m_osize.y, m_osize.z,
                                                                        m_isize.x, m_isize.y, m_isize.z,
                                                                        m_f.x, m_f.y, m_f.z);
    }
    else {
        cplDownsizeFilter3D_kernel<false><<<grids, threads, 0, stream>>>(d_o, d_i,
                                                                         m_osize.x, m_osize.y, m_osize.z,
                                                                         m_isize.x, m_isize.y, m_isize.z,
                                                                         m_f.x, m_f.y, m_f.z);
    }
}

void cplDownsizeFilter3D::filter(cplVector3DArray& d_o, cplVector3DArray& d_i, bool cache, cudaStream_t stream)
{
    this->filter(d_o.x, d_i.x, cache, stream);
    this->filter(d_o.y, d_i.y, cache, stream);
    this->filter(d_o.z, d_i.z, cache, stream);
}

void cplDownsizeFilter3D::printInfo()
{
    std::cout << "Downsize filter "
              << " Input size " << m_isize 
              << " Downsize factor  " << m_f 
              << " Output size " << m_osize << std::endl;
}
