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
#include <cudaRecursiveGaussianFilter.h>
#include <cudaTranspose.h>

void cplRGFilter::init(float sigma, int order)
{
    m_sigma = sigma;
    m_order = order;
    
    // compute filter coefficients
    const float
        nsigma = sigma < 0.1f ? 0.1f : sigma,
        alpha = 1.695f / nsigma,
        ema = (float)std::exp(-alpha),
        ema2 = (float)std::exp(-2*alpha);
    
        
    b1 = -2*ema, b2 = ema2;
    a0 = 0, a1 = 0, a2 = 0, a3 = 0, coefp = 0, coefn = 0;
    switch (order) {
        case 0: {
            const float k = (1-ema)*(1-ema)/(1+2*alpha*ema-ema2);
            a0 = k;
            a1 = k*(alpha-1)*ema;
            a2 = k*(alpha+1)*ema;
            a3 = -k*ema2;
        } break;

        case 1: {
            const float k = (1-ema)*(1-ema)/ema;
            a0 = k*ema;
            a1 = a3 = 0;
            a2 = -a0;
        } break;

        case 2: {
            const float
                ea = (float)std::exp(-alpha),
                k = -(ema2-1)/(2*alpha*ema),
                kn = (-2*(-1+3*ea-3*ea*ea+ea*ea*ea)/(3*ea+1+3*ea*ea+ea*ea*ea));
            a0 = kn;
            a1 = -kn*(1+k*alpha)*ema;
            a2 = kn*(1-k*alpha)*ema;
            a3 = -kn*ema2;
        } break;

        default:
            fprintf(stderr, "gaussianFilter: invalid order parameter!\n");
            return;
    }
    coefp = (a0+a1)/(1+b1+b2);
    coefn = (a2+a3)/(1+b1+b2);
}

////////////////////////////////////////////////////////////////////////////////
//
////////////////////////////////////////////////////////////////////////////////

template<class T, bool clampToEdge>
__global__ void RGFFilter2D_kernel(T* d_o, T* d_i,
                                   int sizeX, int sizeY, 
                                   float a0, float a1, float a2, float a3, float b1, float b2, float coefp, float coefn)
{
    uint x = blockIdx.x*blockDim.x + threadIdx.x;
    if ((x >= sizeX))
        return;

    d_o += x;
    d_i += x;
    
    T xp = (T)0; // previous input
    T yp = (T)0; // previous output
    T yb = (T)0; // previous output by 2
    
    if (clampToEdge){
        xp = *d_i; yb = coefp*xp; yp = yb;
    }

    for (int y = 0; y < sizeY; y++) {
        float xc = *d_i;
        float yc = a0*xc + a1*xp - b1*yp - b2*yb;
		*d_o = yc;

        //shifting around input output 
        xp = xc; yb = yp; yp = yc;

        // move to the next row
        d_i += sizeX; d_o += sizeX;    // move to next rosizeX
    }

    // reset pointers to point to last element in column
    d_i -= sizeX;
    d_o -= sizeX;

    // reverse pass
    // ensures response is symmetrical
    float xn =0.0f, xa = 0.0f, yn = 0.0f, ya = 0.0f;
    
    if (clampToEdge){
        xn = xa = *d_i; yn = coefn*xn; ya = yn;
    }
    
    for (int y = sizeY-1; y >= 0; y--) {
        float xc = *d_i;
        float yc = a2*xn + a3*xa - b1*yn - b2*ya;
        
		*d_o = *d_o +  yc;

        //shifting around input output 
        xa = xn; xn = xc; ya = yn; yn = yc;
        
        d_o -= sizeX; d_i -= sizeX;  // move to previous row
    }
}


void cplRGFilter::filter(float* d_o, float* d_i, int sizeX, int sizeY, float* d_temp, cudaStream_t stream)
{
    dim3 threads(64);
    dim3 grids(iDivUp(sizeX, 64));
    RGFFilter2D_kernel<float, true><<<grids,threads, 0, stream>>>(d_temp, d_i,
                                                                  sizeX, sizeY,
                                                                  a0, a1, a2, a3,
                                                                  b1, b2, coefp, coefn);
    transpose(d_o, d_temp, sizeX, sizeY, stream);
            
    grids.x = iDivUp(sizeY, 64);
    RGFFilter2D_kernel<float, true><<<grids,threads, 0, stream>>>(d_temp, d_o,
                                                                  sizeY, sizeX,
                                                                  a0, a1, a2, a3,
                                                                  b1, b2, coefp, coefn);
    transpose(d_o, d_temp, sizeY, sizeX, stream);
}

////////////////////////////////////////////////////////////////////////////////
//
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
//
////////////////////////////////////////////////////////////////////////////////
template<class T, bool clampToEdge>
__global__ void RGFFilter3D_kernel(T* d_o, T* d_i,
                                   int sizeX, int sizeY, int sizeZ,
                                   float a0, float a1, float a2, float a3, float b1, float b2, float coefp, float coefn)
{
    uint x = blockIdx.x*blockDim.x + threadIdx.x;
    uint y = blockIdx.y*blockDim.y + threadIdx.y;

    if ((x >= sizeX) || (y >= sizeY))
        return;

    uint id = x + y * sizeX;
    const uint planeSize = sizeX * sizeY;

    d_o += id;
    d_i += id;
    
    T xp = (T)0; // previous input
    T yp = (T)0; // previous output
    T yb = (T)0; // previous output by 2
    
    if (clampToEdge){
        xp = *d_i; yb = coefp*xp; yp = yb;
    }

    for (int z = 0; z < sizeZ; z++) {
        T xc = *d_i;
        T yc = a0*xc + a1*xp - b1*yp - b2*yb;
		*d_o = yc;

        //shifting around input output 
        xp = xc; yb = yp; yp = yc;

        // move to next plane
        d_i += planeSize;
        d_o += planeSize;    
    }

    // reset pointers to point to last element in column
    d_i -= planeSize;
    d_o -= planeSize;

    // reverse pass
    // ensures response is symmetrical
    T xn = (T)(0.0f);
    T xa = (T)(0.0f);
    T yn = (T)(0.0f);
    T ya = (T)(0.0f);
    
    if (clampToEdge){
        xn = xa = *d_i; yn = coefn*xn; ya = yn;
    }
        
    for (int z = sizeZ-1; z >= 0; z--) {
        T xc = *d_i;
        T yc = a2*xn + a3*xa - b1*yn - b2*ya;
        *d_o = *d_o + yc;

        //shifting around input output 
        xa = xn;
        xn = xc;
        ya = yn;
        yn = yc;

        // move to previous plane
        d_i -= planeSize;
        d_o -= planeSize;  
    }
}

void cplRGFilter::filter_impl(float* d_o, float* d_i,
                               int sizeX, int sizeY, int sizeZ,
                               float* d_temp, cudaStream_t stream)
{
    
    dim3 threads(16,16);
    dim3 grids(iDivUp(sizeX, threads.x),iDivUp(sizeY, threads.y));
    RGFFilter3D_kernel<float, true><<<grids,threads, 0, stream>>>(d_temp, d_i,
                                                                  sizeX, sizeY, sizeZ,
                                                                  a0, a1, a2, a3,
                                                                  b1, b2, coefp, coefn);
    
    cplShiftCoordinate(d_o, d_temp, sizeX, sizeY, sizeZ, 1, stream);

    grids.x = iDivUp(sizeZ, threads.x);
    grids.y = iDivUp(sizeX, threads.y);
    RGFFilter3D_kernel<float, true><<<grids,threads, 0, stream>>>(d_temp, d_o,
                                                                  sizeZ, sizeX, sizeY,
                                                                  a0, a1, a2, a3,
                                                                  b1, b2, coefp, coefn);
    cplShiftCoordinate(d_o, d_temp, sizeZ, sizeX, sizeY, 1, stream);

    
    grids.x = iDivUp(sizeY, threads.x);
    grids.y = iDivUp(sizeZ, threads.y);
    RGFFilter3D_kernel<float, true><<<grids,threads, 0, stream>>>(d_temp, d_o,
                                                                  sizeY, sizeZ, sizeX,
                                                                  a0, a1, a2, a3,
                                                                  b1, b2, coefp, coefn);
    cplShiftCoordinate(d_o, d_temp, sizeY, sizeZ, sizeX, 1, stream);
}

void cplRGFilter::filter(float* d_o, float* d_i,
                          const Vector3Di& size, float* d_temp, cudaStream_t stream)
{
    unsigned int nElems = size.productOfElements();
    bool need_temp = (d_temp == NULL);
    if (need_temp)
        dmemAlloc(d_temp, nElems);

    filter_impl(d_o, d_i, size.x, size.y, size.z, d_temp, stream);

    if (need_temp)
        dmemFree(d_temp);
}

void cplRGFilter::filter(cplVector3DArray& d_o, cplVector3DArray& d_i,
                          const Vector3Di& size, float* d_temp, cudaStream_t stream)
{
    unsigned int nElems = size.productOfElements();
    bool need_temp = (d_temp == NULL);
    if (need_temp)
        dmemAlloc(d_temp, nElems);

    filter_impl(d_o.x, d_i.x, size.x, size.y, size.z, d_temp, stream);
    filter_impl(d_o.y, d_i.y, size.x, size.y, size.z, d_temp, stream);
    filter_impl(d_o.z, d_i.z, size.x, size.y, size.z, d_temp, stream);

    if (need_temp)
        dmemFree(d_temp);
}

