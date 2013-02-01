#include <cpl.h>

#include <cudaDownSample.h>
#include <cudaVector3DArray.h>
#include <cudaDownsizeFilter3D.h>

#include <cudaRecursiveGaussianFilter.h>
#include <cudaGaussianFilter.h>
#include <cudaImage3D.h>

void computeDSFilterParams(Vector3Df& sigma, Vector3Di& kRadius, int f)
{
    float sig = sqrt(((float)f)/2.0);
    int   kR  = 2*static_cast<int>(std::ceil(sig));

    sigma   = Vector3Df(sig, sig, sig);
    kRadius = Vector3Di(kR, kR, kR);
}

void computeDSFilterParams(Vector3Df& sigma, Vector3Di& kRadius, const Vector3Di& factor)
{
    sigma   = Vector3Df (sqrt(factor.x/2.0), sqrt(factor.y/2.0), sqrt(factor.z/2.0));
    kRadius = Vector3Di(2*static_cast<int>(std::ceil(sigma.x)),
                        2*static_cast<int>(std::ceil(sigma.y)),
                        2*static_cast<int>(std::ceil(sigma.z)));
}


template<class SmoothFilter>
void cplDownSample(float* d_o, float* d_i, const Vector3Di& size,
                    SmoothFilter* smFilter, cplDownsizeFilter3D* dsFilter,
                    float* d_temp0, float* d_temp1, cudaStream_t stream)
{
    unsigned int nElems = size.productOfElements();
    
    bool b0 = (d_temp0 == NULL);
    bool b1 = (d_temp1 == NULL);
    
    if (b0) dmemAlloc( d_temp0, nElems);
    if (b1) dmemAlloc( d_temp1, nElems);

    smFilter->filter(d_temp0, d_i, size, d_temp1, stream);
    dsFilter->filter(d_o, d_temp0, true, stream);
   
    if (b0) dmemFree(d_temp0);
    if (b1) dmemFree(d_temp1);
}

template void cplDownSample(float* d_o, float* d_i, const Vector3Di& size,
                             cplGaussianFilter* smFilter, cplDownsizeFilter3D* dsFilter,
                             float* d_temp0, float* d_temp1, cudaStream_t stream);

template void cplDownSample(float* d_o, float* d_i, const Vector3Di& size,
                             cplRGFilter* smFilter, cplDownsizeFilter3D* dsFilter,
                             float* d_temp0, float* d_temp1, cudaStream_t stream);

template<class SmoothFilter>
void cplDownSample(cplVector3DArray& d_o, cplVector3DArray& d_i, const Vector3Di& size,
                    SmoothFilter* smFilter, cplDownsizeFilter3D* dsFilter,
                    float* d_temp0, float* d_temp1, cudaStream_t stream)
{
    unsigned int nElems = size.productOfElements();
    
    bool b0 = (d_temp0 == NULL);
    bool b1 = (d_temp1 == NULL);
    if (b0) dmemAlloc( d_temp0, nElems);
    if (b1) dmemAlloc( d_temp1, nElems);

    smFilter->filter(d_temp0, d_i.x, size, d_temp1, stream);
    dsFilter->filter(d_o.x, d_temp0, true, stream);

    smFilter->filter(d_temp0, d_i.y, size, d_temp1, stream);
    dsFilter->filter(d_o.y, d_temp0, true, stream);

    smFilter->filter(d_temp0, d_i.z, size, d_temp1, stream);
    dsFilter->filter(d_o.z, d_temp0, true, stream);

    if (b0) dmemFree(d_temp0);
    if (b1) dmemFree(d_temp1);
}

template void cplDownSample(cplVector3DArray& d_o, cplVector3DArray& d_i, const Vector3Di& size,
                             cplGaussianFilter* smFilter, cplDownsizeFilter3D* dsFilter,
                             float* d_temp0, float* d_temp1, cudaStream_t stream);

template void cplDownSample(cplVector3DArray& d_o, cplVector3DArray& d_i, const Vector3Di& size,
                             cplRGFilter* smFilter, cplDownsizeFilter3D* dsFilter,
                             float* d_temp0, float* d_temp1, cudaStream_t stream);


template<class SmoothFilter>
void cplDownSample(float* d_o, float* d_i, const Vector3Di& osize, const Vector3Di& isize,
                   SmoothFilter* smFilter, float* d_temp0, float* d_temp1, cudaStream_t stream)
{
    unsigned int nElems = isize.productOfElements();
    
    bool b0 = (d_temp0 == NULL);
    bool b1 = (d_temp1 == NULL);
    
    if (b0) dmemAlloc( d_temp0, nElems);
    if (b1) dmemAlloc( d_temp1, nElems);

    smFilter->filter(d_temp0, d_i, isize, d_temp1, stream);
    cplResample(d_o, d_temp0, osize, isize, BACKGROUND_STRATEGY_CLAMP, true);
    
    if (b0) dmemFree(d_temp0);
    if (b1) dmemFree(d_temp1);
}

template void cplDownSample(float* d_o, float* d_i, const Vector3Di& osize, const Vector3Di& isize,
                            cplGaussianFilter* smFilter, float* d_temp0, float* d_temp1, cudaStream_t stream);

template void cplDownSample(float* d_o, float* d_i, const Vector3Di& osize, const Vector3Di& isize,
                            cplRGFilter* smFilter, float* d_temp0, float* d_temp1, cudaStream_t stream);


template<class SmoothFilter>
void cplDownSample(cplVector3DArray& d_o, cplVector3DArray& d_i,
                   const Vector3Di& osize, const Vector3Di& isize,
                   SmoothFilter* smFilter, float* d_temp0, float* d_temp1, cudaStream_t stream)
{
    unsigned int nElems = isize.productOfElements();
    
    bool b0 = (d_temp0 == NULL);
    bool b1 = (d_temp1 == NULL);
    if (b0) dmemAlloc( d_temp0, nElems);
    if (b1) dmemAlloc( d_temp1, nElems);

    smFilter->filter(d_temp0, d_i.x, isize, d_temp1, stream);
    cplResample(d_o.x, d_temp0, osize, isize, BACKGROUND_STRATEGY_CLAMP, true);

    smFilter->filter(d_temp0, d_i.y, isize, d_temp1, stream);
    cplResample(d_o.y, d_temp0, osize, isize, BACKGROUND_STRATEGY_CLAMP, true);

    smFilter->filter(d_temp0, d_i.z, isize, d_temp1, stream);
    cplResample(d_o.z, d_temp0, osize, isize, BACKGROUND_STRATEGY_CLAMP, true);

    if (b0) dmemFree(d_temp0);
    if (b1) dmemFree(d_temp1);
}

template void cplDownSample(cplVector3DArray& d_o, cplVector3DArray& d_i,
                            const Vector3Di& osize, const Vector3Di& isize,
                            cplGaussianFilter* smFilter, float* d_temp0, float* d_temp1, cudaStream_t stream);

template void cplDownSample(cplVector3DArray& d_o, cplVector3DArray& d_i,
                            const Vector3Di& osize, const Vector3Di& isize,
                            cplRGFilter* smFilter, float* d_temp0, float* d_temp1, cudaStream_t stream);

