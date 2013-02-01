#include <cpl.h>
#include <cudaGaussianFilter.h>
#include <cudaDownsizeFilter3D.h>
#include <cudaSeparableGaussFilter.h>
#include <cudaTranspose.h>
#include <nddrUtils.h>

cplGaussianFilter::cplGaussianFilter()
    :m_sigma(1.f, 1.f, 1.f), m_kRadius(1, 1, 1)
{

}

void cplGaussianFilter::init(const Vector3Di& size, const Vector3Df& sigma, const Vector3Di& kRadius)
{
    m_sigma   = sigma;
    m_kRadius = kRadius;
    
    // adjust the kernel size if needed
    if (m_kRadius.x > size.x/2 - 1)
        m_kRadius.x = size.x/2 - 1;

    if (m_kRadius.y > size.y/2 - 1)
        m_kRadius.y = size.y/2 - 1;

    if (m_kRadius.z > size.z/2 - 1)
        m_kRadius.z = size.z/2 - 1;
    
    Vector3Di kLength = Vector3Di(m_kRadius.x * 2 + 1,
                                  m_kRadius.y * 2 + 1,
                                  m_kRadius.z * 2 + 1);
    // generate the kernel
    float* h_kX = new float [kLength.x];
    float* h_kY = new float [kLength.y];
    float* h_kZ = new float [kLength.z];

    float* h_sX = new float [m_kRadius.x + 1];
    float* h_sY = new float [m_kRadius.y + 1];
    float* h_sZ = new float [m_kRadius.z + 1];

    generateGaussian(h_kX, h_sX, m_sigma.x, m_kRadius.x);
    generateGaussian(h_kY, h_sY, m_sigma.y, m_kRadius.y);
    generateGaussian(h_kZ, h_sZ, m_sigma.z, m_kRadius.z);

    setConvolutionKernelX(h_kX, kLength.x);
    setConvolutionKernelY(h_kY, kLength.y);
    setConvolutionKernelZ(h_kZ, kLength.z);

    setSupplementKernelX(h_sX, m_kRadius.x + 1);
    setSupplementKernelY(h_sY, m_kRadius.y + 1);
    setSupplementKernelZ(h_sZ, m_kRadius.z + 1);

    delete []h_kX;
    delete []h_kY;
    delete []h_kZ;

    delete []h_sX;
    delete []h_sY;
    delete []h_sZ;
}

void cplGaussianFilter::filter(float* d_o, const float* d_i, const Vector3Di& size,
                                    float* d_temp, cudaStream_t stream)
{
    bool need_temp = (d_temp == NULL);
    if (need_temp)
        dmemAlloc(d_temp, size.productOfElements());
    
    cplConvolutionX3D(d_temp, d_i, m_kRadius.x, size.x, size.y, size.z, stream);
    cplShiftCoordinate(d_o, d_temp, size.x, size.y, size.z, 1, stream);
    
    cplConvolutionX3D(d_temp, d_o, m_kRadius.z, size.z, size.x, size.y, stream);
    cplShiftCoordinate(d_o, d_temp, size.z, size.x, size.y, 1, stream);
    
    cplConvolutionX3D(d_temp, d_o, m_kRadius.y, size.y, size.z, size.x, stream);
    cplShiftCoordinate(d_o, d_temp, size.y, size.z, size.x, 1, stream);

    if (need_temp)
        dmemFree(d_temp);
}

void cplGaussianFilter::filter(cplVector3DArray& d_o, const cplVector3DArray& d_i,
                                    const Vector3Di& size, float* d_temp, cudaStream_t stream)
{
    bool need_temp = (d_temp == NULL);
    if (need_temp)
        dmemAlloc(d_temp, size.productOfElements());

    this->filter(d_o.x, d_i.x, size, d_temp, stream);
    this->filter(d_o.y, d_i.y, size, d_temp, stream);
    this->filter(d_o.z, d_i.z, size, d_temp, stream);
    
    if (need_temp)
        dmemFree(d_temp);
}

