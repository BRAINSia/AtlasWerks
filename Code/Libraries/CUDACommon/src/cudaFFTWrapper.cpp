#include <cpl.h>
#include <cudaFFTWrapper.h>
#include <cudaComplex.h>

void cplFFT3DWrapper::createPlan()
{
    // create FFT plan
    // ATTENTION : the order is reversed from CPLV FFT documentation
    // this is hard to find bug since it is not failed if fft_h = fft_w
    cufftPlan3d(&pR2C, m_size.z, m_size.y, m_size.x, CUFFT_R2C);
    cufftPlan3d(&pC2R, m_size.z, m_size.y, m_size.x, CUFFT_C2R);
    cufftPlan3d(&pC2C, m_size.z, m_size.y, m_size.x, CUFFT_C2C);
    
    hasPlan = true;
}

void cplFFT3DWrapper::clear(){
    cufftDestroy(pC2C);
    cufftDestroy(pR2C);
    cufftDestroy(pC2R);

    hasPlan = false;
    m_size  = Vector3Di(0,0,0);
}


bool cplFFT3DWrapper::checkSize(const Vector3Di& size) const
{
    return ((m_size.x == size.x) && (m_size.y == size.y) && (m_size.z == size.z));
}
    

void cplFFT3DWrapper::forwardFFT(cplComplex* d_o, cplComplex* d_i) const{
    cufftExecC2C(pC2C, d_i, d_o, CUFFT_FORWARD);
    int nElems = m_size.productOfElements();
    float scale = 1.f / nElems;
    cplVectorOpers::MulC_I(d_o, scale, nElems);
}

void cplFFT3DWrapper::forwardFFT(cplComplex* d_o, float* d_i) const{
    cufftExecR2C(pR2C, d_i, d_o);
    int nElems = m_size.productOfElements();
    float scale = 1.f / nElems;
    cplVectorOpers::MulC_I(d_o, scale, nElems);
}

void cplFFT3DWrapper::backwardFFT(cplComplex* d_o, cplComplex* d_i)  const{
    cufftExecC2C(pC2C, d_i, d_o, CUFFT_INVERSE);
}

void cplFFT3DWrapper::backwardFFT(float* d_o, cplComplex* d_i)  const{
    cufftExecC2R(pC2R, d_i, d_o);
}


////////////////////////////////////////////////////////////////////////////////
//
////////////////////////////////////////////////////////////////////////////////
void cplFFT3DConvolutionWrapper::setSize(const Vector3Di& size){
    m_fft.setSize(size);
    dmemAlloc(d_iC, size.productOfElements());
}

void cplFFT3DConvolutionWrapper::clear(){
    if (d_iC)
        dmemFree(d_iC);
}


void cplFFT3DConvolutionWrapper::convolve(float* d_o, float* d_i, cplComplex* d_kernelC) const{
    m_fft.forwardFFT(d_iC, d_i);
    cplComplexMul_I(d_iC, d_kernelC, m_fft.getSize().productOfElements(), 0);
    m_fft.backwardFFT(d_o, d_iC);
}

void cplFFT3DConvolutionWrapper::convolve(float* d_o, float* d_i, cplComplex* d_kernelC, const Vector3Di& size) const
{
    if (m_fft.checkSize(size)){
        convolve(d_o, d_i, d_kernelC);
    } else {
        std::cerr << "Miss match size of input and filter" << std::endl;
    }
}

////////////////////////////////////////////////////////////////////////////////
//
////////////////////////////////////////////////////////////////////////////////

inline int get_id(int x, int y, int z, const Vector3Di& size)
{
    return x + (y + z * size.y) * size.x;
}

void compute3DGaussianKernel(float* kernelImg, const Vector3Di& gridSize, float sigma){
    float var = sigma * sigma;
    for (int k = 0; k < gridSize.z /2; k++)
        for (int j = 0; j < gridSize.y /2; j++)
            for (int i = 0; i < gridSize.x /2; i++) {
                float distsq = i*i + j*j + k*k;
                float g = exp(- distsq / 2 / var);
                
                kernelImg[get_id(i,j,k, gridSize)] = g;
                kernelImg[get_id(i,j,gridSize.z - 1 - k, gridSize)] = g;
                kernelImg[get_id(i,gridSize.y - 1 - j, k, gridSize)] = g;
                kernelImg[get_id(i,gridSize.y - 1 - j, gridSize.z - 1 - k, gridSize)] = g;
                kernelImg[get_id(gridSize.x - 1 - i,j,k, gridSize)] = g;
                kernelImg[get_id(gridSize.x - 1 - i,j,gridSize.z - 1 - k, gridSize)] = g;
                kernelImg[get_id(gridSize.x - 1 - i,gridSize.y - 1 - j, k, gridSize)] = g;
                kernelImg[get_id(gridSize.x - 1 - i,gridSize.y - 1 - j, gridSize.z - 1 - k, gridSize)] = g;
            }
    double sum = 0.f;
    uint totalGridSize = gridSize.x * gridSize.y * gridSize.z;
    for (unsigned i=0; i< totalGridSize; ++i)
        sum += kernelImg[i];

    for (unsigned i=0; i< totalGridSize; ++i)
        kernelImg[i] /= sum;
}

void cplFFTGaussianFilter::init(const Vector3Di& size, float sigma)
{
    int nElems = size.productOfElements();
    
    // Compute the Gaussian kernel on CPU
    float* h_kernel = new float [nElems];
    compute3DGaussianKernel(h_kernel, size, sigma);

    // allocate real gaussian kernel on GPU
    float* d_kernel;
    dmemAlloc(d_kernel, nElems);
    copyArrayToDevice(d_kernel, h_kernel, nElems);

    // Copy to GPU
    dmemAlloc(d_KernelC, nElems);

    // Compute the guassian kernel on frequency domain
    m_fftFilter.setSize(size);
    m_fftFilter.getFFTWrapper().forwardFFT(d_KernelC, d_kernel);

    dmemFree(d_kernel);
    delete []h_kernel;
}

void cplFFTGaussianFilter::filter(float* d_o, float* d_i, const Vector3Di& size) const
{
    m_fftFilter.convolve(d_o, d_i, d_KernelC, size);
}

cplFFTGaussianFilter::cplFFTGaussianFilter()
{
    dmemFree(d_KernelC);
}
