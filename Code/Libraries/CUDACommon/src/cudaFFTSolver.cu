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

#ifndef __CUDA_FFT_SOLVER__CU
#define __CUDA_FFT_SOLVER__CU

#include <cpl.h>
#include <cplFFT.h>

/*------------------------------------------------------------------------------------------*/
#define MAX_LENGTH 512
__device__ __constant__ float sdCosWX[MAX_LENGTH];
__device__ __constant__ float sdSinWX[MAX_LENGTH];
__device__ __constant__ float sdCosWY[MAX_LENGTH];
__device__ __constant__ float sdSinWY[MAX_LENGTH];
__device__ __constant__ float sdCosWZ[MAX_LENGTH];
__device__ __constant__ float sdSinWZ[MAX_LENGTH];

//--------------------------------------------------------------------------------
void LookupTable2D::upload(){
    cudaMemcpyToSymbol(sdCosWX, cosWX, m_xSize * sizeof(float));
    cudaMemcpyToSymbol(sdSinWX, sinWX, m_xSize * sizeof(float));
    cudaMemcpyToSymbol(sdCosWY, cosWY, m_ySize * sizeof(float));
    cudaMemcpyToSymbol(sdSinWY, sinWY, m_ySize * sizeof(float));
}
    
void LookupTable2D::initTable(double dX, double dY){
    // hardcode these for now
    deltaX = dX;
    deltaY = dY;

    //
    // precompute some values
    //
//     double sX = deltaX * 2.0 * M_PI / m_xSize; 
//     double sY = deltaY * 2.0 * M_PI / m_ySize; 
    double sX = 2.0 * M_PI / m_xSize; 
    double sY = 2.0 * M_PI / m_ySize; 

    double deltaXSq = deltaX * deltaX;
    double deltaYSq = deltaY * deltaY;

    //
    // fill in luts
    //
    for (unsigned int x = 0; x < m_xSize; ++x) {
        cosWX[x] = (2.0 * cos(sX * static_cast<float>(x)) - 2.0) / deltaXSq;
        sinWX[x] = sin(sX * static_cast<float>(x)) / deltaX;
    }

    for (unsigned int y = 0; y < m_ySize; ++y) {
        cosWY[y] = (2.0 * cos(sY * static_cast<float>(y)) - 2.0) / deltaYSq;
        sinWY[y] = sin(sY * static_cast<float>(y)) / deltaY;
    }
}
    
void LookupTable2D::allocate(){

    cosWX = new float [MAX_LENGTH];
    cosWY = new float [MAX_LENGTH];

    sinWX = new float [MAX_LENGTH];
    sinWY = new float [MAX_LENGTH];
}

void LookupTable2D::setSize(unsigned int xSize, unsigned int ySize, double sx, double sy){
    m_xSize = xSize; m_ySize = ySize;
    initTable(sx, sy);
}
void LookupTable2D::clear(){
    delete []cosWX;
    delete []cosWY;

    delete []sinWX;
    delete []sinWY;
}

//--------------------------------------------------------------------------------
__global__ void partialNavierStokesSolver2D_C2_kernel(cplComplex* bX, cplComplex* bY,
                                                      const float alpha, const float gamma,
                                                      const int sizeX, const int sizeY)
{
    uint  x = blockIdx.x * blockDim.x + threadIdx.x;
    uint  y = blockIdx.y * blockDim.y + threadIdx.y;

    uint  index     = x + y * sizeX;
    
    if ( x < sizeX && y < sizeY){
        float wx = sdCosWX[x];
        float wy = sdCosWY[y];
        float lambda    = - alpha * (wx + wy) + gamma;
        float invLambda = 1.f / lambda;
        bX[index]   *= invLambda;
        bY[index]   *= invLambda;
    }
}

__global__ void partialNavierStokesSolver2D_C1_kernel(cplComplex* bX,
                                                      const float alpha, const float gamma,
                                                      const int sizeX, const int sizeY)
{
    uint  x = blockIdx.x * blockDim.x + threadIdx.x;
    uint  y = blockIdx.y * blockDim.y + threadIdx.y;

    uint  index     = x + y * sizeX;
    
    if ( x < sizeX && y < sizeY){
        float wx = sdCosWX[x];
        float wy = sdCosWY[y];
        float lambda    = -alpha * (wx + wy) + gamma;
        float invLambda = 1.f / lambda;
        bX[index]   *= invLambda;
    }
}

void laplacianBlurFFTReal_C2(cplComplex* d_data_c1, float* d_idata_r1, float* d_odata_r1,
                             cplComplex* d_data_c2, float* d_idata_r2, float* d_odata_r2,
                             const int fft_w, const int fft_h,
                             const float alpha, const float gamma,
                             cufftHandle& planR2C, cufftHandle& planC2R )
{
    uint nElems  = fft_w * fft_h;
    float scale = 1.f / nElems;

    //1.a Compute FFT of the data (inplace)
    // convert the input from real to frequency(complex image)
    cufftSafeCall(cufftExecR2C(planR2C, d_idata_r1, (cufftComplex*) d_data_c1));
    cufftSafeCall(cufftExecR2C(planR2C, d_idata_r2, (cufftComplex*) d_data_c2));
    
    uint nUpdate = (fft_w / 2 + 1) * fft_h;
    // normalize value after FFT operation

    cplVectorOpers::MulC_I(d_data_c1, scale, nUpdate);
    cplVectorOpers::MulC_I(d_data_c2, scale, nUpdate);
    
    //2.c Solve system
    dim3 threadBlock(16,16);
    dim3 gridBlock(iDivUp(fft_w / 2 + 1,threadBlock.x), iDivUp(fft_h,threadBlock.y));
    partialNavierStokesSolver2D_C2_kernel<<<gridBlock, threadBlock>>>(d_data_c1, d_data_c2,
                                                                      alpha, gamma,
                                                                      fft_w / 2 + 1, fft_h);
    //3.c Get the result using inverse
    cufftExecC2R(planC2R, (cufftComplex*) d_data_c1, d_odata_r1);
    cufftExecC2R(planC2R, (cufftComplex*) d_data_c2, d_odata_r2);
};

void laplacianBlurFFTReal_C1(cplComplex* d_data_c1, float* d_idata_r1, float* d_odata_r1,
                             const int fft_w, const int fft_h,
                             const float alpha, const float gamma,
                             cufftHandle& planR2C, cufftHandle& planC2R)
{
    uint nElems  = fft_w * fft_h;
    float scale = 1.f / nElems;

    //1.a Compute FFT of the data (inplace)
    // convert the input from real to frequency(complex image)
    cufftSafeCall(cufftExecR2C(planR2C, d_idata_r1, (cufftComplex*) d_data_c1));

    // normalize value after FFT operation
    uint nUpdate = (fft_w / 2 + 1) * fft_h;
    cplVectorOpers::MulC_I(d_data_c1, scale, nUpdate);
    
    //2.c Solve system
    dim3 threadBlock(16,16);
    dim3 gridBlock(iDivUp(fft_w / 2 + 1,threadBlock.x), iDivUp(fft_h,threadBlock.y));
    partialNavierStokesSolver2D_C1_kernel<<<gridBlock, threadBlock>>>(d_data_c1,
                                                                      alpha, gamma,
                                                                      fft_w / 2 + 1, fft_h);
    //3.c Get the result using inverse
    cufftExecC2R(planC2R, (cufftComplex*) d_data_c1, d_odata_r1);

};

void FFTSolverPlan2D::setSize(unsigned int w, unsigned h, double sx, double sy){
    fft_w = w;
    fft_h = h;

    // Create look up table 
    lut.setSize(fft_w, fft_h, sx, sy);
    lut.upload();

    // create FFT plan
    // ATTENTION : the order is reversed from CUDA FFT documentation
    // this is hard to find bug since it is not failed if fft_h = fft_w
    cufftSafeCall(cufftPlan2d(&planR2C, fft_h, fft_w, CUFFT_R2C));
    cufftSafeCall(cufftPlan2d(&planC2R, fft_h, fft_w, CUFFT_C2R));
    

}

void FFTSolverPlan2D::clean(){
    // destroy handle
    cufftDestroy(planR2C);
    cufftDestroy(planC2R);
}

void FFTSolverPlan2D::solve(float* d_gv, float* d_gf, float alpha, float gamma){
    // allocate complex force 
    int complexASize = (fft_w /2 + 1) * fft_h * sizeof(cplComplex);

    cplComplex* d_force_FFT;
    cudaMalloc((void**)& d_force_FFT,  complexASize);
    laplacianBlurFFTReal_C1(d_force_FFT, d_gf, d_gv,
                            fft_w, fft_h,
                            alpha, gamma,
                            planR2C, planC2R);
    cudaFree(d_force_FFT);
}

void FFTSolverPlan2D::solve(float* d_gvx, float* d_gfx,
                            float* d_gvy, float* d_gfy,
                            float alpha, float gamma){
    // allocate complex force 
    int complexASize = (fft_w/2 + 1) * fft_h *  sizeof(cplComplex);

    cplComplex* d_force_FFT_x;
    cplComplex* d_force_FFT_y;
    
    cudaMalloc((void**)&d_force_FFT_x,  complexASize);
    cudaMalloc((void**)&d_force_FFT_y,  complexASize);
    
    laplacianBlurFFTReal_C2(d_force_FFT_x, d_gfx, d_gvx,
                            d_force_FFT_y, d_gfy, d_gvy,
                            fft_w, fft_h,
                            alpha, gamma,
                            planR2C, planC2R);
    
    cudaFree(d_force_FFT_x);
    cudaFree(d_force_FFT_y);
}

////////////////////////////////////////////////////////////////////////////////
//
////////////////////////////////////////////////////////////////////////////////
void LookupTable3D::upload(){
    cudaMemcpyToSymbol(sdCosWX, mCosWX, mSize.x * sizeof(float));
    cudaMemcpyToSymbol(sdSinWX, mSinWX, mSize.x * sizeof(float));
    cudaMemcpyToSymbol(sdCosWY, mCosWY, mSize.y * sizeof(float));
    cudaMemcpyToSymbol(sdSinWY, mSinWY, mSize.y * sizeof(float));
    cudaMemcpyToSymbol(sdCosWZ, mCosWZ, mSize.z * sizeof(float));
    cudaMemcpyToSymbol(sdSinWZ, mSinWZ, mSize.z * sizeof(float));
    cutilCheckMsg("LookupTable3D::Upload");
}
    
void LookupTable3D::initTable(double dX, double dY, double dZ){
    if(mSize.x > MAX_LENGTH || 
       mSize.y > MAX_LENGTH || 
       mSize.z > MAX_LENGTH)
    {
        //throw AtlasWerksException(__FILE__,__LINE__,"Error, not enough static memory allocated!");
    }

    //
    // precompute some values
    //
//     double sX = mSp.x * 2.0 * M_PI / mSize.x; 
//     double sY = mSp.y * 2.0 * M_PI / mSize.y; 
//     double sZ = mSp.z * 2.0 * M_PI / mSize.z; 
    double sX = 2.0 * M_PI / mSize.x; 
    double sY = 2.0 * M_PI / mSize.y; 
    double sZ = 2.0 * M_PI / mSize.z; 

    double deltaXSq = mSp.x * mSp.x;
    double deltaYSq = mSp.y * mSp.y;
    double deltaZSq = mSp.z * mSp.z;

    //
    // fill in luts
    //
    for (unsigned int x = 0; x < mSize.x; ++x) {
        mCosWX[x] = (2.0 * cos(sX * static_cast<float>(x)) - 2.0) / deltaXSq;
        mSinWX[x] = sin(sX * static_cast<float>(x)) / mSp.x;
    }

    for (unsigned int y = 0; y < mSize.y; ++y) {
        mCosWY[y] = (2.0 * cos(sY * static_cast<float>(y)) - 2.0) / deltaYSq;
        mSinWY[y] = sin(sY * static_cast<float>(y)) / mSp.y;
    }

    for (unsigned int z = 0; z < mSize.z; ++z) {
        mCosWZ[z] = (2.0 * cos(sZ * static_cast<float>(z)) - 2.0) / deltaZSq;
        mSinWZ[z] = sin(sZ * static_cast<float>(z)) / mSp.z;
    }
}
    
void LookupTable3D::allocate(){
  mCosWX = new float [MAX_LENGTH];
  mCosWY = new float [MAX_LENGTH];
  mCosWZ = new float [MAX_LENGTH];
  
  mSinWX = new float [MAX_LENGTH];
  mSinWY = new float [MAX_LENGTH];
  mSinWZ = new float [MAX_LENGTH];
}

void LookupTable3D::setSize(const Vector3Di& size, const Vector3Df& sp){
    mSize = size;
    mSp   = sp;
    initTable();
    upload();
}

void LookupTable3D::clear(){
    delete []mCosWX;
    delete []mCosWY;
    delete []mCosWZ;

    delete []mSinWX;
    delete []mSinWY;
    delete []mSinWZ;
}

FFTSolverPlan3D::FFTSolverPlan3D(bool saveMem):mSaveMemory(saveMem),
                                   mAlpha(0.01), mGamma(0.001),
                                   mdFFTArrayX(NULL), mdFFTArrayY(NULL), mdFFTArrayZ(NULL),
                                   mdFFTArray(NULL)
{
    mBeta  = (mSaveMemory) ? 0.f : 0.01f;
}

void FFTSolverPlan3D::setSize(const Vector3Di& size, const Vector3Df& sp){
    mSize          = size;
    mSp            = sp;
    mComplexSize   = Vector3Di(mSize.x / 2 + 1, mSize.y, mSize.z);

    mLookupTable.setSize(mSize, mSp);
    this->createCUFFTPlan();
    this->alloc();
}

void FFTSolverPlan3D::setParams(float alpha, float beta, float gamma) {
    mAlpha = alpha;
    mGamma = gamma;
    mBeta  = beta;
    
    if (mSaveMemory){
        if (mBeta != 0.f) {
            fprintf(stderr, "Warning : Solver in the save memory mode did not commit beta value. Set it to 0 ");
            mBeta = 0.f;
        }
    }
}


void FFTSolverPlan3D::createCUFFTPlan(){
    // ATTENTION : the order is reversed from CUDA FFT documentation
    // this is hard to find bug since it is not failed if fft_h = fft_w
    cufftPlan3d(&mPlanR2C, mSize.z, mSize.y, mSize.x, CUFFT_R2C);
    cufftPlan3d(&mPlanC2R, mSize.z, mSize.y, mSize.x, CUFFT_C2R);
    cutilCheckMsg("CUDA FFT Plan Initialization");
}

void FFTSolverPlan3D::destroyCUFFTPlan(){
    // destroy handle
    cufftDestroy(mPlanR2C);
    cufftDestroy(mPlanC2R);
    cutilCheckMsg("CUDA FFT Plan Destruction");
}

void FFTSolverPlan3D::alloc(){
    mAllocateSize = mComplexSize.productOfElements();
    if (mSaveMemory){
        dmemAlloc(mdFFTArray, mAllocateSize);
    } else {
        dmemAlloc(mdFFTArrayX, mAllocateSize);
        dmemAlloc(mdFFTArrayY, mAllocateSize);
        dmemAlloc(mdFFTArrayZ, mAllocateSize);
        mdFFTArray = mdFFTArrayX;
    }
    cutilCheckMsg("Allocate FFT Complex memory");
}

void FFTSolverPlan3D::dealloc(){
    if (mSaveMemory)
        cudaSafeDelete(mdFFTArray);
    else {
        cudaSafeDelete(mdFFTArrayX);
        cudaSafeDelete(mdFFTArrayY);
        cudaSafeDelete(mdFFTArrayZ);
    }
    cutilCheckMsg("Deallocate FFT Complex memory");
}

void FFTSolverPlan3D::clean()
{
    dealloc();
    destroyCUFFTPlan();
}
////////////////////////////////////////////////////////////////////////////////
/// Prescale function - preparing for FFT
////////////////////////////////////////////////////////////////////////////////
void FFTSolverPlan3D::preScale(float* dData, cudaStream_t stream){
    uint nElems = mSize.productOfElements();
    float scale = 1.f / nElems;
    cplVectorOpers::MulC_I(dData, scale, nElems, stream);
}

void FFTSolverPlan3D::preScale(float* d_oData, const float* d_iData, cudaStream_t stream){
    uint nElems = mSize.productOfElements();
    float scale = 1.f / nElems;
    cplVectorOpers::MulC(d_oData, d_iData, scale, nElems, stream);
}

void FFTSolverPlan3D::preScale(float* dDataX, float* dDataY, float* dDataZ, cudaStream_t stream){
    uint nElems = mSize.productOfElements();
    float scale = 1.f / nElems;

    // Scale date before FFT
    cplVectorOpers::MulC_I(dDataX, scale, nElems, stream);
    cplVectorOpers::MulC_I(dDataY, scale, nElems, stream);
    cplVectorOpers::MulC_I(dDataZ, scale, nElems, stream);
}

void FFTSolverPlan3D::preScale(float* d_oDataX, float* d_oDataY, float* d_oDataZ,
                               const float* d_iDataX, const float* d_iDataY, const float* d_iDataZ, cudaStream_t stream)
{
    uint nElems = mSize.productOfElements();
    float scale = 1.f / nElems;

    // Scale date before FFT
    cplVectorOpers::MulC(d_oDataX, d_iDataX, scale, nElems, stream);
    cplVectorOpers::MulC(d_oDataY, d_iDataY, scale, nElems, stream);
    cplVectorOpers::MulC(d_oDataZ, d_iDataZ, scale, nElems, stream);
}

void FFTSolverPlan3D::preScale(cplVector3DArray& d_o, const cplVector3DArray& d_i, cudaStream_t stream){
    uint nElems = mSize.productOfElements();
    float scale = 1.f / nElems;
    cplVector3DOpers::MulC(d_o, d_i, scale, nElems, stream);
}

void FFTSolverPlan3D::preScale(cplVector3DArray& d_data, cudaStream_t stream){
    uint nElems = mSize.productOfElements();
    float scale = 1.f / nElems;
    cplVector3DOpers::MulC_I(d_data, scale, nElems, stream);
}

//--------------------------------------------------------------------------------
// General Navier Stoker solver  with beta is different than 0
//
//--------------------------------------------------------------------------------
__device__ void InverseOperatorMultiply(cplComplex& bX, cplComplex& bY, cplComplex& bZ,
                                        float L00,
                                        float L10, float L11,
                                        float L20, float L21, float L22)
{
    float G00;
    float G10, G11;
    float G20, G21, G22;
    float y0, y1, y2;
    //
    // Given that A is pos-def symetric matrix, solve Ax=b by finding
    // cholesky decomposition GG'=A
    // and then performing 2 back-solves, Gy=b and then G'x=y to get x.
    // 
	   
    // 1. find cholesky decomposition by finding G such that GG'=A.
    //    A must be positive definite symetric (we assume that here)
    //    G is then lower triangular, see algorithm 4.2.1 p142-3
    //    in Golub and VanLoan
    // Note: these are in matlab notation 1:3
    // [ G(1,1)   0      0    ]   [ G(1,1) G(2,1) G(3,1) ]   
    // [ G(2,1) G(2,2)   0    ] * [   0    G(2,2) G(3,2) ] = A matrix
    // [ G(3,1) G(3,2) G(3,3) ]   [   0      0    G(3,3) ]

    float bRealX = bX.x;
    float bRealY = bY.x;
    float bRealZ = bZ.x;
  
    float bImagX = bX.y;
    float bImagY = bY.y;
    float bImagZ = bZ.y;

    float& vRealX = bX.x;
    float& vRealY = bY.x;
    float& vRealZ = bZ.x;
  
    float& vImagX = bX.y;
    float& vImagY = bY.y;
    float& vImagZ = bZ.y;

    G00 = sqrt(L00);
    G10 = L10 / G00;
    G20 = L20 / G00;

    G11 = L11 - G10 * G10;
    G21 = L21 - G20 * G10;
    G11 = sqrt(G11);
    G21 = G21 / G11;

    G22 = L22 - (G20*G20 + G21*G21);
    G22 = sqrt(G22);
    // back-solve Gy=b to get a temporary vector y
    // back-solve G'x=y to get answer in x
    //
    // Note: these are in matlab notation 1:3
    // [ G(1,1)   0      0    ]   [ y(1) ] = b(1)
    // [ G(2,1) G(2,2)   0    ] * [ y(2) ] = b(2)
    // [ G(3,1) G(3,2) G(3,3) ]   [ y(3) ] = b(3)
    //
    // [ G(1,1) G(2,1) G(3,1) ]   [ x(1) ] = y(1)
    // [   0    G(2,2) G(3,2) ] * [ x(2) ] = y(2)
    // [   0      0    G(3,3) ]   [ x(3) ] = y(3)
    y0 = bRealX / G00;
    y1 = (bRealY - G10*y0) / G11;
    y2 = (bRealZ - G20*y0 - G21*y1) / G22;

    vRealZ = y2 / G22;
    vRealY = (y1 - G21*vRealZ) / G11;
    vRealX = (y0 - G10*vRealY - G20*vRealZ) / G00;

    y0 = bImagX / G00;
    y1 = (bImagY - G10*y0) / G11;
    y2 = (bImagZ - G20*y0 - G21*y1) / G22;

    vImagZ = y2 / G22;
    vImagY = (y1 - G21*vImagZ) / G11;
    vImagX = (y0 - G10*vImagY - G20*vImagZ) / G00;
}


//--------------------------------------------------------------------------------
// General Navier Stoker solver  with beta is different than 0
//
//--------------------------------------------------------------------------------
__device__ void OperatorMultiply(cplComplex& bX, cplComplex& bY, cplComplex& bZ,
                                 float L00,
                                 float L10, float L11,
                                 float L20, float L21, float L22)
{
    float bRealX = bX.x;
    float bRealY = bY.x;
    float bRealZ = bZ.x;
  
    float bImagX = bX.y;
    float bImagY = bY.y;
    float bImagZ = bZ.y;

    float& vRealX = bX.x;
    float& vRealY = bY.x;
    float& vRealZ = bZ.x;
  
    float& vImagX = bX.y;
    float& vImagY = bY.y;
    float& vImagZ = bZ.y;

    vRealX = L00*bRealX + L10*bRealY + L20*bRealZ;
    vRealY = L10*bRealX + L11*bRealY + L21*bRealZ;
    vRealZ = L20*bRealX + L21*bRealY + L22*bRealZ;
  
    vImagX = L00*bImagX + L10*bImagY + L20*bImagZ;
    vImagY = L10*bImagX + L11*bImagY + L21*bImagZ;
    vImagZ = L20*bImagX + L21*bImagY + L22*bImagZ;
}


//--------------------------------------------------------------------------------
// CUDA Kernel to solve the Navier Stoke equation
//--------------------------------------------------------------------------------
template<bool inverseOp>
__global__ void fullNavierStokesSolver3D_C3_kernel(cplComplex* bX, cplComplex* bY, cplComplex* bZ,
                                                   float alpha, float beta, float gamma,
                                                   int sizeX,   int sizeY,  int sizeZ)
{
    uint  x = blockIdx.x * blockDim.x + threadIdx.x;
    uint  y = blockIdx.y * blockDim.y + threadIdx.y;

    float lambda;
    float L00;
    float L10, L11;
    float L20, L21, L22;

    uint index     = x + y * sizeX;
    uint planeSize = sizeX * sizeY;
    
    if ( x < sizeX && y < sizeY){
        float wx = sdCosWX[x];
        float wy = sdCosWY[y];
        for (int z=0; z < sizeZ ; ++z, index+=planeSize){
            //
            // compute L (it is symmetric, only need lower triangular part)
            //
            lambda = -alpha * (wx + wy + sdCosWZ[z]) + gamma;
            
            L00 = lambda - beta * sdCosWX[x];
            L11 = lambda - beta * sdCosWY[y];
            L22 = lambda - beta * sdCosWZ[z];
            
            L10 = beta * sdSinWX[x] * sdSinWY[y];
            L20 = beta * sdSinWX[x] * sdSinWZ[z];
            L21 = beta * sdSinWY[y] * sdSinWZ[z];

            if (inverseOp)
                InverseOperatorMultiply(bX[index], bY[index], bZ[index],
                                        L00, L10, L11, L20, L21, L22);
            else
                OperatorMultiply(bX[index], bY[index], bZ[index],
                                 L00, L10, L11, L20, L21, L22);
        }
    }
}

//--------------------------------------------------------------------------------
// General Navier Stoker solver  with beta = 0
//--------------------------------------------------------------------------------
template<bool inverseOp>
__global__ void partialNavierStokesSolver3D_C3_kernel(cplComplex* bX, cplComplex* bY, cplComplex* bZ,
                                                      float alpha, float gamma,
                                                      int sizeX,   int sizeY, int sizeZ)
{
    uint x = blockIdx.x * blockDim.x + threadIdx.x;
    uint y = blockIdx.y * blockDim.y + threadIdx.y;
    uint index     = x + y * sizeX;
    uint planeSize = sizeX * sizeY;

    if (x < sizeX && y < sizeY){
        float wx = sdCosWX[x];
        float wy = sdCosWY[y];
        
        for (int z=0; z < sizeZ ; ++z, index+=planeSize){
            float lambda    = -alpha * (wx + wy + sdCosWZ[z]) + gamma;
            if (inverseOp) {
                float invLambda = 1.f / lambda;
                bX[index] *= invLambda;
                bY[index] *= invLambda;
                bZ[index] *= invLambda;
            } else {
                bX[index] *= lambda;
                bY[index] *= lambda;
                bZ[index] *= lambda;
            }
        }
    }
}

//--------------------------------------------------------------------------------
// General Navier Stoker solver on single channel with beta = 0
//--------------------------------------------------------------------------------
template<bool inverseOp>
__global__ void partialNavierStokesSolver3D_C1_kernel(cplComplex* b,
                                                      float alpha, float gamma,
                                                      int sizeX, int sizeY, int sizeZ)
{
    uint  x = blockIdx.x * blockDim.x + threadIdx.x;
    uint  y = blockIdx.y * blockDim.y + threadIdx.y;

    float lambda;
    uint  index     = x + y * sizeX;
    uint  planeSize = sizeX * sizeY;
    
    if ( x < sizeX && y < sizeY){
        float wx = sdCosWX[x];
        float wy = sdCosWY[y];
        for (uint z=0; z < sizeZ; ++z, index+=planeSize){
            lambda = -alpha * (wx + wy + sdCosWZ[z]) + gamma;
            if (inverseOp) {
                float invLambda = 1.f / lambda;
                b[index]       *= invLambda;
            }
            else b[index]      *= lambda;
        }
    }
}

void FFTSolverPlan3D::applyFull(float* dDataX, float* dDataY, float* dDataZ, bool inverseOp, cudaStream_t stream){
    // Need to run the full solver
    //1.a Compute FFT of the data (inplace)
    // convert the input from real to frequency(complex image)
    cufftExecR2C(mPlanR2C, (cufftReal*)dDataX, (cufftComplex*) mdFFTArrayX);
    cufftExecR2C(mPlanR2C, (cufftReal*)dDataY, (cufftComplex*) mdFFTArrayY);
    cufftExecR2C(mPlanR2C, (cufftReal*)dDataZ, (cufftComplex*) mdFFTArrayZ);
    cutilCheckMsg("Convert from Real to Complex ");

    //2. Solve in the frequency domain
    dim3 threads(16,16);
    dim3 grids(iDivUp(mComplexSize.x,threads.x), iDivUp(mComplexSize.y, threads.y));
    if (inverseOp)
        fullNavierStokesSolver3D_C3_kernel<true><<<grids, threads, 0, stream>>>
            (mdFFTArrayX, mdFFTArrayY, mdFFTArrayZ,
             mAlpha, mBeta, mGamma,
             mComplexSize.x, mComplexSize.y, mComplexSize.z);
    else
        fullNavierStokesSolver3D_C3_kernel<false><<<grids, threads, 0, stream>>>
            (mdFFTArrayX, mdFFTArrayY, mdFFTArrayZ,
             mAlpha, mBeta, mGamma,
             mComplexSize.x, mComplexSize.y, mComplexSize.z);
    cutilCheckMsg("Full Navier Stoke Solver ");
        
    //3.c Get the result using inverse
    cufftExecC2R(mPlanC2R, (cufftComplex*) mdFFTArrayX, dDataX);
    cufftExecC2R(mPlanC2R, (cufftComplex*) mdFFTArrayY, dDataY);
    cufftExecC2R(mPlanC2R, (cufftComplex*) mdFFTArrayZ, dDataZ);
    cutilCheckMsg("Convert from Complex to Real");
}

void FFTSolverPlan3D::applyPartial(float* dDataX, float* dDataY, float* dDataZ, bool inverseOp, cudaStream_t stream){
    //1.a Compute FFT of the data (inplace)
    // convert the input from real to frequency(complex image)
    cufftExecR2C(mPlanR2C, (cufftReal*)dDataX, (cufftComplex*) mdFFTArrayX);
    cufftExecR2C(mPlanR2C, (cufftReal*)dDataY, (cufftComplex*) mdFFTArrayY);
    cufftExecR2C(mPlanR2C, (cufftReal*)dDataZ, (cufftComplex*) mdFFTArrayZ);
    cutilCheckMsg("Convert from Real to Complex ");

    //2. Solve in the frequency domain
    dim3 threads(16,16);
    dim3 grids(iDivUp(mComplexSize.x,threads.x), iDivUp(mComplexSize.y, threads.y));
    if (inverseOp){
        partialNavierStokesSolver3D_C3_kernel<true><<<grids, threads, 0, stream>>>
            (mdFFTArrayX, mdFFTArrayY, mdFFTArrayZ,
             mAlpha, mGamma,
             mComplexSize.x, mComplexSize.y, mComplexSize.z);
    } else {
        partialNavierStokesSolver3D_C3_kernel<false><<<grids, threads, 0, stream>>>
            (mdFFTArrayX, mdFFTArrayY, mdFFTArrayZ,
             mAlpha, mGamma,
             mComplexSize.x, mComplexSize.y, mComplexSize.z);
    }
    cutilCheckMsg("Partial Hemlotz Solver ");

    //3.c Get the result using inverse
    cufftExecC2R(mPlanC2R, (cufftComplex*) mdFFTArrayX, dDataX);
    cufftExecC2R(mPlanC2R, (cufftComplex*) mdFFTArrayY, dDataY);
    cufftExecC2R(mPlanC2R, (cufftComplex*) mdFFTArrayZ, dDataZ);
    cutilCheckMsg("Convert from Complex to Real");
}

void FFTSolverPlan3D::applySingeChannel(float* dData, bool inverseOp, cudaStream_t stream)
{
    //1.a Compute FFT of the data (inplace)
    // convert the input from real to frequency(complex image)
    cufftExecR2C(mPlanR2C, (cufftReal*)dData, (cufftComplex*) mdFFTArray);

    //2. Solve in the frequency domain
    dim3 threads(16,16);
    dim3 grids(iDivUp(mComplexSize.x,threads.x), iDivUp(mComplexSize.y, threads.y));
    if (inverseOp)
        partialNavierStokesSolver3D_C1_kernel<true><<<grids, threads, 0, stream>>>
            (mdFFTArray, mAlpha, mGamma, mComplexSize.x, mComplexSize.y, mComplexSize.z);
    else
        partialNavierStokesSolver3D_C1_kernel<false><<<grids, threads, 0, stream>>>
            (mdFFTArray, mAlpha, mGamma, mComplexSize.x, mComplexSize.y, mComplexSize.z);

    //3.c Get the result using inverse
    cufftExecC2R(mPlanC2R, (cufftComplex*) mdFFTArray, dData);
}

void FFTSolverPlan3D::applyImpl(float* dDataX, float* dDataY, float* dDataZ, bool inverseOp, cudaStream_t stream){
    if (mBeta != 0.f) {
        applyFull(dDataX, dDataY, dDataZ, inverseOp, stream);
    } else {
        if (!mSaveMemory)
            applyPartial(dDataX, dDataY, dDataZ, inverseOp, stream);
        else { // Run the save memory version on each of the channel
            applySingeChannel(dDataX, inverseOp, stream);
            applySingeChannel(dDataY, inverseOp, stream);
            applySingeChannel(dDataZ, inverseOp, stream);
        } 
    }
}

void FFTSolverPlan3D::applyImpl(float* dData, bool inverseOp, cudaStream_t stream){
    if (mBeta != 0.f) {
        //throw AtlasWerksException(__FILE__,__LINE__,"No beta parametter on a single channel image");
    } else {
        applySingeChannel(dData, inverseOp, stream);
    }
}

void FFTSolverPlan3D::apply(float* d_oData, const float* d_iData, bool inverseOp, cudaStream_t stream)
{
    this->preScale(d_oData, d_iData, stream);
    applyImpl(d_oData, inverseOp, stream);
}

void FFTSolverPlan3D::apply(float* d_Data, bool inverseOp, cudaStream_t stream)
{
    this->preScale(d_Data, stream);
    applyImpl(d_Data, inverseOp);
}

void FFTSolverPlan3D::apply(float* d_oDataX, float* d_oDataY, float* d_oDataZ,
                            const float* d_iDataX, const float* d_iDataY, const float* d_iDataZ,
                            bool inverseOp, cudaStream_t stream)
{
    this->preScale(d_oDataX, d_oDataY, d_oDataZ,
                   d_iDataX, d_iDataY, d_iDataZ, stream);
    applyImpl(d_oDataX, d_oDataY, d_oDataZ, inverseOp, stream);
}

void FFTSolverPlan3D::apply(float* dDataX, float* dDataY, float* dDataZ, bool inverseOp, cudaStream_t stream)
{
    this->preScale(dDataX, dDataY, dDataZ, stream);
    applyImpl(dDataX, dDataY, dDataZ, inverseOp, stream);
}

void FFTSolverPlan3D::apply(cplVector3DArray& d_o, const cplVector3DArray& d_i, bool inverseOp, cudaStream_t stream)
{
    this->preScale(d_o, d_i, stream);
    applyImpl(d_o.x, d_o.y, d_o.z, inverseOp, stream);
}

void FFTSolverPlan3D::apply(cplVector3DArray& d_data, bool inverseOp, cudaStream_t stream)
{
    this->preScale(d_data, stream);
    applyImpl(d_data.x, d_data.y, d_data.z, inverseOp, stream);
}

// d_v = L^{-1} d_f
void FFTSolverPlan3D::applyInverseOperator(cplVector3DArray& d_v, const cplVector3DArray& d_f, cudaStream_t stream){
    apply(d_v, d_f, true, stream);
}

void FFTSolverPlan3D::applyInverseOperator(cplVector3DArray& d_f, cudaStream_t stream){
    apply(d_f, true, stream);
}

// d_f = L d_v
void FFTSolverPlan3D::applyOperator(cplVector3DArray& d_f, const cplVector3DArray& d_v, cudaStream_t stream){
    apply(d_f, d_v, false, stream);
}

void FFTSolverPlan3D::applyOperator(cplVector3DArray& d_v, cudaStream_t stream){
    apply(d_v, false, stream);

}
#endif
