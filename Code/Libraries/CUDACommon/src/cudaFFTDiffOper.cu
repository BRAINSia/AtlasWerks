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


#include "DiffOperGPU.h"
//#include <cutil_math.h>
#include <cutil_inline.h>
#include <cutil_comfunc.h>
#include "cudaInterface.h"
#include <cufft.h>

/*------------------------------------------------------------------------------------------*/
#define MAX_LENGTH 512
__device__ __constant__ float sdCosWX[MAX_LENGTH];
__device__ __constant__ float sdSinWX[MAX_LENGTH];
__device__ __constant__ float sdCosWY[MAX_LENGTH];
__device__ __constant__ float sdSinWY[MAX_LENGTH];
__device__ __constant__ float sdCosWZ[MAX_LENGTH];
__device__ __constant__ float sdSinWZ[MAX_LENGTH];

//--------------------------------------------------------------------------------

// ################ LookupTable3D ################ //

void 
LookupTable3D::
Upload(){
  cudaMemcpyToSymbol(sdCosWX, mCosWX, mSize.x * sizeof(float));
  cudaMemcpyToSymbol(sdSinWX, mSinWX, mSize.x * sizeof(float));
  cudaMemcpyToSymbol(sdCosWY, mCosWY, mSize.y * sizeof(float));
  cudaMemcpyToSymbol(sdSinWY, mSinWY, mSize.y * sizeof(float));
  cudaMemcpyToSymbol(sdCosWZ, mCosWZ, mSize.z * sizeof(float));
  cudaMemcpyToSymbol(sdSinWZ, mSinWZ, mSize.z * sizeof(float));
  checkCUDAError("LookupTable3D::Upload");
}
    
void 
LookupTable3D::
InitTable(){

  //
  // Test that MAX_LENGTH is big enough
  //
  if(mSize.x > MAX_LENGTH || 
     mSize.y > MAX_LENGTH || 
     mSize.z > MAX_LENGTH)
    {
        //throw AtlasWerksException(__FILE__,__LINE__,"Error, not enough static memory allocated!");
    }

  //
  // precompute some values
  //
  double sX = mSp.x * 2.0 * M_PI / mSize.x; 
  double sY = mSp.y * 2.0 * M_PI / mSize.y; 
  double sZ = mSp.z * 2.0 * M_PI / mSize.z; 

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
    
void 
LookupTable3D::
Allocate(){
  mCosWX = new float [MAX_LENGTH];
  mCosWY = new float [MAX_LENGTH];
  mCosWZ = new float [MAX_LENGTH];
  
  mSinWX = new float [MAX_LENGTH];
  mSinWY = new float [MAX_LENGTH];
  mSinWZ = new float [MAX_LENGTH];
}

void 
LookupTable3D::
SetSize(const Vector3Di& size, const Vector3Df& spacing){
  mSize = size;
  mSp   = spacing;
  InitTable();
  Upload();
}

void 
LookupTable3D::
Clear(){
  delete []mCosWX;
  delete []mCosWY;
  delete []mCosWZ;

  delete []mSinWX;
  delete []mSinWY;
  delete []mSinWZ;
}

// ################ DiffOperGPU ################ //

// ======== Public Members ======== //

DiffOperGPU::DiffOperGPU(const Vector3Di &size)
{
  // need to initialize internal FFT field
    this->SetSize(size);
}


DiffOperGPU::
~DiffOperGPU()
{
  this->Delete();
}

void
DiffOperGPU::SetSize(const Vector3Di &size, const Vector3Df &spacing)
{
  mSize = size;
  mSp   = spacing;
  
  mComplexSize   = mSize;
  mComplexSize.x = mSize.x/2+1;

  if(mComplexSize.productOfElements() > mMaxComplexSize){
      throw AtlasWerksException(__FILE__,__LINE__,"Error, size exceeds maximum size as specified in constructor");
  }

  // recompute the lookup table
  mLookupTable.SetSize(mSize, mSp);

  // create FFT plan
  // ATTENTION : the order is reversed from CUDA FFT documentation
  // this is hard to find bug since it is not failed if 
  // mComplexSize.z = mComplexSize.x
  cufftPlan3d(&mPlanR2C, mSize.z, mSize.y, mSize.x, CUFFT_R2C);
  cufftPlan3d(&mPlanC2R, mSize.z, mSize.y, mSize.x, CUFFT_C2R);
  checkCUDAError("FFTW Plan Creation");
  
  this->SetParams(param);
}

void 
DiffOperGPU::
SetParams(const DiffOperParam &param){
  // check for unimplemented parameters
  // set values from param
  this->SetAlpha(param.Alpha());
  this->SetBeta(param.Beta());
  this->SetGamma(param.Gamma());
  this->SetLPow(param.LPow());
  this->SetUseEigenLUT(param.UseEigenLUT());
  this->SetDivergenceFree(param.DivergenceFree());
}

DiffOperParam 
DiffOperGPU::
GetParams(){
  DiffOperParam param;
  param.Alpha() = this->GetAlpha();
  param.Beta() = this->GetBeta();
  param.Gamma() = this->GetGamma();
  param.LPow() = this->GetLPow();
  param.UseEigenLUT() = this->GetUseEigenLUT();
  param.DivergenceFree() = this->GetDivergenceFree();
  return param;
}

void DiffOperGPU::
SetAlpha(Real alpha){
  if(alpha != this->mAlpha){
    this->mAlpha = alpha;
  }
}

Real DiffOperGPU::
GetAlpha(){
  return this->mAlpha;
}

void DiffOperGPU::
SetBeta(Real beta){
  if(beta != this->mBeta){
    this->mBeta = beta;
  }
}

Real DiffOperGPU::
GetBeta(){
  return this->mBeta;
}

void DiffOperGPU::
SetGamma(Real gamma){
  if(gamma != this->mGamma){
    this->mGamma = gamma;
  }
}

Real DiffOperGPU::
GetGamma(){
  return this->mGamma;
}

void DiffOperGPU::
SetDivergenceFree(bool df){
  if(df){
      throw AtlasWerksException(__FILE__,__LINE__,"Error, divergence free projection not implemented on the GPU!");
  }
}

bool DiffOperGPU::
GetDivergenceFree(){
  return false;
}

void DiffOperGPU::
SetLPow(Real p){
  if(p != 1.0){
    throw AtlasWerksException(__FILE__,__LINE__,"Error, powers of L not supported on the GPU!");
  }
}

Real DiffOperGPU::
GetLPow(){
  return 1.0f;
}

void DiffOperGPU::
SetUseEigenLUT(bool b){
  if(b){
    throw AtlasWerksException(__FILE__,__LINE__,"Error, eigen LUT not supported on the GPU!");
  }
}

bool DiffOperGPU::
GetUseEigenLUT(){
  return false;
}

void 
DiffOperGPU::
Initialize(const Vector3Di &maxSize)
{
  mMaxComplexSize = (maxSize.x/2 + 1)*maxSize.y*maxSize.z;
  
  int memsize = mMaxComplexSize*sizeof(cplComplex);
  cudaMalloc((void**)&mdFFTArrayX, memsize);
  cudaMalloc((void**)&mdFFTArrayY, memsize);
  cudaMalloc((void**)&mdFFTArrayZ, memsize);
}

void DiffOperGPU::Alloc(){
    mAllocateSize = mComplexSize.productOfElements();
    cudaMalloc((void**)&mdFFTArrayX, mAllocateSize);
    cudaMalloc((void**)&mdFFTArrayY, mAllocateSize);
    cudaMalloc((void**)&mdFFTArrayZ, mAllocateSize);
}

void DiffOperGPU::DeAlloc(){
    cudaSafeDelete(mdFFTArrayX);
    cudaSafeDelete(mdFFTArrayY);
    cudaSafeDelete(mdFFTArrayZ);
}

void 
DiffOperGPU::
Delete()
{
  cufftDestroy(mPlanR2C);
  cufftDestroy(mPlanC2R);
  DeAlloc();
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
  // [ G(2,1) G(2,2)   0    ] * [   0    G(2,2) G(3,2) ] = Amatrix
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

__global__ void fullNavierStokesSolver3D_C3_kernel(cplComplex* bX, cplComplex* bY, cplComplex* bZ,
                                                   const float alpha, const float beta, const float gamma,
                                                   const int sizeX, const int sizeY, const int sizeZ,
						   const bool inverseOp)
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
            

      if(inverseOp){
	InverseOperatorMultiply(bX[index], bY[index], bZ[index],
				L00, L10, L11, L20, L21, L22);
      }else{
	OperatorMultiply(bX[index], bY[index], bZ[index],
			 L00, L10, L11, L20, L21, L22);
      }
    }
  }
}

/**
 * Note: in-place R2C/C2R CUFFT transforms seem to be broken, need to
 * use out-of-place :-(
 */ 
void 
DiffOperGPU::
Apply(float* dDataX, float* dDataY, float* dDataZ, bool inverseOp)
{

  // copy scaled data into FFT array
  uint nElems  = mSize.productOfElements();
  float scale = 1.f / nElems;
  cplVectorOpers::MulC_I(dDataX, scale, nElems);
  cplVectorOpers::MulC_I(dDataY, scale, nElems);
  cplVectorOpers::MulC_I(dDataZ, scale, nElems);

  //1.a Compute FFT of the data (inplace)
  // convert the input from real to frequency(complex image)
  cufftSafeCall(cufftExecR2C(mPlanR2C, (cufftReal*)dDataX, (cufftComplex*) mdFFTArrayX));
  cufftSafeCall(cufftExecR2C(mPlanR2C, (cufftReal*)dDataY, (cufftComplex*) mdFFTArrayY));
  cufftSafeCall(cufftExecR2C(mPlanR2C, (cufftReal*)dDataZ, (cufftComplex*) mdFFTArrayZ));
  
  //2.c Solve system
  dim3 threadBlock(16,16);
  dim3 gridBlock(iDivUp(mComplexSize.x,threadBlock.x), iDivUp(mComplexSize.y, threadBlock.y));
  fullNavierStokesSolver3D_C3_kernel<<<gridBlock, threadBlock>>>(mdFFTArrayX, mdFFTArrayY, mdFFTArrayZ,
								 mAlpha, mBeta, mGamma,
								 mComplexSize.x, mComplexSize.y, mComplexSize.z, inverseOp);
  
  
  cufftExecC2R(mPlanC2R, (cufftComplex*) mdFFTArrayX, dDataX);
  cufftExecC2R(mPlanC2R, (cufftComplex*) mdFFTArrayY, dDataY);
  cufftExecC2R(mPlanC2R, (cufftComplex*) mdFFTArrayZ, dDataZ);

  checkCUDAError("DiffOperGPU::Apply: ");
};

//--------------------------------------------------------------------------------

void 
DiffOperGPU::
ApplyInverseOperator(float* dFx,
		     float* dFy,
		     float* dFz)
{
  Apply(dFx, dFy, dFz, true);
}

void 
DiffOperGPU::
ApplyInverseOperator(cplVector3DArray& dF)
{
  Apply(dF.x, dF.y, dF.z, true);
}

void 
DiffOperGPU::
ApplyOperator(float* dVx,
	      float* dVy,
	      float* dVz)
{
  Apply(dVx, dVy, dVz, false);
}

void 
DiffOperGPU::
ApplyOperator(cplVector3DArray& dV)
{
  Apply(dV.x, dV.y, dV.z, false);
}
