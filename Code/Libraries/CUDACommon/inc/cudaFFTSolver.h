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

#ifndef __CUDA__FFT_SOLVER_H
#define __CUDA__FFT_SOLVER_H

#include <cufft.h>
#include <Vector3D.h>
#include <cudaComplex.h>

class cplVector3DArray;

class LookupTable2D{
public:
    LookupTable2D():cosWX(NULL), cosWY(NULL),
                    sinWX(NULL), sinWY(NULL) {
            allocate();
    };
    ~LookupTable2D(){
        clear();
    }

    void setSize(unsigned w, unsigned h, double sx= 1.0, double sy=1.0);
    void upload();
    
protected:
    void allocate();
    void initTable(double dX, double dY);
    void clear();
    
    // host data 
    float* cosWX,* cosWY;
    float* sinWX,* sinWY;
    
    uint m_xSize, m_ySize;
    double deltaX, deltaY;
};

class FFTSolverPlan2D{
public:
    FFTSolverPlan2D(){};

    ~FFTSolverPlan2D(){
        clean();
    }
    
    LookupTable2D lut;
    void setSize(unsigned int fft_w, unsigned fft_h,
                 double sx =1.0, double sy=1.0);
    
    void solve(float* d_vx, float* d_fx, float alpha, float gamma);
    void solve(float* d_vx, float* d_fx,
               float* d_vy, float* d_fy,
               float alpha, float gamma);

    
    void clean();
    cufftHandle planR2C, planC2R;
    unsigned int fft_w, fft_h;
};

class LookupTable3D{
public:
    LookupTable3D():mCosWX(NULL), mCosWY(NULL), mCosWZ(NULL),
                    mSinWX(NULL), mSinWY(NULL), mSinWZ(NULL)
        {
            allocate();
        };
    
    ~LookupTable3D(){
        clear();
    }
    void setSize(const Vector3Di& size, const Vector3Df& sp=Vector3Df(1.f, 1.f, 1.f));
    void upload();
protected:
    void allocate();
    void initTable(double dX = 1.0, double dY = 1.0, double dZ= 1.0);
    
    void clear();
    
    // host data 
    float* mCosWX,* mCosWY,* mCosWZ;
    float* mSinWX,* mSinWY,* mSinWZ;

    Vector3Di mSize;
    Vector3Df mSp;
};

class FFTSolverPlan3D{
public:
    FFTSolverPlan3D(bool saveMem=false);
    ~FFTSolverPlan3D(){
        clean();
    }
    void setSize(const Vector3Di& size, const Vector3Df& sp=Vector3Df(1.f, 1.f, 1.f));
    void setParams(float alpha, float beta, float gamma);
    void getParams(float& alpha, float& beta, float& gamma) const {
        alpha = mAlpha;
        beta  = mBeta;
        gamma = mGamma;
    }
    
    // General Operator
    void apply(float* d_oData, const float* d_iData, bool inverseOp, cudaStream_t stream=NULL);
    void apply(float* dData, bool inverseOp, cudaStream_t stream=NULL);

    void apply(float* d_oDataX, float* d_oDataY, float* d_oDataZ,
               const float* d_iDataX, const float* d_iDataY, const float* d_iDataZ,
               bool inverseOp, cudaStream_t stream=NULL);
    void apply(float* dDataX, float* dDataY, float* dDataZ, bool inverseOp, cudaStream_t stream=NULL);

    void apply(cplVector3DArray& d_o, const cplVector3DArray& d_i, bool inverseOp, cudaStream_t stream=NULL);
    void apply(cplVector3DArray& d_data, bool inverseOp, cudaStream_t stream=NULL);

    // Vector field operator
    void applyInverseOperator(cplVector3DArray& d_v, const cplVector3DArray& d_f, cudaStream_t stream=NULL);
    void applyInverseOperator(cplVector3DArray& d_f, cudaStream_t stream=NULL);
    void applyOperator(cplVector3DArray& d_f, const cplVector3DArray& d_v, cudaStream_t stream=NULL);
    void applyOperator(cplVector3DArray& d_v, cudaStream_t stream=NULL);
private:
    void alloc();
    void dealloc();
        
    void createCUFFTPlan();
    void destroyCUFFTPlan();

    void clean();

    void preScale(float* dData, cudaStream_t stream=NULL);
    void preScale(float* dDataX, float* dDataY, float* dDataZ, cudaStream_t stream=NULL);

    void preScale(float* d_oData, const float* d_iData, cudaStream_t stream=NULL);
    void preScale(float* d_oDataX, float* d_oDataY, float* d_oDataZ,
                  const float* d_iDataX, const float* d_iDataY, const float* d_iDataZ, cudaStream_t stream=NULL);

    void preScale(cplVector3DArray& d_o, const cplVector3DArray& d_i, cudaStream_t stream=NULL);
    void preScale(cplVector3DArray& d_data, cudaStream_t stream=NULL);
    
    
    void applyFull(float* dDataX, float* dDataY, float* dDataZ, bool inverseOp, cudaStream_t stream=NULL);
    void applyPartial(float* dDataX, float* dDataY, float* dDataZ, bool inverseOp, cudaStream_t stream=NULL);
    void applySingeChannel(float* dData, bool inverseOp, cudaStream_t stream=NULL);

    void applyImpl(float* dDataX, float* dDataY, float* dDataZ, bool inverseOp, cudaStream_t stream=NULL);
    void applyImpl(float* dData, bool inverseOp, cudaStream_t stream=NULL);

    // Size of the processing volume 
    Vector3Di mSize;
    Vector3Di mComplexSize;
    Vector3Df mSp;
    unsigned int mAllocateSize;

    LookupTable3D mLookupTable;
    
    // CUDA FFT handle
    cufftHandle mPlanR2C, mPlanC2R;

    bool mSaveMemory;
    // Fluid param
    float mAlpha, mBeta, mGamma;

    cplComplex* mdFFTArrayX;
    cplComplex* mdFFTArrayY;
    cplComplex* mdFFTArrayZ;
    cplComplex* mdFFTArray;
};



#endif
