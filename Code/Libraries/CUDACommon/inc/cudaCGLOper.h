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

#ifndef __CUDA_CG_LOPER_H
#define __CUDA_CG_LOPER_H

template<class T>
class cplCGLOpers {
public:
    cplCGLOpers() {};
    
    void setSize(const Vector3Di& size, const Vector3Df& sp) {
        mSize = size;
        mSp   = sp;

        mA.setSize(size);
    }
    void setParams(float alpha, float beta, float gamma){
        mAlpha = alpha;
        mBeta  = beta;
        mGamma = mGamma;
        
        mA.setDiffParams(alpha, beta);
    }
    void apply(float* d_oData, const float* d_iData, bool inverseOp, cudaStream_t stream=NULL);
    void apply(float* dData, bool inverseOp, cudaStream_t stream=NULL);
    void apply(float* d_oDataX, float* d_oDataY, float* d_oDataZ,
               const float* d_iDataX, const float* d_iDataY, const float* d_iDataZ,  bool inverseOp, cudaStream_t stream=NULL);
    void apply(float* dDataX, float* dDataY, float* dDataZ, bool inverseOp, cudaStream_t stream=NULL);
    // Vector field operator
    void applyInverseOperator(cplVector3DArray& d_v, const cplVector3DArray& d_f, cudaStream_t stream=NULL);
    void applyInverseOperator(cplVector3DArray& d_f, cudaStream_t stream=NULL);
    void applyOperator(cplVector3DArray& d_f, const cplVector3DArray& d_v, cudaStream_t stream=NULL);
    void applyOperator(cplVector3DArray& d_v, cudaStream_t stream=NULL);
private:
    Vector3Di mSize;
    Vector3Df mSp;
    float mAlpha, mBeta, mGamma;

    class T mA;
};

#endif
