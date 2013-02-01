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

#ifndef __CUDA_LOPERS_H
#define __CUDA_LOPERS_H

class cudaLOpers {
public:
    cudaLOpers() {};
    
    void setSize(const Vector3Di& size, const Vector3Df& sp) {
        mSize = size;
        mSp   = sp;
    }
    void setParams(float alpha, float beta, float gamma){
        mAlpha = alpha;
        mBeta  = beta;
        mGamma = mGamma;
    }
    void apply(float* d_oData, const float* d_iData, bool inverseOp);
    void apply(float* dData, bool inverseOp);
    void apply(float* d_oDataX, float* d_oDataY, float* d_oDataZ,
               const float* d_iDataX, const float* d_iDataY, const float* d_iDataZ,  bool inverseOp);
    void apply(float* dDataX, float* dDataY, float* dDataZ, bool inverseOp);
    // Vector field operator
    void applyInverseOperator(cplVector3DArray& d_v, const cplVector3DArray& d_f);
    void applyInverseOperator(cplVector3DArray& d_f);
    void applyOperator(cplVector3DArray& d_f, const cplVector3DArray& d_v);
    void applyOperator(cplVector3DArray& d_v);

private:
    Vector3Di mSize;
    Vector3Df mSp;
    float mAlpha, mBeta, mGamma;
};


#endif
