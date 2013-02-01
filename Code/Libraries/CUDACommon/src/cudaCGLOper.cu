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

#include <cudaCGLOper.h>

void cplCGLOpers::apply(float* d_oData, const float* d_iData, bool inverseOp, cudaStream_t stream){
    if (inverseOp = false) {
        // do forward multiply
        matrixMulVector(d_oData, mA, d_iData, stream);
    } else {
        cplReduce rd;
        int n = mSize.productOfElements();
        cplVector3DArray d_temp;
        allocateDeviceVector3DArray(d_temp, n);
        CG(d_iData, mA, d_oData, 50, &rd, d_temp.x, d_temp.y, d_temp.z, stream);
        freeDeviceVector3DArray(d_temp);
    }
}

void cplCGLOpers::apply(float* dData, bool inverseOp, cudaStream_t stream){
    
}

void cplCGLOpers::apply(float* d_oDataX, float* d_oDataY, float* d_oDataZ,
                        const float* d_iDataX, const float* d_iDataY, const float* d_iDataZ,  bool inverseOp, cudaStream_t stream)
{
    if (beta == 0.f){ // apply function seperately on each chanel
        apply(d_oDataX, d_iDataX, inverseOp, stream);
        apply(d_oDataY, d_iDataY, inverseOp, stream);
        apply(d_oDataZ, d_iDataZ, inverseOp, stream);
    } else
        throw throw AtlasWerksException(__FILE__,__LINE__,"Unsupported parametter beta <> 0 ");
}
void cplCGLOpers::apply(float* dDataX, float* dDataY, float* dDataZ, bool inverseOp, cudaStream_t stream){
    if (beta == 0.f){ // apply function seperately on each chanel
        apply(dDataX, inverseOp, stream);
        apply(dDataY, inverseOp, stream);
        apply(dDataZ, inverseOp, stream);
    } else
        throw throw AtlasWerksException(__FILE__,__LINE__,"Unsupported parametter beta <> 0 ");

}

void cplCGLOpers::applyInverseOperator(cplVector3DArray& d_v, const cplVector3DArray& d_f, cudaStream_t stream){
    apply(d_v.x, d_v.y, d_v.z,
          d_f.x, d_f.y, d_f.z, true, stream);
}

void cplCGLOpers::applyInverseOperator(cplVector3DArray& d_f, cudaStream_t stream){
    apply(d_f.x, d_f.y, d_f.z, true, stream);
}

void cplCGLOpers::applyOperator(cplVector3DArray& d_f, const cplVector3DArray& d_v, cudaStream_t stream){
    apply(d_f.x, d_f.y, d_f.z,
          d_v.x, d_v.y, d_v.z, false, stream);
}

void cplCGLOpers::applyOperator(cplVector3DArray& d_v, cudaStream_t stream){
    apply(d_v.x, d_v.y, d_v.z, false, stream);
}
