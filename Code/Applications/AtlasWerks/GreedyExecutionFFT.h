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

#ifndef __GREEDY_EXECUTION_FFT_H__
#define __GREEDY_EXECUTION_FFT_H__

#include <VectorMath.h>
#include <cudaReduce.h>
#include <cudaFFTSolver.h>
#include <cudaVector3DArray.h>

void writeDeviceToNrrd(float* d_data, int w, int h, int l, char *name) ;
void writeToNrrd(float* data, int w, int h, int l, char *name);
/*----------------------------------------------------------------------------------------------------*/
class GreedyExecutionFFT{
public:
    float* d_I0;           // device input image
    float* d_I0t;          // device template image at time t

    cplVector3DArray d_v; // Vector field
    cplVector3DArray d_h; // the deformation field at time t

    float mDelta;           // Local or individual delta value
    float mStepL;           // Maximum step length 
    
    int greedyStep(float* d_I1, float* d_temp, cplVector3DArray& d_v3_temp,
                   cplReduce* p_Rd,
                   float& delta,
                   float alpha, float beta, float gamma,
                   uint  w    , uint h    , uint  l,
                   uint niter);

    float computeVectorField(float* d_I1, float* d_temp,
                             cplVector3DArray& d_v3_temp,
                             cplReduce* p_Rd,
                             float alpha, float beta, float gamma,
                             uint  w, uint h, uint l, uint  nIter);
    
    float computeDelta(cplVector3DArray& d_v, cplReduce* p_Rd, float* d_temp, int nP);
    
    
    void updateDeformationField(cplVector3DArray& d_v3_temp,
                                float& delta, uint  w, uint h    , uint  l);
    

    void updateImage(uint w, uint h, uint l);

    
    float step(float* d_I1, float* d_temp, cplVector3DArray& d_v3_temp,
               cplReduce* p_Rd,
               float& delta,
               float alpha, float beta, float gamma, 
               uint  w    , uint h    , uint  l,
               uint niter);

    float step(float* d_I1, float* d_temp, cplVector3DArray& d_v3_temp,
               cplReduce* p_Rd,
               float alpha, float beta, float gamma,
               uint  w    , uint h    , uint  l,
               uint niter);

    FFTSolverPlan3D* pFFTSolver;
    // spacing
    float mSx, mSy, mSz;
    // Step Length that affect the convergence speed
    // it should be chosen to guarantee the small deformation requirement
    // so step length should be strictly < 1

};

#endif // __GREEDY_EXECUTION_FFT_H__
