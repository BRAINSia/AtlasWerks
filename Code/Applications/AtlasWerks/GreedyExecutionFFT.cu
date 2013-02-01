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

#ifndef __GREEDY_EXECUTION__H
#define __GREEDY_EXECUTION__H

#include "GreedyExecutionFFT.h"
#include <cutil_comfunc.h>
#include <cudaInterface.h>
#include <cudaImage3D.h>
#include "cudaComposition.h"
#include "cudaHField3DUtils.h"
#include "libDefine.h"

#define NEW_MAP          0
#define STOP_EPS         1e-4

float GreedyExecutionFFT::computeVectorField(float* d_I1, float* d_temp,
                                             cplVector3DArray& d_v3_temp,
                                             cplReduce* p_Rd,
                                             float alpha, float beta, float gamma,
                                             uint  w, uint h, uint l, uint  nIter)
{
  uint nP = w * h * l;
  ////////////////////////////////////////////////////////////////////////////////
  //Compute force function
  //1. F = -(d_I0t - d_I1) * Grad d_I0t
  ////////////////////////////////////////////////////////////////////////////////
  // a -(d_I0t - d_I1)
  cplVectorOpers::Sub(d_temp, d_I1, d_I0t, nP);

  //  Compute the different to see the convergence 
  float mse = p_Rd->Sum2(d_temp, nP) / nP;

  //fprintf(stderr, "MSE %f \n", mse);

  //2. Compute the gradient Grad d_I0t
  cplVector3DArray d_gd = d_v3_temp;
  ////////////////////////////////////////////////////////////////////////////////
  // this is just the pointer copy no actual data copy
  // We only change the notation so that it is
  // easier to keep track with the algorithm
  ////////////////////////////////////////////////////////////////////////////////
  cplComputeGradient(d_gd.x, d_gd.y, d_gd.z, d_I0t, w, h, l, mSx, mSy, mSz, 0);
    
  //3. Compute the Force F = -(d_I0t - d_I1) * Grad d_I0t
  cplVector3DArray d_F = d_gd;        
  cplVector3DOpers::Mul_I(d_F, d_temp, nP);

#ifdef __DEBUG__
  float sumx = p_Rd->Max(d_F.x, nP);
  float sumy = p_Rd->Max(d_F.y, nP);
  float sumz = p_Rd->Max(d_F.z, nP);
    
  fprintf(stderr,"Force field max (%f, %f, %f) \n", sumx, sumy, sumz);

  sumx = getDeviceSumDouble(d_F.x, nP);
  sumy = getDeviceSumDouble(d_F.y, nP);
  sumz = getDeviceSumDouble(d_F.z, nP);
  fprintf(stderr,"Force field sum (%f, %f, %f) \n", sumx, sumy, sumz);
#endif
    
  //4. Compute the velocity field by FFT method
  // Solve the equation (-Laplace + gamma) v = F
  pFFTSolver->setParams(alpha, beta, gamma);
  pFFTSolver->applyInverseOperator(d_v, d_F);
  
  // pFFTSolver->solve(d_v.x ,d_F.x, 
  //                   d_v.y, d_F.y,
  //                   d_v.z, d_F.z, alpha, beta, gamma);

#ifdef __DEBUG__
  sumx = p_Rd->Max(d_v.x, nP);
  sumy = p_Rd->Max(d_v.y, nP);
  sumz = p_Rd->Max(d_v.z, nP);
  fprintf(stderr,"Vector field  max (%f, %f, %f) \n", sumx, sumy, sumz);
#endif
    
  return mse;
}

////////////////////////////////////////////////////////////////////////////////
// Compute the delta = stepL / maxV for the first iteration
// of each resolution
////////////////////////////////////////////////////////////////////////////////
float GreedyExecutionFFT::computeDelta(cplVector3DArray& d_v, cplReduce* p_Rd, float* d_temp, int nP){
    cplVector3DOpers::Magnitude(d_temp, d_v, nP);
    float maxv = p_Rd->Max(d_temp, nP);
    fprintf(stderr, "delta computation: maxv=%f\n", maxv);
    fprintf(stderr, "delta computation: stepL=%f\n", mStepL);
    return (mStepL / maxv);
}

void GreedyExecutionFFT::updateDeformationField(cplVector3DArray& d_v3_temp,
                                                float& delta, uint  w, uint h, uint  l)
{
  cplVector3DArray d_hPb = d_v3_temp;
  
  // Pullback the field h_(k+1) = h_k( x + delta * v_x)
  cplBackwardMapping(d_hPb, d_h, d_v, Vector3Di(w, h, l), delta, BACKGROUND_STRATEGY_ID);

  /*
  cudaBackwardMap(d_hPb.x,d_hPb.y,d_hPb.z,
                  d_h.x, d_h.y, d_h.z,
                  d_v.x, d_v.y, d_v.z,
                  w, h, l,
                  delta, BACKGROUND_STRATEGY_ID);
  */
  
  int nP = w * h * l;
  cudaMemcpy(d_h.x, d_hPb.x, nP * sizeof(float) , cudaMemcpyDeviceToDevice);
  cudaMemcpy(d_h.y, d_hPb.y, nP * sizeof(float) , cudaMemcpyDeviceToDevice);
  cudaMemcpy(d_h.z, d_hPb.z, nP * sizeof(float) , cudaMemcpyDeviceToDevice);
  
}

void GreedyExecutionFFT::updateImage(uint w, uint h, uint l)
{
  // I0t = I0(h_(k+1))
  //cudaHField3DUtils::cudaHFieldApply(d_I0t, d_I0, d_h.x, d_h.y, d_h.z, w, h, l);
  cudaHField3DUtils::apply(d_I0t, d_I0, d_h.x, d_h.y, d_h.z, w, h, l);
}

////////////////////////////////////////////////////////////////////////////////
//perform the computation with local delta value
////////////////////////////////////////////////////////////////////////////////
float GreedyExecutionFFT::step(float* d_I1, float* d_temp,
                               cplVector3DArray& d_v3_temp,
                               cplReduce* p_Rd,
                               float alpha, float beta, float gamma,
                               uint  w    , uint h    , uint  l,
                               uint niter)
{
  int nP = w * h * l;
  
  // Compute the vector field 
  float mse = computeVectorField(d_I1,  d_temp, d_v3_temp, p_Rd, alpha, beta, gamma, w, h, l, niter);
  
  // Compute the delta if needed, this is the individual delta value
  if (mDelta == 0.f){
    //delta= computeDelta(d_v, p_Rd, nP);
    mDelta = computeDelta(d_v, p_Rd,d_temp, nP);
    fprintf(stderr, "Delta %f \n", mDelta);
  }
  // update the defomation field
  updateDeformationField(d_v3_temp, mDelta, w, h, l);

  // update the image
  updateImage(w, h, l);

  return mse;
}


////////////////////////////////////////////////////////////////////////////////
//perform the computation with global delta value
////////////////////////////////////////////////////////////////////////////////
float GreedyExecutionFFT::step(float* d_I1, float* d_temp,
                               cplVector3DArray& d_v3_temp,
                               cplReduce* p_Rd,
                               float& delta, float alpha, float beta, float gamma,
                               uint  w    , uint h    , uint  l,
                               uint niter)
{

  int nP = w * h * l;
    
  // Compute the vector field 
  float mse = computeVectorField(d_I1,  d_temp, d_v3_temp, p_Rd, alpha, beta, gamma, w, h, l, niter);

  // Compute the delta if needed
  if (delta == 0.f){
    //delta = computeDelta(d_v, p_Rd, nP);
    delta = computeDelta(d_v, p_Rd,d_temp, nP);
  }

  // update the defomation field
  updateDeformationField(d_v3_temp, delta, w, h, l);

  // update the image
  updateImage(w, h, l);

  return mse;
}


#endif
