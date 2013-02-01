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

#ifndef __CUDA_SOR_SOLVER_H
#define __CUDA_SOR_SOLVER_H

#include <Vector3D.h>
#include <cuda_runtime.h>
class cplVector3DArray;

void cplSORSolver3D(float* d_gv, float* d_gf,
                     float alpha, float gamma,
                     int w, int h, int l,
                     int nIters, float* d_temp, cudaStream_t stream=NULL);

void cplSORSolver3D(float* d_gv, float* d_gf,
                     float alpha, float gamma,
                     const Vector3Di& size, int nIters, float* d_temp, cudaStream_t stream=NULL);

void cplSORSolver3D(cplVector3DArray& d_gv, cplVector3DArray& d_gf,
                     float alpha, float gamma, const Vector3Di& size, int nIters, float* d_temp, cudaStream_t stream=NULL);

void cplSORSolver3D_shared(float* d_gv, float* d_gf,
                            float alpha, float gamma,
                            const Vector3Di& size, int nIters, float* d_temp, cudaStream_t stream=NULL);

void testSOR(int sizeX, int sizeY, int sizeZ);

#endif
