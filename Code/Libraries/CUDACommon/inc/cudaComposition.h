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

#ifndef __CUDA_COMPOSITION_H
#define __CUDA_COMPOSITION_H

#include <Vector3D.h>
#include <cuda_runtime.h>

class cplVector3DArray;

/**
 * h(x) = f(x + delta*g(x))
 */
void cplBackwardMapping(float* d_hX, const float* d_fX, 
                        const float* d_gX, const float* d_gY, const float* d_gZ,
                        int sizeX, int sizeY, int sizeZ,
                        float delta,
                        int backgroundStrategy, cudaStream_t stream=NULL);

void cplBackwardMapping(float* d_hX, const float* d_fX, 
                        const cplVector3DArray& d_g, const Vector3Di& size, 
                        float delta,
                        int backgroundStrategy, cudaStream_t stream=NULL);

/**
 * h(x) = f(x + delta*g(x))
 */
void cplBackwardMapping(float* d_hX, float* d_hY, float* d_hZ,
                        const float* d_fX, const float* d_fY, const float* d_fZ,
                        const float* d_gX, const float* d_gY, const float* d_gZ,
                        int sizeX, int sizeY, int sizeZ,
                        float delta,
                        int backgroundStrategy, cudaStream_t stream=NULL);

void cplBackwardMapping(cplVector3DArray& d_h, const cplVector3DArray& d_f, const cplVector3DArray& d_g,
                        const Vector3Di& size,
                        float delta,
                        int backgroundStrategy, cudaStream_t stream=NULL);

/**
 * h(x) = f(x + delta*g(x))
 */
void cplBackwardMapping_tex(float* d_hX, float* d_hY, float* d_hZ,
                            const float* d_fX, const float* d_fY, const float* d_fZ,
                            const float* d_gX, const float* d_gY, const float* d_gZ,
                            int sizeX, int sizeY, int sizeZ,
                            float delta,
                            int backgroundStrategy, cudaStream_t stream=NULL);

void cplBackwardMapping_tex(cplVector3DArray& d_h, const cplVector3DArray& d_f, const cplVector3DArray& d_g,
                            const Vector3Di& size,
                            float delta,
                            int backgroundStrategy, cudaStream_t stream=NULL);
#endif
