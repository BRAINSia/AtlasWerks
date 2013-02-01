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

#ifndef __CUDA_MESH_OPERS_H
#define __CUDA_MESH_OPERS_H

#include <cudaReduce.h>
#include <cudaVector3DArray.h>
#include <cudaIndex3DArray.h>
#include <Vector3D.h>

namespace cplMeshOper {
    void   computeBoundingBox(cplReduce& rd, Vector3Df& minPnt, Vector3Df& maxPnt, cplVector3DArray& d_pnt, int n);
    Vector3Df computeMeanPoint(cplReduce& rd, const cplVector3DArray& d_pnt, int n);
    
    void computeFaceNormal(cplVector3DArray& fnorm, cplIndex3DArray & d_idx, int nF,
                           float4* d_pnt, int nV, cudaStream_t stream=cudaStream_t(NULL));

    void computeWeightedFaceNormal(cplVector3DArray& fnorm, cplIndex3DArray & d_idx, int nF,
                                   float4* d_pnt, int nV, cudaStream_t stream=cudaStream_t(NULL));

    
    void computeCentroid(cplVector3DArray& d_cen, cplIndex3DArray & d_idx, int nF, float4* d_pnt, int nV, cudaStream_t stream=cudaStream_t(NULL));
    
    void computeCentroidNormal(cplVector3DArray& d_cen, cplVector3DArray& d_norm, cplIndex3DArray & d_idx, int nF,
                               float4* d_pnt, int nV, cudaStream_t stream=cudaStream_t(NULL));

    void computeCentroidWeightedNormal(cplVector3DArray& d_cen, cplVector3DArray& d_norm,
                                       cplIndex3DArray & d_idx, int nF, float4* d_pnt, int nV, cudaStream_t stream=cudaStream_t(NULL));
    

    void computeVertexNormal_fixedPoint(cplVector3DArray& d_vNorm, float4* d_pnt, int nV,
                                        cplIndex3DArray& d_i, int nF, float avgEdgeLength, cudaStream_t stream = 0);

    float computeMaxEdgeLength(cplReduce& rd, float4* d_pnt, int nV,
                               cplIndex3DArray& d_i, int nF, cudaStream_t stream=cudaStream_t(NULL));

    float computeAvgEdgeLength(cplReduce& rd, float4* d_pnt, int nV,
                               cplIndex3DArray& d_i, int nF, cudaStream_t stream=cudaStream_t(NULL));


    void cplSplatNormal(cplVector3DArray& d_on, const Vector3Di& size,
                         cplVector3DArray& d_in, cplVector3DArray& d_pos, int nP, cudaStream_t stream=cudaStream_t(NULL));

    void cplInterpolateFromGrid(float* d_o, cplVector3DArray& d_pos, int nP,
                                 float* d_grid, int sizeX, int sizeY, int sizeZ, cudaStream_t stream=cudaStream_t(NULL));
    
    void cplInterpolateFromGrid(float* d_o,    cplVector3DArray& d_pos, int nP,
                                 float* d_grid, const Vector3Di& size, cudaStream_t stream=cudaStream_t(NULL));

    void cplInterpolateFromGrid(cplVector3DArray& d_o, cplVector3DArray& d_pos, int nP,
                                 cplVector3DArray& d_grid,
                                 int sizeX, int sizeY, int sizeZ, cudaStream_t stream=cudaStream_t(NULL));
    
    void cplInterpolateFromGrid(cplVector3DArray& d_o, cplVector3DArray& d_pos, int nP,
                                 cplVector3DArray& d_grid,
                                 const Vector3Di& size, cudaStream_t stream=cudaStream_t(NULL));

    void cplUpdateNormalVertexInfluence(cplVector3DArray& d_norm,
                                         float4* d_pnt, float4* d_Kn, int nV,
                                         cplIndex3DArray& d_i, int nF, cudaStream_t stream=cudaStream_t(NULL));
    
    void cplUpdateNormalVertexInfluence_fixedPoint(cplVector3DArray& d_norm,
                                                    float4* d_pnt, float4* d_Kn, int nV,
                                                    cplIndex3DArray& d_i, int nF, cudaStream_t stream=cudaStream_t(NULL));
    

 
 
};
#endif
