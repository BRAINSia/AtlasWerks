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

#include <VectorMath.h>
#include <cudaInterface.h>
#include <cutil_comfunc.h>
#include <cplMacro.h>
#include <cudaSplat.h>
#include <cudaImage3D.h>
#include <mesh/cudaMeshOper.h>
#include <VectorMath.h>

// Atomic floating point function by SPWorley CUDA Forum
__inline__ __device__ void atomicAddf(float* addr, float data){
    while (data) data=atomicExch(addr, data+atomicExch(addr, 0.0f));
}

namespace cplMeshOper {
    
    texture<float4, 1, cudaReadModeElementType> cgl_VertexTexture;
    texture<int4  , 1, cudaReadModeElementType> cgl_IndexTexture;

    ////////////////////////////////////////////////////////////////////////////////
// Compute the bounding box of a set of point
////////////////////////////////////////////////////////////////////////////////
    void computeBoundingBox(cplReduce& rd,
                            Vector3Df& minPnt, Vector3Df& maxPnt,
                            cplVector3DArray& d_pnt, int n)
    {
        MaxMin(rd, maxPnt, minPnt, d_pnt, n);
    }

////////////////////////////////////////////////////////////////////////////////
// compute the mean point of the set of point 
////////////////////////////////////////////////////////////////////////////////
    Vector3Df computeMeanPoint(cplReduce& rd, const cplVector3DArray& d_pnt, int n)
    {
        return Sum(rd, d_pnt, n) / n;
    }

    inline __device__ __host__ float3 faceNormal(float3 pa, float3 pb, float3 pc)
    {
        // compute the normal
        float3 e0 = pb - pa;
        float3 e1 = pc - pa;
        return normalize(cross(e0, e1));
    }

    inline __device__ __host__ float3 faceWeightedNormal(float3 pa, float3 pb, float3 pc)
    {
        // compute the normal
        float3 e0 = pb - pa;
        float3 e1 = pc - pa;
        float3 un = cross(e0, e1) / 2.f;
        return un;
    }


    template<bool normalized>
    __global__ void computeFaceNormal_kernel(float* nX, float* nY, float* nZ,
                                             uint* aIdx, uint* bIdx, uint* cIdx,
                                             int n)
    {
        uint blockId = get_blockID();
        uint id      = get_threadID(blockId);

        if (id < n){
            // read the 3 index
            uint aid = aIdx[id], bid = bIdx[id], cid = cIdx[id];
        
            float4 a = tex1Dfetch(cgl_VertexTexture, aid);
            float4 b = tex1Dfetch(cgl_VertexTexture, bid);
            float4 c = tex1Dfetch(cgl_VertexTexture, cid);

            float3 n;
            if (normalized)
                n = faceNormal(make_float3(a), make_float3(b), make_float3(c));
            else 
                n = faceWeightedNormal(make_float3(a), make_float3(b), make_float3(c));
        
            // return the normal
            nX[id] = n.x;
            nY[id] = n.y;
            nZ[id] = n.z;
        }
    }

    void computeFaceNormal(cplVector3DArray& fnorm, 
                           cplIndex3DArray & d_idx, int nF,
                           float4* d_pnt, int nV, cudaStream_t stream)
    {
        dim3 threads(256);
        dim3 grids(iDivUp(nF, threads.x));
        checkConfig(grids);

        cudaBindTexture(0, cgl_VertexTexture, d_pnt, nV * sizeof(float4));
        computeFaceNormal_kernel<true><<<grids, threads, 0, stream>>>(fnorm.x, fnorm.y, fnorm.z,
                                                                      d_idx.aIdx, d_idx.bIdx, d_idx.cIdx, nF);
    }

    void computeWeightedFaceNormal(cplVector3DArray& fnorm, 
                                   cplIndex3DArray & d_idx, int nF,
                                   float4* d_pnt, int nV, cudaStream_t stream)
    {
        dim3 threads(256);
        dim3 grids(iDivUp(nF, threads.x));
        checkConfig(grids);

        cudaBindTexture(0, cgl_VertexTexture, d_pnt, nV * sizeof(float4));
        computeFaceNormal_kernel<false><<<grids, threads, 0, stream>>>(fnorm.x, fnorm.y, fnorm.z,
                                                                       d_idx.aIdx, d_idx.bIdx, d_idx.cIdx, nF);
    }


//////////////////////////////////////////////////////////////////////////////////
// Compute the compute the face centroid
//////////////////////////////////////////////////////////////////////////////////
    __global__ void computeCentroid_kernel(float* cX, float* cY, float* cZ,
                                           uint* aIdx, uint* bIdx, uint* cIdx,
                                           int n)
    {
        uint blockId = get_blockID();
        uint id      = get_threadID(blockId);

        if (id < n){
            // read the 3 index
            uint aid = aIdx[id], bid = bIdx[id], cid = cIdx[id];
        
            float4 a = tex1Dfetch(cgl_VertexTexture, aid);
            float4 b = tex1Dfetch(cgl_VertexTexture, bid);
            float4 c = tex1Dfetch(cgl_VertexTexture, cid);

            // return the normal
            cX[id] = (a.x  + b.x + c.x) / 3.f;
            cY[id] = (a.y  + b.y + c.y) / 3.f;
            cZ[id] = (a.z  + b.z + c.z) / 3.f;
        }
    }

    void computeCentroid(cplVector3DArray& d_cen, 
                         cplIndex3DArray & d_idx, int nF,
                         float4* d_pnt, int nV, cudaStream_t stream)
    {
        dim3 threads(256);
        dim3 grids(iDivUp(nF, threads.x));
        checkConfig(grids);

        cudaBindTexture(0, cgl_VertexTexture, d_pnt, nV * sizeof(float4));
        computeCentroid_kernel<<<grids, threads, 0, stream>>>(d_cen.x, d_cen.y, d_cen.z,
                                                              d_idx.aIdx, d_idx.bIdx, d_idx.cIdx, nF);
    }

//////////////////////////////////////////////////////////////////////////////////
// Compute the compute the face centroid and facenormal the sametime
//////////////////////////////////////////////////////////////////////////////////
    template<bool normalized>
    __global__ void computeCentroidNormal_kernel(float* d_c, float* d_n, int nAlign,
                                                 uint* aIdx, uint* bIdx, uint* cIdx,
                                                 int n)
    {
        uint blockId = get_blockID();
        uint id      = get_threadID(blockId);

        if (id < n){
            // read the 3 index
            uint aid = aIdx[id], bid = bIdx[id], cid = cIdx[id];
        
            float4 a = tex1Dfetch(cgl_VertexTexture, aid);
            float4 b = tex1Dfetch(cgl_VertexTexture, bid);
            float4 c = tex1Dfetch(cgl_VertexTexture, cid);

            float3 normal;
            if (normalized)
                normal = faceNormal(make_float3(a), make_float3(b), make_float3(c));
            else
                normal = faceWeightedNormal(make_float3(a), make_float3(b), make_float3(c));

            // return the normal
            d_n[id]              = normal.x;
            d_n[id + nAlign]     = normal.y;
            d_n[id + 2 * nAlign] = normal.z;

            d_c[id]              = (a.x  + b.x + c.x) / 3.f;
            d_c[id + nAlign]     = (a.y  + b.y + c.y) / 3.f;
            d_c[id + 2 * nAlign] = (a.z  + b.z + c.z) / 3.f;
        }
    }


    void computeCentroidNormal(cplVector3DArray& d_cen, cplVector3DArray& d_norm,
                               cplIndex3DArray & d_idx, int nF,
                               float4* d_pnt, int nV, cudaStream_t stream)
    {
        dim3 threads(256);
        dim3 grids(iDivUp(nF, threads.x));
        checkConfig(grids);

        cudaBindTexture(0, cgl_VertexTexture, d_pnt, nV * sizeof(float4));

        computeCentroidNormal_kernel<true><<<grids, threads, 0, stream>>>(d_cen.x, d_norm.x, d_cen.nAlign,
                                                                          d_idx.aIdx, d_idx.bIdx, d_idx.cIdx, nF);
    }

    void computeCentroidWeightedNormal(cplVector3DArray& d_cen, cplVector3DArray& d_norm,
                                       cplIndex3DArray & d_idx, int nF,
                                       float4* d_pnt, int nV, cudaStream_t stream)
    {
        dim3 threads(256);
        dim3 grids(iDivUp(nF, threads.x));
        checkConfig(grids);

        cudaBindTexture(0, cgl_VertexTexture, d_pnt, nV * sizeof(float4));

        computeCentroidNormal_kernel<false><<<grids, threads, 0, stream>>>(d_cen.x, d_norm.x, d_cen.nAlign,
                                                                           d_idx.aIdx, d_idx.bIdx, d_idx.cIdx, nF);
    }

    template<int weightedAVG>
    __global__ void computeVertexNormal_fixed_kernel(int* nX, int* nY, int* nZ,
                                                     uint* aIdx, uint* bIdx, uint* cIdx, int n,
                                                     float avgEdgeLength2)
    {
        uint blockId = get_blockID();
        uint id      = get_threadID(blockId);

        if (id >= n)
            return;

        // read the 3 index
        uint aid = aIdx[id], bid = bIdx[id], cid = cIdx[id];
        
        float3 p0 = make_float3(tex1Dfetch(cgl_VertexTexture, aid));
        float3 p1 = make_float3(tex1Dfetch(cgl_VertexTexture, bid));
        float3 p2 = make_float3(tex1Dfetch(cgl_VertexTexture, cid));
        
        float3 a = p1-p2, b = p2-p0, c = p0-p1;
        float3 facenormal = cross(a, b);

        if (weightedAVG){
            float l2a = dot(a,a), l2b = dot(b,b), l2c = dot(c,c);
            float3 normDev0 = facenormal * (avgEdgeLength2 / (l2b * l2c));
            atomicAdd(nX + aid, S2p20(normDev0.x));
            atomicAdd(nY + aid, S2p20(normDev0.y));
            atomicAdd(nZ + aid, S2p20(normDev0.z));

            float3 normDev1 = facenormal * (avgEdgeLength2 / (l2c * l2a));
            atomicAdd(nX + bid, S2p20(normDev1.x));
            atomicAdd(nY + bid, S2p20(normDev1.y));
            atomicAdd(nZ + bid, S2p20(normDev1.z));

            float3 normDev2 = facenormal * (avgEdgeLength2 / (l2a * l2b));
            atomicAdd(nX + cid, S2p20(normDev2.x));
            atomicAdd(nY + cid, S2p20(normDev2.y));
            atomicAdd(nZ + cid, S2p20(normDev2.z));
        }
        else {
            atomicAdd(nX + aid, S2p20(facenormal.x));
            atomicAdd(nY + aid, S2p20(facenormal.y));
            atomicAdd(nZ + aid, S2p20(facenormal.z));

            atomicAdd(nX + bid, S2p20(facenormal.x));
            atomicAdd(nY + bid, S2p20(facenormal.y));
            atomicAdd(nZ + bid, S2p20(facenormal.z));

            atomicAdd(nX + cid, S2p20(facenormal.x));
            atomicAdd(nY + cid, S2p20(facenormal.y));
            atomicAdd(nZ + cid, S2p20(facenormal.z));
        }
    }
    
    void computeVertexNormal_fixedPoint(cplVector3DArray& d_vNorm, float4* d_pnt, int nV,
                                        cplIndex3DArray& d_i, int nF, float avgEdgeLength, cudaStream_t stream)
    {
        dim3 threads(256);
        dim3 grids(iDivUp(nF, threads.x));
        checkConfig(grids);
        
        cplVector3DOpers::SetMem(d_vNorm, 0, nV);
        cudaBindTexture(0, cgl_VertexTexture, d_pnt, nV * sizeof(float4));
        int* d_nX = (int*) d_vNorm.x;
        int* d_nY = (int*) d_vNorm.y;
        int* d_nZ = (int*) d_vNorm.z;
    
        computeVertexNormal_fixed_kernel<1><<<grids, threads, 0, stream>>>(d_nX, d_nY, d_nZ,
                                                                           d_i.aIdx, d_i.bIdx, d_i.cIdx,
                                                                           nF, avgEdgeLength * avgEdgeLength);
        cplVector3DOpers::FixedToFloatingNormalize(d_vNorm, nV);
    }

    __global__ void computeMaxEdgeLength_kernel(float* d_len, uint* aIdx, uint* bIdx, uint* cIdx,
                                                int n)
    {
        uint blockId = get_blockID();
        uint id      = get_threadID(blockId);

        if (id >= n)
            return;

        // read the 3 index
        uint aid = aIdx[id], bid = bIdx[id], cid = cIdx[id];
        
        float3 p0 = make_float3(tex1Dfetch(cgl_VertexTexture, aid));
        float3 p1 = make_float3(tex1Dfetch(cgl_VertexTexture, bid));
        float3 p2 = make_float3(tex1Dfetch(cgl_VertexTexture, cid));
        
        float3 a = p1-p2, b = p2-p0, c = p0-p1;

        d_len[id] = fmaxf(dot(a,a),fmaxf(dot(b,b), dot(c,c)));
    }

    float computeMaxEdgeLength(cplReduce& rd, float4* d_pnt, int nV,
                               cplIndex3DArray& d_i, int nF, cudaStream_t stream)
    {
        dim3 threads(256);
        dim3 grids(iDivUp(nF, threads.x));
        checkConfig(grids);
        float* d_maxLength;
        dmemAlloc(d_maxLength, nF);
        
        cudaBindTexture(0, cgl_VertexTexture, d_pnt, nV * sizeof(float4));
        computeMaxEdgeLength_kernel<<<grids, threads, 0, stream>>>(d_maxLength,
                                                                   d_i.aIdx, d_i.bIdx, d_i.cIdx,
                                                                   nF);
        float maxLength = rd.Max(d_maxLength, nF);
        dmemFree(d_maxLength);
        fprintf(stderr, "Max length %f \n", sqrt(maxLength));
        return sqrt(maxLength);
    }


    __global__ void computeAvgEdgeLength_kernel(float* d_len, uint* aIdx, uint* bIdx, uint* cIdx,
                                                int n)
    {
        uint blockId = get_blockID();
        uint id      = get_threadID(blockId);

        if (id >= n)
            return;

        // read the 3 index
        uint aid = aIdx[id], bid = bIdx[id], cid = cIdx[id];
        
        float3 p0 = make_float3(tex1Dfetch(cgl_VertexTexture, aid));
        float3 p1 = make_float3(tex1Dfetch(cgl_VertexTexture, bid));
        float3 p2 = make_float3(tex1Dfetch(cgl_VertexTexture, cid));
        
        float3 a = p1-p2, b = p2-p0, c = p0-p1;

        d_len[id] = (dot(a,a) + dot(b,b) + dot(c,c))/3.f;
    }

    float computeAvgEdgeLength(cplReduce& rd, float4* d_pnt, int nV,
                               cplIndex3DArray& d_i, int nF, cudaStream_t stream)
    {
        dim3 threads(256);
        dim3 grids(iDivUp(nF, threads.x));
        checkConfig(grids);
        float* d_maxLength;
        dmemAlloc(d_maxLength, nF);
        
        cudaBindTexture(0, cgl_VertexTexture, d_pnt, nV * sizeof(float4));
        computeAvgEdgeLength_kernel<<<grids, threads, 0, stream>>>(d_maxLength,
                                                                   d_i.aIdx, d_i.bIdx, d_i.cIdx,
                                                                   nF);
        float avgLength = rd.Sum(d_maxLength, nF) / nF;
        dmemFree(d_maxLength);
        fprintf(stderr, "Average length %f \n", sqrt(avgLength));
        return sqrt(avgLength);
    }

    void cplSplatNormal(cplVector3DArray& d_on, const Vector3Di& size,
                         cplVector3DArray& d_in, cplVector3DArray& d_pos, int nP, cudaStream_t stream)
    {
        cplSplat3D(d_on, size, d_in, d_pos, nP, stream);
    }

    ///
    /// Interpolation of the value from the grid value
    ///
    
    void cplInterpolateFromGrid(float* d_o, cplVector3DArray& d_pos, int nP,
                                 float* d_grid, int sizeX, int sizeY, int sizeZ, cudaStream_t stream)
    {
        
        triLerp(d_o, d_pos, nP, d_grid, sizeX, sizeY, sizeZ, stream);
    }
    
    void cplInterpolateFromGrid(float* d_o,    cplVector3DArray& d_pos, int nP,
                                 float* d_grid, const Vector3Di& size, cudaStream_t stream)
    {
        triLerp(d_o, d_pos, nP, d_grid, size.x, size.y, size.z, stream);
    }
    
    ///
    /// Interpolation of the field value from the grid of field value
    ///
    void cplInterpolateFromGrid(cplVector3DArray& d_o, cplVector3DArray& d_pos, int nP,
                                 cplVector3DArray& d_grid,
                                 int sizeX, int sizeY, int sizeZ, cudaStream_t stream)
    {
        triLerp(d_o.x, d_pos, nP, d_grid.x, sizeX, sizeY, sizeZ, stream);
        triLerp(d_o.y, d_pos, nP, d_grid.y, sizeX, sizeY, sizeZ, stream);
        triLerp(d_o.z, d_pos, nP, d_grid.z, sizeX, sizeY, sizeZ, stream);
    }

    void cplInterpolateFromGrid(cplVector3DArray& d_o, cplVector3DArray& d_pos, int nP,
                                 cplVector3DArray& d_grid,
                                 const Vector3Di& size, cudaStream_t stream)
    {
        triLerp(d_o.x, d_pos, nP, d_grid.x, size.x, size.y, size.z, stream);
        triLerp(d_o.y, d_pos, nP, d_grid.y, size.x, size.y, size.z, stream);
        triLerp(d_o.z, d_pos, nP, d_grid.z, size.x, size.y, size.z, stream);
    }

    texture<float4, 1, cudaReadModeElementType> cgl_KnVertexTexture;
    __global__ void cplUpdateNormalVertexInfluence_kernel(
        float* nX, float* nY, float* nZ,
        uint* aIdx, uint* bIdx, uint* cIdx, int n)
    {
        uint blockId = get_blockID();
        uint id      = get_threadID(blockId);

        if (id < n){
            // read the 3 index
            uint aid = aIdx[id], bid = bIdx[id], cid = cIdx[id];
        
            float3 a = make_float3(tex1Dfetch(cgl_VertexTexture, aid));
            float3 b = make_float3(tex1Dfetch(cgl_VertexTexture, bid));
            float3 c = make_float3(tex1Dfetch(cgl_VertexTexture, cid));

            float3 kn0 = make_float3(tex1Dfetch(cgl_KnVertexTexture, aid));
            float3 kn1 = make_float3(tex1Dfetch(cgl_KnVertexTexture, bid));
            float3 kn2 = make_float3(tex1Dfetch(cgl_KnVertexTexture, cid));

            float3 e0 = b - c, e1 = c - a, e2 = a - b;

            float3 normDev0 = cross(e0, kn0);
            float3 normDev1 = cross(e1, kn1);
            float3 normDev2 = cross(e2, kn2);
        
            atomicAddf(nX + aid, normDev0.x);
            atomicAddf(nY + aid, normDev0.y);
            atomicAddf(nZ + aid, normDev0.z);

            atomicAddf(nX + bid, normDev1.x);
            atomicAddf(nY + bid, normDev1.y);
            atomicAddf(nZ + bid, normDev1.z);

            atomicAddf(nX + cid, normDev2.x);
            atomicAddf(nY + cid, normDev2.y);
            atomicAddf(nZ + cid, normDev2.z);
        }
    }

    void cplUpdateNormalVertexInfluence(cplVector3DArray& d_norm,
                                         float4* d_pnt, float4* d_Kn, int nV,
                                         cplIndex3DArray& d_i, int nF, cudaStream_t stream)
    {
        dim3 threads(256);
        dim3 grids(iDivUp(nF, threads.x));
        checkConfig(grids);

        cudaBindTexture(0, cgl_VertexTexture, d_pnt, nV * sizeof(float4));
        cudaBindTexture(0, cgl_KnVertexTexture, d_Kn, nV * sizeof(float4));
        
        cplUpdateNormalVertexInfluence_kernel<<<grids, threads, 0, stream>>>
            (d_norm.x, d_norm.y, d_norm.z, d_i.aIdx, d_i.bIdx, d_i.cIdx, nF);
    }

    __global__ void cplUpdateNormalVertexInfluence_fixed_kernel(
        int* nX, int* nY, int* nZ,
        uint* aIdx, uint* bIdx, uint* cIdx, int n)
    {
        uint blockId = get_blockID();
        uint id      = get_threadID(blockId);
        if (id < n){
            // read the 3 index
            uint aid = aIdx[id], bid = bIdx[id], cid = cIdx[id];
        
            float3 a = make_float3(tex1Dfetch(cgl_VertexTexture, aid));
            float3 b = make_float3(tex1Dfetch(cgl_VertexTexture, bid));
            float3 c = make_float3(tex1Dfetch(cgl_VertexTexture, cid));

            float3 kn0 = make_float3(tex1Dfetch(cgl_KnVertexTexture, aid));
            float3 kn1 = make_float3(tex1Dfetch(cgl_KnVertexTexture, bid));
            float3 kn2 = make_float3(tex1Dfetch(cgl_KnVertexTexture, cid));

            float3 e0 = b - c, e1 = c - a, e2 = a - b;

            float3 normDev0 = cross(e0, kn0);
            float3 normDev1 = cross(e1, kn1);
            float3 normDev2 = cross(e2, kn2);

            atomicAdd(nX + aid, S2p20(normDev0.x));
            atomicAdd(nY + aid, S2p20(normDev0.y));
            atomicAdd(nZ + aid, S2p20(normDev0.z));

            atomicAdd(nX + bid, S2p20(normDev1.x));
            atomicAdd(nY + bid, S2p20(normDev1.y));
            atomicAdd(nZ + bid, S2p20(normDev1.z));
        
            atomicAdd(nX + cid, S2p20(normDev2.x));
            atomicAdd(nY + cid, S2p20(normDev2.y));
            atomicAdd(nZ + cid, S2p20(normDev2.z));
        }
    }

    void cplUpdateNormalVertexInfluence_fixedPoint(cplVector3DArray& d_norm,
                                                    float4* d_pnt, float4* d_Kn, int nV,
                                                    cplIndex3DArray& d_i, int nF, cudaStream_t stream)
    {
        dim3 threads(256);
        dim3 grids(iDivUp(nF, threads.x));
        checkConfig(grids);

        cudaBindTexture(0, cgl_VertexTexture, d_pnt, nV * sizeof(float4));
        cudaBindTexture(0, cgl_KnVertexTexture, d_Kn, nV * sizeof(float4));

        int* d_nX = (int*) d_norm.x;
        int* d_nY = (int*) d_norm.y;
        int* d_nZ = (int*) d_norm.z;
    
        cplUpdateNormalVertexInfluence_fixed_kernel<<<grids, threads, 0, stream>>>
            (d_nX, d_nY, d_nZ, d_i.aIdx, d_i.bIdx, d_i.cIdx, nF);
        cplVectorOpers::FixedToFloating(d_norm.x, d_nX, nV);
        cplVectorOpers::FixedToFloating(d_norm.y, d_nY, nV);
        cplVectorOpers::FixedToFloating(d_norm.z, d_nZ, nV);
    }
   
}; // End of cplMesh namespace 
