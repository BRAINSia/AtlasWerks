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

#include <assert.h>
#include <stdio.h>
#include <iostream>
#include <cudaInterface.h>
#include <cudaDataConvert.h>
#include <mesh/cudaMesh.h>
#include <mesh/cudaMeshOper.h>

cplMesh::cplMesh(float* h_verts, int nV,
                   int*   h_faces, int nF)
{
    m_nV = nV;
    m_nF = nF;
    int nMax = max(nV, nF);

    float* h_vx = new float [nMax];
    float* h_vy = new float [nMax];
    float* h_vz = new float [nMax];

    allocateDeviceVector3DArray(d_pnt, m_nV);
    allocateDeviceVector3DArray(d_idx, m_nF);
        
    for (int i=0; i< m_nV; ++i){
        h_vx[i] = h_verts[i * 3    ];
        h_vy[i] = h_verts[i * 3 + 1];
        h_vz[i] = h_verts[i * 3 + 2];
    }

    copyArrayToDevice(d_pnt.x, h_vx, m_nV);
    copyArrayToDevice(d_pnt.y, h_vy, m_nV);
    copyArrayToDevice(d_pnt.z, h_vz, m_nV);

    uint* h_aIdx = (uint*) h_vx;
    uint* h_bIdx = (uint*) h_vy;
    uint* h_cIdx = (uint*) h_vz;

    for (int i=0; i< m_nF; ++i){
        h_aIdx[i] = h_faces[i * 3    ];
        h_bIdx[i] = h_faces[i * 3 + 1];
        h_cIdx[i] = h_faces[i * 3 + 2];
    }
    
    copyArrayToDevice(d_idx.aIdx, h_aIdx, m_nF);
    copyArrayToDevice(d_idx.bIdx, h_bIdx, m_nF);
    copyArrayToDevice(d_idx.cIdx, h_cIdx, m_nF);

    delete []h_vx;
    delete []h_vy;
    delete []h_vz;
    
    has_face_centroid = false;
    has_face_normal = false;
    has_scalar = false;
    has_vert_normal = false;

    m_maxEdgeLength = 0.f;
    m_avgEdgeLength = 0.f;
}

cplMesh::~cplMesh()
{
    // TODO
}

void cplMesh::loadScalaField(float* h_f){
    dmemAlloc(d_f, m_nV);
    copyArrayToDevice(d_f, h_f, m_nV);
    has_scalar = true;
}

void cplMesh::need_face_centroid(float4* d_temp4){
    bool need_temp = (d_temp4 == NULL);
    if (need_temp){
        dmemAlloc(d_temp4, m_nV);
    }
    convertX_Y_ZtoXYZW(d_temp4, d_pnt.x, d_pnt.y, d_pnt.z, m_nV);
    // alocate the face centroid

    allocateDeviceVector3DArray(d_cFace, m_nF);
    
    has_face_centroid = true;
    cplMeshOper::computeCentroid(d_cFace, d_idx, m_nF, d_temp4, m_nV);

    if (need_temp)
        dmemFree(d_temp4);
}

void cplMesh::need_face_normal(bool normalized, float4* d_temp4){
    bool need_temp = (d_temp4 == NULL);
    if (need_temp){
        dmemAlloc(d_temp4, m_nV);
    }
    convertX_Y_ZtoXYZW(d_temp4, d_pnt.x, d_pnt.y, d_pnt.z, m_nV);
    
    // alocate the face normal
    allocateDeviceVector3DArray(d_nFace, m_nF);

    has_face_normal = true;
    if (normalized)
        cplMeshOper::computeFaceNormal(d_nFace, d_idx, m_nF, d_temp4, m_nV);
    else
        cplMeshOper::computeWeightedFaceNormal(d_nFace, d_idx, m_nF, d_temp4, m_nV);

    if (need_temp)
        dmemFree(d_temp4);
};

void cplMesh::need_face_centroid_and_normal(bool normalized, float4* d_temp4)
{
    bool need_temp = (d_temp4 == NULL);
    if (need_temp){
        dmemAlloc(d_temp4, m_nV);
    }
    convertX_Y_ZtoXYZW(d_temp4, d_pnt.x, d_pnt.y, d_pnt.z, m_nV);
    
    allocateDeviceVector3DArray(d_nFace, m_nF);
    allocateDeviceVector3DArray(d_cFace, m_nF);
    has_face_normal = true;
    has_face_centroid = true;
    
    if (normalized){
        cplMeshOper::computeCentroidNormal(d_cFace, d_nFace,
                                            d_idx, m_nF, d_temp4, m_nV);

    }else {
        cplMeshOper::computeCentroidWeightedNormal(d_cFace, d_nFace,
                                                    d_idx, m_nF, d_temp4, m_nV);
    }
    if (need_temp)
        dmemFree(d_temp4);
}

void cplMesh::need_bounding_box(cplReduce& rd){
    cplMeshOper::computeBoundingBox(rd, m_vMinP, m_vMaxP, d_pnt, m_nV);
    fprintf(stderr, "Bounding box [(%f,%f,%f),(%f,%f,%f)] \n",
            m_vMinP.x, m_vMinP.y, m_vMinP.z,
            m_vMaxP.x, m_vMaxP.y, m_vMaxP.z);
}

void cplMesh::need_mean_point(cplReduce& rd){
    m_vCentroid = cplMeshOper::computeMeanPoint(rd, d_pnt, m_nV);
}

void cplMesh::need_scalar_range(cplReduce& rd){
    assert(has_scalar);
    rd.MaxMin(m_maxS, m_minS, d_f, m_nV);
}


void cplMesh::need_vert_normal(float4* d_temp4){
    bool need_temp = (d_temp4 == NULL);
    if (need_temp){
        dmemAlloc(d_temp4, m_nV);
    }
    convertX_Y_ZtoXYZW(d_temp4, d_pnt.x, d_pnt.y, d_pnt.z, m_nV);
    allocateDeviceVector3DArray(d_nVert, m_nV);

    if (m_avgEdgeLength == 0.f)
    {
        fprintf(stderr, "Need to compute the average edge length");
    }
    cplMeshOper::computeVertexNormal_fixedPoint(d_nVert, d_temp4, m_nV, d_idx, m_nF,
                                                 m_avgEdgeLength/2);
    
    if (need_temp)
        dmemFree(d_temp4);
}


void cplMesh::adjustOriginAndScale(const Vector3Df& org, const Vector3Df& scale)
{
    cplVector3DOpers::AddCMulC_I(d_pnt, -org, scale, m_nV);
}


void cplMesh::flipFaces()
{
    uint* d_temp = d_idx.aIdx;
    d_idx.aIdx = d_idx.bIdx;
    d_idx.bIdx = d_temp;
}

////////////////////////////////////////////////////////////////////////////////
// Measure on the bounding box
////////////////////////////////////////////////////////////////////////////////

void cplMesh::getBoundingBoxExt(float& x, float& y, float& z) const{
    x = m_vMaxP.x - m_vMinP.x;
    y = m_vMaxP.y - m_vMinP.y;
    z = m_vMaxP.z - m_vMinP.z;
}

void cplMesh::getBoundingBoxCentroid(float& x, float& y, float& z) const{
    x = (m_vMaxP.x + m_vMinP.x) * 0.5;
    y = (m_vMaxP.y + m_vMinP.y) * 0.5;
    z = (m_vMaxP.z + m_vMinP.z) * 0.5;
}

const Vector3Df cplMesh::getBoundingBoxCentroid() const       {
    return (m_vMaxP + m_vMinP) * 0.5;
}


const Vector3Df cplMesh::getBoundingBoxExt() const
{
    return (m_vMaxP - m_vMinP);
}

float cplMesh::getMaxLength() const{
    Vector3Df ext = m_vMaxP - m_vMinP;
    float maxLength = (ext.x > ext.y) ? ext.x   : ext.y;
    maxLength = (maxLength > ext.z) ? maxLength : ext.z;
    return maxLength;
}

float cplMesh::getDiagLength() const{
    return getBoundingBoxExt().length();
}

void cplMesh::need_max_edge_length(cplReduce& rd, float4* d_temp4)
{
    bool need_temp = (d_temp4 == NULL);
    if (need_temp){
        dmemAlloc(d_temp4, m_nV);
    }
    convertX_Y_ZtoXYZW(d_temp4, d_pnt.x, d_pnt.y, d_pnt.z, m_nV);

    m_maxEdgeLength =  cplMeshOper::computeMaxEdgeLength(rd, d_temp4, m_nV,
                                                          d_idx, m_nF, 0);
    if (need_temp)
        dmemFree(d_temp4);
}


void cplMesh::need_avg_edge_length(cplReduce& rd, float4* d_temp4)
{
    bool need_temp = (d_temp4 == NULL);
    if (need_temp){
        dmemAlloc(d_temp4, m_nV);
    }
    convertX_Y_ZtoXYZW(d_temp4, d_pnt.x, d_pnt.y, d_pnt.z, m_nV);

    m_avgEdgeLength =  cplMeshOper::computeAvgEdgeLength(rd, d_temp4, m_nV,
                                                          d_idx, m_nF, 0);
    if (need_temp)
        dmemFree(d_temp4);
}
