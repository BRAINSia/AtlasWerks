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

#ifndef __CUDA_MESH__H
#define __CUDA_MESH__H

#include <cudaVector3DArray.h>
#include <cudaIndex3DArray.h>
#include <cudaReduce.h>

class cplMesh {
public:
    cplMesh(float* h_verts, int nV,
             int*   h_faces, int nF);

    ~cplMesh();
    
    cplVector3DArray d_pnt;// Input point set
    cplVector3DArray d_nVert; // normal at the vertex
    
    cplIndex3DArray  d_idx;// Input index array
    cplVector3DArray d_cFace; // centroid of the face            
    cplVector3DArray d_nFace; // normal of the face
    float*            d_f;     // scalar field at the vertex
    
    void getBoundingBoxExt(float& x, float& y, float& z) const;
    void getBoundingBoxCentroid(float& x, float& y, float& z) const;
    
    const Vector3Df getBoundingBoxExt() const;
    const Vector3Df getBoundingBoxCentroid() const;

    float getMaxLength() const;
    float getDiagLength() const;

    float getMaxEdgeLength() const { return m_maxEdgeLength; };
    float getAvgEdgeLength() const { return m_avgEdgeLength; };
    
public:
    
    unsigned int getNumVertices() const { return m_nV;}
    unsigned int getNumFaces() const { return m_nF;}

    void need_face_centroid_and_normal(bool normalized, float4* d_temp4);
    void need_face_centroid(float4* d_temp4);
    void need_face_normal(bool normalized, float4* d_temp4);
    void need_vert_normal(float4* d_temp4);

    void need_bounding_box(cplReduce& rd);
    void need_mean_point(cplReduce& rd);
    void need_scalar_range(cplReduce& rd);
    
    void need_max_edge_length(cplReduce& rd, float4* d_temp4);
    void need_avg_edge_length(cplReduce& rd, float4* d_temp4);
    
    void adjustOriginAndScale(const Vector3Df& org, const Vector3Df& scale);

    void flipFaces();
    void loadScalaField(float* h_f);

protected :
    
    int m_nV;
    int m_nF;

    Vector3Df m_vCentroid;  // Centroid of the mesh 
    Vector3Df m_vMinP;      // Bounding box infomation
    Vector3Df m_vMaxP;      //

    float m_maxS;         // Scalar range   
    float m_minS;

    bool has_face_centroid;
    bool has_face_normal;
    bool has_scalar;
    bool has_vert_normal;

    float m_maxEdgeLength;
    float m_avgEdgeLength;
};


#endif
