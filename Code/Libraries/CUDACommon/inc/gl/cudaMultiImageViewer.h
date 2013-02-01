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

#ifndef __CUDA_MULTI_IMAGE_VIEWER__H
#define __CUDA_MULTI_IMAGE_VIEWER__H

#include <cutil_math.h>
#include <vector>
#include <gl/cudaBufferObject.h>
#include <gl/GLRenderInterface.h>

class CUDAMultiImageViewer: public GLRenderInterface
{
public:
    CUDAMultiImageViewer();
    virtual ~CUDAMultiImageViewer();
    
    void setImageList(const std::vector<float* >& imgList);
    void setImageSize(unsigned int x, unsigned int y, unsigned int z) {
        sizeX = x; sizeY = y; sizeZ = z;
    }
    unsigned int getNumDisplayImages() const { return m_nImgs; };
    void setNumDisplayImages(unsigned int n);
public:
    virtual void initializeGL();
    virtual void paintGL();
    virtual void resizeGL(int w, int h);
    virtual void idle();
    virtual void keyPressEvent(int key);
    
    virtual void mouseReleaseEvent(int button, int x, int y){};
    virtual void mousePressEvent(int button, int x, int y){};
    virtual void mouseMoveEvent(int x, int y){};
private:

    unsigned int screenWidth;
    unsigned int screenHeight;
    
    unsigned int sizeX, sizeY, sizeZ;
    unsigned int nCols, nRows;
    
    unsigned int nSlice;
    unsigned int m_nImgs;
        
    cudaPBO* pbo;
    float* d_scratchI;
    
    unsigned int texId;
    GLuint shader;
    float** m_arrayPtr;
};


#endif
