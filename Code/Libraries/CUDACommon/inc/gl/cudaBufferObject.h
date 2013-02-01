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

#ifndef __CUDA_BUFFER_OBJECT__H
#define __CUDA_BUFFER_OBJECT__H

#include "GLSL_shader.h"
#include <cuda_runtime.h>
class cudaGraphicsResource;

class GLBObject{
public:
    enum GLB_kind {GLB_VBO, GLB_PBO};

    GLBObject(unsigned int kind);
    virtual ~GLBObject();
    
    explicit GLBObject(unsigned int kind,
                       unsigned int type, unsigned int sizeInBytes, GLenum usage);


    unsigned int getId()         {return m_id;   };
    unsigned int getKind()       {return m_kind;   };
    unsigned int getType()       {return m_type; };
    unsigned int getAccessMode() {return m_mode; };
    unsigned int getBufferSize() {return m_sizeInByte; };

public:
    bool create(unsigned int type, unsigned int sizeInBytes, GLenum usage);
    void destroy();

    void bind();
    void unbind();

    void copyFromHost(void* h_i, unsigned int sizeInBytes, unsigned int offetInBytes=0);
    void copyToHost(void* h_o, unsigned int sizeInBytes, unsigned int offetInBytes=0);
    
    void* getHostPointerToRead();
    void* getHostPointerToWrite();
    void* getHostPointerToChange();
private:
    void discardCurrentBuffer();
    bool checkValidBufferType();
    bool checkValidAccessMode();
    bool checkValidSize(unsigned int sizeInBytes, unsigned int offsetInBytes);

    unsigned int m_id;
    unsigned int m_kind;
    unsigned int m_type;
    unsigned int m_mode;
    unsigned int m_sizeInByte;
};

class VBObject : public GLBObject
{
public:
    VBObject();
    VBObject(unsigned int type, unsigned int sizeInBytes, GLenum usage);
    virtual ~VBObject();
};

class PBObject : public GLBObject
{
public:
    PBObject();
    PBObject(unsigned int type, unsigned int sizeInBytes, GLenum usage);
    
    void readFromScreen(int x, int y, unsigned int width, unsigned int height, 
                        unsigned int format, unsigned int type);

    void readFromTexture(unsigned int textureId, unsigned int format, unsigned int type);
    
    void writeToScreen(unsigned int width, unsigned int height,
                       unsigned int format, unsigned int type);

    void writeToTexture(unsigned int textureId, 
                        int x, int y, unsigned int width, unsigned int height, 
                        unsigned int format, unsigned int type);
    
    void setPixelFormat(unsigned int format) { m_pixel_format = format; };
    void setPixelType(unsigned int type) { m_data_type = type; }
    unsigned int getPixelFormat() { return m_pixel_format;};
    unsigned int getDataType()    { return m_data_type;};
    
    virtual ~PBObject();
protected:
    unsigned int m_pixel_format;
    unsigned int m_data_type;
};

class cudaVBO
{
public:
    cudaVBO();
    explicit cudaVBO(unsigned int type, unsigned int sizeInBytes, GLenum usage);
    ~cudaVBO();

    void bind();
    void unbind();
    void cudaRegister();
    void cudaUnregister();
    
    void copyFromHost(void* h_i, unsigned int sizeInBytes, unsigned int offsetInBytes=0);
    void copyToHost(void* h_o, unsigned int sizeInBytes, unsigned int offsetInBytes=0);

    void copyFromDevice(void* d_i, unsigned int sizeInBytes, unsigned int offsetInBytes=0);
    void copyToDevice(void* d_o, unsigned int sizeInBytes, unsigned int offsetInBytes=0);

    void* getHostPointerToRead();
    void* getHostPointerToWrite();

    void* getDevicePointer();
    void* acquireDevicePointer();
    void  releaseDevicePointer();

    void setWorkingStream(cudaStream_t stream);
    unsigned int getId() { return m_vbo.getId(); }
private:
    VBObject m_vbo;
    cudaGraphicsResource *m_vbo_res;
    unsigned int m_res_flag;
    cudaStream_t m_stream;
    size_t m_accessSize;
};


class cudaPBO
{
public:
    cudaPBO();
    explicit cudaPBO(unsigned int type, unsigned int sizeInBytes, GLenum usage);
    ~cudaPBO();

    void bind();
    void unbind();
    void cudaRegister();
    void cudaUnregister();
    
    void copyFromHost(void* h_i, unsigned int sizeInBytes, unsigned int offsetInBytes=0);
    void copyToHost(void* h_o, unsigned int sizeInBytes, unsigned int offsetInBytes=0);

    void copyFromDevice(void* d_i, unsigned int sizeInBytes, unsigned int offsetInBytes=0);
    void copyToDevice(void* d_o, unsigned int sizeInBytes, unsigned int offsetInBytes=0);

    void* getHostPointerToRead();
    void* getHostPointerToWrite();

    void* getDevicePointer();
    void* acquireDevicePointer();
    void  releaseDevicePointer();
            
    unsigned int getId() { return m_pbo.getId(); }
public:

    void readFromScreen(int x, int y, unsigned int width, unsigned int height, 
                        unsigned int format, unsigned int type);

    void readFromTexture(unsigned int textureId, unsigned int format, unsigned int type);
    
    void writeToScreen(unsigned int width, unsigned int height,
                       unsigned int format, unsigned int type);

    void writeToTexture(unsigned int textureId, 
                        int x, int y, unsigned int width, unsigned int height, 
                        unsigned int format, unsigned int type);

    void setWorkingStream(cudaStream_t stream);
private:
    PBObject m_pbo;
    cudaGraphicsResource *m_pbo_res;
    unsigned int m_res_flag;
    cudaStream_t m_stream;
    size_t m_accessSize;
};

#endif
