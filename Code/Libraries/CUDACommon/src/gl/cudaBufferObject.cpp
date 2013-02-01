#include <gl/cudaBufferObject.h>
#include <cuda_runtime_api.h>
#include <cutil_inline.h>
#include <cutil_gl_inline.h>
#include <cuda_gl_interop.h>
#include <cudaInterface.h>

GLBObject::GLBObject(unsigned int kind):m_id(0), m_kind(kind){
    
}

GLBObject::GLBObject(unsigned int kind,
                     unsigned int type, unsigned int sizeInBytes, GLenum usage)
{
    m_kind = kind;
    create(type, sizeInBytes, usage);
}

GLBObject::~GLBObject(){
    this->destroy();
}

void GLBObject::bind()
{
#ifdef __CHECK_BIND__
    unsigned int bindType = GL_ARRAY_BUFFER_BINDING;
    if (m_type == GL_ELEMENT_ARRAY_BUFFER)
        bindType = GL_ELEMENT_ARRAY_BUFFER_BINDING;
    unsigned int id;
    glGetIntergerv(bindType, &id);
    if (id == m_id)
        return;
#endif
    glBindBuffer(m_type, m_id);
}

void GLBObject::unbind()
{
    glBindBuffer(m_type, 0);
}

bool GLBObject::create(unsigned int type, unsigned int sizeInBytes, GLenum usage)
{
    m_type       = type;
    m_mode       = usage;
    m_sizeInByte = sizeInBytes;

    if (checkValidBufferType() && checkValidAccessMode()){
        glGenBuffers(1, &m_id);
        this->bind();
        glBufferData(m_type, sizeInBytes, 0, m_mode);
        this->unbind();
        return true;
    }
    return false;
}

void GLBObject::destroy()
{
    if (m_id){
        this->bind();
        glDeleteBuffers(1, &m_id);
        m_id = 0;
    }
}

void GLBObject::copyFromHost(void* h_i, unsigned int sizeInBytes, unsigned int offsetInBytes){
#if 1
    char* ptr = (char*)glMapBuffer(m_type, GL_WRITE_ONLY);
    memcpy(ptr + offsetInBytes, h_i, sizeInBytes);
    glUnmapBuffer(m_type);
#else
    glBufferSubData(m_type, offsetInBytes, sizeInBytes, h_i);
#endif
}

void GLBObject::copyToHost(void* h_o, unsigned int sizeInBytes, unsigned int offsetInBytes){
#if 1
    char* ptr = (char*)glMapBuffer(m_type, GL_READ_ONLY);
    memcpy(h_o, ptr + offsetInBytes, sizeInBytes);
    glUnmapBuffer(m_type);
#else
    glGetBufferSubData(m_type, offsetInBytes, sizeInBytes, h_o);
#endif
}

void* GLBObject::getHostPointerToRead()
{
    return glMapBuffer(m_type, GL_READ_ONLY);
}

void GLBObject::discardCurrentBuffer(){
    glBufferDataARB(m_type, m_sizeInByte, 0, m_mode);
}
    
////////////////////////////////////////////////////////////////////////////////
//  GLBObject::getHostPointerToWrite()
// map the buffer object into client's memory
// Note that glMapBufferARB() causes sync issue.
// If GPU is working with this buffer, glMapBufferARB() will wait(stall)
// for GPU to finish its job. To avoid waiting (stall), you can call
// first glBufferDataARB() with NULL pointer before glMapBufferARB().
// If you do that, the previous data in PBO will be discarded and
// glMapBufferARB() returns a new allocated pointer immediately
// even if GPU is still working with the previous data.
////////////////////////////////////////////////////////////////////////////////

void* GLBObject::getHostPointerToWrite()
{
    return glMapBuffer(m_type, GL_WRITE_ONLY);
}

void* GLBObject::getHostPointerToChange()
{
    return glMapBuffer(m_type, GL_READ_WRITE);
}

bool GLBObject::checkValidBufferType(){
    if (m_kind == GLB_VBO)//VBO
        return ((m_type == GL_ARRAY_BUFFER) || (m_type == GL_ELEMENT_ARRAY_BUFFER));
    else 
        return ((m_type == GL_PIXEL_PACK_BUFFER) || (m_type == GL_PIXEL_UNPACK_BUFFER));
}


bool GLBObject::checkValidAccessMode(){
    bool check = ((m_mode == GL_STREAM_DRAW) || (m_mode == GL_STREAM_READ) || (m_mode == GL_STREAM_COPY) ||
                  (m_mode == GL_STATIC_DRAW) || (m_mode == GL_STATIC_READ) || (m_mode == GL_STATIC_COPY) ||
                  (m_mode == GL_DYNAMIC_DRAW) || (m_mode == GL_DYNAMIC_READ) || (m_mode == GL_DYNAMIC_COPY));
    if (!check)
        fprintf(stderr, "Invalid buffer access mode");
    return check;
}

bool GLBObject::checkValidSize(unsigned int sizeInBytes, unsigned int offsetInBytes){
    bool check = (offsetInBytes + sizeInBytes < m_sizeInByte);
    if (!check)
        fprintf(stderr, "Warning: Access out of bound of allocated buffer size");
    return check;
}


////////////////////////////////////////////////////////////////////////////////
// VBObject
////////////////////////////////////////////////////////////////////////////////

VBObject::VBObject():GLBObject(GLBObject::GLB_VBO){

};

VBObject::VBObject(unsigned int type, unsigned int sizeInBytes, GLenum usage):GLBObject(GLBObject::GLB_VBO, type, sizeInBytes, usage)
{

};


VBObject::~VBObject(){
    
}

////////////////////////////////////////////////////////////////////////////////
// PBObject
////////////////////////////////////////////////////////////////////////////////

PBObject::PBObject():GLBObject(GLBObject::GLB_PBO){

};

PBObject::PBObject(unsigned int type, unsigned int sizeInBytes, GLenum usage):GLBObject(GLBObject::GLB_PBO, type, sizeInBytes, usage)
{

};

PBObject::~PBObject(){
    
}

void PBObject::readFromScreen(int x, int y, unsigned int width, unsigned int height, 
                              unsigned int format, unsigned int type)
{
    glReadPixels(x, y, width, height, format, type, 0);
}

void PBObject::writeToScreen(unsigned int width, unsigned int height,
                             unsigned int format, unsigned int type)
{
    glDrawPixels(width, height, format, type, 0);
}

void PBObject::readFromTexture(unsigned int textureId, unsigned int format, unsigned int type)
{
    glBindTexture(GL_TEXTURE_2D, textureId);
    glGetTexImage(GL_TEXTURE_2D, 0, format, type, 0);
    glBindTexture(GL_TEXTURE_2D, 0);
}

void PBObject::writeToTexture(unsigned int textureId, 
                              int x, int y, unsigned int width, unsigned int height, 
                              unsigned int format, unsigned int type)
{
    glBindTexture(GL_TEXTURE_2D, textureId);
    glTexSubImage2D(GL_TEXTURE_2D, 0, x, y, width, height, format, type, 0);
    glBindTexture(GL_TEXTURE_2D, 0);
}

unsigned int ResourceFlagHint(int mode)
{
    if ((mode == GL_STREAM_DRAW) || (mode == GL_STATIC_DRAW) || (mode == GL_DYNAMIC_DRAW))
        return cudaGraphicsMapFlagsWriteDiscard;
    if ((mode == GL_STREAM_READ) || (mode == GL_STATIC_READ) || (mode == GL_DYNAMIC_READ))
        return cudaGraphicsMapFlagsReadOnly;
    return cudaGraphicsMapFlagsNone;
}
////////////////////////////////////////////////////////////////////////////////
//cudaVBO
////////////////////////////////////////////////////////////////////////////////
cudaVBO::cudaVBO():m_vbo(),m_stream(NULL){
    
}

cudaVBO::cudaVBO(unsigned int type, unsigned int sizeInBytes, GLenum usage)
    :m_vbo(type, sizeInBytes, usage), m_stream(NULL)
{
    m_res_flag = ResourceFlagHint(usage);
}

void cudaVBO::bind()
{
    m_vbo.bind();
}

void cudaVBO::cudaRegister()
{
    //DEPRECATED: cudaGLRegisterBufferObject(m_vbo.getId());
    cutilSafeCall(cudaGraphicsGLRegisterBuffer(&m_vbo_res, m_vbo.getId(), m_res_flag));
}

void cudaVBO::cudaUnregister(){
    cudaGraphicsUnregisterResource(m_vbo_res);
    //DEPRECATED: cudaGLUnregisterBufferObject(m_vbo.getId());
}
    
void cudaVBO::unbind(){
    m_vbo.unbind();
}

void cudaVBO::copyFromHost(void* h_i, unsigned int sizeInBytes, unsigned int offsetInBytes)
{
    cudaUnregister();
    m_vbo.copyFromHost(h_i, sizeInBytes, offsetInBytes);
    cudaRegister();
}

void cudaVBO::copyToHost(void* h_o, unsigned int sizeInBytes, unsigned int offsetInBytes)
{
    // The buffer don't need to be binded
    cudaUnregister();
    m_vbo.copyToHost(h_o, sizeInBytes, offsetInBytes);
    cudaRegister();
}

void cudaVBO::copyFromDevice(void* d_i, unsigned int sizeInBytes, unsigned int offsetInBytes)
{
    // The buffer don't need to be binded
    char* d_ptr = (char*) acquireDevicePointer();
    cudaMemcpy(d_ptr + offsetInBytes, d_i, sizeInBytes, cudaMemcpyDeviceToDevice);
    releaseDevicePointer();
}

void cudaVBO::copyToDevice(void* d_o, unsigned int sizeInBytes, unsigned int offsetInBytes)
{
    char* d_ptr = (char*) acquireDevicePointer();
    cudaMemcpy(d_o, d_ptr + offsetInBytes, sizeInBytes, cudaMemcpyDeviceToDevice);
    releaseDevicePointer();
}

void* cudaVBO::getHostPointerToRead()
{
    cudaUnregister();
    return m_vbo.getHostPointerToRead();
}

void* cudaVBO::getHostPointerToWrite()
{
    cudaUnregister();
    return m_vbo.getHostPointerToWrite();
}

void* cudaVBO::acquireDevicePointer()
{
    // The buffer don't need to be bind but need to be registered
    cudaRegister();
    void* d_ptr;
    // DEPRECATED:cudaGLMapBufferObject((void**)&d_ptr, m_vbo.getId());
    
    cutilSafeCall(cudaGraphicsMapResources(1, &m_vbo_res, m_stream));
    cutilSafeCall(cudaGraphicsResourceGetMappedPointer((void **)&d_ptr, &m_accessSize,  
                                                       m_vbo_res));
    return d_ptr;
}
void cudaVBO::releaseDevicePointer()
{
    // DEPRECATED: cudaGLUnmapBufferObject(m_vbo.getId());
    cutilSafeCall(cudaGraphicsUnmapResources(1, &m_vbo_res, m_stream));
    cudaUnregister();
}

void* cudaVBO::getDevicePointer()
{
    acquireDevicePointer();
}

void cudaVBO::setWorkingStream(cudaStream_t stream){
    m_stream = stream;
}

////////////////////////////////////////////////////////////////////////////////
//cudaPBO
////////////////////////////////////////////////////////////////////////////////
cudaPBO::cudaPBO():m_pbo(), m_stream(NULL){
    
}

cudaPBO::~cudaPBO()
{
    
}

cudaPBO::cudaPBO(unsigned int type, unsigned int sizeInBytes, GLenum usage)
    :m_pbo(type, sizeInBytes, usage), m_stream(NULL)
{
    m_res_flag = ResourceFlagHint(usage);
}

void cudaPBO::bind()
{
    m_pbo.bind();
}

void cudaPBO::cudaRegister()
{
    //cudaGLRegisterBufferObject(m_pbo.getId());
    cutilSafeCall(cudaGraphicsGLRegisterBuffer(&m_pbo_res, m_pbo.getId(), m_res_flag));
}

void cudaPBO::cudaUnregister(){
    //cudaGLUnregisterBufferObject(m_pbo.getId());
    cudaGraphicsUnregisterResource(m_pbo_res);
}
    
void cudaPBO::unbind(){
    m_pbo.unbind();
}

void cudaPBO::copyFromHost(void* h_i, unsigned int sizeInBytes, unsigned int offsetInBytes)
{
    cudaUnregister();
    m_pbo.copyFromHost(h_i, sizeInBytes, offsetInBytes);
    cudaRegister();
}

void cudaPBO::copyToHost(void* h_o, unsigned int sizeInBytes, unsigned int offsetInBytes)
{
    // The buffer don't need to be binded
    cudaUnregister();
    m_pbo.copyToHost(h_o, sizeInBytes, offsetInBytes);
    cudaRegister();
}

void cudaPBO::copyFromDevice(void* d_i, unsigned int sizeInBytes, unsigned int offsetInBytes)
{
    // The buffer don't need to be binded
    char* d_ptr = (char*) acquireDevicePointer();
    cudaMemcpy(d_ptr + offsetInBytes, d_i, sizeInBytes, cudaMemcpyDeviceToDevice);
    releaseDevicePointer();
}

void cudaPBO::copyToDevice(void* d_o, unsigned int sizeInBytes, unsigned int offsetInBytes)
{
    char* d_ptr = (char*) acquireDevicePointer();
    cudaMemcpy(d_o, d_ptr + offsetInBytes, sizeInBytes, cudaMemcpyDeviceToDevice);
    releaseDevicePointer();
}

void* cudaPBO::getHostPointerToRead()
{
    cudaUnregister();
    return m_pbo.getHostPointerToRead();
}

void* cudaPBO::getHostPointerToWrite()
{
    cudaUnregister();
    return m_pbo.getHostPointerToWrite();
}

void* cudaPBO::acquireDevicePointer()
{
    // The buffer don't need to be bind but need to be registered
    cudaRegister();
    void* d_ptr;

    // DEPRECATED:cudaGLMapBufferObject((void**)&d_ptr, m_pbo.getId());
    cutilSafeCall(cudaGraphicsMapResources(1, &m_pbo_res, m_stream));
    cutilSafeCall(cudaGraphicsResourceGetMappedPointer((void **)&d_ptr, &m_accessSize,  
                                                       m_pbo_res));
    return d_ptr;
}
void cudaPBO::releaseDevicePointer()
{
    // DEPRECATED: cudaGLUnmapBufferObject(m_pbo.getId());
    cutilSafeCall(cudaGraphicsUnmapResources(1, &m_pbo_res, m_stream));
    cudaUnregister();
}

void* cudaPBO::getDevicePointer()
{
    acquireDevicePointer();
}

void cudaPBO::readFromScreen(int x, int y, unsigned int width, unsigned int height, 
                             unsigned int format, unsigned int type)
{
    cudaRegister();
    m_pbo.readFromScreen(x,y,width, height, format, type);
}

void cudaPBO::readFromTexture(unsigned int textureId, unsigned int format, unsigned int type)
{
    cudaRegister();
    m_pbo.readFromTexture(textureId, format, type);
}


void cudaPBO::writeToScreen(unsigned int width, unsigned int height,
                            unsigned int format, unsigned int type)
{
    cudaRegister();
    m_pbo.writeToScreen(width, height, format, type);
}

void cudaPBO::writeToTexture(unsigned int textureId, 
                             int x, int y, unsigned int width, unsigned int height, 
                             unsigned int format, unsigned int type)
{
    cudaRegister();
    m_pbo.writeToTexture(textureId, x, y, width, height, format, type);
}
    
void cudaPBO::setWorkingStream(cudaStream_t stream){
    m_stream = stream;
}
