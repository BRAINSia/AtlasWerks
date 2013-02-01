#include <CommonShaderSource.h>
#include <gl/GLSL_shader.h>
#include <gl/cudaMultiImageViewer.h>
#include <cpl.h>
#include <imageDisplay.h>

// shader for displaying floating-point texture
const int nColsLut[12] = {1, 2, 2, 2, 3, 3, 4, 4, 3, 4, 4, 4};
const int nRowsLut[12] = {1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3};

CUDAMultiImageViewer::CUDAMultiImageViewer():pbo(NULL),
                                             sizeX(0), sizeY(0), sizeZ(0),
                                             m_nImgs(0), m_arrayPtr(NULL){
    nSlice = 0;
    nCols  = 1;
    nRows  = 1;
}

void CUDAMultiImageViewer::setNumDisplayImages(unsigned int n)
{
    m_nImgs = n;
    m_arrayPtr = new float* [m_nImgs];
}   
CUDAMultiImageViewer::~CUDAMultiImageViewer(){
    delete pbo;
    if (m_arrayPtr){
        delete []m_arrayPtr;
    }
}

void CUDAMultiImageViewer::setImageList(const std::vector<float* >& imgList)
{
    assert(m_arrayPtr != NULL);
    for (uint i=0; i< m_nImgs; ++i)
        m_arrayPtr[i] = imgList[i];
}

void CUDAMultiImageViewer::initializeGL()
{
    fprintf(stderr, "Initialized GL ");
    assert((sizeX > 0) && (sizeY > 0) && (sizeZ > 0) && (m_nImgs > 0));
        
    // compute the number of cols and numeber of row
    nCols = nColsLut[m_nImgs-1];
    nRows = nRowsLut[m_nImgs-1];
    uint nElems = nCols * sizeX * nRows * sizeY;
    
    // create the pbo
    pbo = new cudaPBO(GL_PIXEL_UNPACK_BUFFER_ARB, nElems * sizeof(float), GL_STREAM_DRAW);
    
    // create texture for display
    glGenTextures(1, &texId);
    glBindTexture(GL_TEXTURE_2D, texId);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE32F_ARB,
                 sizeX * nCols, sizeY * nRows, 0, GL_LUMINANCE, GL_FLOAT, NULL);
    glBindTexture(GL_TEXTURE_2D, 0);
    
    // load shader program
    shader = compileASMShader(GL_FRAGMENT_PROGRAM_ARB, fsTextureDisplay);

    dmemAlloc(d_scratchI, nElems);
}

void CUDAMultiImageViewer::paintGL(){
    //concate the image to the scratch space
    concateVolumeSlices(d_scratchI, m_arrayPtr, m_nImgs,
                        nSlice, sizeX, sizeY, sizeZ);
    cutilCheckMsg("Concate volume slice");
    
    float* d_img = (float*) pbo->acquireDevicePointer();
    cutilCheckMsg("Acquire pointer");

    cplVectorOpers::SetMem(d_img, 0.f, sizeX * nCols * sizeY * nRows);
    cutilCheckMsg("Clear the images");
    
    //write  image to the pbo
    reshapeImages(d_img, d_scratchI, sizeX, sizeY, m_nImgs, nCols);
    cutilCheckMsg("Concate volume slice");
    
    pbo->releaseDevicePointer();
    cutilCheckMsg("Release pointer");
    
    //copy the result to the texture 
    pbo->bind();
    pbo->writeToTexture( texId, 0, 0, nCols * sizeX, nRows * sizeY, GL_LUMINANCE, GL_FLOAT);
    pbo->unbind();
    cutilCheckMsg("Write to texture");
    
    // Bind the texture shader
    glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
    glBindProgramARB(GL_FRAGMENT_PROGRAM_ARB, shader);
    glEnable(GL_FRAGMENT_PROGRAM_ARB);

    // Display the texture
    glBindTexture(GL_TEXTURE_2D, texId);
    glDisable(GL_DEPTH_TEST);
    glBegin(GL_QUADS);
    glVertex2f(0, 0); glTexCoord2f(0, 0);
    glVertex2f(0, 1); glTexCoord2f(1, 0);
    glVertex2f(1, 1); glTexCoord2f(1, 1);
    glVertex2f(1, 0); glTexCoord2f(0, 1);
    glEnd();
    glBindTexture(GL_TEXTURE_2D, 0);
    glDisable(GL_FRAGMENT_PROGRAM_ARB);

    glutSwapBuffers();
}

void CUDAMultiImageViewer::resizeGL(int w, int h){
    screenWidth = w;
    screenHeight= h;

    glViewport(0, 0, screenWidth, screenHeight);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glutPostRedisplay();
}

void CUDAMultiImageViewer::idle(){
    //glutPostRedisplay();
}

void CUDAMultiImageViewer::keyPressEvent(int key){
     switch(key)
    {
        case 27:
            exit(0);
            break;
        case '>':
            ++nSlice;
            nSlice = min(nSlice, sizeZ - 1);
            fprintf(stderr, "Current slice %d \r", nSlice);
            break;
        case '<':
            --nSlice;
            nSlice = max(nSlice,(uint) 0);
            fprintf(stderr, "Current slice %d \r", nSlice);
            break;
        default:
            break;
    }
    glutPostRedisplay();
    
}
