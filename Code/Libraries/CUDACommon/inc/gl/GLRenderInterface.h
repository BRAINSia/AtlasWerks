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

#ifndef __GL_RENDER_INTERFACE_H
#define __GL_RENDER_INTERFACE_H

#include <cstddef>

class GLRenderInterface{
public:
    GLRenderInterface(){};
    virtual ~GLRenderInterface(){};

    static void displayFunc(){
        m_objPtr->paintGL();
    };
    static void reshapeFunc(int w, int h){
        m_objPtr->resizeGL(w, h);
    };
    static void keyboardFunc(unsigned char key, int x, int y){ m_objPtr->keyPressEvent(key); }
    static void specialFunc(int key, int x, int y) { m_objPtr->keyPressEvent(key); };
    static void idleFunc(){ m_objPtr->idle();};
    static void motionFunc(int x, int y){  m_objPtr->mouseMoveEvent(x, y);};
    static void mouseFunc(int button, int state, int x, int y) {
        if (state == 0) // GLUT_DOWN
            m_objPtr->mousePressEvent(button, x, y);
        else
            m_objPtr->mouseReleaseEvent(button, x, y);
    }                        

    void makeCurrent(){
        setInterfacePtr(this);
    }


    void updateGL() { paintGL();}

    virtual void initializeGL()=0;
    virtual void paintGL()=0;
    virtual void resizeGL(int w, int h)=0;
    virtual void idle()=0;

    virtual void mouseReleaseEvent(int button, int x, int y)=0;
    virtual void mousePressEvent(int button, int x, int y)=0;
    virtual void mouseMoveEvent(int x, int y)=0;
    virtual void keyPressEvent(int key)=0;
protected:
    void setInterfacePtr(GLRenderInterface* objPtr) { m_objPtr = objPtr; };
private:
     static GLRenderInterface* m_objPtr;
};



#endif
