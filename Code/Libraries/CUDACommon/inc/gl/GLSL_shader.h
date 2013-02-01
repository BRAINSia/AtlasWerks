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

// Copyright (c) 2005, 2006
// University of Utah
// All rights reserved.
//
// This software is licensed under the BSD open-source license. See
// http://www.opensource.org/licenses/bsd-license.php for more detail.
//
// *************************************************************
// Redistribution and use in source and binary forms, with or 
// without modification, are permitted provided that the following 
// conditions are met:
//
// Redistributions of source code must retain the above copyright notice, 
// this list of conditions and the following disclaimer. 
//
// Redistributions in binary form must reproduce the above copyright notice, 
// this list of conditions and the following disclaimer in the documentation 
// and/or other materials provided with the distribution. 
//
// Neither the name of the University of Utah nor the names of 
// the contributors may be used to endorse or promote products derived 
// from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS 
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT 
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS 
// FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL 
// THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, 
// INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL 
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE 
// GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS 
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, 
// WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT 
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF 
// THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY 
// OF SUCH DAMAGE.

#ifndef _GLSL_SHADER_H_
#define _GLSL_SHADER_H_


#include <string>
#include <vector>
#include <map>

#ifdef _WIN32
#include<GL/glee.h>
#else
#define GL_GLEXT_PROTOTYPES
#endif

#include<GL/glut.h>
#include<GL/glu.h>
#include<GL/gl.h>

/**
 *  The following classes define the GLSL implementation of the initial ARB-based implementation.
 *  The class GLSLShaderProgram is the workhorse of the system.  It manages all of the various 
 *  shaders necessary to complete a rendering.  The class GLSLShader is used to form the vertex
 *  and fragment shaders (GLSLVertexShader and GLSLFragmentShader respectively).  Two of these 
 *  objects are required to use a ShaderProgram as they must be attached in order to function 
 *  properly.  Note:  The GLSLShaderProgram class does NOT enforce persistence of these objects.
 *  They should ONLY be destroyed AFTER they are no longer needed; however, they can be deleted 
 *  after being loaded onto the video card as the system guarantees their persistence there.
 */
class GLSLShader
{
public:
	GLSLShader(unsigned int type);
	GLSLShader(const std::string& fname, const unsigned int type);
	virtual ~GLSLShader();

    bool create(std::string& filename);
    bool compile(const char* srcStr);
        
    void destroy();
    inline unsigned int getId() { return id_;};
	inline unsigned int getType() { return type_;};
    
protected:
	bool readProgram(std::string& filename);
    bool createObject();
    bool compile();

    unsigned int 	id_;
    std::string 	programFile_;
    
	unsigned int	type_;
    std::string 	programDef_;
	int				size_;
};

class GLSLVertexShader : public GLSLShader
{
public:
	GLSLVertexShader();
	GLSLVertexShader(const std::string& name);
	virtual ~GLSLVertexShader();
};

class GLSLGeometryShader : public GLSLShader
{
public:
	GLSLGeometryShader();
	GLSLGeometryShader(const std::string& name);
	virtual ~GLSLGeometryShader();

};

class GLSLFragmentShader : public GLSLShader
{
public:
	GLSLFragmentShader();
	GLSLFragmentShader(const std::string& name);
	virtual ~GLSLFragmentShader();

};

#define MAX_ATTACHED_SHADER 10

class GLSLShaderProgram
{
public:
	/*  Note:  The default constructor is available, but the second constructor is preferred.
     *  However, since there is a mechanism in place for resetting shaders, the default constructor
     *  can easily be used if necessary.
	 */
	GLSLShaderProgram();
	virtual ~GLSLShaderProgram();

    void attachVertexShader(GLSLVertexShader& shader);
    void attachGeometryShader(GLSLGeometryShader& shader,
                              GLenum gsInput, GLenum gsOutput, unsigned int nOPrims);
    void attachFragmentShader(GLSLFragmentShader& shader);
    
    void detachShader(GLSLShader& shader);

public:
    void build();
    void load();
	void reload();
	void unload();
	void destroy();

public:
    void setTextureUnit(std::string name, int texunit);
	void setParameters(std::string name, float* val, int size);

    void setUniform1f(const char *name, float value);
    void setUniform2f(const char *name, float x, float y);
    void setUniform3f(const char *name, float x, float y, float z);
    void setUniform4f(const char *name, float x, float y, float z, float w);
    void setUniformMatrix4fv(const GLchar *name, GLfloat *m, bool transpose);
    
    
    GLuint getVertexAttrib(std::string name);
	inline unsigned int getProgId() { return progId_; }
	inline bool isLinked() { return isLinked_;};

protected:
	unsigned int progId_;
	bool		 isLinked_;
    void setGeometryParameters(GLenum gsInput, GLenum gsOutput, unsigned int nOPrims);
    void attachShader(GLSLShader& shader);
    
};


GLuint compileASMShader(GLenum program_type, const char *code);
    
#endif // GLSL_Shader_h
