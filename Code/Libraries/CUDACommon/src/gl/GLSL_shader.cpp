//
// Copyright (c) 2006, University of Utah
// All rights reserved.
//
// This software is licensed under the BSD open-source license. See
// http://www.opensource.org/licenses/bsd-license.php for more detail.
//
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

#include <gl/GLSL_shader.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <stdlib.h>
#include <string.h>
/** 
 *  GLSLShader implementation is below:  This represents all of the basic structure
 *  and functionality for both sub-types of shaders.  The implementation follows the
 *  OpenGL 2.0 specification and the functionality is undefined for non 2.0 implementations:
 *  Note:  In most cases, non 2.0 implementations will cause failure to compile.

 */

/*  type of shader
	GL_VERTEX_SHADER ; vertex program
	GL_FRAGMENT_SHADER ; vertex program	

*/
GLSLShader::GLSLShader(unsigned int type) : id_(0), type_(type)
{ 
	
}

/**
 *  GLSLShader ctor:  Creates and empty shader object and associates it with the 
 */
GLSLShader::GLSLShader(const std::string& fname, const unsigned int type)
  : id_(0), programFile_(fname), type_(type)
{
	create(programFile_);
}

/** 
 *  GLSLShader dtor:  Destroys the shader
 */
GLSLShader::~GLSLShader() 
{
	id_=0;
	destroy();
}

/** 
 *  GLSLShader readProgram(std::string&):  Read in the program from the specified file in preparation 
 *  for its compilation.
 */
bool GLSLShader::readProgram(std::string& filename)
{
    std::ifstream f(filename.c_str(), std::ios::in | std::ios::binary);
  	if (f.fail())
  	{
   	 	fprintf(stderr, "[ShaderProgramARB(\"%s\")::reload] "
			"Failed to open file: %s\n", filename.c_str(), filename.c_str());
   		return false;
  	}
  	// get length of file
  	f.seekg(0, std::ios::end);
  	int length = f.tellg();
  	f.seekg(0, std::ios::beg);
  	char* buffer = new char[length+1];
 	
	f.read(buffer, length);
  	buffer[length] = 0;
  	f.close();

    programDef_ = std::string(buffer);
  	delete [] buffer;
	return true;
}

bool GLSLShader::compile(const char* srcStr)
{
    programDef_ = std::string(srcStr);
    if (!createObject())
        return false;
    return compile();
}

bool GLSLShader::createObject(){
    id_ = glCreateShaderObjectARB(type_);
    std::cout << "Shader ID " << id_ << std::endl;
    if(id_ == 0)
	{
		std::cerr << "Could not generate a valid identifier for the shader!" << std::endl;
		return false;
	}
    return true;
}

bool GLSLShader::compile()
{
    const char* p = programDef_.c_str();
    glShaderSourceARB(id_, 1, &p, NULL);
    glCompileShaderARB(id_);
    int val;
	glGetShaderiv(id_, GL_COMPILE_STATUS, &val);
	if(val == GL_FALSE)
	{
        std::cerr << "Could not compile the shader " << std::endl;
		char buf[1024];
		int size;
		glGetShaderInfoLog(id_, 1024, &size, buf);
		std::cerr << "Shader infolog as follows:" << std::endl;
		std::cerr << buf << std::endl;
		return false;
	}
    return true;
}
/** 
 *  GLSLShader create(std::string):  Create a shader from a given file of source.  This will 
 *  Compile the shader, but it will NOT link or attach it.
 */
bool GLSLShader::create(std::string& filename)
{
    if (!readProgram(filename))
		return false;
    if (!createObject())
        return false;
    return compile();
}

/** 
 *  GLSLShader destroy():  Unloads and destroys the shader.  This does nto destroy the shader object!
 */
void GLSLShader::destroy()
{
	if(glIsShader(id_))
		glDeleteShader(id_);
}



/** 
 *  The following are the sub-class specific implementation details for the GLSLShader super-class.
 *  The only real difference is that the GL shader types are respected.
 */
GLSLVertexShader::GLSLVertexShader(const std::string& name) : 
    GLSLShader(name, GL_VERTEX_SHADER_ARB)
{
	if(name.size() < 1) {
		std::cerr << "Vertex Shader could not be loaded :: Filename is invalid" << std::endl;
		exit(1);
	}
}

GLSLVertexShader::GLSLVertexShader() : GLSLShader(GL_VERTEX_SHADER_ARB)
{
    
}

GLSLVertexShader::~GLSLVertexShader() 
{

}

GLSLGeometryShader::GLSLGeometryShader(const std::string& name) : 
GLSLShader(name, GL_GEOMETRY_SHADER_EXT)
{
	if(name.size() < 1) {
		std::cerr << "Geometry Shader could not be loaded :: Filename is invalid" << std::endl;
		exit(1);
	}
}

GLSLGeometryShader::GLSLGeometryShader() : GLSLShader(GL_GEOMETRY_SHADER_EXT)
{
    
}

GLSLGeometryShader::~GLSLGeometryShader() 
{

}


GLSLFragmentShader::GLSLFragmentShader(const std::string& name):
    GLSLShader( name, GL_FRAGMENT_SHADER_ARB)
{
	if(name.size() < 1) {
		std::cerr << "Fragment Shader could not be loaded :: Filename is invalid" << std::endl;
		exit(1);
	}
}

GLSLFragmentShader::GLSLFragmentShader() : GLSLShader(GL_FRAGMENT_SHADER_ARB)
{

}

GLSLFragmentShader::~GLSLFragmentShader() 
{

}

/*-----------------------------------------------------------------------------------------------------------------------------------------------------*/

/** 
 *  GLSLShaderProgram ctor (default):  Default constructor for the shader program.
 */

GLSLShaderProgram::GLSLShaderProgram() : isLinked_(false)
{
	progId_ = glCreateProgram();
	if (progId_ == 0)
	{
		std::cerr << "Shader program could not be created " << std::endl;
		exit(1);
	}
}

/*  GLSLShaderProgram dtor:  Destroys any attached shaders as well as itself. */
GLSLShaderProgram::~GLSLShaderProgram()
{
	destroy();
}

/*  GLSLShaderProgram attachShader(GLSLShader& shader):  Attach the given shader to the 
 *  GL pipeline.  This (generally) will be called using the Shader members of the 
 *  GLSLShaderProgram object.
 */
void GLSLShaderProgram::attachShader(GLSLShader& shader)
{
    unsigned int id = shader.getId();
    unsigned int shaderIDs[MAX_ATTACHED_SHADER];
    int count;
	glGetAttachedShaders(progId_, MAX_ATTACHED_SHADER, &count, shaderIDs);
	for(int i = 0; i < count; ++i)
	{
		if(shaderIDs[i] == id)
			return;
	}
    glAttachShader(progId_, id);
}

void GLSLShaderProgram::attachVertexShader(GLSLVertexShader& shader){
    attachShader(shader);
}

void GLSLShaderProgram::attachFragmentShader(GLSLFragmentShader& shader){
    attachShader(shader);
}

/*  GLSLShaderProgram attachShader(GLSLShader& shader):  Attach the given shader to the 
 *  GL pipeline.  This (generally) will be called using the Shader members of the 
 *  GLSLShaderProgram object.
 */
void GLSLShaderProgram::attachGeometryShader(GLSLGeometryShader& shader,
                                             GLenum gsInput, GLenum gsOutput, unsigned int nOPrims)
{
    if (shader.getType() != GL_GEOMETRY_SHADER_EXT){
        std::cerr << "Wrong shader type" <<  std::endl;
        exit(1);
    }
    attachShader(shader);
    //glProgramParameter Must Be Called Before the Shaders are Linked
    setGeometryParameters(gsInput, gsOutput, nOPrims);
}

/*  GLSLShaderProgram attachShader(GLSLShader& shader):  Attach the given shader to the 
 *  GL pipeline.  This (generally) will be called using the Shader members of the 
 *  GLSLShaderProgram object.
 */
void GLSLShaderProgram::detachShader(GLSLShader& shader)
{
	unsigned int shaderIDs[MAX_ATTACHED_SHADER];
	unsigned int id = shader.getId();
	int count;
	glGetAttachedShaders(progId_, MAX_ATTACHED_SHADER, &count, shaderIDs);
	for(int i = 0; i < count; ++i)
	{
		if(shaderIDs[i] == id)
		{
			glDetachShader(progId_, id);
			return;
		}
	}
}

/*  GLSLShaderProgram build():  Build a GLSL shader program and the shaders associated with it
 *  This will create the various shaders and if needed, compile and link the shaders.
 *  It will NOT bind the shader.
 */
void GLSLShaderProgram::build()
{
	glLinkProgram(progId_);
	GLint val;
	glGetProgramiv(progId_, GL_LINK_STATUS, &val);
	if(val == GL_FALSE)
	{
		std::cerr << "Failed to link the program for use!" << std::endl;
		char buf[4096];
		int len;
		glGetProgramInfoLog(progId_, 4096, &len, buf);
		std::cerr << "Linker error log is as follows:" << std::endl << buf << std::endl;
	}
}


/*  GLSLShaderProgram load():  Load a GLSL shader program and the shaders associated with it
 *  This will create the various shaders if needed, compile, link, and bind the shader
 *  The proper usage of this class can be as simple as:
 *  	GLSLShaderProgram* prog = new GLSLShaderProgram(fragFilename, vertFilename);
 *  	prog->load();
 *  After the load call, the shaders are ready to go and need no further work to execute.
 */
void GLSLShaderProgram::load()
{
	glUseProgram(progId_);
}

/*  GLSLShaderProgram reload(): Reload a GLSLShader program and the shaders it contains.
 */
void GLSLShaderProgram::reload()
{
	build();
	load();
}

/*  GLSLShaderProgram unload():  Unload a GLSL shader program ant the shaders associated with it.
 *  This will NOT destroy the shaders, it merely unbinds them to prevent their execution.
 */
void GLSLShaderProgram::unload()
{
	glUseProgram(0);
}

/*  GLSLShaderProgram destroy():  Destroys the shader program and the shaders it uses.
 *  Note:  This is called by the dtor;  it unloads and then deletes the shaders.
 */
void GLSLShaderProgram::destroy()
{
	glDeleteProgram(progId_);
	progId_ = 0;
	isLinked_ = false;
}

/** 
 *  GLSLShader setParameter(std::string&, float):  Set (and bind if necessary) the attribute 
 *  given by the string the value defined by the float.  
 */
void GLSLShaderProgram::setParameters(std::string name, float* val, int size)
{
	glUseProgram(progId_);
	int id = glGetUniformLocation(progId_, name.c_str());

	if(id == -1){
		return;
	}
	if(size == 1) {
		glUniform1fv(id, 1, val);
	}else if(size == 2) {
		glUniform2fv(id, 1, val);
	}else if(size == 3) {
		glUniform3fv(id, 1, val);
	}else if(size == 4) {
		glUniform4fv(id, 1, val);
	}
}


/** 
 *  GLSLShader setParameter(std::string&, float):  Set (and bind if necessary) the attribute 
 *  given by the string the value defined by the float.  
 */
GLuint GLSLShaderProgram::getVertexAttrib(std::string name)
{
	glUseProgram(progId_);
	return glGetAttribLocation(progId_,name.c_str());
}


/** 
 *  setTextureUnit(std::string&, int):  Set a texture unit up in the fragment shader.
 *  The string parameter is the name of the sampler in the fragment shader, the int
 *  is the texture unit ID we wish to bind to.
 */
void GLSLShaderProgram::setTextureUnit(std::string name, int texunit)
{
	int links;
	glGetProgramiv(progId_, GL_LINK_STATUS, &links);
	if(links != GL_TRUE) {
		std::cout << "setTextureUnit sees link status failed" << std::endl;
        return;
    }

	int id = glGetUniformLocation(progId_, name.c_str());
	glUniform1i(id, texunit);
}


/** 
 *  Set the parameters for Geometry shader 
 *  with gsInput is any type of inputs GL_POINTS, GL_LINES ...
 *  however it is must be compatible with the input call
 *  gsOutput is one of three type GL_POINTS, GL_LINE_STRIP, GL_TRIANGLES_STRIP
 *  nOPrims the size of output primitives  
 */
void GLSLShaderProgram::setGeometryParameters(GLenum gsInput, GLenum gsOutput, unsigned int nOPrims)
{
    if ((gsOutput != GL_POINTS) && (gsOutput != GL_LINE_STRIP) && (gsOutput != GL_TRIANGLE_STRIP))
    {
        std::cout << "setGeometryParameters: Invalid geometry output type" <<std::endl;
        exit(1);
    }
    glProgramParameteriEXT(progId_, GL_GEOMETRY_INPUT_TYPE_EXT, gsInput);
    glProgramParameteriEXT(progId_, GL_GEOMETRY_OUTPUT_TYPE_EXT, gsOutput); 
    glProgramParameteriEXT(progId_, GL_GEOMETRY_VERTICES_OUT_EXT, nOPrims); 
}


void
GLSLShaderProgram::setUniform1f(const char *name, float value)
{
    GLint loc = glGetUniformLocation(progId_, name);
    if (loc >= 0) {
        glUniform1f(loc, value);
    } else {
#if _DEBUG
        fprintf(stderr, "Error setting parameter '%s'\n", name);
#endif
    }
}

void
GLSLShaderProgram::setUniform2f(const char *name, float x, float y)
{
    GLint loc = glGetUniformLocation(progId_, name);
    if (loc >= 0) {
        glUniform2f(loc, x, y);
    } else {
#if _DEBUG
        fprintf(stderr, "Error setting parameter '%s'\n", name);
#endif
    }
}

void
GLSLShaderProgram::setUniform3f(const char *name, float x, float y, float z)
{
    GLint loc = glGetUniformLocation(progId_, name);
    if (loc >= 0) {
        glUniform3f(loc, x, y, z);
    } else {
#if _DEBUG
        fprintf(stderr, "Error setting parameter '%s'\n", name);
#endif
    }
}

void
GLSLShaderProgram::setUniform4f(const char *name, float x, float y, float z, float w)
{
    GLint loc = glGetUniformLocation(progId_, name);
    if (loc >= 0) {
        glUniform4f(loc, x, y, z, w);
    } else {
#if _DEBUG
        fprintf(stderr, "Error setting parameter '%s'\n", name);
#endif
    }
}

void
GLSLShaderProgram::setUniformMatrix4fv(const GLchar *name, GLfloat *m, bool transpose)
{
    GLint loc = glGetUniformLocation(progId_, name);
    if (loc >= 0) {
        glUniformMatrix4fv(loc, 1, transpose, m);
    } else {
#if _DEBUG
        fprintf(stderr, "Error setting parameter '%s'\n", name);
#endif
    }
}

GLuint compileASMShader(GLenum program_type, const char *code)
{
    GLuint program_id;
    glGenProgramsARB(1, &program_id);
    glBindProgramARB(program_type, program_id);
    glProgramStringARB(program_type, GL_PROGRAM_FORMAT_ASCII_ARB, (GLsizei) strlen(code), (GLubyte *) code);

    GLint error_pos;
    glGetIntegerv(GL_PROGRAM_ERROR_POSITION_ARB, &error_pos);
    if (error_pos != -1) {
        const GLubyte *error_string;
        error_string = glGetString(GL_PROGRAM_ERROR_STRING_ARB);
        fprintf(stderr, "Program error at position: %d\n%s\n", (int)error_pos, error_string);
        return 0;
    }
    return program_id;
}
