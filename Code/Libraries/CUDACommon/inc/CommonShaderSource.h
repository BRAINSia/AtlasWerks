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

#ifndef __COMMON_SHADER_SOURCE_H
#define __COMMON_SHADER_SOURCE_H

static const char* fsColorPassThrough = "\n"
"/////////////////////////////////////////////////////////////////\n"
"//\n"
"//      Normal visualization passthrough fragment shader\n"
"//      Author: Linh Ha\n"
"//      lha [at] sci [dot] utah [dot] edu\n"
"//\n"
"/////////////////////////////////////////////////////////////////\n"
"\n"
"#version 120\n"
"void main(void)\n"
"{\n"
"	gl_FragColor=gl_Color;\n"
"}\n"
;

static const char* gsNormalDisplay= "\n"
"/////////////////////////////////////////////////////////////////\n"
"//\n"
"//      Normal visualization geometry shader\n"
"//      Author: Linh Ha\n"
"//      lha [at] sci [dot] utah [dot] edu\n"
"//\n"
"/////////////////////////////////////////////////////////////////\n"
"\n"
"#version 120\n"
"#extension GL_EXT_geometry_shader4 : enable\n"
"// need the bracket in Geometry shader \n"
"// and the keyword in to work\n"
"varying in vec3 vnorm[]; \n"
"// the vnorm must be declared in vertex shader\n"
"uniform float bbScale;\n"
"void main(void)\n"
"{\n"
"    vec4 point    = gl_PositionIn[0];\n"
"    gl_FrontColor = gl_FrontColorIn[0];\n"
"    gl_Position   = gl_ModelViewProjectionMatrix*point;\n"
"	EmitVertex();\n"
"    gl_Position   = gl_ModelViewProjectionMatrix*(point + 0.02 * bbScale * vec4(vnorm[0], 0));\n"
"	EmitVertex();\n"
"}\n"
"\n"
;
static const char* vsNormalPassThrough= "\n"
"/////////////////////////////////////////////////////////////////\n"
"//\n"
"//      Normal visualization passthrough vertex shader\n"
"//      Author: Linh Ha\n"
"//      lha [at] sci [dot] utah [dot] edu\n"
"//\n"
"/////////////////////////////////////////////////////////////////\n"
"\n"
"#version 120\n"
"varying vec3 vnorm;\n"
"void main(void)\n"
"{\n"
"	//just send the things as they are\n"
"	gl_Position    = gl_Vertex;\n"
"   gl_FrontColor  = gl_Color;\n"
"   vnorm = gl_Normal;\n"
"}\n"
;

static const char *fsTextureDisplay = 
"!!ARBfp1.0\n"
"TEX result.color, fragment.texcoord, texture[0], 2D; \n"
"END";

#endif
