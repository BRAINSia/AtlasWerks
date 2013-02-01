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

#ifndef __CUDA_MATRIX_H
#define __CUDA_MATRIX_H

#include <cutil_math.h>
#include <iostream>

struct  float3x3{
    float3 m[3];
};

struct float3x4{
    float4 m[3];
};

struct float4x4{
    void print(const char* name);
    float4 m[4];
};

// Missing function from cutil_math.h
/*
static __inline__ __host__ __device__ float2 fminf(float2 a, float2 b)
{
	return make_float2(fminf(a.x,b.x), fminf(a.y,b.y));
}

// max
static __inline__ __host__ __device__ float2 fmaxf(float2 a, float2 b)
{
	return make_float2(fmaxf(a.x,b.x), fmaxf(a.y,b.y));
}
*/

// min
static __inline__ __host__ __device__ int2 min(int2 a, int2 b)
{
	return make_int2(min(a.x,b.x), min(a.y,b.y));
}

static __inline__ __host__ __device__ int2 max(int2 a, int2 b)
{
	return make_int2(max(a.x,b.x), max(a.y,b.y));
}

inline __host__ __device__ float4 operator*(float4 a, float4 b)
{
    return make_float4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
}    


// Working function with CUDA matrix 
inline __host__ __device__ float4x4 transpose(float4x4 a){
    float4x4 b;
    b.m[0] = make_float4(a.m[0].x, a.m[1].x, a.m[2].x, a.m[3].x);
    b.m[1] = make_float4(a.m[0].y, a.m[1].y, a.m[2].y, a.m[3].y);
    b.m[2] = make_float4(a.m[0].z, a.m[1].z, a.m[2].z, a.m[3].z);
    b.m[3] = make_float4(a.m[0].w, a.m[1].w, a.m[2].w, a.m[3].w);

    return b;
}

inline __host__ __device__ float3 operator*(float3x3 a, float3 b){
    float x = dot(a.m[0],b);
    float y = dot(a.m[1],b);
    float z = dot(a.m[2],b);
    return make_float3(x,y,z);
}

inline __host__ __device__ float4 operator*(float4x4 a, float4 b)
{
    float x = dot(a.m[0],b);
    float y = dot(a.m[1],b);
    float z = dot(a.m[2],b);
    float w = dot(a.m[3],b);
    return make_float4(x,y,z,w);
}

inline __host__ __device__ float4x4 operator*(float4x4 a, float4x4 b)
{
    float4x4 c;

    c.m[0].x = a.m[0].x * b.m[0].x + a.m[0].y * b.m[1].x +  a.m[0].z * b.m[2].x + a.m[0].w * b.m[3].x;
    c.m[0].y = a.m[0].x * b.m[0].y + a.m[0].y * b.m[1].y +  a.m[0].z * b.m[2].y + a.m[0].w * b.m[3].y;
    c.m[0].z = a.m[0].x * b.m[0].z + a.m[0].y * b.m[1].z +  a.m[0].z * b.m[2].z + a.m[0].w * b.m[3].z;
    c.m[0].w = a.m[0].x * b.m[0].w + a.m[0].y * b.m[1].w +  a.m[0].z * b.m[2].w + a.m[0].w * b.m[3].w;

    c.m[1].x = a.m[1].x * b.m[0].x + a.m[1].y * b.m[1].x +  a.m[1].z * b.m[2].x + a.m[1].w * b.m[3].x;
    c.m[1].y = a.m[1].x * b.m[0].y + a.m[1].y * b.m[1].y +  a.m[1].z * b.m[2].y + a.m[1].w * b.m[3].y;
    c.m[1].z = a.m[1].x * b.m[0].z + a.m[1].y * b.m[1].z +  a.m[1].z * b.m[2].z + a.m[1].w * b.m[3].z;
    c.m[1].w = a.m[1].x * b.m[0].w + a.m[1].y * b.m[1].w +  a.m[1].z * b.m[2].w + a.m[1].w * b.m[3].w;

    c.m[2].x = a.m[2].x * b.m[0].x + a.m[2].y * b.m[1].x +  a.m[2].z * b.m[2].x + a.m[2].w * b.m[3].x;
    c.m[2].y = a.m[2].x * b.m[0].y + a.m[2].y * b.m[1].y +  a.m[2].z * b.m[2].y + a.m[2].w * b.m[3].y;
    c.m[2].z = a.m[2].x * b.m[0].z + a.m[2].y * b.m[1].z +  a.m[2].z * b.m[2].z + a.m[2].w * b.m[3].z;
    c.m[2].w = a.m[2].x * b.m[0].w + a.m[2].y * b.m[1].w +  a.m[2].z * b.m[2].w + a.m[2].w * b.m[3].w;

    c.m[3].x = a.m[3].x * b.m[0].x + a.m[3].y * b.m[1].x +  a.m[3].z * b.m[2].x + a.m[3].w * b.m[3].x;
    c.m[3].y = a.m[3].x * b.m[0].y + a.m[3].y * b.m[1].y +  a.m[3].z * b.m[2].y + a.m[3].w * b.m[3].y;
    c.m[3].z = a.m[3].x * b.m[0].z + a.m[3].y * b.m[1].z +  a.m[3].z * b.m[2].z + a.m[3].w * b.m[3].z;
    c.m[3].w = a.m[3].x * b.m[0].w + a.m[3].y * b.m[1].w +  a.m[3].z * b.m[2].w + a.m[3].w * b.m[3].w;

    return c;
}

inline __host__ __device__ float4x4 float4x4_Identity(){
    float4x4 IM;
    IM.m[0] = make_float4(1.f, 0.f, 0.f, 0.f);
    IM.m[1] = make_float4(0.f, 1.f, 0.f, 0.f);
    IM.m[2] = make_float4(0.f, 0.f, 1.f, 0.f);
    IM.m[3] = make_float4(0.f, 0.f, 0.f, 1.f);
    return IM;
}

inline __host__ __device__ float4x4 float4x4_Translation(float x, float y, float z){
    float4x4 TM;
    TM.m[0] = make_float4(1.f, 0.f, 0.f, x  );
    TM.m[1] = make_float4(0.f, 1.f, 0.f, y  );
    TM.m[2] = make_float4(0.f, 0.f, 1.f, z  );
    TM.m[3] = make_float4(0.f, 0.f, 0.f, 1.f);
    return TM;
}

inline __host__ __device__ float4x4 float4x4_Scale(float sx, float sy, float sz){
    float4x4 TM;
    TM.m[0] = make_float4(sx, 0.f, 0.f, 0  );
    TM.m[1] = make_float4(0.f, sy, 0.f, 0  );
    TM.m[2] = make_float4(0.f, 0.f, sz, 0  );
    TM.m[3] = make_float4(0.f, 0.f, 0.f, 1.f);
    return TM;
}

inline __host__ __device__ float4x4 float4x4_RotateX(float angle){
    float4x4 RMx;
    float c = cos(angle * M_PI / 180);
    float s = sin(angle * M_PI / 180);

    RMx.m[0] = make_float4(1.f, 0.f, 0.f, 0.f);
    RMx.m[1] = make_float4(0.f, c  , -s , 0.f);
    RMx.m[2] = make_float4(0.f, s  ,  c , 0.f);
    RMx.m[3] = make_float4(0.f, 0.f, 0.f, 1.f);
    return RMx;
}

inline __host__ __device__ float4x4 float4x4_RotateY(float angle){
    float4x4 RMy;
    float c = cos(angle * M_PI / 180);
    float s = sin(angle * M_PI / 180);

    RMy.m[0] = make_float4( c , 0.f,  s , 0.f);
    RMy.m[1] = make_float4(0.f, 1.f, 0.f, 0.f);
    RMy.m[2] = make_float4(-s , 0.f,  c , 0.f);
    RMy.m[3] = make_float4(0.f, 0.f, 0.f, 1.f);

    return RMy;
}

inline __host__ __device__ float4x4 float4x4_RotateZ(float angle){
    float4x4 RMz;
    float c = cos(angle * M_PI / 180);
    float s = sin(angle * M_PI / 180);

    RMz.m[0] = make_float4( c ,  -s, 0.f, 0.f);
    RMz.m[1] = make_float4( s ,   c, 0.f, 0.f);
    RMz.m[2] = make_float4(0.f, 0.f, 1.f, 0.f);
    RMz.m[3] = make_float4(0.f, 0.f, 0.f, 1.f);

    return RMz;
}

inline __host__ __device__ float4x4 float4x4_Rotate(float angle, float x, float y, float z)
{
    float4x4 RM;
    float c = cos(angle * M_PI / 180);
    float s = sin(angle * M_PI / 180);

    RM.m[0] = make_float4(c+(1-c)*x*x  , (1-c)*y*x-s*z, (1-c)*z*x+s*y, 0.f);
    RM.m[1] = make_float4((1-c)*x*y+s*z, c+(1-c)*y*y  , (1-c)*z*y-s*x, 0.f);
    RM.m[2] = make_float4((1-c)*x*z-s*y, (1-c)*y*z+s*x, c+(1-c)*z*z  , 0.f);
    RM.m[3] = make_float4(0.f, 0.f, 0.f, 1.f);
    
    return RM;
}




inline __host__ __device__ float4 mul(const float4x4 &M, const float4 &v)
{
    float4 r;
    r.x = dot(v, M.m[0]);
    r.y = dot(v, M.m[1]);
    r.z = dot(v, M.m[2]);
    r.w = dot(v, M.m[3]);
    return r;
}

inline __host__ __device__ float3 mul(const float3x4 &M, const float3 &v)
{
    float3 r;
    r.x = dot(v, make_float3(M.m[0]));
    r.y = dot(v, make_float3(M.m[1]));
    r.z = dot(v, make_float3(M.m[2]));
    return r;
}

// transform vector by matrix with translation
inline __host__ __device__ float4 mul(const float3x4 &M, const float4 &v)
{
    float4 r;
    r.x = dot(v, M.m[0]);
    r.y = dot(v, M.m[1]);
    r.z = dot(v, M.m[2]);
    r.w = 1.0f;
    return r;
}

inline __host__ __device__ float4x4 float4x4_Frustum(float left, float right, float bottom, float top, float near, float far)
{
    float4x4 FM;

    float A = (right + left) / (right - left);
    float B = (top + bottom) / (top - bottom);
    float C = -(far + near)  / (far - near);
    float D = -2 * far * near / (far - near);

    FM.m[0] = make_float4( 2 * near / (right - left), 0, A, 0);
    FM.m[1] = make_float4( 0, 2 * near / (top - bottom), B, 0);
    FM.m[2] = make_float4( 0, 0, C,  D);
    FM.m[3] = make_float4( 0, 0, -1, 0);

    return FM;
}


inline __host__ __device__ float4x4 float4x4_Perspective(float fov, float aspect, float near, float far)
{
    float4x4 PM;

    float f = 1.f / tan( fov * M_PI / 180 / 2.f);

    PM.m[0] = make_float4(f/aspect, 0 , 0, 0);
    PM.m[1] = make_float4(0, f , 0, 0);
    PM.m[2] = make_float4(0, 0 , (near + far) / (near-far), 2 * far * near / (near-far));
    PM.m[3] = make_float4(0, 0 , -1.f, 0);

//     PM.m[2] = make_float4(0, 0 , far / far - near, far * near / (near-far));
//     PM.m[3] = make_float4(0, 0 , 1.f, 0);

    return PM;
}

inline __host__ __device__ float4x4 float4x4_Orthographic(float l, float r,
                                                          float b, float t,
                                                          float n, float f)
{
    float4x4 OM;
    
    OM.m[0] = make_float4(2/(r-l), 0       , 0        , -(r+l)/(r-l));
    OM.m[1] = make_float4(0      , 2/(t-b) , 0        , -(t+b)/(t-b));
    OM.m[2] = make_float4(0      , 0       , 2/(f - n), -(f+n)/(f-n));
    OM.m[3] = make_float4(0      , 0       , 0        , 1);
    
    return OM;
}

inline __host__ __device__ float4x4 float4x4_LookAt(float3 eye,
                                                    float3 cen,
                                                    float3 up)
{
    float3 f = cen - eye;
    f  = normalize(f);
    up = normalize(up);

    float3 s = cross(f, up);
    float3 u = cross(s,f);

    float4x4 M;

    M.m[0] = make_float4(s.x, s.y, s.z, 0);
    M.m[1] = make_float4(u.x, u.y, u.z, 0);
    M.m[2] = make_float4(-f.x, -f.y, -f.z, 0);
    M.m[3] = make_float4( 0, 0, 0, 1);

    return M;
}

std::ostream& operator<<(std::ostream& output, const float4x4& v);
std::ostream& operator<<(std::ostream& output, const float3x3& v);
std::ostream& operator<<(std::ostream& output, const float3x4& v);

#endif
