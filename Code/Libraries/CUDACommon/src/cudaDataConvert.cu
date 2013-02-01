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

#include <cpl.h>
#include <cudaDataConvert.h>

// template void convertX_YtoXY_new(float2* d_xy, float* d_x, float* d_y, uint n, cudaStream_t stream);
// template void convertX_YtoXY_new(uint2* d_xy, uint* d_x, uint* d_y, uint n, cudaStream_t stream);
template void convertXYtoX_Y(float* d_x, float* d_y, float2* d_xy, uint n, cudaStream_t stream);
template void convertX_YtoXY(float2* d_xy, float* d_x, float* d_y, uint n, cudaStream_t stream);

template void convertXYZtoX_Y_Z(float* d_x, float* d_y, float* d_z, float3* d_xyz, uint n, cudaStream_t stream);
template void convertX_Y_ZtoXYZ(float3* d_xyz, float* d_x, float* d_y, float* d_z, uint n, cudaStream_t stream);

template void convertXYZWtoX_Y_Z_W(float* d_x, float* d_y, float* d_z, float* d_w, float4* d_xyzw, uint n, cudaStream_t stream);
template void convertX_Y_Z_WtoXYZW(float4* d_xyzw, float* d_x, float* d_y, float* d_z, float* d_w, uint n, cudaStream_t stream);

template void convertXYZWtoX_Y_Z(float* d_x, float* d_y, float* d_z, float4* d_xyzw, uint n, cudaStream_t stream);
template void convertX_Y_ZtoXYZW(float4* d_xyz, float* d_x, float* d_y, float* d_z, uint n, cudaStream_t stream);

template void convertXYtoX_Y(int* d_x, int* d_y, int2* d_xy, uint n, cudaStream_t stream);
template void convertX_YtoXY(int2* d_xy, int* d_x, int* d_y, uint n, cudaStream_t stream);

template void convertXYZtoX_Y_Z(int* d_x, int* d_y, int* d_z, int3* d_xyz, uint n, cudaStream_t stream);
template void convertX_Y_ZtoXYZ(int3* d_xyz, int* d_x, int* d_y, int* d_z, uint n, cudaStream_t stream);

template void convertXYZWtoX_Y_Z_W(int* d_x, int* d_y, int* d_z, int* d_w, int4* d_xyzw, uint n, cudaStream_t stream);
template void convertX_Y_Z_WtoXYZW(int4* d_xyzw, int* d_x, int* d_y, int* d_z, int* d_w, uint n, cudaStream_t stream);

template void convertXYZWtoX_Y_Z(int* d_x, int* d_y, int* d_z, int4* d_xyzw, uint n, cudaStream_t stream);
template void convertX_Y_ZtoXYZW(int4* d_xyz, int* d_x, int* d_y, int* d_z, uint n, cudaStream_t stream);

template void convertXYtoX_Y(uint* d_x, uint* d_y, uint2* d_xy, uint n, cudaStream_t stream);
template void convertX_YtoXY(uint2* d_xy, uint* d_x, uint* d_y, uint n, cudaStream_t stream);

template void convertXYZtoX_Y_Z(uint* d_x, uint* d_y, uint* d_z, uint3* d_xyz, uint n, cudaStream_t stream);
template void convertX_Y_ZtoXYZ(uint3* d_xyz, uint* d_x, uint* d_y, uint* d_z, uint n, cudaStream_t stream);

template void convertXYZWtoX_Y_Z_W(uint* d_x, uint* d_y, uint* d_z, uint* d_w, uint4* d_xyzw, uint n, cudaStream_t stream);
template void convertX_Y_Z_WtoXYZW(uint4* d_xyzw, uint* d_x, uint* d_y, uint* d_z, uint* d_w, uint n, cudaStream_t stream);

template void convertXYZWtoX_Y_Z(uint* d_x, uint* d_y, uint* d_z, uint4* d_xyzw, uint n, cudaStream_t stream);
template void convertX_Y_ZtoXYZW(uint4* d_xyz, uint* d_x, uint* d_y, uint* d_z, uint n, cudaStream_t stream);


#include <cudaVector3DArray.h>

void convertXYZtoX_Y_Z(cplVector3DArray& d_o, float3* d_i, uint n, cudaStream_t stream){
    convertXYZtoX_Y_Z(d_o.x, d_o.y, d_o.z, d_i, n, stream);
};

void convertX_Y_ZtoXYZ(float3* d_o, cplVector3DArray& d_i, uint n, cudaStream_t stream){
    convertX_Y_ZtoXYZ(d_o, d_i.x, d_i.y, d_i.z, n, stream);
};

void convertX_Y_ZtoXYZW(float4* d_o, cplVector3DArray& d_i, uint n, cudaStream_t stream){
    convertX_Y_ZtoXYZW(d_o, d_i.x, d_i.y, d_i.z, n, stream);
}

void convertXYZWtoX_Y_Z(cplVector3DArray& d_o, float4* d_i, uint n, cudaStream_t stream){
    convertXYZWtoX_Y_Z(d_o.x, d_o.y, d_o.z, d_i, n, stream);
}

#include "cudaDataConvertDef.h"
void testDataConvert(int n) {
    int2* d_o_xy;
    int4* d_o_xyzw;

    int* d_i_x, *d_i_y, *d_i_z, *d_i_w;
    
    dmemAlloc(d_i_x, n);
    dmemAlloc(d_i_y, n);
    dmemAlloc(d_i_z, n);
    dmemAlloc(d_i_w, n);

    cplVectorOpers::SetMem(d_i_x, 0, n);
    cplVectorOpers::SetMem(d_i_y, 1, n);
    cplVectorOpers::SetMem(d_i_z, 2, n);
    cplVectorOpers::SetMem(d_i_w, 3, n);

    printDeviceArray1D(d_i_x, 32, "X");
    printDeviceArray1D(d_i_y, 32, "Y");

    dmemAlloc(d_o_xy, n);
    dmemAlloc(d_o_xyzw, n);

    convertX_YtoXY(d_o_xy, d_i_x, d_i_y, n);
    printDeviceArray1D((int*)d_o_xy, 64, "XY");

    convertX_Y_Z_WtoXYZW(d_o_xyzw, d_i_x, d_i_y, d_i_z, d_i_w, n);
    printDeviceArray1D((int*)d_o_xyzw, 128, "XYZW");

    convertXYtoX_Y(d_i_x, d_i_y, d_o_xy,n);
    printDeviceArray1D(d_i_x, 32, "X");
    printDeviceArray1D(d_i_y, 32, "Y");
    
    convertXYZWtoX_Y_Z_W(d_i_x, d_i_y, d_i_z, d_i_w, d_o_xyzw, n);
    printDeviceArray1D(d_i_x, 32, "X");
    printDeviceArray1D(d_i_y, 32, "Y");
    printDeviceArray1D(d_i_z, 32, "Z");
    printDeviceArray1D(d_i_w, 32, "W");
}



