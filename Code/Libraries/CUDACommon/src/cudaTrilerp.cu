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

#ifndef __CUDA_TRILERP_CUH
#define __CUDA_TRILERP_CUH

#include "libDefine.h"

texture<float , 1, cudaReadModeElementType> com_tex_float;
texture<float , 1, cudaReadModeElementType> com_tex_float_x;
texture<float , 1, cudaReadModeElementType> com_tex_float_y;
texture<float , 1, cudaReadModeElementType> com_tex_float_z;

//////////////////////////////////////////////////////////////////////////////////////////
// Get value from the 1D texture 
//////////////////////////////////////////////////////////////////////////////////////////
inline void cache_bind(const float* d_x, const float* d_y, const float* d_z)
{
    cudaBindTexture(NULL, com_tex_float_x, d_x);
    cudaBindTexture(NULL, com_tex_float_y, d_y);
    cudaBindTexture(NULL, com_tex_float_z, d_z);
}

inline void cache_bind_x(const float* d_i) {
    cudaBindTexture(NULL, com_tex_float_x, d_i);
}

inline void cache_bind_y(const float* d_i) {
    cudaBindTexture(NULL, com_tex_float_y, d_i);
}

inline void cache_bind_z(const float* d_i) {
    cudaBindTexture(NULL, com_tex_float_z, d_i);
}

__inline__ __device__ float getTexVal(int index){
    return tex1Dfetch(com_tex_float, index);
}

__inline__ __device__ float getTexValX(int index){
    return tex1Dfetch(com_tex_float_x, index);
}

__inline__ __device__ float getTexValY(int index){
    return tex1Dfetch(com_tex_float_y, index);
}

__inline__ __device__ float getTexValZ(int index){
    return tex1Dfetch(com_tex_float_z, index);
}


//////////////////////////////////////////////////////////////////////////////////////////
// Get the pixel from 3D array 
//////////////////////////////////////////////////////////////////////////////////////////
__inline__ __device__ float getPixel3D(int x, int y, int z,
                                       const float* d_i,
                                       int sizeX, int sizeY, int sizeZ)
{
    int index = (z * sizeY + y) * sizeX + x;
    return d_i[index];
}

__inline__ __device__ float getPixel3D(int x, int y, int z,
                                       int sizeX, int sizeY, int sizeZ)
{
    int index = (z * sizeY + y) * sizeX + x;
    return tex1Dfetch(com_tex_float, index);
}

__inline__ __device__ void getPixel3D(float& vx, float& vy, float& vz,
                                      int x, int y, int z,
                                      const float* d_iX, const float* d_iY, const float* d_iZ,
                                      int sizeX, int sizeY, int sizeZ)
{
    int index = (z * sizeY + y) * sizeX + x;

    vx = d_iX[index];
    vy = d_iY[index];
    vz = d_iZ[index];
}

__inline__ __device__ void getPixel3D(float& vx, float& vy, float& vz,
                                      int x, int y, int z,
                                      int sizeX, int sizeY, int sizeZ)
{
    int index = (z * sizeY + y) * sizeX + x;

    vx = tex1Dfetch(com_tex_float_x, index);
    vy = tex1Dfetch(com_tex_float_y, index);
    vz = tex1Dfetch(com_tex_float_z, index);
}

__inline__ __device__ void getPixel3D_tex(float& vx, float& vy, float& vz,
                               int x, int y, int z,
                               int sizeX, int sizeY, int sizeZ)
{
    int index = (z * sizeY + y) * sizeX + x;

    vx = tex1Dfetch(com_tex_float_x, index);
    vy = tex1Dfetch(com_tex_float_y, index);
    vz = tex1Dfetch(com_tex_float_z, index); 
}


// This simple trilerp works when the point is completely inside the volume
// and can not handle sophisicated strategy 
__inline__ __device__ float simple_triLerp_kernel(const float *image ,
                                       float x, float y, float z,
                                       uint w, uint wh)
{
    int xInt, yInt, zInt;
    float dx, dy, dz;
    float dxy, oz;

    xInt = int(x);
    yInt = int(y);
    zInt = int(z);

    dx = x - xInt;
    dy = y - yInt;
    dz = z - zInt;

    dxy = dx * dy;
    oz = 1.f - dz;

    int id = xInt  + yInt  * w + zInt  * wh;
    float x0y0z0 = image[id    ];
    float x0y0z1 = image[id    +wh];
        
    float x1y0z0 = image[id+1  ];
    float x1y0z1 = image[id+1  +wh];

    float x0y1z0 = image[id+w  ];
    float x0y1z1 = image[id+w  +wh];
    
    float x1y1z0 = image[id+w+1];
    float x1y1z1 = image[id+w+1+wh];

    float b1 = (x1y0z0* (dx - dxy) + x0y1z0 * (dy - dxy) + x1y1z0*dxy + x0y0z0 * (1-dy -(dx-dxy)));
    float b2 = (x1y0z1* (dx - dxy) + x0y1z1 * (dy - dxy) + x1y1z1*dxy + x0y0z1 * (1-dy -(dx-dxy)));

    return (b1 * oz + b2 * dz);
}

__inline__ __device__ float simple_triLerpTex_kernel(const float* image, 
					  float x, float y, float z,
                                          uint w, uint wh)
{
    int xInt, yInt, zInt;
    float dx, dy, dz;
    float dxy, oz;

    xInt = int(x);
    yInt = int(y);
    zInt = int(z);

    dx = x - xInt;
    dy = y - yInt;
    dz = z - zInt;

    dxy = dx * dy;
    oz = 1.f - dz;

    int id = xInt  + yInt  * w + zInt  * wh;

    float x0y0z0 = image[id];
    
    float x0y1z0 = tex1Dfetch(com_tex_float,id+w);
    float x0y0z1 = tex1Dfetch(com_tex_float,id+wh);
    float x0y1z1 = tex1Dfetch(com_tex_float,id+wh+w);

    float x1y0z0 = tex1Dfetch(com_tex_float,id+1  );
    float x1y1z0 = tex1Dfetch(com_tex_float,id+1+w);
    
    float x1y0z1 = tex1Dfetch(com_tex_float,id+1+wh);
    float x1y1z1 = tex1Dfetch(com_tex_float,id+1+wh+w);

    float b1 = (x1y0z0* (dx - dxy) + x0y1z0 * (dy - dxy) + x1y1z0*dxy + x0y0z0 * (1-dy -(dx-dxy)));
    float b2 = (x1y0z1* (dx - dxy) + x0y1z1 * (dy - dxy) + x1y1z1*dxy + x0y0z1 * (1-dy -(dx-dxy)));

    return (b1 * oz + b2 * dz);
}

//////////////////////////////////////////////////////////////////////////////////////////
// Wrap around strategy 
//////////////////////////////////////////////////////////////////////////////////////////
__inline__ __device__ void wrapBackground(int& floorX, int& ceilX, int sizeX){

    if(floorX < 0) floorX = sizeX + floorX;
    else if(floorX >= sizeX) floorX = floorX % sizeX;
        
    if(ceilX < 0) ceilX = sizeX + ceilX;
    else if(ceilX >= sizeX) ceilX = ceilX % sizeX;
}

__inline__ __device__ void wrapBackground(int& floorX,int& floorY,int& floorZ,
                               int& ceilX, int& ceilY, int& ceilZ,
                               int  sizeX, int  sizeY, int  sizeZ){

    wrapBackground(floorX, ceilX, sizeX);
    wrapBackground(floorY, ceilY, sizeY);
    wrapBackground(floorZ, ceilZ, sizeZ);
}


//////////////////////////////////////////////////////////////////////////////////////////
// Clamp strategy
//////////////////////////////////////////////////////////////////////////////////////////
__inline__ __device__ void clampBackground(int& floorX, int& ceilX, int sizeX){
    if(floorX < 0) floorX = 0;
    else if(floorX >= sizeX) floorX = sizeX-1;
    
    if(ceilX < 0) ceilX = 0;
    else if(ceilX >= sizeX) ceilX = sizeX-1;
}

__inline__ __device__ void clampBackground(int& floorX,int& floorY,int& floorZ,
                                int& ceilX, int& ceilY, int& ceilZ,
                                int  sizeX, int  sizeY, int  sizeZ){

    clampBackground(floorX, ceilX, sizeX);
    clampBackground(floorY, ceilY, sizeY);
    clampBackground(floorZ, ceilZ, sizeZ);
}


//////////////////////////////////////////////////////////////////////////////////////////
// Check if the point is completely inside the boundary
//////////////////////////////////////////////////////////////////////////////////////////

__inline__ __device__ int isInside(int floorX,int floorY,int floorZ,
                        int ceilX, int ceilY, int ceilZ,
                        int  sizeX, int  sizeY, int  sizeZ){
    
    return (floorX >= 0 && ceilX < sizeX &&
            floorY >= 0 && ceilY < sizeY &&
            floorZ >= 0 && ceilZ < sizeZ);
}



//////////////////////////////////////////////////////////////////////////////////////////
// Trilerp function for single array input 
//////////////////////////////////////////////////////////////////////////////////////////
template<int backgroundStrategy>
__device__ void triLerp(float& hx, float& hy, float& hz,
                        const float* imgX, const float* imgY, const float* imgZ,
                        float x, float y, float z,
                        int sizeX, int sizeY, int sizeZ){
    
    int floorX = (int)(x);
    int floorY = (int)(y);
    int floorZ = (int)(z);

    if (x < 0 && x != (int)(x)) --floorX;
    if (y < 0 && y != (int)(y)) --floorY;
    if (z < 0 && z != (int)(z)) --floorZ;

    // this is not truly ceiling, but floor + 1, which is usually ceiling    
    int ceilX = floorX + 1;
    int ceilY = floorY + 1;
    int ceilZ = floorZ + 1;

    float t = x - floorX;
	float u = y - floorY;
	float v = z - floorZ;

    float oneMinusT = 1.f - t;
	float oneMinusU = 1.f - u;
    float oneMinusV = 1.f - v;

    float v0X=0.f, v0Y=0.f, v0Z=0.f;
    float v1X=0.f, v1Y=0.f, v1Z=0.f;
    float v2X=0.f, v2Y=0.f, v2Z=0.f;
    float v3X=0.f, v3Y=0.f, v3Z=0.f;
    float v4X=0.f, v4Y=0.f, v4Z=0.f;
    float v5X=0.f, v5Y=0.f, v5Z=0.f;
    float v6X=0.f, v6Y=0.f, v6Z=0.f;
    float v7X=0.f, v7Y=0.f, v7Z=0.f;

    int inside = 1;

    // adjust the position of the sample point if required
    if (backgroundStrategy == BACKGROUND_STRATEGY_WRAP){
        wrapBackground(floorX, floorY, floorZ,
                       ceilX, ceilY, ceilZ,
                       sizeX, sizeY, sizeZ);
    }
    else if (backgroundStrategy == BACKGROUND_STRATEGY_CLAMP){
        clampBackground(floorX, floorY, floorZ,
                        ceilX, ceilY, ceilZ,
                        sizeX, sizeY, sizeZ);
    }
    else {
        inside = isInside(floorX, floorY, floorZ,
                          ceilX, ceilY, ceilZ,
                          sizeX, sizeY, sizeZ);
    }

    
    if (inside){
        getPixel3D(v0X, v0Y, v0Z, floorX, floorY, floorZ, imgX, imgY, imgZ, sizeX, sizeY, sizeZ);
        getPixel3D(v1X, v1Y, v1Z, ceilX, floorY, floorZ,  imgX, imgY, imgZ, sizeX, sizeY, sizeZ);
        getPixel3D(v2X, v2Y, v2Z, ceilX, ceilY, floorZ,  imgX, imgY, imgZ, sizeX, sizeY, sizeZ);
        getPixel3D(v3X, v3Y, v3Z, floorX, ceilY, floorZ,  imgX, imgY, imgZ, sizeX, sizeY, sizeZ);

        getPixel3D(v4X, v4Y, v4Z, floorX, ceilY, ceilZ,  imgX, imgY, imgZ, sizeX, sizeY, sizeZ);
        getPixel3D(v5X, v5Y, v5Z, ceilX, ceilY, ceilZ,  imgX, imgY, imgZ, sizeX, sizeY, sizeZ);
        getPixel3D(v6X, v6Y, v6Z, ceilX, floorY, ceilZ,  imgX, imgY, imgZ, sizeX, sizeY, sizeZ);
        getPixel3D(v7X, v7Y, v7Z, floorX, floorY, ceilZ,  imgX, imgY, imgZ, sizeX, sizeY, sizeZ);
    }else if (backgroundStrategy == BACKGROUND_STRATEGY_ID){
            //
            // coordinate is not inside volume, return identity
            //
            hx = x; hy = y; hz = z;
            return;
    } else if (backgroundStrategy == BACKGROUND_STRATEGY_ZERO) {
            hx = 0; hy = 0; hz = 0;
            return;
    } else if (backgroundStrategy == BACKGROUND_STRATEGY_PARTIAL_ID ||
	       backgroundStrategy == BACKGROUND_STRATEGY_PARTIAL_ZERO){
	//
	// coordinate is not inside volume; initialize cube
	// corners to identity/zero then set any corners of cube that
	// fall on volume boundary
	//
	if(backgroundStrategy == BACKGROUND_STRATEGY_PARTIAL_ID){
	  v0X = floorX;  v0Y = floorY;  v0Z = floorZ;
      v1X = ceilX;   v1Y = floorY;  v1Z = floorZ;
      v2X = ceilX;   v2Y = ceilY;   v2Z = floorZ;
      v3X = floorX;  v3Y = ceilY;   v3Z = floorZ;
      v4X = floorX;  v4Y = ceilY;   v4Z = ceilZ;
      v5X = ceilX;   v5Y = ceilY;   v5Z = ceilZ;
      v6X = ceilX;   v6Y = floorY;  v6Z = ceilZ;
      v7X = floorX;  v7Y = floorY;  v7Z = ceilZ;
	  // 			v0X = x; v0Y = y; v0Z = z;
	  // 			v1X = x; v1Y = y; v1Z = z;
	  // 			v2X = x; v2Y = y; v2Z = z;
	  // 			v3X = x; v3Y = y; v3Z = z;
	  // 			v4X = x; v4Y = y; v4Z = z;
	  // 			v5X = x; v5Y = y; v5Z = z;
	  // 			v6X = x; v6Y = y; v6Z = z;
	  // 			v7X = x; v7Y = y; v7Z = z;
	}
	else{ // BACKGROUND_STRATEGY_PARTIAL_ZERO
	  v0X = 0; v0Y = 0; v0Z = 0;
	  v1X = 0; v1Y = 0; v1Z = 0;
	  v2X = 0; v2Y = 0; v2Z = 0;
	  v3X = 0; v3Y = 0; v3Z = 0;
	  v4X = 0; v4Y = 0; v4Z = 0;
	  v5X = 0; v5Y = 0; v5Z = 0;
	  v6X = 0; v6Y = 0; v6Z = 0;
	  v7X = 0; v7Y = 0; v7Z = 0;
	}
	bool floorXIn = (floorX >= 0) && (floorX < sizeX);
	bool floorYIn = (floorY >= 0) && (floorY < sizeY);
	bool floorZIn = (floorZ >= 0) && (floorZ < sizeZ);
	
	bool ceilXIn = (ceilX >= 0) && (ceilX < sizeX);
	bool ceilYIn = (ceilY >= 0) && (ceilY < sizeY);
	bool ceilZIn = (ceilZ >= 0) && (ceilZ < sizeZ);

	if (floorXIn && floorYIn && floorZIn)
	  getPixel3D(v0X, v0Y, v0Z, floorX, floorY, floorZ, imgX, imgY, imgZ, sizeX, sizeY, sizeZ);
            
	if (ceilXIn && floorYIn && floorZIn)
	  getPixel3D(v1X, v1Y, v1Z, ceilX, floorY, floorZ,  imgX, imgY, imgZ, sizeX, sizeY, sizeZ);
            
	if (ceilXIn && ceilYIn && floorZIn)
	  getPixel3D(v2X, v2Y, v2Z, ceilX, ceilY, floorZ,  imgX, imgY, imgZ, sizeX, sizeY, sizeZ);

	if (floorXIn && ceilYIn && floorZIn)
	  getPixel3D(v3X, v3Y, v3Z, floorX, ceilY, floorZ,  imgX, imgY, imgZ, sizeX, sizeY, sizeZ);

	if (floorXIn && ceilYIn && ceilZIn)
	  getPixel3D(v4X, v4Y, v4Z, floorX, ceilY, ceilZ,  imgX, imgY, imgZ, sizeX, sizeY, sizeZ);
	
	if (ceilXIn && ceilYIn && ceilZIn)
	  getPixel3D(v5X, v5Y, v5Z, ceilX, ceilY, ceilZ,  imgX, imgY, imgZ, sizeX, sizeY, sizeZ);
	
	if (ceilXIn && floorYIn && ceilZIn)
	  getPixel3D(v6X, v6Y, v6Z, ceilX, floorY, ceilZ,  imgX, imgY, imgZ, sizeX, sizeY, sizeZ);
	
	if (floorXIn && floorYIn && ceilZIn)
	  getPixel3D(v7X, v7Y, v7Z, floorX, floorY, ceilZ,  imgX, imgY, imgZ, sizeX, sizeY, sizeZ);
    }
    

    //
    // do trilinear interpolation
    //
    
    //
    // this is the basic trilerp function...
    //
    //     h = 
    //       v0 * (1 - t) * (1 - u) * (1 - v) +
    //       v1 * t       * (1 - u) * (1 - v) +
    //       v2 * t       * u       * (1 - v) +
    //       v3 * (1 - t) * u       * (1 - v) +
    //       v4 * (1 - t) * u       * v       +
    //       v5 * t       * u       * v       +
    //       v6 * t       * (1 - u) * v       +
    //       v7 * (1 - t) * (1 - u) * v;
    //
    // the following nested version saves 30 multiplies.
    //
    hx = 
        oneMinusT * (oneMinusU * (v0X * oneMinusV + v7X * v)  +
                     u         * (v3X * oneMinusV + v4X * v)) +
        t         * (oneMinusU * (v1X * oneMinusV + v6X * v)  +
                     u         * (v2X * oneMinusV + v5X * v));
    
    hy = 
        oneMinusT * (oneMinusU * (v0Y * oneMinusV + v7Y * v)  +
                     u         * (v3Y * oneMinusV + v4Y * v)) +
        t         * (oneMinusU * (v1Y * oneMinusV + v6Y * v)  +
                     u         * (v2Y * oneMinusV + v5Y * v));
    
    hz = 
        oneMinusT * (oneMinusU * (v0Z * oneMinusV + v7Z * v)  +
                     u         * (v3Z * oneMinusV + v4Z * v)) +
        t         * (oneMinusU * (v1Z * oneMinusV + v6Z * v)  +
                     u         * (v2Z * oneMinusV + v5Z * v));

}

template<int backgroundStrategy>
__device__ void triLerp_tex(float& hx, float& hy, float& hz,
                            float x, float y, float z,
                            int sizeX, int sizeY, int sizeZ){
    
    int floorX = (int)(x);
    int floorY = (int)(y);
    int floorZ = (int)(z);

    if (x < 0 && x != (int)(x)) --floorX;
    if (y < 0 && y != (int)(y)) --floorY;
    if (z < 0 && z != (int)(z)) --floorZ;

    // this is not truly ceiling, but floor + 1, which is usually ceiling    
    int ceilX = floorX + 1;
    int ceilY = floorY + 1;
    int ceilZ = floorZ + 1;

    float t = x - floorX;
	float u = y - floorY;
	float v = z - floorZ;

    float oneMinusT = 1.f - t;
	float oneMinusU = 1.f - u;
    float oneMinusV = 1.f - v;

    float v0X=0.f, v0Y=0.f, v0Z=0.f;
    float v1X=0.f, v1Y=0.f, v1Z=0.f;
    float v2X=0.f, v2Y=0.f, v2Z=0.f;
    float v3X=0.f, v3Y=0.f, v3Z=0.f;
    float v4X=0.f, v4Y=0.f, v4Z=0.f;
    float v5X=0.f, v5Y=0.f, v5Z=0.f;
    float v6X=0.f, v6Y=0.f, v6Z=0.f;
    float v7X=0.f, v7Y=0.f, v7Z=0.f;

    int inside = 1;

    // adjust the position of the sample point if required
    if (backgroundStrategy == BACKGROUND_STRATEGY_WRAP){
        wrapBackground(floorX, floorY, floorZ,
                       ceilX, ceilY, ceilZ,
                       sizeX, sizeY, sizeZ);
    }
    else if (backgroundStrategy == BACKGROUND_STRATEGY_CLAMP){
        clampBackground(floorX, floorY, floorZ,
                        ceilX, ceilY, ceilZ,
                        sizeX, sizeY, sizeZ);
    }
    else {
        inside = isInside(floorX, floorY, floorZ,
                          ceilX, ceilY, ceilZ,
                          sizeX, sizeY, sizeZ);
    }

    if (inside){
        getPixel3D(v0X, v0Y, v0Z, floorX, floorY, floorZ, sizeX, sizeY, sizeZ);
        getPixel3D(v1X, v1Y, v1Z, ceilX, floorY, floorZ,  sizeX, sizeY, sizeZ);
        getPixel3D(v2X, v2Y, v2Z, ceilX, ceilY, floorZ,   sizeX, sizeY, sizeZ);
        getPixel3D(v3X, v3Y, v3Z, floorX, ceilY, floorZ,  sizeX, sizeY, sizeZ);

        getPixel3D(v4X, v4Y, v4Z, floorX, ceilY, ceilZ,   sizeX, sizeY, sizeZ);
        getPixel3D(v5X, v5Y, v5Z, ceilX, ceilY, ceilZ,    sizeX, sizeY, sizeZ);
        getPixel3D(v6X, v6Y, v6Z, ceilX, floorY, ceilZ,   sizeX, sizeY, sizeZ);
        getPixel3D(v7X, v7Y, v7Z, floorX, floorY, ceilZ,  sizeX, sizeY, sizeZ);
    }else if (backgroundStrategy == BACKGROUND_STRATEGY_ID){
            //
            // coordinate is not inside volume, return identity
            //
            hx = x; hy = y; hz = z;
            return;
    } else if (backgroundStrategy == BACKGROUND_STRATEGY_ZERO) {
            hx = 0; hy = 0; hz = 0;
            return;
    } else if (backgroundStrategy == BACKGROUND_STRATEGY_PARTIAL_ID){
			//
			// coordinate is not inside volume; initialize cube
			// corners to identity then set any corners of cube that
			// fall on volume boundary
			//
			v0X = x; v0Y = y; v0Z = z;
			v1X = x; v1Y = y; v1Z = z;
			v2X = x; v2Y = y; v2Z = z;
			v3X = x; v3Y = y; v3Z = z;
			v4X = x; v4Y = y; v4Z = z;
			v5X = x; v5Y = y; v5Z = z;
			v6X = x; v6Y = y; v6Z = z;
			v7X = x; v7Y = y; v7Z = z;

            bool floorXIn = (floorX >= 0) && (floorX < sizeX);
            bool floorYIn = (floorY >= 0) && (floorY < sizeY);
            bool floorZIn = (floorZ >= 0) && (floorZ < sizeZ);

            bool ceilXIn = (ceilX >= 0) && (ceilX < sizeX);
            bool ceilYIn = (ceilY >= 0) && (ceilY < sizeY);
            bool ceilZIn = (ceilZ >= 0) && (ceilZ < sizeZ);

			if (floorXIn && floorYIn && floorZIn)
                getPixel3D(v0X, v0Y, v0Z, floorX, floorY, floorZ, sizeX, sizeY, sizeZ);
            
			if (ceilXIn && floorYIn && floorZIn)
                getPixel3D(v1X, v1Y, v1Z, ceilX, floorY, floorZ,  sizeX, sizeY, sizeZ);
            
			if (ceilXIn && ceilYIn && floorZIn)
                getPixel3D(v2X, v2Y, v2Z, ceilX, ceilY, floorZ,   sizeX, sizeY, sizeZ);

            if (floorXIn && ceilYIn && floorZIn)
                getPixel3D(v3X, v3Y, v3Z, floorX, ceilY, floorZ,  sizeX, sizeY, sizeZ);

            if (floorXIn && ceilYIn && ceilZIn)
                getPixel3D(v4X, v4Y, v4Z, floorX, ceilY, ceilZ,   sizeX, sizeY, sizeZ);

            if (ceilXIn && ceilYIn && ceilZIn)
                getPixel3D(v5X, v5Y, v5Z, ceilX, ceilY, ceilZ,    sizeX, sizeY, sizeZ);

            if (ceilXIn && floorYIn && ceilZIn)
                getPixel3D(v6X, v6Y, v6Z, ceilX, floorY, ceilZ,   sizeX, sizeY, sizeZ);

            if (floorXIn && floorYIn && ceilZIn)
                getPixel3D(v7X, v7Y, v7Z, floorX, floorY, ceilZ,  sizeX, sizeY, sizeZ);
    }
    

    //
    // do trilinear interpolation
    //
    
    //
    // this is the basic trilerp function...
    //
    //     h = 
    //       v0 * (1 - t) * (1 - u) * (1 - v) +
    //       v1 * t       * (1 - u) * (1 - v) +
    //       v2 * t       * u       * (1 - v) +
    //       v3 * (1 - t) * u       * (1 - v) +
    //       v4 * (1 - t) * u       * v       +
    //       v5 * t       * u       * v       +
    //       v6 * t       * (1 - u) * v       +
    //       v7 * (1 - t) * (1 - u) * v;
    //
    // the following nested version saves 30 multiplies.
    //
    hx = 
        oneMinusT * (oneMinusU * (v0X * oneMinusV + v7X * v)  +
                     u         * (v3X * oneMinusV + v4X * v)) +
        t         * (oneMinusU * (v1X * oneMinusV + v6X * v)  +
                     u         * (v2X * oneMinusV + v5X * v));
    
    hy = 
        oneMinusT * (oneMinusU * (v0Y * oneMinusV + v7Y * v)  +
                     u         * (v3Y * oneMinusV + v4Y * v)) +
        t         * (oneMinusU * (v1Y * oneMinusV + v6Y * v)  +
                     u         * (v2Y * oneMinusV + v5Y * v));
    
    hz = 
        oneMinusT * (oneMinusU * (v0Z * oneMinusV + v7Z * v)  +
                     u         * (v3Z * oneMinusV + v4Z * v)) +
        t         * (oneMinusU * (v1Z * oneMinusV + v6Z * v)  +
                     u         * (v2Z * oneMinusV + v5Z * v));

}


template<int backgroundStrategy>
__device__ float triLerp_tex(
    float x, float y, float z,
    int sizeX, int sizeY, int sizeZ){
    
    int floorX = (int)(x);
    int floorY = (int)(y);
    int floorZ = (int)(z);

    if (x < 0 && x != (int)(x)) --floorX;
    if (y < 0 && y != (int)(y)) --floorY;
    if (z < 0 && z != (int)(z)) --floorZ;

    // this is not truly ceiling, but floor + 1, which is usually ceiling    
    int ceilX = floorX + 1;
    int ceilY = floorY + 1;
    int ceilZ = floorZ + 1;

    float t = x - floorX;
	float u = y - floorY;
	float v = z - floorZ;

    float oneMinusT = 1.f- t;
	float oneMinusU = 1.f- u;
    float oneMinusV = 1.f- v;

    float v0, v1, v2, v3, v4, v5, v6, v7;
    int inside = 1;

    // adjust the position of the sample point if required
    if (backgroundStrategy == BACKGROUND_STRATEGY_WRAP){
        wrapBackground(floorX, floorY, floorZ,
                       ceilX, ceilY, ceilZ,
                       sizeX, sizeY, sizeZ);
    }
    else if (backgroundStrategy == BACKGROUND_STRATEGY_CLAMP){
        clampBackground(floorX, floorY, floorZ,
                        ceilX, ceilY, ceilZ,
                        sizeX, sizeY, sizeZ);
    }
    else {
        inside = isInside(floorX, floorY, floorZ,
                          ceilX, ceilY, ceilZ,
                          sizeX, sizeY, sizeZ);
    }

    
    if (inside){
        v0 = getPixel3D(floorX, floorY, floorZ, sizeX, sizeY, sizeZ);
        v1 = getPixel3D(ceilX, floorY, floorZ,  sizeX, sizeY, sizeZ);
        v2 = getPixel3D(ceilX, ceilY, floorZ,   sizeX, sizeY, sizeZ);
        v3 = getPixel3D(floorX, ceilY, floorZ,  sizeX, sizeY, sizeZ);

        v4 = getPixel3D(floorX, ceilY, ceilZ,   sizeX, sizeY, sizeZ);
        v5 = getPixel3D(ceilX, ceilY, ceilZ,    sizeX, sizeY, sizeZ);
        v6 = getPixel3D(ceilX, floorY, ceilZ,   sizeX, sizeY, sizeZ);
        v7 = getPixel3D(floorX, floorY, ceilZ,  sizeX, sizeY, sizeZ);
    }else {
        bool floorXIn = floorX >= 0 && floorX < sizeX;
        bool floorYIn = floorY >= 0 && floorY < sizeY;
        bool floorZIn = floorZ >= 0 && floorZ < sizeZ;
      
        bool ceilXIn = ceilX >= 0 && ceilX < sizeX;
        bool ceilYIn = ceilY >= 0 && ceilY < sizeY;
        bool ceilZIn = ceilZ >= 0 && ceilZ < sizeZ;

        v0 = (floorXIn && floorYIn && floorZIn) ? getPixel3D(floorX, floorY, floorZ, sizeX, sizeY, sizeZ): 0.f;
        v1 = (ceilXIn && floorYIn && floorZIn)  ? getPixel3D(ceilX, floorY, floorZ,  sizeX, sizeY, sizeZ): 0.f;
        v2 = (ceilXIn && ceilYIn && floorZIn)   ? getPixel3D(ceilX, ceilY, floorZ,   sizeX, sizeY, sizeZ): 0.f;
        v3 = (floorXIn && ceilYIn && floorZIn)  ? getPixel3D(floorX, ceilY, floorZ,  sizeX, sizeY, sizeZ): 0.f;
        
        v4 = (floorXIn && ceilYIn && ceilZIn)   ? getPixel3D(floorX, ceilY, ceilZ,  sizeX, sizeY, sizeZ): 0.f;
        v5 = (ceilXIn && ceilYIn && ceilZIn)    ? getPixel3D(ceilX, ceilY, ceilZ,   sizeX, sizeY, sizeZ): 0.f;
        v6 = (ceilXIn && floorYIn && ceilZIn)   ? getPixel3D(ceilX, floorY, ceilZ,  sizeX, sizeY, sizeZ): 0.f;
        v7 = (floorXIn && floorYIn && ceilZIn)  ? getPixel3D(floorX, floorY, ceilZ, sizeX, sizeY, sizeZ): 0.f;
    }

    //
    // do trilinear interpolation
    //
    
    //
    // this is the basic trilerp function...
    //
    //     h = 
    //       v0 * (1 - t) * (1 - u) * (1 - v) +
    //       v1 * t       * (1 - u) * (1 - v) +
    //       v2 * t       * u       * (1 - v) +
    //       v3 * (1 - t) * u       * (1 - v) +
    //       v4 * (1 - t) * u       * v       +
    //       v5 * t       * u       * v       +
    //       v6 * t       * (1 - u) * v       +
    //       v7 * (1 - t) * (1 - u) * v;
    //
    // the following nested version saves 30 multiplies.
    //
    return  oneMinusT * (oneMinusU * (v0 * oneMinusV + v7 * v)  +
                         u         * (v3 * oneMinusV + v4 * v)) +
        t         * (oneMinusU * (v1 * oneMinusV + v6 * v)  +
                     u         * (v2 * oneMinusV + v5 * v));
}

template<int backgroundStrategy>
__device__ float triLerp(const float* img,
			 float x, float y, float z,
			 int sizeX, int sizeY, int sizeZ)
{
    
    int floorX = (int)(x);
    int floorY = (int)(y);
    int floorZ = (int)(z);

    if (x < 0 && x != (int)(x)) --floorX;
    if (y < 0 && y != (int)(y)) --floorY;
    if (z < 0 && z != (int)(z)) --floorZ;

    // this is not truly ceiling, but floor + 1, which is usually ceiling    
    int ceilX = floorX + 1;
    int ceilY = floorY + 1;
    int ceilZ = floorZ + 1;

    float t = x - floorX;
	float u = y - floorY;
	float v = z - floorZ;

    float oneMinusT = 1.f- t;
	float oneMinusU = 1.f- u;
    float oneMinusV = 1.f- v;

    float v0, v1, v2, v3, v4, v5, v6, v7;
    int inside = 1;

    // adjust the position of the sample point if required
    if (backgroundStrategy == BACKGROUND_STRATEGY_WRAP){
        wrapBackground(floorX, floorY, floorZ,
                       ceilX, ceilY, ceilZ,
                       sizeX, sizeY, sizeZ);
    }
    else if (backgroundStrategy == BACKGROUND_STRATEGY_CLAMP){
        clampBackground(floorX, floorY, floorZ,
                        ceilX, ceilY, ceilZ,
                        sizeX, sizeY, sizeZ);
    }
    else {
        inside = isInside(floorX, floorY, floorZ,
                          ceilX, ceilY, ceilZ,
                          sizeX, sizeY, sizeZ);
    }

    
    if (inside){
        v0 = getPixel3D(floorX, floorY, floorZ, img, sizeX, sizeY, sizeZ);
        v1 = getPixel3D(ceilX, floorY, floorZ,  img, sizeX, sizeY, sizeZ);
        v2 = getPixel3D(ceilX, ceilY, floorZ,   img, sizeX, sizeY, sizeZ);
        v3 = getPixel3D(floorX, ceilY, floorZ,  img, sizeX, sizeY, sizeZ);

        v4 = getPixel3D(floorX, ceilY, ceilZ,  img, sizeX, sizeY, sizeZ);
        v5 = getPixel3D(ceilX, ceilY, ceilZ,   img, sizeX, sizeY, sizeZ);
        v6 = getPixel3D(ceilX, floorY, ceilZ,  img, sizeX, sizeY, sizeZ);
        v7 = getPixel3D(floorX, floorY, ceilZ, img, sizeX, sizeY, sizeZ);
    }else {
        bool floorXIn = floorX >= 0 && floorX < sizeX;
        bool floorYIn = floorY >= 0 && floorY < sizeY;
        bool floorZIn = floorZ >= 0 && floorZ < sizeZ;
      
        bool ceilXIn = ceilX >= 0 && ceilX < sizeX;
        bool ceilYIn = ceilY >= 0 && ceilY < sizeY;
        bool ceilZIn = ceilZ >= 0 && ceilZ < sizeZ;

        v0 = (floorXIn && floorYIn && floorZIn) ? getPixel3D(floorX, floorY, floorZ, img, sizeX, sizeY, sizeZ): 0.f;
        v1 = (ceilXIn && floorYIn && floorZIn)  ? getPixel3D(ceilX, floorY, floorZ,  img, sizeX, sizeY, sizeZ): 0.f;
        v2 = (ceilXIn && ceilYIn && floorZIn)   ? getPixel3D(ceilX, ceilY, floorZ,   img, sizeX, sizeY, sizeZ): 0.f;
        v3 = (floorXIn && ceilYIn && floorZIn)  ? getPixel3D(floorX, ceilY, floorZ,  img, sizeX, sizeY, sizeZ): 0.f;
        
        v4 = (floorXIn && ceilYIn && ceilZIn)   ? getPixel3D(floorX, ceilY, ceilZ,  img, sizeX, sizeY, sizeZ): 0.f;
        v5 = (ceilXIn && ceilYIn && ceilZIn)    ? getPixel3D(ceilX, ceilY, ceilZ,   img, sizeX, sizeY, sizeZ): 0.f;
        v6 = (ceilXIn && floorYIn && ceilZIn)   ? getPixel3D(ceilX, floorY, ceilZ,  img, sizeX, sizeY, sizeZ): 0.f;
        v7 = (floorXIn && floorYIn && ceilZIn)  ? getPixel3D(floorX, floorY, ceilZ, img, sizeX, sizeY, sizeZ): 0.f;
    }

    //
    // do trilinear interpolation
    //
    
    //
    // this is the basic trilerp function...
    //
    //     h = 
    //       v0 * (1 - t) * (1 - u) * (1 - v) +
    //       v1 * t       * (1 - u) * (1 - v) +
    //       v2 * t       * u       * (1 - v) +
    //       v3 * (1 - t) * u       * (1 - v) +
    //       v4 * (1 - t) * u       * v       +
    //       v5 * t       * u       * v       +
    //       v6 * t       * (1 - u) * v       +
    //       v7 * (1 - t) * (1 - u) * v;
    //
    // the following nested version saves 30 multiplies.
    //
    return  oneMinusT * (oneMinusU * (v0 * oneMinusV + v7 * v)  +
                         u         * (v3 * oneMinusV + v4 * v)) +
        t         * (oneMinusU * (v1 * oneMinusV + v6 * v)  +
                     u         * (v2 * oneMinusV + v5 * v));
}


#endif
