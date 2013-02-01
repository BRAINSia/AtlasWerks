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

#ifndef __CPU_IMAGE3D_H
#define __CPU_IMAGE3D_H

class cpuArray3D
{
public:
    cpuArray3D():data(0), sizeX(0), sizeY(0), sizeZ(0){

    }
    cpuArray3D(float* data, int sizeX, int sizeY, int sizeZ):data(data), sizeX(sizeX), sizeY(sizeY), sizeZ(sizeZ){

    }
            
    void setSize(unsigned int sX, unsigned int sY, unsigned int sZ){
        sizeX = sX;
        sizeY = sY;
        sizeZ = sZ;
    }
    
    float& operator()(unsigned int i, unsigned int j, unsigned int k){
        return data[i + (j + k * sizeY) * sizeX];
    }

    
    float* data;

    int sizeX;
    int sizeY;
    int sizeZ;
};

/*-----------------------------------------------------------------------
  Gradient function of the grids (old version with zero gradient boundray)
 -----------------------------------------------------------------------*/
void gradient_cpu3D(float * h_gdx, float * h_gdy, float * h_gdz,
                    float * h_src, int w, int h, int l);


/*-----------------------------------------------------------------------
  Gradient function of the grids (full version - SHOULD use this instead)
  Inputs : d_in scalar fields of the grids
           sizeX, sizeY, sizeZ    : size of the grid
           spX, spY, spZ          : spacing units (normaly use spX = spY = spZ = 1)
           wrap                   : boundary condition (cyclic or not)
             0: use central difference on the inside, one side difference for the boundary 
             1: use central difference every where + cyclic boundary
           When do not know use wrap = 0;
  Ouput  : h_gradX gradient on x direction 
           h_gradY gradient on y direction
           h_gradZ gradient on y direction
-----------------------------------------------------------------------*/

void cpuComputeGradient(float* h_gradX, float* h_gradY, float* h_gradZ,
                        float* h_in,
                        int sizeX, int sizeY, int sizeZ,
                        float spX=1.0f, float spY=1.0f, float spZ=1.0f,
                        bool wrap=false);
////////////////////////////////////////////////////////////////////////////////
//  I(t) = I(x+ v(x))
//  Zero boundary condition that mean I(x) = 0 with the point on the boundary
////////////////////////////////////////////////////////////////////////////////

void cpuBackwardMapping3D(float * h_o,
                          float * h_i,
                          float * vx, float * vy, float* vz,
                          int w, int h, int l);

//////////////////////////////////////////////////////////////////////////////////
// Set the boundary of an image with Zero
//////////////////////////////////////////////////////////////////////////////////
void cpuMakeZeroVolumeBoundary(float* data, const int data_w, const int data_h, const int data_l);

//////////////////////////////////////////////////////////////////////////////////
// Generate random image with zero boundary
//  b : is the size of the boundary 
//////////////////////////////////////////////////////////////////////////////////

void genRandomImageWithZeroBound(float* img, int w, int h, int l, int b);

//////////////////////////////////////////////////////////////////////////////////
// Generate random vector field with zero boundary
//  b is the size of the boundary
//////////////////////////////////////////////////////////////////////////////////
void genRandomVelocityWithZeroBound(float* vec, int w, int h, int l, int b);


#endif
