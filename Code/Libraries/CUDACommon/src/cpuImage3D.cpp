#include <cpuImage3D.h>
#include <stdlib.h>

void gradient_cpu3D(float * h_gdx, float * h_gdy, float * h_gdz,
                    float * h_src, int w, int h, int l)
{
    int i, j, k, id =0;
    int wh = w * h;

    for(k = 0; k < l; k++)
        for(j = 0; j < h; j++)
            for(i = 0; i < w; i++, ++id){
                float x0yz, x1yz, xy0z, xy1z, xyz0, xyz1;

                xyz1 = ( k == l - 1) ? 0 : h_src[id + wh];
                xy1z = ( j == h - 1) ? 0 : h_src[id + w];
                x1yz = ( i == w - 1) ? 0 : h_src[id + 1];

                xyz0 = ( k == 0) ? 0 : h_src[id - wh];
                xy0z = ( j == 0) ? 0 : h_src[id - w];
                x0yz = ( i == 0) ? 0 : h_src[id - 1];

                h_gdx[id] = 0.5f * (x1yz - x0yz);
                h_gdy[id] = 0.5f * (xy1z - xy0z);
                h_gdz[id] = 0.5f * (xyz1 - xyz0);
            }
}

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
                        float spX, float spY, float spZ, bool wrap)
{
    cpuArray3D aIn(h_in, sizeX, sizeY, sizeZ);
    cpuArray3D gdX(h_gradX, sizeX, sizeY, sizeZ);
    cpuArray3D gdY(h_gradY, sizeX, sizeY, sizeZ);
    cpuArray3D gdZ(h_gradZ, sizeX, sizeY, sizeZ);
    
    unsigned int xIndex, yIndex, zIndex;
    // Process interior points
    for(zIndex = 0; zIndex < sizeZ; zIndex++) {
        for(yIndex = 0; yIndex < sizeY; yIndex++) {
            for(xIndex = 1; xIndex < sizeX - 1; xIndex++) {
                gdX(xIndex, yIndex, zIndex) =
                    (aIn(xIndex + 1, yIndex, zIndex) -
                     aIn(xIndex - 1, yIndex, zIndex)) / (2.0 * spX);
            }
        }
    }
    
    for(zIndex = 0; zIndex < sizeZ; zIndex++) {
        for(yIndex = 1; yIndex < sizeY - 1; yIndex++) {
            for(xIndex = 0; xIndex < sizeX; xIndex++) {
                gdY(xIndex, yIndex, zIndex) =
                    (aIn(xIndex, yIndex + 1, zIndex) - 
                     aIn(xIndex, yIndex - 1, zIndex)) / (2.0 * spY);
            }
        }
    }
    for(zIndex = 1; zIndex < sizeZ - 1; zIndex++) {
        for(yIndex = 0; yIndex < sizeY; yIndex++) {
            for(xIndex = 0; xIndex < sizeX; xIndex++) {
                gdZ(xIndex, yIndex, zIndex) =
                    (aIn(xIndex, yIndex, zIndex + 1) - 
                     aIn(xIndex, yIndex, zIndex - 1)) / (2.0 * spZ);
            }
        }
    }
    
    if(wrap){
        // Process the voxels on the x-dimensional edge
        for(zIndex = 0; zIndex < sizeZ; zIndex++) {
            for(yIndex = 0; yIndex < sizeY; yIndex++) {
                gdX(0, yIndex, zIndex) = 
                    (aIn(1, yIndex, zIndex) - 
                     aIn(sizeX - 1, yIndex, zIndex)) / (2.0*spX);
                gdX(sizeX - 1, yIndex, zIndex) = 
                    (aIn(0, yIndex, zIndex) - 
                     aIn(sizeX - 2, yIndex, zIndex)) / (2.0*spX);
            }
        }
      
        // Process the voxel on the y-dimensional edge
        for(zIndex = 0; zIndex < sizeZ; zIndex++) {
            for(xIndex = 0; xIndex < sizeX; xIndex++) {
                gdY(xIndex, 0, zIndex) = 
                    (aIn(xIndex, 1, zIndex) - 
                     aIn(xIndex, sizeY - 1, zIndex)) / (2.0*spY);
                gdY(xIndex, sizeY - 1, zIndex) = 
                    (aIn(xIndex, 0, zIndex) - 
                     aIn(xIndex, sizeY - 2, zIndex)) / (2.0*spY);
            }
        }
      
        // Process the voxel on the z-dimensional edge
        for(yIndex = 0; yIndex < sizeY; yIndex++) {
            for(xIndex = 0; xIndex < sizeX; xIndex++) {
                gdZ(xIndex, yIndex, 0) = 
                    (aIn(xIndex, yIndex, 1) - 
                     aIn(xIndex, yIndex, sizeZ - 1)) / (2.0*spZ);
                gdZ(xIndex, yIndex, sizeZ - 1) = 
                    (aIn(xIndex, yIndex, 0) - 
                     aIn(xIndex, yIndex, sizeZ - 2)) / (2.0*spZ);
            }
        }
    }else{
        // Process the voxels on the x-dimensional edge
        for(zIndex = 0; zIndex < sizeZ; zIndex++) {
            for(yIndex = 0; yIndex < sizeY; yIndex++) {
                gdX(0, yIndex, zIndex) = 
                    (aIn(1, yIndex, zIndex) - aIn(0, yIndex, zIndex)) / spX;
                gdX(sizeX - 1, yIndex, zIndex) =
                    (aIn(sizeX - 1, yIndex, zIndex) - aIn(sizeX - 2, yIndex, zIndex)) / spX;
            }
        }
        // Process the voxel on the y-dimensional edge
        for(zIndex = 0; zIndex < sizeZ; zIndex++) {
            for(xIndex = 0; xIndex < sizeX; xIndex++) {
                gdY(xIndex, 0, zIndex) = 
                    (aIn(xIndex, 1, zIndex) - aIn(xIndex, 0, zIndex)) / spY;
                gdY(xIndex, sizeY - 1, zIndex) = 
                    (aIn(xIndex, sizeY - 1, zIndex) - aIn(xIndex, sizeY - 2, zIndex)) / spY;
            }
        }
      
        // Process the voxel on the z-dimensional edge
        for(yIndex = 0; yIndex < sizeY; yIndex++) {
            for(xIndex = 0; xIndex < sizeX; xIndex++) {
                gdZ(xIndex, yIndex, 0) = 
                    (aIn(xIndex, yIndex, 1) - aIn(xIndex, yIndex, 0)) / spZ;
                gdZ(xIndex, yIndex, sizeZ - 1) = 
                    (aIn(xIndex, yIndex, sizeZ - 1) - aIn(xIndex, yIndex, sizeZ - 2)) / spZ;
            }
        }
    }// end if not wrap
}

////////////////////////////////////////////////////////////////////////////////
//  I(t) = I(x+ v(x))
//  Zero boundary condition that mean I(x) = 0 with the point on the boundary
////////////////////////////////////////////////////////////////////////////////

void cpuBackwardMapping3D(float * h_o,
                          float * h_i,
                          float * vx, float * vy, float* vz,
                          int w, int h, int l)
{
    int xInt, yInt, zInt, xIntP, yIntP, zIntP;
    float x, y, z;
    float dx, dy, dz;
    float dxy, oz;

    int i, j, k, index = 0;
    const int wh = w * h;
    for (k = 0; k < l; ++k)
        for(j = 0; j < h; ++j)
            for(i = 0; i < w; ++i, ++index){
                x = (float)(i) + vx[index];
                y = (float)(j) + vy[index];
                z = (float)(k) + vz[index];

                if (x >= 0 && x < w - 1 &&
                    y >= 0 && y < h - 1 &&
                    z >= 0 && z < l - 1){

                    xInt = (int)x;
                    yInt = (int)y;
                    zInt = (int)z;

                    xIntP = xInt + 1;
                    yIntP = yInt + 1;
                    zIntP = zInt + 1;

                    dx = x - xInt;
                    dy = y - yInt;
                    dz = z - zInt;

                    dxy = dx * dy;
                    oz = 1.f - dz;

                    float x0y0z0 = h_i[xInt  + yInt  * w + zInt  * wh];
                    float x1y0z0 = h_i[xIntP + yInt  * w + zInt  * wh];
                    float x0y1z0 = h_i[xInt  + yIntP * w + zInt  * wh];
                    float x1y1z0 = h_i[xIntP + yIntP * w + zInt  * wh];

                    float x0y0z1 = h_i[xInt  + yInt  * w + zIntP * wh];
                    float x1y0z1 = h_i[xIntP + yInt  * w + zIntP * wh];
                    float x0y1z1 = h_i[xInt  + yIntP * w + zIntP * wh];
                    float x1y1z1 = h_i[xIntP + yIntP * w + zIntP * wh];

                    h_o[index]=(x1y0z0* (dx - dxy) + x0y1z0 * (dy - dxy) + x1y1z0*dxy + x0y0z0 * (1-dy -(dx-dxy))) * oz +
                        (x1y0z1* (dx - dxy) + x0y1z1 * (dy - dxy) + x1y1z1*dxy + x0y0z1 * (1-dy -(dx-dxy))) * dz;
                }
                else
                    h_o[index]=0.0;
            }
}



//////////////////////////////////////////////////////////////////////////////////
// Set the boundary of an image with Zero
//////////////////////////////////////////////////////////////////////////////////

void cpuMakeZeroVolumeBoundary(float* data, const int data_w, const int data_h, const int data_l){
    const int planeSize = data_w * data_h;
    int i, j , k ;
    for (j = 0; j < data_h; ++j)
        for (i = 0; i < data_w; ++i){
            data[j * data_w + i] = 0.f;
            data[(data_l-1) * planeSize + j * data_w + i] = 0.f;
        }

    for (k = 0; k < data_l; ++k)
        for (i = 0; i < data_w; ++i){
            data[ k * planeSize + i] = 0.f;
            data[ k * planeSize + (data_h-1) * data_w + i] = 0.f;
        }
    
    for (k = 0; k < data_l; ++k)
        for (j = 0; j < data_h; ++j){
            data[ k * planeSize + j * data_w] = 0.f;
            data[ k * planeSize + j * data_w + data_w -1] = 0.f;
        }
}


//////////////////////////////////////////////////////////////////////////////////
// Generate random image with zero boundary
//  b : is the size of the boundary 
//////////////////////////////////////////////////////////////////////////////////

void genRandomImageWithZeroBound(float* img, int w, int h, int l, int b){
    int id = 0;
    for (int k=0; k< l; ++k) 
        for (int j=0; j< h; ++j) 
            for (int i=0; i< w; ++i, ++id){
                if ( (i>=b) && (i < w - b) &&
                     (j>=b) && (j < h - b) &&
                     (k>=b) && (k < l - b)){
                    img[id] = rand() % 256;
                }
                else
                    img[id] = 0;
            }
}

//////////////////////////////////////////////////////////////////////////////////
// Generate random vector field with zero boundary
//  b is the size of the boundary
//////////////////////////////////////////////////////////////////////////////////
void genRandomVelocityWithZeroBound(float* vec, int w, int h, int l, int b){
    int id = 0;
    for (int k=0; k< l; ++k) 
        for (int j=0; j< h; ++j) 
            for (int i=0; i< w; ++i, ++id){
                if ( (i>=b) && (i < w - b) &&
                     (j>=b) && (j < h - b) &&
                     (k>=b) && (k < l - b)){
                    vec[id] = (float(rand()) / RAND_MAX - 0.5f) * 2.f;
                }
                else
                    vec[id] = 0;
            }
}

