/****************************************************************************
 ** File: Gaussianfilter2D.cxx                                             **
 ** Description: 2D gaussian blur                                          **
 ** Author: Matthieu Jomier <mat-dev@jomier.com>                           **
 ** Version: 1.0                                                           **
 **                                                                        **
 ** ---------------------------------------------------------------------- **
 ** History: 09/24/2003            | Main application v1.0                 **
 ** Modif:   06/11/2004 dprigent                                           **
 **          06/25/2008 JDH: Converted to 2D                               **
 ****************************************************************************/

#include <assert.h>

#include "GaussianFilter2D.h"
//#include "Array2DIO.h"
#include <iostream>

#include <memory.h>
#include <stdlib.h>

using std::cerr;


GaussianFilter2D::GaussianFilter2D()
{
  unsigned int dim;
  for(dim = 0; dim < 2; dim++) {
    sigma[dim] = 1;
    kernelSize[dim] = 1;
  }
  DownsizeFactorX = 1;
  DownsizeFactorY = 1;
  //  DownsizeFactorZ = 1;
}


GaussianFilter2D::~GaussianFilter2D()
{
}


void GaussianFilter2D::SetInput(const Array2D<float>& _inarray)
{
  inarray = &_inarray;
}

Array2D<float>& GaussianFilter2D::GetOutput()
{
  return outarray;
}

void GaussianFilter2D::Update()
{
  filter_symmetric_full(*inarray,outarray,sigma,kernelSize);
}

void GaussianFilter2D::setSigma(float s1,float s2)
{
  sigma[0] = s1;
  sigma[1] = s2;
}

void GaussianFilter2D::setFactor(int _factorX, int _factorY)
{
  DownsizeFactorX = _factorX;
  DownsizeFactorY = _factorY;
  //  DownsizeFactorZ = _factorZ;
}

void GaussianFilter2D::setKernelSize(int s1,int s2)
{
  kernelSize[0] = s1;//kernel size
  kernelSize[1] = s2;
  //  kernelSize[2] = s3;
}


/******************************************************************************
 *
 * generate a matrix of (1D) gauss filters
 *
 * - only half the filter is calculated (symmetry)
 * - normalized so that the sum of the elements is 1.0
 * - filter[0] is the normal gauss filter
 * - filter[i] is the filter for filtering the elements in a distance (size-i)
 *   from the edge of the image
 * - returns an array[size+1] of Vectors of the length size+1
 * 
 *****************************************************************************/
float** GaussianFilter2D::
filter_generate_gauss_full( float  sigma, /* sigma of gauss function */
                            int    size)  /* half size of filter */
{
  float**     filter;
  float       exp_scale,scale;
  int         i,j;
   
  filter = (float**) malloc((size+1)*sizeof(float*));
  if (!filter)
  {
    cerr << "Could not allocate filter" << std::endl;
    return 0;
  }
        
  filter[0] = (float *)malloc((size+1)*sizeof(float));
  exp_scale = 1.0/(2.0 * sigma * sigma);
  scale= filter[0][0] = 1.0;
  for (i = 1; i <= size; i++)
  {
    filter[0][i] = exp(-(float)(i*i) * exp_scale);
    scale += 2.0*filter[0][i];//symmetry
  }

  scale = 1.0/scale;
  for (i=0; i<= size; i++)
  {
    /* rescale filter */
    filter[0][i] *= scale;
  }

  /* calculate filters for edge */
  scale = 0.0;
  for (j=1; j <= size; j++)
  {
    filter[j] = (float *)malloc((size+1)*sizeof(float));
    /* sum of "missing" elements */
    scale += filter[0][size+1-j];
    for(i=0;i<=size;i++)
    {
      filter[j][i] = filter[0][i]/(1.-scale);
    }
  }
#if 0
  //output debug 
  std::cout<<"filter with a kernel size = "<<size<<std::endl;
  for (int x = 0 ; x <=size ; x++){
    for (int y = 0 ; y <=size ; y++){
      std::cout<<""<<filter[x][y];
    }
    std::cout<<""<<std::endl;
  }
  //output debug
#endif
  return filter;
}




/******************************************************************************
 *
 * filter input field in 1-dim with symmetric filter along one
 * (hyper-)plane
 *
 * similar to filter_symmetric, but does also filter the edges
 * the filter must be a 2-dim array, the fist row contains the normal filter
 * the i-th row contains the filter for element in the distance size-i
 * from the edge 
 * 
 *****************************************************************************/
void GaussianFilter2D::
filter_symmetric_full(const Array2D<float> in, /* scalar field */
                      Array2D<float>& out, /* filtered field */
                      float sigma[2], /* sigma of gauss filter */
                      int kernelSize[2])
{
  register float*       f;      
  register int x,y,i;

  float** filter;
  Array2D<float> buf;

  buf.resize(in.getSize());
  buf.fill(0);

  //  buf2.resize(in.getSize());
  //buf2.fill(0);
   
  out.resize(in.getSize());
  out.fill(0);
        
  int inSizeX =  in.getSizeX();
  int inSizeY =  in.getSizeY();
  //  int inSizeZ =  in.getSizeZ();

  // If the size of the kernel is larger than half the dimension of
  // the image, minus 1, a memory error results.  Foskey 5/5/04.
  Array2D<float>::SizeType inSize = in.getSize();
  for(i = 0; i < 2; i++) {
    assert(kernelSize[i] >= 0);
    if ((unsigned int) kernelSize[i] > inSize[i] / 2 - 1) 
    {
      kernelSize[i] = inSize[i] / 2 - 1;
    }
  }

  /* do x-direction */
  
  filter = filter_generate_gauss_full(sigma[0],kernelSize[0]);
  //  for (z = 0; z < inSizeZ; z++)
    for (y = 0; y < inSizeY; y++)
    {   
      f = filter[0];
      
      //based on the downsample factor, we only compute the voxels which are going to be picked
      register int startX;
      (((kernelSize[0]%DownsizeFactorX) == 0) ? startX = kernelSize[0] : startX = (kernelSize[0]+1));

      for ( x = startX; x < inSizeX-kernelSize[0]; x = x+DownsizeFactorX)
      {
        out(x,y) = in(x,y)*f[0];
        for (i=1; i<= kernelSize[0]; i++)
        {
          out(x,y) += (in(x-i,y) + in(x+i,y)) * f[i];
        }       
      }
                  
                  
      /* filtering of edge elements */
      for (x=kernelSize[0]-1;x>=0;x--)
      {
        
        f = filter[kernelSize[0]-x];
        /* left edge */
        // Use mirroring to take care of edge artifact       
        //sysmetric part of the filter that overlap the image
        
        //based on the downsample factor, we only compute the voxels which are going to be picked
        if ((x%DownsizeFactorX) == 0)
        {
          out(x,y) = in(x,y)*f[0];    /* central point */
          for (i=1; i<= x; i++)
          { 
            out(x,y) += (in(x-i,y) + in(x+i,y)) * f[i];
          }
          //right tail the kernel which doesn't the corresponding on th left size (off the image)                          
          for(i=x+1;i<=kernelSize[0];i++){
            
            out(x,y) += in(x+i,y)*f[i];   
          }
        }
                                  
        /* right edge */
        // Use mirroring to take care of edge artifact 
        // index = row_index + (n_el-1-el)*inc_el;
        
        //based on the downsample factor, we only compute the voxels which are going to be picked
        if (((inSizeX-1-x)%DownsizeFactorX) == 0)
        {    
          out(inSizeX-1-x,y) = in(inSizeX-1-x,y)*f[0];  /* central point */
          for (i=1; i<= x; i++)
          {
            out(inSizeX-1-x,y) += (in(inSizeX-1-x+i,y) + 
                                     in(inSizeX-1-x-i,y)) * f[i];
          }
          
          for(i=x+1;i<=kernelSize[0];i++)
          {
            out(inSizeX-1-x,y) += in(inSizeX-1-x-i,y)*f[i];
          }
          
        }
        
      }
    }


  for (i=0;i<=kernelSize[0];i++) {
    free(filter[i]);
  }
  free(filter);


  /* do y-direction */
  filter = filter_generate_gauss_full(sigma[1],kernelSize[1]);
  //for (z = 0; z < inSizeZ; z++)
    for (x = 0; x < inSizeX; x=x+DownsizeFactorX)
    {   
      f = filter[0];   
      
      register int startY;
      (((kernelSize[1]%DownsizeFactorY) == 0) ? startY = kernelSize[1] : startY = (kernelSize[1]+1));

      for (y = startY; y < inSizeY-kernelSize[1]; y = y + DownsizeFactorY)
      { 
        buf(x,y) = out(x,y)*f[0];
        for (i=1; i <= kernelSize[1]; i++)
        {
          buf(x,y) += (out(x,y-i) + out(x,y+i)) * f[i];
        } 
      }
                  
                  
      /* filtering of edge elements */
      for (y=kernelSize[1]-1;y>=0;y--)
      {
        
        f = filter[kernelSize[1]-y];
        /* left edge */
        if ((y%DownsizeFactorY) == 0)
        {
          buf(x,y) = out(x,y)*f[0];   /* central point */
          for (i=1; i<= y; i++)
          {
            buf(x,y) += (out(x,y-i) + out(x,y+i)) * f[i];
          }
          
          for(i=y+1;i<=kernelSize[1];i++){
            buf(x,y) += out(x,y+i)*f[i];
          }  
        }
        
        /* right edge */
        // index = row_index + (n_el-1-el)*inc_el;
        if (((inSizeY-1-y)%DownsizeFactorY) == 0)
        {
          buf(x,inSizeY-1-y) = out(x,inSizeY-1-y)*f[0];  /* central point */
          
          for (i=1; i<= y; i++)
          {
            buf(x,inSizeY-1-y) += (out(x,inSizeY-1-y-i) + 
                                     out(x,inSizeY-1-y+i)) * f[i];
          }
           
          for(i=y+1;i<=kernelSize[1];i++)
          {
            buf(x,inSizeY-1-y) += out(x,inSizeY-1-y-i)*f[i];
          } 
        }  
      } 
    }

  for (i=0;i<=kernelSize[1];i++) {
    free(filter[i]);
  }
  free(filter);

//   /* do z-direction */
//   filter = filter_generate_gauss_full(sigma[2],kernelSize[2]);
//   for (x = 0; x < inSizeX; x=x+DownsizeFactorX)
//     for (y = 0; y < inSizeY; y = y+DownsizeFactorY)
//     {   
//       f = filter[0];

//       register int startZ;
//       (((kernelSize[2]%DownsizeFactorZ) == 0) ? startZ = kernelSize[2] : startZ = (kernelSize[2]+1));

//       for (z = startZ; z < inSizeZ-kernelSize[2]; z= z + DownsizeFactorZ)
//       { 
//         buf2(x,y,z) = buf(x,y,z)*f[0];
//         for (i=1; i <= kernelSize[2]; i++)
//         {
//           buf2(x,y,z) += (buf(x,y,z-i) + buf(x,y,z+i)) * f[i];
//         }
//       }
                  
                  
//       /* filtering of edge elements */
//       for (z=kernelSize[2]-1;z>=0;z--)
//       {
//         f = filter[kernelSize[2]-z];
//         /* left edge */
//         if ((z%DownsizeFactorZ) == 0)
//         {
//           buf2(x,y,z) = buf(x,y,z)*f[0];   /* central point */
//           for (i=1; i<= z; i++)
//           {
//             buf2(x,y,z) += (buf(x,y,z-i) + buf(x,y,z+i)) * f[i];
//           }
          
//           for(i=z+1;i<=kernelSize[2];i++){
//             buf2(x,y,z) += buf(x,y,z+i)*f[i];
//           }    
//         }
        
//         /* right edge */
//         // index = row_index + (n_el-1-el)*inc_el;
//         if (((inSizeZ-1-z)%DownsizeFactorZ) == 0)
//         {
//           buf2(x,y,inSizeZ-1-z) = buf(x,y,inSizeZ-1-z)*f[0];   /* central point */
          
//           for (i=1; i<= z; i++)
//           {
//             buf2(x,y,inSizeZ-1-z) += (buf(x,y,inSizeZ-1-z-i) +
//                                       buf(x,y,inSizeZ-1-z+i)) * f[i]; 
//           }
          
//           for(i=z+1;i<=kernelSize[2];i++)
//           {
//             buf2(x,y,inSizeZ-1-z) += buf(x,y,inSizeZ-1-z-i)*f[i];
//           }  
//         }
//       }
      
//     }
//   for (i=0;i<=kernelSize[2];i++) {
//     free(filter[i]);
//   }
//   free(filter);
 
  out.setData(buf);

}


