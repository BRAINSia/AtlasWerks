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

#include <stdio.h>
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "Image3D.hxx"

// Gets linear interpolated value at x,y,z
Real Image3D::getVal(Real x, Real y, Real z) const
{
  Real val = 0.0;

  Real d000, d001, d010, d011, d100, d101, d110, d111;
  Real d00x, d01x, d10x, d11x, d0yx, d1yx;

  Real voxelX = x / deltaX;
  Real voxelY = y / deltaY;
  Real voxelZ = z / deltaZ;

  unsigned int xInt = static_cast<unsigned int>(voxelX + xDim) % xDim;
  unsigned int yInt = static_cast<unsigned int>(voxelY + yDim) % yDim;
  unsigned int zInt = static_cast<unsigned int>(voxelZ + zDim) % zDim;

  unsigned int xIntP = (xInt + 1) % xDim;
  unsigned int yIntP = (yInt + 1) % yDim;
  unsigned int zIntP = (zInt + 1) % zDim;

  Real xFrac = voxelX - static_cast<Real>(xInt);
  Real yFrac = voxelY - static_cast<Real>(yInt);
  Real zFrac = voxelZ - static_cast<Real>(zInt);

  if(xInt >= 0 && xInt < xDim && yInt >= 0 && yInt < yDim &&
     zInt >= 0 && zInt < zDim)
  {
    d000 = data[xDim * (yDim * zInt + yInt) + xInt];
    d001 = data[xDim * (yDim * zInt + yInt) + xIntP];
    d010 = data[xDim * (yDim * zInt + yIntP) + xInt];
    d011 = data[xDim * (yDim * zInt + yIntP) + xIntP];

    d100 = data[xDim * (yDim * zIntP + yInt) + xInt];
    d101 = data[xDim * (yDim * zIntP + yInt) + xIntP];
    d110 = data[xDim * (yDim * zIntP + yIntP) + xInt];
    d111 = data[xDim * (yDim * zIntP + yIntP) + xIntP];

    d00x = d000 + xFrac * (d001 - d000);
    d01x = d010 + xFrac * (d011 - d010);
    d10x = d100 + xFrac * (d101 - d100);
    d11x = d110 + xFrac * (d111 - d110);
    d0yx = d00x + yFrac * (d01x - d00x);
    d1yx = d10x + yFrac * (d11x - d10x);

    val = d0yx + zFrac * (d1yx - d0yx);
  }

  return val;
}

void copyImage(Image3D & result, const Image3D & im)
{
  unsigned int size = result.xDim * result.yDim * result.zDim;
  unsigned int i;

  for(i = 0; i < size; i++)
    result.data[i] = im.data[i];
}

// Returns result = result + im
void addImage(Image3D & result, const Image3D & im)
{
  unsigned int size = result.xDim * result.yDim * result.zDim;
  unsigned int i;

  for(i = 0; i < size; i++)
    result.data[i] += im.data[i];
}

// Returns result = result + a
void addImage(Image3D & result, Real a)
{
  unsigned int size = result.xDim * result.yDim * result.zDim;
  unsigned int i;

  for(i = 0; i < size; i++)
    result.data[i] += a;
}

// Returns result = result - im
void subImage(Image3D & result, const Image3D & im)
{
  unsigned int size = result.xDim * result.yDim * result.zDim;
  unsigned int i;

  for(i = 0; i < size; i++)
    result.data[i] -= im.data[i];
}

// Returns im = a * im
void multImage(Image3D & im, Real a)
{
  unsigned int size = im.xDim * im.yDim * im.zDim;
  unsigned int i;

  for(i = 0; i < size; i++)
    im.data[i] *= a;
}

// Returns im1 = im1 * im2
void multImage(Image3D & im1, Image3D & im2)
{
  unsigned int size = im1.xDim * im1.yDim * im1.zDim;
  unsigned int i;

  for(i = 0; i < size; i++)
    im1.data[i] *= im2.data[i];
}

Real l2normSqr(const Image3D & im)
{
  unsigned int i, size = im.xDim * im.yDim * im.zDim;
  Real norm = 0.0;
  Real fact = im.deltaX * im.deltaY * im.deltaZ;

  for(i = 0; i < size; i++)
    norm += im.data[i] * im.data[i] * fact;

  return norm;
}

Real l2DotProd(const Image3D & im1, const Image3D & im2)
{
  unsigned int i, size = im1.xDim * im1.yDim * im1.zDim;
  Real dotProd = 0.0;
  Real fact = im1.deltaX * im1.deltaY * im1.deltaZ;

  for(i = 0; i < size; i++)
    dotProd += im1.data[i] * im2.data[i] * fact;

  return dotProd;
}

Image3D * downsample(Image3D * image, unsigned int factor)
{
  unsigned int xDim = image->xDim;
  unsigned int yDim = image->yDim;
  unsigned int zDim = image->zDim;

  unsigned int newXDim = xDim / factor;
  unsigned int newYDim = yDim / factor;
  unsigned int newZDim = zDim / factor;

  unsigned int i, j, k;

  unsigned int size = xDim * yDim * zDim;
  unsigned int newSize = size / (factor * factor * factor);

  Image3D * newImage = new Image3D(newXDim, newYDim, newZDim);
  newImage->deltaX = image->deltaX * factor;
  newImage->deltaY = image->deltaY * factor;
  newImage->deltaZ = image->deltaZ * factor;
  Real * newData = newImage->data;

  for(i = 0; i < newSize; i++)
    newData[i] = 0.0;

  unsigned int index = 0;
  unsigned int newIndex;
  Real divFactor = static_cast<Real>(factor * factor * factor);
  for(k = 0; k < zDim; k++)
  {
    for(j = 0; j < yDim; j++)
    {
      for(i = 0; i < xDim; i++)
      {
        newIndex = i / factor + newXDim * ((j / factor) + newYDim * (k / factor));
        newData[newIndex] += image->data[index] / divFactor;

        index++;
      }
    }
  }

  return newImage;
}

Image3D * upsample(Image3D * image, unsigned int factor)
{
  unsigned int xDim = image->xDim;
  unsigned int yDim = image->yDim;
  unsigned int zDim = image->zDim;

  unsigned int newXDim = xDim * factor;
  unsigned int newYDim = yDim * factor;
  unsigned int newZDim = zDim * factor;

  unsigned int i, j, k;

  Image3D * newImage = new Image3D(newXDim, newYDim, newZDim);
  newImage->deltaX = image->deltaX / factor;
  newImage->deltaY = image->deltaY / factor;
  newImage->deltaZ = image->deltaZ / factor;
  Real * newData = newImage->data;
  Real * data = image->data;

  Real xFrac, yFrac, zFrac;
  Real x, y, z;
  Real d000, d001, d010, d011, d100, d101, d110, d111;
  Real d00x, d01x, d10x, d11x, d0yx, d1yx;
  unsigned int xInt, yInt, zInt, xIntP, yIntP, zIntP;
  unsigned int index = 0;
  for(k = 0; k < newZDim; k++)
  {
    // Add zDim to z coordinate to make sure it is positive (it's removed by mod later)
    z = (static_cast<Real>(k) - 0.5) / static_cast<Real>(factor) + static_cast<Real>(zDim);
    zInt = static_cast<unsigned int>(z);
    zFrac = z  - static_cast<Real>(zInt);
    zInt = zInt % zDim;
    //zFrac = static_cast<Real>(k % factor) / static_cast<Real>(factor);
    for(j = 0; j < newYDim; j++)
    {
      y = (static_cast<Real>(j) - 0.5) / static_cast<Real>(factor) + static_cast<Real>(yDim);
      yInt = static_cast<unsigned int>(y);
      yFrac = y  - static_cast<Real>(yInt);
      yInt = yInt % yDim;
      //yFrac = static_cast<Real>(j % factor) / static_cast<Real>(factor);
      for(i = 0; i < newXDim; i++)
      {
        x = (static_cast<Real>(i) - 0.5) / static_cast<Real>(factor) + static_cast<Real>(xDim);
        xInt = static_cast<unsigned int>(x);
        xFrac = x  - static_cast<Real>(xInt);
        xInt = xInt % xDim;
        //xFrac = static_cast<Real>(i % factor) / static_cast<Real>(factor);

        //xInt = (i / factor + xDim) % xDim;
        //yInt = (j / factor + yDim) % yDim;
        //zInt = (k / factor + zDim) % zDim;

        xIntP = (xInt + 1) % xDim;
        yIntP = (yInt + 1) % yDim;
        zIntP = (zInt + 1) % zDim;

        d000 = data[xDim * (yDim * zInt + yInt) + xInt];
        d001 = data[xDim * (yDim * zInt + yInt) + xIntP];
        d010 = data[xDim * (yDim * zInt + yIntP) + xInt];
        d011 = data[xDim * (yDim * zInt + yIntP) + xIntP];
        
        d100 = data[xDim * (yDim * zIntP + yInt) + xInt];
        d101 = data[xDim * (yDim * zIntP + yInt) + xIntP];
        d110 = data[xDim * (yDim * zIntP + yIntP) + xInt];
        d111 = data[xDim * (yDim * zIntP + yIntP) + xIntP];

        d00x = d000 + xFrac * (d001 - d000);
        d01x = d010 + xFrac * (d011 - d010);
        d10x = d100 + xFrac * (d101 - d100);
        d11x = d110 + xFrac * (d111 - d110);
        d0yx = d00x + yFrac * (d01x - d00x);
        d1yx = d10x + yFrac * (d11x - d10x);

        newData[index] = d0yx + zFrac * (d1yx - d0yx);
        index++;
      }
    }
  }

  return newImage;
}
//This only works for even dimensional image say 32x64x8 and not 35x64x5 etc
Image3D * upsampleSinc(Image3D * image, unsigned int factor)
{
  unsigned int xDim = image->xDim;
  unsigned int yDim = image->yDim;
  unsigned int zDim = image->zDim;

  unsigned int newXDim = xDim * factor;
  unsigned int newYDim = yDim * factor;
  unsigned int newZDim = zDim * factor;

  Image3D * newImage = new Image3D(newXDim, newYDim, newZDim);

  newImage->deltaX = image->deltaX / factor;
  newImage->deltaY = image->deltaY / factor;
  newImage->deltaZ = image->deltaZ / factor;

  Real * scratch, * newScratch,* newImageComplexData,* oldImageComplexData;
  unsigned int sizeComplexOld = (xDim+1) * (yDim+1) * (zDim+1) * 2;
  unsigned int sizeComplexNew = (newXDim+1) * (newYDim+1) * (newZDim+1) * 2;
  scratch = new Real[sizeComplexOld];
  newScratch = new Real[sizeComplexNew];
  oldImageComplexData = new Real[sizeComplexOld];
  newImageComplexData = new Real[sizeComplexNew];
  fftwf_plan fftwForwardPlan;
  fftwf_plan fftwBackwardPlan;
  fftwf_plan_with_nthreads(2);

  unsigned int index, INDEX;  
  unsigned int x, y, z;
  int X, Y, Z;
  index = 0;INDEX = 0;
  for ( z = 0; z < zDim+1; ++z)
  {
    for ( y = 0; y < yDim+1; ++y)
    {
      for ( x = 0; x < xDim+1; ++x)
      {
	index = (x+((xDim+1) * (y + (yDim+1) * z)));

	if(x==0||y==0||z==0)
	{
  	  oldImageComplexData[2*index] = static_cast<Real>(0); 
	}
	else
	{
  	  INDEX = (x-1+(xDim * (y-1 + yDim * (z-1))));
  	  oldImageComplexData[2*index] = image->data[INDEX]; 
	}
	  oldImageComplexData[2*index+1] = static_cast<Real>(0);
      }
    }
  }  

  int dims[3];
  dims[0] = zDim+1;
  dims[1] = yDim+1;
  dims[2] = xDim+1;
  int newDims[3];
  newDims[0] = newZDim+1;
  newDims[1] = newYDim+1;
  newDims[2] = newXDim+1;
  fftwForwardPlan = fftwf_plan_dft(3,dims ,(fftwf_complex *)oldImageComplexData, (fftwf_complex *)(scratch),-1, FFTW_ESTIMATE);
  fftwBackwardPlan = fftwf_plan_dft(3,newDims, (fftwf_complex *)(newScratch), (fftwf_complex *)newImageComplexData,+1, FFTW_ESTIMATE);
  std::cout << "Plans of interpolation for upsampling image done" << std::endl;

  fftwf_execute(fftwForwardPlan);

  //Pad the frequency response with zeros
  index = 0;INDEX = 0;
  unsigned int lowerX, upperX, lowerY, upperY, lowerZ, upperZ;
  lowerX = static_cast<unsigned int>(xDim/2+1); upperX = static_cast<unsigned int>((factor-0.5)*xDim);
  lowerY = static_cast<unsigned int>(yDim/2+1); upperY = static_cast<unsigned int>((factor-0.5)*yDim);
  lowerZ = static_cast<unsigned int>(zDim/2+1); upperZ = static_cast<unsigned int>((factor-0.5)*zDim);

  for ( z = 0; z < newZDim+1; ++z)
  {
    for ( y = 0; y < newYDim+1; ++y)
	{
	  for ( x = 0; x < newXDim+1; ++x)
	  {
		index = 2*(x+((newXDim+1) * (y + (newYDim+1) * z)));
		
		if(x < lowerX && y<lowerY && z< lowerZ)
		{
			X = x; Y = y; Z = z;
			INDEX = 2*(X+((xDim+1) * (Y + (yDim+1) * Z)));
			newScratch[index] = scratch[INDEX];
			newScratch[index+1] = scratch[INDEX+1];

		}			
		else if(x > upperX && y>upperY && z> upperZ)
		{
			X = x-(factor-1)*xDim; Y = y-(factor-1)*yDim; Z = z-(factor-1)*zDim;
			INDEX = 2*(X+((xDim+1) * (Y + (yDim+1) * Z)));
			newScratch[index] = scratch[INDEX];
			newScratch[index+1] = scratch[INDEX+1];
		}
		else if(x < lowerX && y<lowerY && z> upperZ)
		{
			X = x; Y = y; Z = z-(factor-1)*zDim;
			INDEX = 2*(X+((xDim+1) * (Y + (yDim+1) * Z)));
			newScratch[index] = scratch[INDEX];
			newScratch[index+1] = scratch[INDEX+1];
		}
		else if(x < lowerX && y>upperY && z< lowerZ)
		{
			X = x; Y = y-(factor-1)*yDim; Z = z;
			INDEX = 2*(X+((xDim+1) * (Y + (yDim+1) * Z)));
			newScratch[index] = scratch[INDEX];
			newScratch[index+1] = scratch[INDEX+1];
		}
		else if(x > upperX && y<lowerY && z< lowerZ)
		{
			X = x-(factor-1)*xDim; Y = y; Z = z;
			INDEX = 2*(X+((xDim+1) * (Y + (yDim+1) * Z)));
			newScratch[index] = scratch[INDEX];
			newScratch[index+1] = scratch[INDEX+1];
 		}
		else if(x < lowerX && y>upperY && z> upperZ)
		{
			X = x; Y = y-(factor-1)*yDim; Z = z-(factor-1)*zDim;
			INDEX = 2*(X+((xDim+1) * (Y + (yDim+1) * Z)));
			newScratch[index] = scratch[INDEX];
			newScratch[index+1] = scratch[INDEX+1];
		}
		else if(x > upperX && y<lowerY && z> upperZ)
		{
			X = x-(factor-1)*xDim; Y = y; Z = z-(factor-1)*zDim;
			INDEX = 2*(X+((xDim+1) * (Y + (yDim+1) * Z)));
			newScratch[index] = scratch[INDEX];
			newScratch[index+1] = scratch[INDEX+1];
		}
		else if(x > upperX && y>upperY && z< lowerZ)
		{
			X = x-(factor-1)*xDim; Y = y-(factor-1)*yDim; Z = z;
			INDEX = 2*(X+((xDim+1) * (Y + (yDim+1) * Z)));
			newScratch[index] = scratch[INDEX];
			newScratch[index+1] = scratch[INDEX+1];
		}		
		else
		{
			newScratch[index] = static_cast<Real>(0);
			newScratch[index+1] = static_cast<Real>(0);
		}

	  }
	}
  }
  
  
  fftwf_execute(fftwBackwardPlan);

  for ( z = 0; z < newZDim+1; ++z)
  {
    for ( y = 0; y < newYDim+1; ++y)
    {
      for ( x = 0; x < newXDim+1; ++x)
      {
	index = (x+((newXDim+1) * (y + (newYDim+1) * z)));
	if(!(x==0||y==0||z==0))
	{
	  INDEX = (x-1+(newXDim * (y-1 + newYDim * (z-1))));
	  //Ignoring the imaginary part as we know that the result would be real
  	  newImage->data[INDEX] = newImageComplexData[2*index];
	}
      }
    }
  }
  multImage(*newImage, 1/static_cast<Real>(xDim*yDim*zDim));    
  delete [] scratch;
  delete [] newScratch;
  delete [] oldImageComplexData;
  delete [] newImageComplexData;

  return newImage;
}

Image3D * gaussBlur(Image3D * image, unsigned int radius, Real sigma)
{
  unsigned int i, j, k, kk, index;

  unsigned int xDim = image->xDim;
  unsigned int yDim = image->yDim;
  unsigned int zDim = image->zDim;
  Real * data = image->data;

  Real dist, kernelSum = 0.0f;
  unsigned int kernelWidth = radius * 2 + 1;
  Real * kernel = new Real[kernelWidth];
  Image3D * result =
    new Image3D(xDim, yDim, zDim, image->deltaX, image->deltaY, image->deltaZ);

  Real * resultData = result->data;

  // Generate kernel
  for(i = 0; i < kernelWidth; i++)
  {
    dist = static_cast<Real>(i - radius);
    kernel[i] = exp(- dist * dist / (2.0 * sigma * sigma));
    kernelSum += kernel[i];
  }
  
  for(i = 0; i < kernelWidth; i++)
    kernel[i] /= kernelSum;

  // Convolve in x
  for(k = 0; k < zDim; k++)
  {
    for(j = 0; j < yDim; j++)
    {
      for(i = 0; i < xDim; i++)
      {
        index = xDim * (j + yDim * k);
        resultData[index + i] = kernel[0] * data[index + (i + radius) % xDim];
        for(kk = 1; kk < kernelWidth; kk++)
          resultData[index + i] += kernel[kk] *
            data[index + (-kk + radius + i + xDim) % xDim];
      }
    }
  }

  // Convolve in y
  for(k = 0; k < zDim; k++)
  {
    for(j = 0; j < yDim; j++)
    {
      for(i = 0; i < xDim; i++)
      {
        index = i + xDim * yDim * k;
        data[index + j * xDim] =
          kernel[0] * resultData[index + ((j + radius) % yDim) * xDim];
        for(kk = 1; kk < kernelWidth; kk++)
          data[index + j * xDim] += kernel[kk] *
            resultData[index + ((-kk + radius + j + yDim) % yDim) * xDim];
      }
    }
  }

  // Convolve in z
  for(k = 0; k < zDim; k++)
  {
    for(j = 0; j < yDim; j++)
    {
      for(i = 0; i < xDim; i++)
      {
        index = i + xDim * j;
        resultData[index + k * xDim * yDim] =
          kernel[0] * data[index + ((k + radius) % zDim) * xDim * yDim];
        for(kk = 1; kk < kernelWidth; kk++)
          resultData[index + k * xDim * yDim] += kernel[kk] *
            data[index + ((-kk + radius + k + zDim) % zDim) * xDim * yDim];
      }
    }
  }

  delete [] kernel;

  return result;
}

Image3D * readMetaImage(const char * filename)
{
  if(filename == NULL)
    return NULL;

  typedef itk::Image<Real, 3> ImageType;
  typedef itk::ImageFileReader<ImageType> ReaderType;
  ReaderType::Pointer reader = ReaderType::New();

  reader->SetFileName(filename);
  try
  {
    reader->Update();
  }
  catch(itk::ExceptionObject & e)
  {
    std::cerr << "Error reading image: " << e << std::endl;
    return NULL;
  }

  ImageType::Pointer itkImage = reader->GetOutput();
  ImageType::SizeType size = itkImage->GetLargestPossibleRegion().GetSize();
  Image3D * image = new Image3D(size[0], size[1], size[2]);

  unsigned int i, j, k, index = 0;
  for(k = 0; k < size[2]; k++)
  {
    for(j = 0; j < size[1]; j++)
    {
      for(i = 0; i < size[0]; i++)
      {
        ImageType::IndexType itkIndex;
        itkIndex[0] = i; itkIndex[1] = j; itkIndex[2] = k;
        image->data[index] = itkImage->GetPixel(itkIndex);
        index++;
      }
    }
  }

  return image;
}

void writeMetaImage(const char * filePrefix, Image3D * image)
{
  if(filePrefix == NULL)
    return;

  typedef itk::Image<Real, 3> ImageType;
  typedef itk::ImageFileWriter<ImageType> WriterType;
  WriterType::Pointer writer = WriterType::New();

  ImageType::Pointer itkImage = ImageType::New();
  ImageType::SizeType size;
  ImageType::RegionType region;
  ImageType::SpacingType spacing;
  size[0] = image->xDim; size[1] = image->yDim; size[2] = image->zDim;
  region.SetSize(size);
  itkImage->SetRegions(region);
  spacing[0] = image->deltaX;
  spacing[1] = image->deltaY;
  spacing[2] = image->deltaZ;
  itkImage->SetSpacing(spacing);
  itkImage->Allocate();

  unsigned int i, j, k, index = 0;
  for(k = 0; k < size[2]; k++)
  {
    for(j = 0; j < size[1]; j++)
    {
      for(i = 0; i < size[0]; i++)
      {
        ImageType::IndexType itkIndex;
        itkIndex[0] = i; itkIndex[1] = j; itkIndex[2] = k;
        itkImage->SetPixel(itkIndex, image->data[index]);
        //std::cout << image->data[index] << " " << itkImage->GetPixel(itkIndex) << std::endl;
        index++;
      }
    }
  }

  writer->SetFileName(filePrefix);
  writer->UseCompressionOn();
  writer->SetInput(itkImage);
  try
  {
    writer->Update();
  }
  catch(itk::ExceptionObject & e)
  {
    std::cerr << "Error writing image: " << e << std::endl;
    return;
  }
}
