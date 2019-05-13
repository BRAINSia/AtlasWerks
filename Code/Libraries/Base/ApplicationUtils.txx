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

#ifndef ApplicationUtils_txx
#define ApplicationUtils_txx

#include "DataTypes/Image.h"
#include "Image2D.h"
#include "AtlasWerksException.h"
#include "StringUtils.h"

#include <itkImage.h>
#include <itkIndex.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkImportImageFilter.h>
#include <math.h>
#include <string.h>

template <class VoxelType>
void 
ApplicationUtils::
LoadImageITK(const char *fileName, Image<VoxelType>& image)
{
  //
  // load itk image
  //
  typedef itk::Image<VoxelType, 3>        ImageType;
  typedef typename ImageType::Pointer     ImagePointer;
  typedef itk::ImageFileReader<ImageType> VolumeReaderType;
  typename VolumeReaderType::Pointer reader = VolumeReaderType::New();
  reader->SetFileName(fileName);
  ImagePointer imagePtr = reader->GetOutput();

  try
  {
    reader->Update();
  }
  catch(itk::ImageFileReaderException &exc)
  {
    throw AtlasWerksException(__FILE__, __LINE__, "ITK File Reader Exception", &exc);
  }
  catch(itk::ExceptionObject &exc)
  {
    throw AtlasWerksException(__FILE__, __LINE__, "ITK Exception", &exc);
  }
  catch(...)
  {
    std::string errMsg = StringUtils::strPrintf("Unknown exception loading: %s", fileName);
    throw AtlasWerksException(__FILE__, __LINE__, errMsg);
  }

  //
  // copy data to Image
  //
  // instead we should just steal the pointer
  int sizeX = imagePtr->GetLargestPossibleRegion().GetSize()[0];
  int sizeY = imagePtr->GetLargestPossibleRegion().GetSize()[1];
  int sizeZ = imagePtr->GetLargestPossibleRegion().GetSize()[2];
  image.resize(sizeX, sizeY, sizeZ);
  memcpy(image.getDataPointer(),
         imagePtr->GetBufferPointer(),
         sizeX * sizeY * sizeZ * sizeof(VoxelType));

  //
  // copy origin and spacing
  //
  image.setOrigin(imagePtr->GetOrigin()[0],
                  imagePtr->GetOrigin()[1],
                  imagePtr->GetOrigin()[2]);

  image.setSpacing(imagePtr->GetSpacing()[0],
                   imagePtr->GetSpacing()[1],
                   imagePtr->GetSpacing()[2]);
}

template <class VoxelType>
void 
ApplicationUtils::
SaveImageITK(const char *fileName, const Image<VoxelType>& image)
{
  //
  // create an itk version of the image 
  //
  typedef itk::Image<VoxelType, 3>               ImageType;
  typedef itk::ImportImageFilter<VoxelType, 3>   ImportFilterType;
  typename ImportFilterType::Pointer importFilter = ImportFilterType::New();

  // construction image region (basically just the dimensions)
  typename ImportFilterType::IndexType start;
  start.Fill(0);
  typename ImportFilterType::SizeType size;
  size[0] = image.getSize()[0];
  size[1] = image.getSize()[1];
  size[2] = image.getSize()[2];

  typename ImportFilterType::RegionType region;
  region.SetSize(size);
  region.SetIndex(start);
  importFilter->SetRegion(region);

  // set origin and spacing 
  typename ImportFilterType::OriginType origin;
  origin[0] = image.getOrigin()[0];
  origin[1] = image.getOrigin()[1];
  origin[2] = image.getOrigin()[2];
  importFilter->SetOrigin(origin);

  typename ImportFilterType::SpacingType spacing;
  spacing[0] = image.getSpacing()[0];
  spacing[1] = image.getSpacing()[1];
  spacing[2] = image.getSpacing()[2];
  importFilter->SetSpacing(spacing);

  // copy the data into the itk image
  const bool importImageFilterWillOwnTheBuffer = false;
  importFilter->SetImportPointer((VoxelType*)image.getDataPointer(),
                                 image.getNumElements(),
                                 importImageFilterWillOwnTheBuffer);

  //
  // create writer
  //
  typedef itk::ImageFileWriter<ImageType> VolumeWriterType;
  typename VolumeWriterType::Pointer writer = VolumeWriterType::New();
  writer->SetFileName(fileName);
  writer->SetInput(importFilter->GetOutput());

  //
  // write image
  //
  try
  {
    writer->Update();
  }
  catch(itk::ImageFileWriterException &exc)
  {
    throw AtlasWerksException(__FILE__, __LINE__, "ITK File Writer Exception", &exc);
  }
  catch(itk::ExceptionObject &exc)
  {
    throw AtlasWerksException(__FILE__, __LINE__, "ITK Exception", &exc);
  }
  catch(...)
  {
    std::string errMsg = StringUtils::strPrintf("Unknown exception saving: %s", fileName);
    throw AtlasWerksException(__FILE__, __LINE__, errMsg);
  }
}
template <class VoxelType>
int
ApplicationUtils::
SaveImageSlice(const char *fileName, const Image<VoxelType>& image3d, const char* slicePlaneType, unsigned int sliceNumber)
{

  //
  // create an itk version of the image
  //
  typedef unsigned char VxType; 
  typedef itk::Image<VxType, 2>               ImageType;
  typedef itk::ImportImageFilter<VxType, 2>   ImportFilterType;
  typename ImportFilterType::Pointer importFilter = ImportFilterType::New();

  // set origin and spacing
  typename ImportFilterType::OriginType origin;
  typename ImportFilterType::SpacingType spacing;
  typename ImportFilterType::IndexType start;
  typename ImportFilterType::RegionType region;
  typename ImportFilterType::SizeType size;
  start.Fill(0);

  Image2D<VxType> image2d;

  if(!strcmp(slicePlaneType,"AXIAL") || !strcmp(slicePlaneType,"axial"))
  {
        // check if slice number is in the bounds
        if(!((sliceNumber>=0) && (sliceNumber<=image3d.getSize()[2])))
	{
                std::cout<<"Invalid Slice Number"<<std::endl;
		return EXIT_FAILURE;
	}

        size[0]=image3d.getSize()[0];
        size[1]=image3d.getSize()[1];

        origin[0] = image3d.getOrigin()[0];
        origin[1] = image3d.getOrigin()[1];

        spacing[0] = image3d.getSpacing()[0];
        spacing[1] = image3d.getSpacing()[1];

	image2d.resize(size[0],size[1]);
	VoxelType max=-1;
        for(unsigned int i=0;i<size[0];i++)
        {
                for(unsigned int j=0;j<size[1];j++)
                {
			//index = {i, j};
			if(image3d.get(i,j,sliceNumber) > max)
				max = image3d.get(i,j,sliceNumber);
                }
        }
        for(unsigned int i=0;i<size[0];i++)
        {
                for(unsigned int j=0;j<size[1];j++)
                {
                        image2d.set(i,j,(VxType)lround(image3d.get(i,j,sliceNumber) / ((max * 1.15)/255)));
                }
        }

  }
  else if(!strcmp(slicePlaneType,"CORONAL") || !strcmp(slicePlaneType,"coronal"))
  {
        // check if slice number is in the bounds
        if(!((sliceNumber>=0) && (sliceNumber<=image3d.getSize()[1])))
	{
                std::cout<<"Invalid Slice Number"<<std::endl;
		return EXIT_FAILURE;
		
	}

        size[0]=image3d.getSize()[0];
        size[1]=image3d.getSize()[2];

        origin[0] = image3d.getOrigin()[0];
        origin[1] = image3d.getOrigin()[2];

        spacing[0] = image3d.getSpacing()[0];
        spacing[1] = image3d.getSpacing()[2];


	image2d.resize(size[0],size[1]);

	VoxelType max=-1;
        for(unsigned int i=0;i<size[0];i++)
        {
                for(unsigned int j=0;j<size[1];j++)
                {
			//index = {i, j};
			if(image3d.get(i,sliceNumber,j) > max)
				max = image3d.get(i,sliceNumber,j);
                }
        }
        for(unsigned int i=0;i<size[0];i++)
        {
                for(unsigned int j=0;j<size[1];j++)
                {
                        image2d.set(i,j,(VxType)lround(image3d.get(i,sliceNumber,j) / ((max * 1.15)/255)));
                }
        }

  }
  else if(!strcmp(slicePlaneType,"SAGITTAL") || !strcmp(slicePlaneType,"sagittal"))
  {

        // check if slice number is in the bounds
        if(!((sliceNumber>=0) && (sliceNumber<=image3d.getSize()[0])))
	{
                std::cout<<"Invalid Slice Number"<<std::endl;
		return EXIT_FAILURE;
	}

        size[0]=image3d.getSize()[1];
        size[1]=image3d.getSize()[2];


        origin[0] = image3d.getOrigin()[1];
        origin[1] = image3d.getOrigin()[2];

        spacing[0] = image3d.getSpacing()[1];
        spacing[1] = image3d.getSpacing()[2];

	image2d.resize(size[0],size[1]);

	VoxelType max=-1;
        for(unsigned int i=0;i<size[0];i++)
        {
                for(unsigned int j=0;j<size[1];j++)
                {
			//index = {i, j};
			if(image3d.get(sliceNumber,i,j) > max)
				max = image3d.get(sliceNumber,i,j);
                }
        }
        for(unsigned int i=0;i<size[0];i++)
        {
                for(unsigned int j=0;j<size[1];j++)
                {
                        image2d.set(i,j,(VxType)lround(image3d.get(sliceNumber,i,j) / ((max * 1.15)/255)));
                }
        }
  }
  else
  {
	std::cout<<"Wrong slice plane to save........!!!!!!!\n"<<"Only these planes are expected: \"AXIAL\", \"CORONAL\", or \"SAGITTAL\" "<<std::endl;
        return EXIT_FAILURE;
  }


  region.SetSize(size);
  region.SetIndex(start);
  importFilter->SetRegion(region);

  // set origin and spacing
  importFilter->SetOrigin(origin);
  importFilter->SetSpacing(spacing);





  // copy the data into the itk image
  const bool importImageFilterWillOwnTheBuffer = false;
  importFilter->SetImportPointer((VxType*)image2d.getDataPointer(),
                                 image2d.getNumElements(),
                                 importImageFilterWillOwnTheBuffer);

  //
  // create writer
  //
  typedef itk::ImageFileWriter<ImageType> VolumeWriterType;
  typename VolumeWriterType::Pointer writer = VolumeWriterType::New();
  writer->SetFileName(fileName);
  writer->SetInput(importFilter->GetOutput());

  //
  // write image
  //
  try
  {
    writer->Update();
  }
  catch(itk::ImageFileWriterException &exc)
  {
    throw AtlasWerksException(__FILE__, __LINE__, "ITK File Writer Exception", &exc);
  }
  catch(itk::ExceptionObject &exc)
  {
    throw AtlasWerksException(__FILE__, __LINE__, "ITK Exception", &exc);
  }
  catch(...)
  {
    std::string errMsg = StringUtils::strPrintf("Unknown exception saving: %s", fileName);
    throw AtlasWerksException(__FILE__, __LINE__, errMsg);
  }

  return EXIT_SUCCESS;

}


template <class VoxelType>
void        
ApplicationUtils::
LoadHFieldITK(const char* fileNamePrefix,
	      const char* extension, 
	      Array3D<Vector3D<VoxelType> >& h)
{
  char fNameX[256], fNameY[256], fNameZ[256];
  sprintf(fNameX, "%s_x.%s", fileNamePrefix, extension);
  sprintf(fNameY, "%s_y.%s", fileNamePrefix, extension);
  sprintf(fNameZ, "%s_z.%s", fileNamePrefix, extension);
  LoadHFieldITK(fNameX, fNameY, fNameZ, h);
}

template <class VoxelType>
void        
ApplicationUtils::
LoadHFieldITK(const char* fileNameX,
	      const char* fileNameY, 
	      const char* fileNameZ, 
	      Array3D<Vector3D<VoxelType> >& h)
{
  const char *fname[3];
  fname[0] = fileNameX;
  fname[1] = fileNameY;
  fname[2] = fileNameZ;
  Image<VoxelType> dimImage;
  for(int dim=0;dim<3;dim++){
    LoadImageITK(fname[dim], dimImage);
    if(dim == 0) h.resize(dimImage.getSize());
    for (unsigned int z = 0; z < h.getSize().z; ++z) {
      for (unsigned int y = 0; y < h.getSize().y; ++y) {
        for (unsigned int x = 0; x < h.getSize().x; ++x) {
          h(x,y,z)[dim] = dimImage(x,y,z);
        }
      }
    }
  } // end loop over dims
}

template <class VoxelType>
void        
ApplicationUtils::
LoadHFieldITK(const char* fileName,
	      Array3D<Vector3D<VoxelType> >& h)
{
  OriginType dummyOrigin;
  SpacingType dummySpacing;
  LoadHFieldITK(fileName, dummyOrigin, dummySpacing, h);
}

template <class VoxelType>
void        
ApplicationUtils::
LoadHFieldITK(const char* fileName,
	      OriginType &hOrigin,
	      SpacingType &hSpacing,
	      Array3D<Vector3D<VoxelType> >& h)
{
  typedef itk::Vector<VoxelType, 3> VectorType;
  typedef itk::Image<VectorType, 3> VectorImageType;
  typedef itk::ImageFileReader< VectorImageType > ReaderType;
  typename ReaderType::Pointer reader = ReaderType::New();

  reader->SetFileName(fileName);
  try
  {
    reader->Update();
  }
  catch(itk::ImageFileReaderException &exc)
  {
    // TEST
    std::cout << "Testing exception output:" << exc << std::endl;
    AtlasWerksException ce(__FILE__, __LINE__, "ITK File Reader Exception", &exc);
    //    ce << "Testing adding extra output" << ", and then some more";
    throw ce;
    // END TEST
    //throw AtlasWerksException("ITK File Reader Exception", "ApplicationUtils::LoadHFieldITK", &exc);
  }
  catch(itk::ExceptionObject &exc)
  {
    throw AtlasWerksException(__FILE__, __LINE__, "ITK Exception", &exc);
  }
  catch(...)
  {
    std::string errMsg = StringUtils::strPrintf("Unknown exception loading: %s", fileName);
    throw AtlasWerksException(__FILE__, __LINE__, errMsg);
  }
  
  typename VectorImageType::Pointer itkImage = reader->GetOutput();
  typename VectorImageType::SizeType itkSize = itkImage->GetLargestPossibleRegion().GetSize();
  Vector3D<unsigned int> size(itkSize[0],itkSize[1],itkSize[2]);
  h.resize(size);
  
  unsigned int x, y, z;
  for(z = 0; z < size.z; z++){
    for(y = 0; y < size.y; y++){
      for(x = 0; x < size.x; x++){
	typename VectorImageType::IndexType itkIndex;
	itkIndex[0] = x; itkIndex[1] = y; itkIndex[2] = z;
	VectorType pixel = itkImage->GetPixel(itkIndex);
	h(x,y,z).set(pixel[0],pixel[1],pixel[2]);
      }
    }
  }

  typename VectorImageType::PointType itkOrigin;
  itkOrigin = itkImage->GetOrigin();
  hOrigin = OriginType(itkOrigin[0], itkOrigin[1], itkOrigin[2]);

  typename VectorImageType::SpacingType itkSpacing;
  itkSpacing = itkImage->GetSpacing();
  hSpacing = SpacingType(itkSpacing[0], itkSpacing[1], itkSpacing[2]);

}


template <class VoxelType>
void        
ApplicationUtils::
SaveHFieldITK(const char* fileNamePrefix,
	      const char* extension, 
	      const Array3D<Vector3D<VoxelType> >& h)
{
  const char *dimChars = "xyz";
  char fname[256];
  Image<VoxelType> dimImage(h.getSize());
  for(int dim=0;dim<3;dim++){
    for (unsigned int z = 0; z < h.getSize().z; ++z) {
      for (unsigned int y = 0; y < h.getSize().y; ++y) {
        for (unsigned int x = 0; x < h.getSize().x; ++x) {
          dimImage(x,y,z) = h(x,y,z)[dim];
        }
      }
    }
    sprintf(fname, "%s_%c.%s", fileNamePrefix, dimChars[dim], extension);
    SaveImageITK(fname, dimImage);
  } // end loop over dims
}

template <class VoxelType>
void        
ApplicationUtils::
SaveHFieldITK(const char *fileName,
	      const Array3D<Vector3D<VoxelType> > &h,
	      const OriginType &origin,
	      const SpacingType &spacing)
{
  typedef itk::Vector<VoxelType, 3> VectorType;
  typedef itk::Image<VectorType, 3> VectorImageType;
  typename VectorImageType::Pointer image = VectorImageType::New();


  Vector3D<unsigned int> size = h.getSize();
  typename VectorImageType::SizeType itkSize;
  itkSize[0] = size.x;
  itkSize[1] = size.y;
  itkSize[2] = size.z;
  typename VectorImageType::IndexType orig;
  orig[0] = 0;
  orig[1] = 0;
  orig[2] = 0;
  typename VectorImageType::RegionType region;
  region.SetSize(itkSize);
  region.SetIndex(orig);
  image->SetRegions(region);
  image->Allocate();
  typename VectorImageType::SpacingType itkSpacing;
  itkSpacing[0] = spacing.x;
  itkSpacing[1] = spacing.y;
  itkSpacing[2] = spacing.z;
  image->SetSpacing(itkSpacing);
  typename VectorImageType::PointType itkOrigin;
  itkOrigin[0] = origin.x;
  itkOrigin[1] = origin.y;
  itkOrigin[2] = origin.z;
  image->SetOrigin(itkOrigin);
  
  typename VectorImageType::IndexType index;
  typename VectorImageType::PixelType val;
  unsigned int x, y, z;
  for(z = 0; z < size.z; z++)
  {
    for(y = 0; y < size.y; y++)
    {
      for(x = 0; x < size.x; x++)
      {
        index[0] = x;
        index[1] = y;
        index[2] = z;

	const Vector3D<VoxelType> &pixel = h(x,y,z);

        val[0] = pixel.x;
        val[1] = pixel.y;
        val[2] = pixel.z;
        image->SetPixel(index, val);
      }
    }
  }

  typename itk::ImageFileWriter<VectorImageType>::Pointer VolWriter
    = itk::ImageFileWriter<VectorImageType>::New();
  VolWriter->SetFileName(fileName);
  VolWriter->SetInput(image);
  //VolWriter->UseCompressionOn();
  try
    {
      VolWriter->Update();
    }
  catch(itk::ImageFileWriterException &exc)
    {
      throw AtlasWerksException(__FILE__, __LINE__, "ITK File Writer Exception", &exc);
    }
  catch(itk::ExceptionObject &exc)
    {
      throw AtlasWerksException(__FILE__, __LINE__, "ITK Exception", &exc);
    }
  catch(...)
    {
      std::string errMsg = StringUtils::strPrintf("Unknown exception saving: %s", fileName);
      throw AtlasWerksException(__FILE__, __LINE__, errMsg);
    }
}

#endif // ApplicationUtils_txx
