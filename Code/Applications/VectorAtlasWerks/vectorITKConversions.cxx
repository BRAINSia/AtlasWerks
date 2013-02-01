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


#include "vectorITKConversions.h"

#include "itkImageRegionIteratorWithIndex.h"

// Convert Image to itk::VectorImage
itk::VectorImage<float>::Pointer
convertImageVolume(Image< vnl_vector<float> >* img)
{
  typedef itk::VectorImage<float> ITKImageType;

  ITKImageType::Pointer itkimg = ITKImageType::New();

  ITKImageType::SizeType size;
  size[0] = img->getSizeX();
  size[1] = img->getSizeY();
  size[2] = img->getSizeZ();

  ITKImageType::RegionType region;
  region.SetSize(size);

  unsigned int vectorDim = (img->get(0, 0, 0)).size();

  itkimg->SetVectorLength(vectorDim);
  itkimg->SetRegions(region);
  itkimg->Allocate();

  for (unsigned int k = 0; k < size[2]; k++)
    for (unsigned int j = 0; j < size[1]; j++)
      for (unsigned int i = 0; i < size[0]; i++)
      {
        ITKImageType::IndexType ind;
        ind[0] = i;
        ind[1] = j;
        ind[2] = k;

        vnl_vector<float> v = img->get(i, j, k);

        ITKImageType::PixelType w(vectorDim);
        for (unsigned int d = 0; d < vectorDim; d++)
          w[d] = v[d];

        itkimg->SetPixel(ind, w);
      }

  ITKImageType::SpacingType spacing;
  spacing[0] = img->getSpacingX();
  spacing[1] = img->getSpacingY();
  spacing[2] = img->getSpacingZ();
  itkimg->SetSpacing(spacing);

  ITKImageType::PointType origin;
  origin[0] = img->getOriginX();
  origin[1] = img->getOriginY();
  origin[2] = img->getOriginZ();
  itkimg->SetOrigin(origin);

  return itkimg;
}

// Convert itk::VectorImage to Image
Image< vnl_vector<float> >*
convertITKVolume(itk::VectorImage<float>* itkimg)
{
  typedef vnl_vector<float> VectorType;

  typedef itk::VectorImage<float> ITKImageType;

  ITKImageType::SizeType itksize = itkimg->GetLargestPossibleRegion().GetSize();

  Image<VectorType>::SizeType size;
  size.x = itksize[0];
  size.y = itksize[1];
  size.z = itksize[2];

  Image<VectorType>* image = new Image<VectorType>(size);

  image->setOrigin(
    itkimg->GetOrigin()[0],
    itkimg->GetOrigin()[1],
    itkimg->GetOrigin()[2]);
  image->setSpacing(
    itkimg->GetSpacing()[0],
    itkimg->GetSpacing()[1],
    itkimg->GetSpacing()[2]);

  image->setDataType(Image<VectorType>::VectorDataType);

  unsigned int vectorDim = itkimg->GetVectorLength();

  // Copy data from itk::VectorImage iteratively
  // Cannot use memcpy as array storage may be different (?)
  typedef itk::ImageRegionIteratorWithIndex<ITKImageType> IteratorType;
  IteratorType it(itkimg, itkimg->GetLargestPossibleRegion());
  for (it.GoToBegin(); !it.IsAtEnd(); ++it)
  {
    ITKImageType::IndexType ind = it.GetIndex();

     ITKImageType::PixelType v = it.Get();

     VectorType w(vectorDim);
     for (unsigned int dim = 0; dim < vectorDim; dim++)
       w[dim] = v[dim];

    image->set(ind[0], ind[1], ind[2], w);
  }

  return image;

}
