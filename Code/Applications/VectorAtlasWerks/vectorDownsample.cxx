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


#include "vectorDownsample.h"
#include "vectorITKConversions.h"

#include "itkImageRegionIteratorWithIndex.h"
#include "itkNeighborhoodIterator.h"
#include "itkVectorImage.h"
#include "itkVectorLinearInterpolateImageFunction.h"

#include <cmath>

Image< vnl_vector<float> >*
vectorDownsample(
  Image< vnl_vector<float> >* image,
  double factor,
  double sigma,
  double kernelSize)
{
  typedef vnl_vector<float> VoxelType;

  typedef itk::VectorImage<float> ITKImageType;

  unsigned int vectorDim = image->get(0, 0, 0).size();

  ITKImageType::Pointer itkimg = convertImageVolume(image);

  ITKImageType::PointType origin = itkimg->GetOrigin();
  ITKImageType::SpacingType spacing = itkimg->GetSpacing();
  ITKImageType::SizeType size = itkimg->GetLargestPossibleRegion().GetSize();

  ITKImageType::SpacingType downspacing = itkimg->GetSpacing();
  ITKImageType::SizeType downsize = size;
  for (int dim = 0; dim < 3; dim++)
  {
    downsize[dim] = (int)(size[dim] / factor);
    downspacing[dim] = spacing[dim]*size[dim] / downsize[dim];
  }

  // Blur image
  typedef itk::NeighborhoodIterator<ITKImageType> NeighIteratorType;

  ITKImageType::SizeType radii;
  radii.Fill((int)floor(kernelSize+0.5));

  NeighIteratorType nIt(radii, itkimg, itkimg->GetLargestPossibleRegion());

  // Build kernel weights
  double* kernelWeights = new double[nIt.Size()];
  double sumW = 1e-20;
  double var = sigma*sigma;
  for (unsigned int i = 0; i < nIt.Size(); ++i)
  {
    ITKImageType::OffsetType offt = nIt.GetOffset(i);
    double dist = 0;
    for (int dim = 0; dim < 3; dim++)
    {
      double d = spacing[dim]*offt[dim];
      dist += d*d;
    }
    double w = exp(-0.5 * dist / var);
    kernelWeights[i] = w;
    sumW += w;
  }
  for (unsigned int i = 0; i < nIt.Size(); ++i)
    kernelWeights[i] /= sumW;

  ITKImageType::Pointer filtimg = ITKImageType::New();
  filtimg->SetVectorLength(vectorDim);
  filtimg->SetRegions(itkimg->GetLargestPossibleRegion());
  filtimg->Allocate();
  filtimg->SetOrigin(itkimg->GetOrigin());
  filtimg->SetSpacing(itkimg->GetSpacing());

  for (nIt.GoToBegin(); !nIt.IsAtEnd(); ++nIt)
  {
    ITKImageType::IndexType ind = nIt.GetIndex();
    ITKImageType::PixelType v(vectorDim);
    v.Fill( 0.0);
    for (unsigned int i = 0; i < nIt.Size(); i++)
      v += kernelWeights[i] * nIt.GetPixel(i);
    filtimg->SetPixel(ind, v);
  }

  delete [] kernelWeights;

  // Downsample image
  ITKImageType::RegionType downregion = itkimg->GetLargestPossibleRegion();
  downregion.SetSize(downsize);

  ITKImageType::Pointer downimg = ITKImageType::New();
  downimg->SetVectorLength(vectorDim);
  downimg->SetRegions(downregion);
  downimg->Allocate();
  downimg->SetSpacing(downspacing);
  downimg->SetOrigin(origin);

  ITKImageType::PixelType zerov(vectorDim);
  zerov.Fill(0.0);

  typedef itk::ImageRegionIteratorWithIndex<ITKImageType> IteratorType;
  IteratorType downIt(downimg, downimg->GetLargestPossibleRegion());

  for (downIt.GoToBegin(); !downIt.IsAtEnd(); ++downIt)
  {
    ITKImageType::PointType p;
    downimg->TransformIndexToPhysicalPoint(downIt.GetIndex(), p);

    ITKImageType::IndexType mappedInd;
    mappedInd[0] = (long)floor( (p[0] - origin[0]) / spacing[0] + 0.5 );
    mappedInd[1] = (long)floor( (p[1] - origin[1]) / spacing[1] + 0.5 );
    mappedInd[2] = (long)floor( (p[2] - origin[2]) / spacing[2] + 0.5 );

    bool outside = false;
    for (int dim = 0; dim < 3; dim++)
    {
      if (mappedInd[dim] < 0 || mappedInd[dim] >= (int)size[dim])
        outside = true;
    }

    if (!outside)
      downIt.Set(itkimg->GetPixel(mappedInd));
    else
      downIt.Set(zerov);
  }

  return convertITKVolume(downimg);
}

