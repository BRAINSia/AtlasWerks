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

#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkStatisticsImageFilter.h"

int main( int argc, char ** argv )
{
  // Verify the number of parameters in the command line
  if( argc < 2 )
    {
      std::cerr << "Usage: " << std::endl;
      std::cerr << argv[0] << " inputImageFile"
		<< std::endl;
      return EXIT_FAILURE;
    }


  typedef float PixelType;
  const unsigned int Dimension = 3;

  typedef itk::Image< PixelType, Dimension >  ImageType;
  typedef itk::ImageFileReader< ImageType >  ReaderType;
  typedef itk::StatisticsImageFilter< ImageType >  StatisticsFilterType;
  
  typedef ImageType::RegionType RegionType;
  typedef ImageType::SizeType SizeType;
  typedef ImageType::SpacingType SpacingType;
  typedef ImageType::PointType PointType;
  typedef ImageType::IndexType IndexType;

  ReaderType::Pointer reader = ReaderType::New();
  StatisticsFilterType::Pointer statisticsFilter = StatisticsFilterType::New();

  RegionType region;
  SizeType regionSize;
  IndexType regionIndex;
  PointType origin;
  SpacingType spacing;

  const char *inputFileName = argv[1];

  reader->SetFileName( inputFileName  );

  reader->Update();
  
  ImageType::Pointer image = reader->GetOutput();

  region = image->GetLargestPossibleRegion();
  regionSize = region.GetSize();
  regionIndex = region.GetIndex();

  origin = image->GetOrigin();
  spacing = image->GetSpacing();

  statisticsFilter->SetInput(image);
  statisticsFilter->Update();

  double range[2];
  range[0] = statisticsFilter->GetMinimum();
  range[1] = statisticsFilter->GetMaximum();

  std::cout << "Origin: " << origin << std::endl; 
  std::cout << "Spacing: " << spacing << std::endl; 
  std::cout << "Index: " << regionIndex << std::endl;
  std::cout << "Size: " << regionSize << std::endl;
  std::cout << "Range: [" << range[0] << ", " << range[1] << "]" << std::endl;

  return EXIT_SUCCESS;
}



