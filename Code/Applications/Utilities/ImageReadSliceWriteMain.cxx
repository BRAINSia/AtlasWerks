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
#include "itkImageFileWriter.h"
#include "itkExtractImageFilter.h"
#include "itkRescaleIntensityImageFilter.h"

/**
 * Simple application to extract a slice from a 3D image and write it
 * to a png or other format.
 */
int main( int argc, char ** argv )
{
  // Verify the number of parameters in the command line
  if( argc < 5 )
    {
      std::cerr << "Usage: " << std::endl;
      std::cerr << argv[0] << " inputImageFile outputImageFile"
		<< " -slice slicenum"
		<< " [-dim {x|y|z}]" 
		<< std::endl;
      return EXIT_FAILURE;
    }


  typedef float PixelType;
  typedef unsigned char SlicePixelType;
  const unsigned int Dimension = 3;
  const unsigned int SliceDimension = 2;

  typedef itk::Image< PixelType, Dimension >  ImageType;
  typedef itk::Image< SlicePixelType, Dimension >  CastType;
  typedef itk::Image< SlicePixelType, SliceDimension >  SliceType;
  typedef itk::ImageFileReader< ImageType >  ReaderType;
  typedef itk::ImageFileWriter< SliceType >  WriterType;
  typedef itk::RescaleIntensityImageFilter< ImageType, CastType >  RescaleType;
  typedef itk::ExtractImageFilter< CastType, SliceType >  ExtractImageType;
  
  typedef ImageType::RegionType RegionType;
  typedef ImageType::SizeType SizeType;
  typedef ImageType::IndexType IndexType;

  ReaderType::Pointer reader = ReaderType::New();
  WriterType::Pointer writer = WriterType::New();
  ExtractImageType::Pointer extractSlice = ExtractImageType::New();
  RescaleType::Pointer rescale = RescaleType::New();

  RegionType extractRegion;
  SizeType regionSize;
  IndexType regionIndex;

  const char *inputFileName = argv[1];
  const char *outputFileName = argv[2];
  int slice = -1;

  int curArg = 3;
  int dim = 2; // 2=Z
  while(curArg < argc){
    if(argv[curArg][0] != '-'){
      std::cerr << "Error, cannot parse arg " << argv[curArg] << std::endl;
      std::exit(-1);
    }else{
      if(strcmp("-slice",argv[curArg]) == 0){
	slice = atoi(argv[curArg+1]);
	curArg += 2;
	std::cout << "Extracting slice " << slice << std::endl;
      }else if(strcmp("-dim",argv[curArg]) == 0){
	switch(argv[curArg+1][0]){
	case 'x':
	case 'X':
	  dim = 0;
	  std::cout << "Using X-dim slices" << std::endl;
	  break;
	case 'y':
	case 'Y':
	  dim = 1;
	  std::cout << "Using Y-dim slices" << std::endl;
	  break;
	case 'z':
	case 'Z':
	  dim = 2;
	  std::cout << "Using Z-dim slices" << std::endl;
	  break;
	default:
	  std::cerr << "Error, unknown dimension: " << argv[curArg+1] << std::endl;
	  std::exit(-1);
	}
	curArg += 2;
      }
    }
  }

  reader->SetFileName( inputFileName  );

  reader->Update();
  
  rescale->SetInput(reader->GetOutput());
  rescale->SetOutputMinimum(0);
  rescale->SetOutputMaximum(255);
  rescale->Update();

  extractRegion = reader->GetOutput()->GetLargestPossibleRegion();

  regionSize = extractRegion.GetSize();
  regionIndex = extractRegion.GetIndex();

  if(slice < 0 || slice >= (int)regionSize[dim]){
    std::cerr << "Error, slice " << slice << " out of bounds: [0," << regionSize[dim]-1 << "]" << std::endl;
    std::exit(-1);
  }

  regionSize[dim] = 0;
  regionIndex[dim] += slice;
  
  extractRegion.SetSize(regionSize);
  extractRegion.SetIndex(regionIndex);

  extractSlice->SetExtractionRegion(extractRegion);
  extractSlice->SetInput(rescale->GetOutput());
  extractSlice->Update();
  
  writer->SetFileName( outputFileName );
  writer->SetInput( extractSlice->GetOutput() );

  try 
    { 
      writer->Update(); 
    } 
  catch( itk::ExceptionObject & err ) 
    { 
      std::cerr << "ExceptionObject caught !" << std::endl; 
      std::cerr << err << std::endl; 
      return EXIT_FAILURE;
    } 

  return EXIT_SUCCESS;
}



