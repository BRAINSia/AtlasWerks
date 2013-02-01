//////////////////////////////////////////////////////////////////////
//
// File: DicomITK.cxx
// 
//////////////////////////////////////////////////////////////////////
#include <iostream>

#include "DicomITK.h"

//////////////////////////////////////////////////////////////////////
//
// DicomITK::DicomITK
//
//////////////////////////////////////////////////////////////////////
DicomITK::DicomITK()
{
} // DicomITK::DicomITK()


//////////////////////////////////////////////////////////////////////
//
// DicomITK::~DicomITK
//
//////////////////////////////////////////////////////////////////////
DicomITK::~DicomITK()
{
} // DicomITK::~DicomITK
 

//////////////////////////////////////////////////////////////////////
//
// DicomITK::PreLoad
//
//////////////////////////////////////////////////////////////////////
int
DicomITK::PreLoad( std::string fileName )
{
  ///////////////
  // Load data //
  ///////////////
  int numVolumes = dicomImage.OpenDICOMFile( (char*)fileName.c_str() );
  if( numVolumes < 1 ) 
    {
      std::cerr << "[DicomITK::Load] no volumes loaded!" << std::endl;
    }
  std::cerr << "[DicomITK::Load] loaded " << numVolumes << " volumes." 
	    << std::endl;
  
  return( numVolumes );
}// DicomITK::PreLoad


//////////////////////////////////////////////////////////////////////
//
// DicomITK::LoadSelection
//
//////////////////////////////////////////////////////////////////////
DICOM_ITK_ImageType::Pointer 
  DicomITK::LoadSelection( int selection ){

  ////////////////////////////////////////////////////
  // Do something with the selected volume/slice ?? //
  ////////////////////////////////////////////////////
  dicomImage.SelectImage( selection );
  
  ////////////////
  // Diagnostic //
  ////////////////
  std::cout << "Dimensions (x,y,z) = " << dicomImage.getsizeX() << ", " <<
    dicomImage.getsizeY() << ", " << dicomImage.getsizeZ() << std::endl;
  std::cout << "Spacing (x,y,z) = " << dicomImage.PixsizeX() << ", " <<
    dicomImage.PixsizeY() << ", " << dicomImage.PixsizeZ() << std::endl;

  ///////////////////////////////////////////////////////
  // Will DICOM data always be of type unsigned short? //
  ///////////////////////////////////////////////////////
  int numVoxels = 
    dicomImage.getsizeZ() *
    dicomImage.getsizeY() *
    dicomImage.getsizeX();

  DICOM_ITK_VoxelType* data = new unsigned short[numVoxels];
  std::cout << "Loading DICOM data...";
  dicomImage.LoadTheImage((void *) data );
  std::cout << "Done." << std::endl;

  DICOM_ITK_ImageType::Pointer image = DICOM_ITK_ImageType::New();

  DICOM_ITK_ImageType::SizeType imageSize;
  imageSize[0] = dicomImage.getsizeX();
  imageSize[1] = dicomImage.getsizeY();
  imageSize[2] = dicomImage.getsizeZ();
  image->SetRegions( imageSize );

  double imageSpacing[3];
  imageSpacing[0] = dicomImage.PixsizeX();
  imageSpacing[1] = dicomImage.PixsizeY();
  imageSpacing[2] = dicomImage.PixsizeZ();
  image->SetSpacing( imageSpacing );

  double origin[3];
  // we only have z info from dicom reader
  origin[0] = 0.0;
  origin[1] = 0.0;
  origin[2] = dicomImage.get_Zoffset();
  image->SetOrigin(origin);
	  	
  image->Allocate();

  ////////////////////////
  // Populate the image //
  ////////////////////////
  std::cout << "Populating ITK image...";
  DICOM_ITK_ImageType::IndexType imageIndex;
  DICOM_ITK_VoxelType* dataPtr = data;
  for( int zPos = 0; zPos < dicomImage.getsizeZ(); zPos++ ) {
    imageIndex[2] = zPos;
    for( int yPos = 0; yPos < dicomImage.getsizeY(); yPos++ ) {
      imageIndex[1] = yPos;
      for( int xPos = 0; xPos < dicomImage.getsizeX(); xPos++ ) {
	imageIndex[0] = xPos;
	
	image->SetPixel( imageIndex, *dataPtr++ );
      }
    }
  }
  std::cout << "Done." << std::endl;

  delete [] data;

   
  return( image );
} // DicomITK::LoadSelection



DICOMimage DicomITK::getDICOMimage(){
  return dicomImage;
}

//////////////////////////////////////////////////////////////////////
//
// DicomITK::Load
//
//////////////////////////////////////////////////////////////////////
DICOM_ITK_ImageType::Pointer 
DicomITK::Load( std::string fileName )
{
    
  int selectedVolume= 0;
  int maxSlice = 0;
  
  DICOMimage dicomImage;

  ///////////////
  // Load data //
  ///////////////
  int numVolumes = dicomImage.OpenDICOMFile( (char*)fileName.c_str() );
  if( numVolumes < 1 ) {
    std::cerr << "[DicomITK::Load] no volumes loaded!" << std::endl;
    return( (DICOM_ITK_ImageType::Pointer)0 );
  }
  std::cerr << "[DicomITK::Load] loaded " << numVolumes << " volumes." <<
    std::endl;

  ////////////////////////////////////////////////////
  // Find the volume/slice with the largest z size? //
  ////////////////////////////////////////////////////
  for( int volume = 0; volume < numVolumes; volume++ ) {
    dicomImage.SelectImage( volume );

    if( dicomImage.getsizeZ() > maxSlice ) {
      maxSlice = dicomImage.getsizeZ();
      selectedVolume = volume;
    }
  }

  ////////////////////////////////////////////////////
  // Do something with the selected volume/slice ?? //
  ////////////////////////////////////////////////////
  dicomImage.SelectImage( selectedVolume );
  
  ////////////////
  // Diagnostic //
  ////////////////
  std::cout << "Dimensions (x,y,z) = " << dicomImage.getsizeX() << ", " <<
    dicomImage.getsizeY() << ", " << dicomImage.getsizeZ() << std::endl;
  std::cout << "Spacing (x,y,z) = " << dicomImage.PixsizeX() << ", " <<
    dicomImage.PixsizeY() << ", " << dicomImage.PixsizeZ() << std::endl;

  ///////////////////////////////////////////////////////
  // Will DICOM data always be of type unsigned short? //
  ///////////////////////////////////////////////////////
  int numVoxels = 
    dicomImage.getsizeZ() *
    dicomImage.getsizeY() *
    dicomImage.getsizeX();
    
  DICOM_ITK_VoxelType* data = new unsigned short[numVoxels];
  std::cout << "Loading DICOM data...";
  dicomImage.LoadTheImage( data );
  std::cout << "Done." << std::endl;

  DICOM_ITK_ImageType::Pointer image = DICOM_ITK_ImageType::New();
  

  DICOM_ITK_ImageType::SizeType imageSize;
  imageSize[0] = dicomImage.getsizeX();
  imageSize[1] = dicomImage.getsizeY();
  imageSize[2] = dicomImage.getsizeZ();
  image->SetRegions( imageSize );

  double imageSpacing[3];
  imageSpacing[0] = dicomImage.PixsizeX();
  imageSpacing[1] = dicomImage.PixsizeY();
  imageSpacing[2] = dicomImage.PixsizeZ();
  image->SetSpacing( imageSpacing );

  double origin[3];
  // we only have z info from dicom reader
  origin[0] = 0.0;
  origin[1] = 0.0;
  origin[2] = dicomImage.get_Zoffset();
  image->SetOrigin(origin);

  image->Allocate();

  ////////////////////////
  // Populate the image //
  ////////////////////////
  std::cout << "Populating ITK image...";
  DICOM_ITK_ImageType::IndexType imageIndex;
  DICOM_ITK_VoxelType* dataPtr = data;
  for( int zPos = 0; zPos < dicomImage.getsizeZ(); zPos++ ) {
    imageIndex[2] = zPos;
    for( int yPos = 0; yPos < dicomImage.getsizeY(); yPos++ ) {
      imageIndex[1] = yPos;
      for( int xPos = 0; xPos < dicomImage.getsizeX(); xPos++ ) {
	imageIndex[0] = xPos;
	
	image->SetPixel( imageIndex, *dataPtr++ );
      }
    }
  }
  std::cout << "Done." << std::endl;

  delete data;
  

  return( image );
} // DicomITK::Load
