//////////////////////////////////////////////////////////////////////
//
// File: DicomITK.h
// 
//////////////////////////////////////////////////////////////////////
#ifndef _DICOM_ITK_H_
#define _DICOM_ITK_H_

#include <string>

#include <itkImage.h>

#include "dicomImage.h"
#include "dicomContour.h"

typedef unsigned short DICOM_ITK_VoxelType;
typedef itk::Image< DICOM_ITK_VoxelType, 3 > DICOM_ITK_ImageType;

class DicomITK {
 public:
  DicomITK();
  ~DicomITK();

  DICOM_ITK_ImageType::Pointer Load( std::string fileName );
  
  int PreLoad( std::string fileName );
  DICOM_ITK_ImageType::Pointer LoadSelection( int selection );

  DICOMimage getDICOMimage();

  void setDICOMimage(DICOMimage _Dicom)
  {dicomImage = _Dicom;}
  
private:
  
  DICOMimage dicomImage;
  
}; // class DicomITK

#endif // #ifndef _DICOM_ITK_H_
