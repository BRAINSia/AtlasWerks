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

#ifndef Dicom_Window_Class_HDR
#define Dicom_Window_Class_HDR 

#include <FL/Fl_Double_Window.H>
#include <FL/Fl.H>
#include <dicomImage.h>

class DicomWindowClass : public Fl_Double_Window {

  DICOMimage DICOM;

public :
  DicomWindowClass(int W, int H, const char *L):Fl_Double_Window(W,H,L){
  };

  DICOMimage getDicom(){
    return DICOM;
  }

  void setDicom(DICOMimage _DICOM){
    DICOM = _DICOM;
  }

};

#endif
