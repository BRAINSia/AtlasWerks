#ifndef IMAGE_IO_H
#define IMAGE_IO_H

// 'identifier' : identifier was truncated to 'number' characters in the
// debug information
#ifdef WIN32
#pragma warning ( disable : 4786 )
#endif

//#include <strings.h>
#include <string> 
#include <iostream>
#include <string.h> 
#include <stdio.h>

#include "PlanIM/PlanIO.h"
#include "PlanIM/PlanIM_Functions.h"
#include "Dicom/dicomImage.h"
#include "Dicom/dicomContour.h"

#include <Array3D.h>
#include <Image.h>
#include <Vector3D.h>
#include "dbh.h"
#include "IDLdefines.h"

#ifndef MAXPATHLEN 
#define MAXPATHLEN 256
#endif

using std::string;
//using std::istream;



// Defined in image_hdr.h
// enum Orientation {
//     LAI,LAS,LPI,LPS,RAI,RAS,RPI,RPS,
//     RSP,RSA,RIP,RIA,LSP,LSA,LIP,LIA,
//     AIR,AIL,ASR,ASL,PIR,PIL,PSR,PSL
//   };

typedef unsigned short u_short;
typedef unsigned char   u_char;

class ImageIO
{
protected :

  // Flag for Byte swapping
  bool Need_BS;

  // Data for planIM functions

  plan_im_header IM_input_header;
  float* IM_z_positions;
  float* IM_z_diffs;
  float IM_z_zero, IM_slice_thickness;
  int IM_num_slices;
  PIXELTYPE IM_background_fill;

  // Data for DICOM
  int _selected_DICOM;
  DICOMimage dicomImage;
  int nb_DICOM;
  bool UniformSpacing;

  void _dicomReslicing(Image<float> &image3D, float newSpacing);
  
public:

  enum ImageType {analyze,gipl,meta,PLUNC_dose_grid,planIM,dicom,RTOG,unknown};

  ImageIO(){ Need_BS = false; UniformSpacing = true;};


  // Guess the format of an image given the path
  ImageType GuessImageFormat(string filename);

  // Load an image guessing its format
  void LoadThisImage(string filename, Image<float> &image3D,
                     ImageType extension = unknown);

  // Save an image guessing its format
  void SaveImage(string filename, Image<float> &image3D);


  // Analyze Images
  /*****************/

  void ConstructAnalyzeNames(const char *filename,char *ret_prefix,
			     char *ret_hdrname,char *ret_imgname);

  // Load

  // Load data depending on the data type
  void LoadData(Image<u_char> &dat, std::istream &file);
  void LoadData(Image<u_short> &dat, std::istream &file);
  void LoadData(Image<float> &dat, std::istream &file);
  
  // Read the header and fill our image_hdr
  void  ReadAnalyzeHeader(const char *hdrname,Image<float> &header);
  void  ReadAnalyzeHeader(const char *hdrname,Image<unsigned short> &header);
  
  // Load Analyze image parameter can be an Array3D or a pointer to an
  // Array3D
  void LoadAnalyze(string filename,Image<float> &image3D);
  void LoadAnalyze(string filename,Image<unsigned short> &image3D);
  void LoadAnalyze(string filename,Image<float> *image3D){
    LoadAnalyze(filename,*&image3D);
  }
  
// Save
  
  // Write Array3D in a file
  void SaveData(Image<float> dat, std::ostream &file);
  // Open a file and call SaveData
  void SaveRaw(string fname,Image<float> &image3D);
  // Save the header for an analyze image
  void SaveAnalyzeHeader(string hdrname,Image<float> &image3D);

  // Write an analyze image in a file
  void SaveAnalyze(string filename,Image<float> &image3D);


// DICOM Images
/***************/

  // First step to load a dicom, 
  // Set the DICOM parameters of Image IO (DICOM and nb_DICOM)
  void LoadDicom(string filename);
  // check if the spacing is uniform for the selected 
  bool CheckSpacingSelectedDicom(int selected_DICOM);
  // Load the selected image (called after user's choice)
  void LoadSelectedDicom(Image<float> &image3D, float newSpacing = 1);
  void LoadDicomContour(DICOMcontour *dcont);
 
  
// PlanIM Images
/***************/

  // Load PlanIM image parameter can be an Array3D or a pointer to an
  // Array3D
  // If the image is irregularly sliced, it calls a function that will guess
  // default parameters for the reslicing.
  // These reslicing parameters are parameters of imageIO
  // (IM_input_header, IM_z_positions, IM_z_diffs, IM_z_zero,
  // IM_slice_thickness, IM_num_slices, IM_background_fill)
  void LoadPlanIM(string filename,Image<float> &image3D);
  void LoadPlanIM(string filename,Image<float> *image3D){
    LoadPlanIM(filename,*&image3D);
   }
  // Called once the user as set the new reslicing paramater 
  // If there is no user interface part, the parameters used here
  // will be the one obtained from guessParameters
  void LoadPlanIMIrreg(string filename,Image<float> &image3D);

  //Save
  
  void getMinMax(float &minret, float &maxret,Image<float> image3D);

  void convertToUnsignedShort(Image<float> image3D,Array3D<u_short> &u_short_dat);
  void convertToShort(Image<float> image3D,Array3D<short> &short_dat);
  
  void SavePlanIM(string filename,Image<float> &image3D, float offset_z);

// RTOG
/***************/

  //
  void LoadRTOGHeader(string filename);
  void LoadRTOGScan();
  void LoadRTOGStructure(int structnum);
  void LoadRTOGDose();

  void SaveRTOG(string RTOGDir);

// PLUNC dose grids
/***************/

  void LoadDoseGrid(string filename,Image<float> &image3D);
  void LoadDoseGrid(string filename,Image<float> *image3D){
    LoadDoseGrid(filename,*&image3D);
  }

  void SaveDoseGrid(string filename, Image<float>& image3D);

// PlanIM : get and set functions
// This part is needed for the user interface part

  plan_im_header get_IM_input_header(){
    return IM_input_header;
  }
  float* get_IM_z_positions(){
    return IM_z_positions;
  }
  float* get_IM_z_diffs(){
    return IM_z_diffs;
  }
  float get_IM_z_zero(){
    return IM_z_zero;
  }
  float get_IM_slice_thickness(){
    return IM_slice_thickness;
  }
  int get_IM_num_slices(){
    return IM_num_slices;
  }
  PIXELTYPE get_IM_background_fill(){
    return IM_background_fill;
  }

  void set_IM_z_zero(float val){
    IM_z_zero=val;
  }
  void set_IM_slice_thickness(float val){
    IM_slice_thickness=val;
  }
  void set_IM_num_slices(int val){
    IM_num_slices=val;
  }
  void set_IM_background_fill(PIXELTYPE val){
    IM_background_fill=val;
  }
  
// DICOM : get functions
// This part is needed for the user interface part
  
  DICOMimage get_DICOM(){
    return dicomImage;
  }

  int get_nb_DICOM(){
    return nb_DICOM;
  }

  void setDICOM(DICOMimage dicom){
  dicomImage = dicom;
  }

};

#endif
