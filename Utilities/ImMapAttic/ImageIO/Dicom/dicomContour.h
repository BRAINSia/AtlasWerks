#ifndef __DICOMCONTOUR_H__
#define __DICOMCONTOUR_H__

#include "macros.h"
#include "IDLdefines.h"
#include "dicom_.h"
#include "dicomImage.h"

typedef unsigned long U32;

class DICOMcontour : public DICOM
{
  public :
  DICOMcontour();
    DICOMcontour(char*);
  DICOMcontour(DICOMimage *ref);
  ~DICOMcontour();
  int  OpenDICOMFile();
  void set_ref(DICOMimage *ref ) { image 	= ref;
                                   infile = ref->get_infile();
                                   num_files = ref->get_num_files(); };
  inline int get_num_anastruct() 		{ return(num_anastruct); }
  inline char* get_contourName(int num_c)       { return(ana[num_c].label);}
  inline char* get_patientName() 		{ return(patient_name);}
  inline DICOMimage* get_image() 		{ return(image);}
  inline ANASTRUCT& get_AnaStruct(int num_c)     {return(ana[num_c]);}
  
  private :
    int get_current_anastruct(char*);
  int read_op(int,int);
  void getMinMax();
  
  float		pixel_size;
  float		zpos;
  float		xCenter,yCenter;
  DICOMimage	*image;
  int		cur_anastruct;
  int 		num_anastruct; 
  char		patient_name[50];
  ANASTRUCT	*ana;
};

#endif // DICOMCONTOUR_CLASS_H
