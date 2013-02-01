#ifndef __DICOMIMAGE_H__
#define __DICOMIMAGE_H__

#include "dicom_.h"

// Necessary because PDATE is already defined in this file
//#include "../bradlib/gen.h"
#include "gen.h"
#include <string>
#include <vector>



// This class is derived from dicom_ and add some info cvoncerning the image
// the header for instance, the get functions are defined here.



// typedef struct
// {
//   int day	; 	/* one based: 1 = first day of month */
//   int month; 	/* one based: 1 = Jan */
//   int year; 	/* 1900 based: 97 = 1997 (year 2000 problem) */
//   int dow ; 	/* zero based: 0 = Sunday */
// } PDATE;


typedef struct _DICOM_header
{
  char   unit_number[20];	/* hospital id # */
  char   patient_name[100];
  PDATE   date;
  char   comment[100];
  char 	 machine_id[100];	/* currently of only potential use */
  char 	 image_type[100];	
  char	 str_orient[3];
  char	 modality[10];
  char	 study_time[10];
  float  pixel_size;		/* absolute value - square assumed */
  float  pixel_sizeZ;	
  float	 *z_pos;
  float  z_offset;
  int    x_dim;
  int    y_dim;
  int    z_dim;
  int    min;			/* min CT number in study */
  int    max;			/* max CT number in study */

} DICOM_header;

class DICOMimage : public DICOM
{
  public :
    DICOMimage();
  ~DICOMimage();
  int OpenDICOMSelectedFiles(std::string DirectoryName, std::vector<std::string> SelectedFiles);
  int  OpenDICOMFile( char*);
  int OpenDICOMFile();
  void LoadTheImage(void *);

  inline void SelectImage(int _select) { select = _select; }
  inline DICOMimage*  getObject() { return(this); }
  inline char** get_infile()	  { return(infile); }
  inline int get_num_files() 	  { return(num_files); }
  inline int getsizeX() 	  { return(header[select].x_dim); }
  inline int getsizeY() 	  { return(header[select].y_dim); }
  inline int getsizeZ() 	  { return(header[select].z_dim); }
  inline int get_min()            { return(header[select].min);   }
  inline int get_max()            { return(header[select].max);   } 
  inline float PixsizeX() 	  { return(header[select].pixel_size); }
  inline float PixsizeY() 	  { return(header[select].pixel_size); }
  inline float PixsizeZ()	  { return(header[select].pixel_sizeZ); }
  inline float get_Zpos(int slice){return (header[select].z_pos[slice]);}
  inline float get_Zoffset()	  { return(header[select].z_offset); }
  inline char* get_orient() 	  { return(header[select].str_orient); }
  inline char* get_image_type()   { return(header[select].image_type); }
  inline char* get_name() 	  { return(header[select].patient_name); }
  inline char* get_comment() 	  { return(header[select].comment); }
  inline char* get_machine_ID()   { return(header[select].machine_id); }
  inline char* get_study_time()   { return(header[select].study_time); }
  inline char* get_unit_number()  { return(header[select].unit_number);}
  inline char* get_modality()     { return(header[select].modality);}
  char* get_Date();
  bool CheckUniformSpacing();
  BOOLEAN loadcontours;
		

  private :
    int read_op(int fd,int num);

  DICOM_header	*header;
  int		*data_pos;
  int		num_volumes;
  int		*tab_select;
  int		select;
  char          date_str[11];
};
		
#endif // DICOM_CLASS_H

