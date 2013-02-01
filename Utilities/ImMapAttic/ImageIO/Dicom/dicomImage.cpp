#include <iostream.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <ctype.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

#ifndef WIN32
#include <unistd.h>

#else

#include <io.h>

#endif
#include <dicomImage.h>

int		z_num;
int		aq_num;
int             bits_alloc; 
int		bits_stores; 
int		bits_highbit;
int		debug 		= FALSE;
DICOM_header	*tmp_header;
float           tmp_z;

// This class is derived from dicom_ and add some info cvoncerning the image
// the header for instance, the get functions are defined here.


typedef unsigned short u_short;

DICOMimage::DICOMimage()
{
  loadcontours	= TRUE;
  data_pos	= NULL;
  header		= NULL;
  tab_select	= NULL;		
  num_volumes 	= 0;
}

DICOMimage::~DICOMimage()
{
}

int DICOMimage::OpenDICOMFile(char *_basename) 
{ 
  int 	loop;
  int 	fd;
  int 	nitems;
  int 	i,j;
  float z_diff;

  basename = _basename;

  if(!basename)
    return 0;

  get_fileList(basename);

  if(!infile)
    return 0;

  header 	= new DICOM_header[10]; 
  tmp_header 	= new DICOM_header;
  tab_select 	= new int[num_files];
  data_pos 	= new int[num_files];
  float *z_pos  = new float[num_files];

  for (loop = 0; loop < num_files; loop++)
    {
      if (loop == num_files) 
	break;

      if (debug) 
	cout<<"name : "<<infile[loop]<<endl;

      if ((fd = open(infile[loop], O_RDONLY|O_BINARY, 0)) < 0)
	{
	  infile[loop] = NULL;
	  continue;
	}
	
      if (guess_swap(fd)<0)		/* set swap1 & swap2 */
	{
	  if(debug)
	    cout<<infile[loop]<<"Guess swap failed!! is not a DICOM file"<<endl;
	  infile[loop] = NULL;
	  close(fd);
	  continue;
	}


      lseek(fd, 0, SEEK_SET);	/* rewind */

      nitems = 0;	    		
      for (;;)
	{
	  if (read_op(fd,loop)) 
	    break;
	  nitems++;
	}

      if(!nitems)
	{
	  if(debug)
	    cout<<infile[loop]<<"no items "<<endl;
	  infile[loop] = NULL;
	  close(fd);
	  continue;
	}

      data_pos[loop] = lseek(fd,0,SEEK_CUR);
      z_pos[loop]    = tmp_z;
      close(fd);
    }


  for(i=0;i<num_volumes;i++)	
    {
      if(!header[i].patient_name[0])
	strcpy(header[i].patient_name,"No Name");

      header[i].z_dim = 0;
      header[i].z_pos = NULL;

      for (loop = 0; loop < num_files; loop++)
	{
	  if((!infile[loop]) || (tab_select[loop] != i))
	    continue;
	  header[i].z_dim++;
	  header[i].z_pos = (float*)realloc(header[i].z_pos,header[i].z_dim*sizeof(float));
	  header[i].z_pos[header[i].z_dim-1] = z_pos[loop];
	}

      if(header[i].z_dim==1)
	header[i].pixel_sizeZ = 0;
      else {
	if(PLAN_FABS(header[i].z_pos[1]-header[i].z_pos[0])<1.0) //if z_pos is in cm
	  for (j = 0; j < header[i].z_dim; j++)
	    header[i].z_pos[j] *=10.0;   //convert z_pos in mm
	
	for (j = 1; j < header[i].z_dim; j++)
	  {
	    z_diff = PLAN_FABS(header[i].z_pos[j]-header[i].z_pos[j-1]);
	    if (debug) 
	      cout<<"volume "<<i<<" z pos "<<header[i].z_pos[j]<<" diff "<<z_diff<<endl;
	  }
	header[i].pixel_sizeZ = z_diff;
      }
      header[i].z_offset = header[i].z_pos[0];
    }
	


  delete tmp_header;
  delete z_pos;
  return(num_volumes);
}


void DICOMimage::LoadTheImage(void* data)
{
  u_short *scan_data = NULL;
  int 	loop;
  int   len;
  int 	fd;
  int	sx,sy;

  sx = header[select].x_dim;
  sy = header[select].y_dim;

  len = sx*sy*sizeof(short);
  scan_data = (u_short*)malloc(len);

  for (loop = 0; loop < num_files; loop++)
    {
      
      if((!infile[loop]) || (tab_select[loop] != select)) {
	infile[loop] = NULL;
	continue;
      }
      
      fd = open(infile[loop], O_RDONLY|O_BINARY, 0);
      guess_swap(fd);
      lseek(fd, data_pos[loop], SEEK_SET);		
      read(fd, scan_data, len);
      
      if (swap2 == 1) 
	my_swab((char*)scan_data, (char*)scan_data, len);

      close(fd);
      
      memcpy((u_short*)data,scan_data,len);
      data = (u_short*)data +sx*sy;				
    }

  if(scan_data)
    delete [] scan_data;
}


char *DICOMimage::get_Date()
{
  sprintf(date_str,"%d/%d/%d",header[select].date.month,
	  header[select].date.day,header[select].date.year);
  return date_str; 
}


/* interpret next opcode in file. 
   if pixel size, patient name or resolution change, print err msg & exit(1)*/

int DICOMimage::read_op(int fd,int num)
{   
  int	val;
  char	buf[MAX_PRIVBUFLEN];
  char 	temp[4];
  char	tmpc;
  float tmp;

  val = read_single_op(fd, buf);

  if (val == -1) 
    {
      if((tmp_header->x_dim != header[num_volumes-1].x_dim) ||
	 (tmp_header->y_dim != header[num_volumes-1].y_dim) ||
	 (tmp_header->pixel_size != header[num_volumes-1].pixel_size) ||
	 (strcmp(tmp_header->patient_name,header[num_volumes-1].patient_name))) 
	{
	  header[num_volumes] = *tmp_header;
	  tmp_header = new DICOM_header;
	  num_volumes++;
	}   	
      tab_select[num] = num_volumes-1;
      return -1;
    }

  if (val == -2){
    return -1;

  }



  int i = 0;

  switch (opcode) {
  case OP_Z_POSITION:
    sscanf(buf,"%f%c%f%c%f",&tmp,&tmpc,&tmp,&tmpc,&tmp);
    //tmp_z = PLAN_FABS(tmp);
	tmp_z = tmp;
    break;
  case OP_PAT_ID:
    strcpy(tmp_header->unit_number, buf);
    tmp_header->unit_number[19] = '\0';
    strReplace(tmp_header->unit_number, PLAN_MIN(oplen, 18), '_');
    if (debug) 
      cout<<"id: "<<tmp_header->unit_number<<endl;
    break;
  case OP_PAT_NAME:
    strcpy(tmp_header->patient_name, buf);
    if (debug) 
      cout<<"name: "<< tmp_header->patient_name<<endl;
    break;
  case OP_DATE:
    for(i=0;i<4;i++)
      temp[i]=buf[i];
    tmp_header->date.year  = atoi(temp);
    temp[2]=0;
    temp[3]=0;			
    if (buf[4] == '.') {
      temp[0]=buf[5];
      temp[1]=buf[6];				
      tmp_header->date.month = atoi(temp);
      temp[0]=buf[8];
      temp[1]=buf[9];				
      tmp_header->date.day   = atoi(temp);
    }
    else {
      temp[0]=buf[4];
      temp[1]=buf[5];				
      tmp_header->date.month = atoi(temp);
      temp[0]=buf[6];
      temp[1]=buf[7];				
      tmp_header->date.day   = atoi(temp);
    }
    if (debug) 
      cout<<"date: "<<tmp_header->date.month<<"/"<<tmp_header->date.day<<"/"<<
	tmp_header->date.year<<endl;
    break;
  case OP_SCANNER:
    strcpy(tmp_header->machine_id, buf);
    if (debug)
      cout<<"scanner: "<<tmp_header->machine_id<<endl;
    break;
  case OP_STUDY_DESC:
    strcpy(tmp_header->comment, buf);
    if (debug) 
      cout<<"study desc: "<<tmp_header->comment<<endl;
    break;
  case OP_MODALITY:
    strcpy(tmp_header->modality, buf);
    if (debug) 
      cout<<"modality: "<<tmp_header->modality<<endl;
    break;
  case  OP_IM_TYPE:
    strcpy(tmp_header->image_type, buf);
    if (debug)
      cout<<"Image type: "<<tmp_header->image_type<<endl;
    break;
  case OP_ROWS:
    tmp_header->x_dim = get_short((short*)buf);
    if (debug) 
      cout<<"x_dim: "<<tmp_header->x_dim<<endl;
    break;
  case OP_COLS:
    tmp_header->y_dim = get_short((short*)buf);
    if (debug) 
      cout<<"y_dim: "<<tmp_header->y_dim<<endl;
    break;
  case OP_PIXEL_SIZE:
    tmp_header->pixel_size = atof(buf);
    if (debug) 
      cout<<"pixel size: "<<tmp_header->pixel_size<<endl;
    break;		
  case OP_PIXEL_ALLOC:
    bits_alloc = get_short((short*)buf);
    if (debug) 
      cout<<"bits allocated: "<<bits_alloc <<endl;
    break;		
  case OP_PIXEL_STORED:
    bits_stores = get_short((short*)buf);
    if (debug) 
      cout<<"bits stored: "<<bits_stores <<endl;
    break;		
  case OP_PIXEL_HIGHBIT:
    bits_highbit = get_short((short*)buf);
    if (debug) 
      cout<<"high bit (zero rel): "<<bits_highbit <<endl;
    break;		
  case OP_PIXEL_MIN:
    tmp_header->min = get_short((short*)buf);
    if (debug) 
      cout<<"min: "<<tmp_header->min<<endl;
    break;
  case OP_PIXEL_MAX:
    tmp_header->max = get_short((short*)buf);
    if (debug) 
      cout<<"max: "<<tmp_header->max<<endl;
    break;
  case OP_IM_NUM:
    z_num = atoi(buf);
    if (debug) 
      cout<<"im_num: "<<z_num<<endl;
    break;
  case OP_AQ_NUM:
    aq_num = atoi(buf);
    if (debug) 
      cout<<"aq_num: "<< aq_num<<endl;
    break;
  case OP_ORIENTATION:
    switch (buf[0]) {
    case 'H' :
      tmp_header->str_orient[2] = 'S';
      break;
    case 'F' :
      tmp_header->str_orient[2] = 'I';
      break;
    }
    switch (buf[2]) {
    case 'S' :
      tmp_header->str_orient[0] = 'L';
      tmp_header->str_orient[1] = 'A';
      break;
    case 'P' :
      tmp_header->str_orient[0] = 'R';
      tmp_header->str_orient[1] = 'P';
      break;
    }
    tmp_header->str_orient[3] = '\0';
    if (debug) 
      cout<<"orientation: "<<tmp_header->str_orient<<endl;
    break;
  case  OP_STUDY_TIME:			
    sprintf(tmp_header->study_time,"%c%c:%c%c:%c%c",buf[0],buf[1],buf[2],buf[3],buf[4],buf[5]);
    if (debug) 
      cout<<"Study Time: "<<tmp_header->study_time<<endl;
    break;
  }

  return(0);
}



