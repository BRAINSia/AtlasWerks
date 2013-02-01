#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <string.h>
#include <math.h>

#ifndef WIN32
#include <unistd.h>
#else
#include <io.h>
#endif

#include <ctype.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include "dicomContour.h"

int 	curve_dim;
int	nb_points;
int	z;
int 	dimx,dimy;


DICOMcontour::DICOMcontour()
{
  ana	 	= NULL;
  image 	= NULL;
  cur_anastruct	= 0;
  num_anastruct	= 0;
}

DICOMcontour::DICOMcontour(char *_basename)
{
  ana	 	= NULL;
  image 	= NULL;
  cur_anastruct	= 0;
  num_anastruct	= 0;
  basename 	= _basename;
}

DICOMcontour::DICOMcontour(DICOMimage *ref)
{
  ana	 	= NULL;
  image 	= ref;
  cur_anastruct	= 0;
  num_anastruct	= 0;
  infile	= ref->get_infile();
  num_files	= ref->get_num_files();
}

DICOMcontour::~DICOMcontour()
{
}

int DICOMcontour::OpenDICOMFile()
{
  int 	loop;
  int 	fd;
  // unused var // int 	len;

  if(!image) 
    {
      if(!basename)
	return 0;
      get_fileList(basename);
    }

  if(!infile)
    return 0;
  z = 0;

  for (loop = 0; loop < num_files; loop++)
    {
      if (loop == num_files)
	break;

      if(!infile[loop])
	continue;

#ifdef WIN32
      if ((fd = open(infile[loop], O_RDONLY|O_BINARY, 0)) < 0)
	continue;
#else
      if ((fd = open(infile[loop], O_RDONLY, 0)) < 0)
	continue;
#endif
	
      if (guess_swap(fd)<0)		/* set swap1 & swap2 */
	{
	  close(fd);
	  continue;
	}

      lseek(fd, 0, SEEK_SET);	/* rewind */

      for (;;)
	if (read_op(fd,loop)) 
	  break;
      z++;

      close(fd);
    }

  getMinMax();

  if(!patient_name[0])
    strcpy(patient_name,"No Name");
#ifdef NOTDEFINED
  for(int i=0;i<num_anastruct;i++)	
    {
      len = strcspn(ana[i].label, "   ");
      ana[i].label[len] = '\0';
    }
#endif
  return(num_anastruct);
}


int DICOMcontour::get_current_anastruct(char* name)
{
  for(int i=0;i<num_anastruct;i++){
    if(!strcmp((char*)ana[i].label,name)){
      ana[i].contour_count++;
      return i;
    }
  }
  num_anastruct++;
  ana = (ANASTRUCT*)realloc(ana,num_anastruct*sizeof(ANASTRUCT));
  strcpy((char*)ana[num_anastruct - 1].label,name);
  ana[num_anastruct-1].contours = NULL;
  ana[num_anastruct-1].contour_count = 1;
  return(num_anastruct-1);
}


int DICOMcontour::read_op(int fd,int num)
{   
  char	buf[MAX_PRIVBUFLEN];
  // unused var//char 	temp[4];
  int 	size;
  int 	w;
  float *f;
  // unused var//char 	*wc;
  float	x,y;
  float maxx,maxy,minx,miny;
  float tmp;
  char	tmpc;
	
  maxx = 100000;
  maxy = 100000;
  minx = -100000;
  miny = -100000;

  if(read_single_op(fd, buf))
    return -1;

  switch (opcode) {
  case OP_Z_POSITION:
    sscanf(buf,"%f%c%f%c%f",&tmp,&tmpc,&tmp,&tmpc,&tmp);
    //zpos = ABS(tmp);

	zpos = tmp/10;

    break;
  case OP_PIXEL_SIZE:
    pixel_size = atof(buf) * 0.1;
    break;
  case OP_ROWS:
    dimx = get_short((short*)buf);
    break;
  case OP_COLS:
    dimy = get_short((short*)buf);
    break;
  case OP_PAT_NAME:
    strcpy(patient_name, buf);
    break;
  case  CURVEDIMENSIONS:
    curve_dim = get_short((short*)buf);
    break;
  case  CURVENUMBEROFPOINTS:
    if(curve_dim!=3)
      nb_points = get_short((short*)buf);
    break;
  case  CURVEDESCRIPTION:
    if(curve_dim!=3)
      {
	xCenter = pixel_size * dimx/2;
	yCenter = pixel_size * dimy;
	cur_anastruct = get_current_anastruct(buf);
	ana[cur_anastruct].contours =  
	  (CONTOUR*)realloc((void*)ana[cur_anastruct].contours,
			    (ana[cur_anastruct].contour_count)*sizeof(CONTOUR));	
	ana[cur_anastruct].contours[ana[cur_anastruct].contour_count-1].vertex_count = nb_points;
	ana[cur_anastruct].contours[ana[cur_anastruct].contour_count-1].x = new float[nb_points];
	ana[cur_anastruct].contours[ana[cur_anastruct].contour_count-1].y = new float[nb_points];
	ana[cur_anastruct].contours[ana[cur_anastruct].contour_count-1].z = zpos;
	ana[cur_anastruct].contours[ana[cur_anastruct].contour_count-1].max.z = zpos  + 0.25;
	ana[cur_anastruct].contours[ana[cur_anastruct].contour_count-1].min.z = zpos  - 0.25;
	ana[cur_anastruct].contours[ana[cur_anastruct].contour_count-1].density = 1.0;
	ana[cur_anastruct].contours[ana[cur_anastruct].contour_count-1].slice_number = z;
      }
    break;
  case  CURVEDATA:
    if(curve_dim!=3)
      {
	std::cerr <<"dicomContour loading: " <<ana[cur_anastruct].label<<std::endl;
	// cout<<"swap1 "<<swap1<<endl;
	// cout<<"swap2 "<<swap2<<endl;
	
	f = (float*)(buf);
	size = oplen / sizeof(U32) / 2;

	std::cerr << "\tcontour number: " 
		  << ana[cur_anastruct].contour_count << std::endl;
	std::cerr << "\tnum vertices: " << size << std::endl;


	for (w = 0; w < size; w++, f += 2)
	  {  
	    x = get_float(f);
	    y = get_float(f+1);

//	    x = x*pixel_size - xCenter;
//
//	    y = yCenter - y*pixel_size;

	    ana[cur_anastruct].contours[ana[cur_anastruct].contour_count - 1].x[w] = x;
	    ana[cur_anastruct].contours[ana[cur_anastruct].contour_count - 1].y[w] = y;
	    
	    
	    //cout<<"x,y,w : "<<x<<" "<< y<<" "<< w<<endl;
	    
	    
	    if((x>maxx)||(!w))
	      maxx = x;
	    if((x<minx)||(!w))
	      minx = x;
	    if((y>maxy)||(!w))
	      maxy = y;
	    if((y<miny)||(!w))
	      miny = y;
	  }
	
	
	ana[cur_anastruct].contours[ana[cur_anastruct].contour_count - 1].max.x = maxx;
	ana[cur_anastruct].contours[ana[cur_anastruct].contour_count - 1].min.x = minx;
	ana[cur_anastruct].contours[ana[cur_anastruct].contour_count - 1].max.y = maxy;
	ana[cur_anastruct].contours[ana[cur_anastruct].contour_count - 1].min.y = miny;
      }
    break;

  }

  return(0);
}

void DICOMcontour::getMinMax()
{
  for(int i = 0; i < num_anastruct; i++)	
    {
      if (ana[i].contour_count == 0)
	{
	  ana[i].max.x = 0;
	  ana[i].max.y = 0;
	  ana[i].max.z = 0;
	  ana[i].min.x = 0;
	  ana[i].min.y = 0;
	  ana[i].min.z = 0;	  
	}
      else
	{
	  float maxX = ana[i].contours[0].max.x;
	  float minX = ana[i].contours[0].min.x;
	  float maxY = ana[i].contours[0].max.y;
	  float minY = ana[i].contours[0].min.y;
	  float maxZ = ana[i].contours[0].max.z;
	  float minZ = ana[i].contours[0].min.z;

	  for(int j = 1; j < ana[i].contour_count; j++)
	    {
	      if(ana[i].contours[j].max.x > maxX)
		{
		  maxX = ana[i].contours[j].max.x;
		}
	      if(ana[i].contours[j].max.y > maxY)
		{
		  maxY = ana[i].contours[j].max.y;
		}
	      if(ana[i].contours[j].max.z > maxZ)
		{
		  maxZ = ana[i].contours[j].max.z;
		}
	      if(ana[i].contours[j].min.x < minX)
		{
		  minX = ana[i].contours[j].min.x;
		}
	      if(ana[i].contours[j].min.y < minY)
		{
		  minY = ana[i].contours[j].min.y;
		}
	      if(ana[i].contours[j].min.z < minZ)
		{
		  minZ = ana[i].contours[j].min.z;
		}
	    }
	  ana[i].max.x = maxX;
	  ana[i].max.y = maxY;
	  ana[i].max.z = maxZ;
	  ana[i].min.x = minX;
	  ana[i].min.y = minY;
	  ana[i].min.z = minZ;
	}
    }
}
