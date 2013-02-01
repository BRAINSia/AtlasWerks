
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <ctype.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include "dicom_.h"
#include "BinaryIO.h"
#include <iostream>
#include <fstream>

#ifndef WIN32
#include <unistd.h>

#else

#include <io.h>

#endif

//#define PLAN_BIG_ENDIAN

// This class define a DICOM image. 
// We derived this class in dicomIMAGE and we use this one

DICOM::DICOM()
{
  num_files	= 0;
  swap1 	= 0;	
  swap2 	= FALSE;
  fileList 	= NULL;
  infile  	= NULL;
  basename	= NULL;

}

DICOM::~DICOM()
{
}

int DICOM::read_int1(int fd, int *ptr)
{   
  int	len;
  short	temp;
  short	*sptr;
  char	ctemp;
  char	*cptr;
  len 	= read(fd, ptr, 4);

  if (len < 4) 	
    return(-1);	/* err */

  if (swap1 & 1) {
    sptr = (short *)ptr;
    temp = *sptr;
    *sptr = *(sptr+1);
    *(sptr+1) = temp;
  }
  if (swap1 & 2) {
    cptr = (char *)ptr;
    ctemp = *cptr;
    *cptr = *(cptr+1);
    *(cptr+1) = ctemp;

    cptr += 2;
    ctemp = *cptr;
    *cptr = *(cptr+1);
    *(cptr+1) = ctemp;
  }
  return(0);
}


int DICOM::read_int2(int fd, int *ptr)
{   
  int	len;
  short	temp;
  short	*sptr;
  char	ctemp;
  char	*cptr;
  len	= read(fd, ptr, 4);

  if (len < 4) 
    return(-1);

  if (swap2 == 1) {
    sptr = (short *)ptr;	/* swap 2 16-bit values */
    temp = *sptr;
    *sptr = *(sptr+1);
    *(sptr+1) = temp;

    cptr = (char *)ptr;	/* swap first 2 8-bit values */
    ctemp = *cptr;
    *cptr = *(cptr+1);
    *(cptr+1) = ctemp;

    cptr += 2;		/* swap last 2 8-bit values */
    ctemp = *cptr;
    *cptr = *(cptr+1);
    *(cptr+1) = ctemp;
  }
  return(0);
}


void DICOM::my_swab(char *src, char *dest, int num)
{   
  int	i;
  char	temp;

  for (i = 0; i < num; i += 2) {
    temp = src[i];
    dest[i] = src[i+1];
    dest[i+1] = temp;
  }
}


int DICOM::get_short(short *ptr)
{   
  int	val;

  if (swap2 == 1) 
    my_swab((char*)ptr, (char*)ptr, 2);
  val = *ptr;
  return(val);
}

float DICOM::get_float(float *ptr)
{
  float val;
  val = *ptr;
  
  if (swap1 == 3) {
    my_swab((char*)ptr, (char*)ptr, 4);
    val = *ptr;
  }
  else if (swap1 == 0) {
    char tmp[4];
    tmp[0] = ((char*)ptr)[2];
    tmp[1] = ((char*)ptr)[3];
    tmp[2] = ((char*)ptr)[0];
    tmp[3] = ((char*)ptr)[1];
    val = *((float *)tmp);
  }
  else {
  }

  return(val);
}


void DICOM::strReplace(char *str_p, int nchars, char to)
{
  int ch;
  for (ch=0; ch < nchars; ch++)
    {
      if (str_p[ch] == '\0')	/* end of conversion */
	break;
      if (str_p[ch] == '-')	/* denotes a negative number; skip */
	continue;
      if (str_p[ch] == '+')	/* denotes a positive number; skip */
	continue;
      if (str_p[ch] == '.')	/* decimal point; skip */
	continue;
      if (str_p[ch] == ',')	/* Euro decimal point; skip */
	continue;
      if (isspace(str_p[ch]) || ispunct(str_p[ch]) || iscntrl(str_p[ch]))
	str_p[ch] = to;
    }
}	

int DICOM::get_fileList(char* basename) 
{
  char 	cmd[1000];
  // unused var//char result[200000];
  // unused var//int 	fdes;
  // unused var//int 	nchars;
  char 	tmpfile[1024];
  int fileCount=0;

 
	infile = NULL;
#ifdef WIN32
	char *p = basename;
	while (*p){
	if (*p == '/') *p = '\\';
		*p++;
	}
     strcpy(tmpfile, "filelist.dispimage.txt");
    // 'dir /b /s' is bare-format (no dir header info) and subdir's
    // (forces dirname prepended to filename)
	sprintf(cmd, "dir /b /s /O:N \"%s\" > %s\n", basename, tmpfile);
	printf("%s\n",cmd);
#else
  tmpnam(tmpfile);
  //  mkstemp(tmpfile);
  sprintf(cmd, "/bin/ls -1 %s > %s", basename, tmpfile);
#endif

  system(cmd);
  std::ifstream fileListInput(tmpfile);
  if (fileListInput.fail())
	{
		return 0;
	}
	
    while (!fileListInput.eof())
    {
	  infile = (char **)realloc((char*)infile, (fileCount+1)*sizeof(char *));
      infile[fileCount] = (char*) malloc(1000);
	  fileListInput.getline(infile[fileCount],1000);
      if (fileListInput.gcount() > 0) fileCount++;
    }
	fileListInput.close();
 
	   
//remove(tmpfile);

  /* strip newline from end */
	num_files = fileCount;
 return fileCount;
}


/* expand wildcard or basename spec to a list of files */
/* Ret: 1: ok   0: err (out of mem) */
#if 0
int DICOM::get_fileList (char* basename)
{
  int 	len;

  expand_filename(basename); 
		
  /* copy space-delimited filenames into infile array */
  while ((len = strcspn(fileList, " ")) > 4)
    {
      infile = (char **)realloc((char*)infile, (num_files+1)*sizeof(char *));	
      infile[num_files] = (fileList);				
      num_files++;
      /* mark end of this filename */
      fileList[len] = '\0';
      fileList += len + 1;
    }

  return num_files;
}
#endif
//
// this is my attempt to make this function work
// why use swap1 and swap2 and not 
// bool _doWordSwap, _doByteSwap
// i dont know
//
// im not sure this takes care of all the cases
//

int
DICOM
::guess_swap(int fd)
{   
  int buffer;
  int bytesRead = read(fd, &buffer, 4);
  if (bytesRead != 4) 
    {
      // couldnt read 4 bytes
      return(-1);
    }

  // try to decide based on first 4 bytes of file
  if (buffer == 0x00000008)
    {
      // no swap
      swap1 = 0;
      swap2 = 0;
    }
  else if (buffer == 0x00080000)
    {
      // do word swap
      swap1 = 1;
      swap2 = 0;
    }
  else if (buffer == 0x00000800)
    {
      // do byte swap (how do these numbers work?)
      swap1 = 2;
      swap2 = 0;
    }
  else if (buffer == 0x08000000)
    {
      // do word and byte swap (again im confused...)
      swap1 = 3;
      swap2 = 1;
    }
  else
    {
      // decide based on machine endianness
      // these numbers just seem to work
      if (BinaryIO::computeHostEndian() == BinaryIO::big_endian)
	{
	  swap1 = 3;
	  swap2 = 1;
	}
      else
	{
	  swap1 = 0;
	  swap2 = 0;
	}
    }  

  // No return codes have been defined
  return( 0 );
}

/* Guess byte-swap and word-swap. Based on static first bytes of ACR */
/* file or BIG_ENDIAN file order for DICOM file. */
/* Side effect: file is rewound */
/* Ret: -1 if err, else 0 */

#ifdef NOT_DEFINED
int DICOM::guess_swap(int fd)			/* taste this open file */
{   
  int     buf;
  int     p;			/* path through code */
  int 	len;

  len = read(fd, &buf, 4);
  if (len < 4) 
    return(-1);		/* err */

  /* compare first opcode in file to constants. If it is the static */
  /* ACR magic number, accept it, else assume it's a DICOM file and send */
  /* back swapping appropriate to PLAN_BIG_ENDIAN */


  if (buf == 0x00000008) {swap1 = 0; swap2 = 0;p=1;} /* no swap */
  else if (buf == 0x00080000) {swap1 = 1; swap2 = 0;p=2;} /* do word-swap */
  else if (buf == 0x00000800) {swap1 = 2; swap2 = 0;p=3;} /* do byte-swap */
  else if (buf == 0x08000000) {swap1 = 3; swap2 = 1;p=4;} /* do word-swap & byte-swap */
  else {
    /* DICOM order is fixed; only variable is CPU arch. "program" */
    /* order is based on group/element #ifdef's above (eg: OP_DATE) */
#ifdef PLAN_BIG_ENDIAN
    p=5;
    swap1 = 3; swap2 = 1;	/* big -> program */
#else
    p=6;
    swap1 = 1; swap2 = 1;	/* little -> program */
    //swap1 = 0; swap2 = 0;	/* big -> program */
    swap1 = 1; swap2 = 0;	/* big -> program */
#endif
  }
  std::cout << "swap1 = " << swap1 << ", swap2 = " << swap2 << std::endl;

  return 0;
}  
#endif

int DICOM::read_single_op(int fd, char *buf)
{
  int	group, element;

  if (read_int1(fd, &opcode)) {
	  std::cout<<"bad read opcode: "<<opcode<<std::endl;
    return(-2);		/* err */
  }

  group = opcode & 0xffff;
  element = opcode >> 16;

  read_int2(fd, &oplen);

  if (opcode == OP_PIXEL_DATA)
    return(-1);

  if(oplen>MAX_PRIVBUFLEN)
    {
	  std::cout<<"bad read oplen: "<<oplen<<std::endl;
      return(-2);		/* err */
    }

  read(fd, buf, oplen); 
  buf[oplen] = 0;
 
  if ((group & 0xff00) == 0x5000)
    opcode &= 0xffffff00;

  return 0;
}
