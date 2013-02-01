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
#include <dicom_.h>
#define PLAN_BIG_ENDIAN

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

void DICOM::expand_filename(char* basename) 
{
  char 	cmd[1000], result[200000];
  int 	fdes;
  int 	nchars;
  char 	tmpfile[L_tmpnam + 1];
 
  tmpnam(tmpfile);
  //  mkstemp(tmpfile);

  sprintf(cmd, "echo %s > %s", basename, tmpfile);

  system(cmd);

  fdes = open(tmpfile, O_RDONLY,0);
  nchars = read(fdes, result, sizeof(result)-1);
	   
  close(fdes);
  remove(tmpfile);

  /* strip newline from end */
  if (result[nchars - 1] == '\n')
    result[nchars - 1] = '\0';
  result[nchars] = '\0';
		
  fileList = (char*)malloc(strlen(result)+1);
  strcpy(fileList, result);
}


/* expand wildcard or basename spec to a list of files */
/* Ret: 1: ok   0: err (out of mem) */

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


/* Guess byte-swap and word-swap. Based on static first bytes of ACR */
/* file or BIG_ENDIAN file order for DICOM file. */
/* Side effect: file is rewound */
/* Ret: -1 if err, else 0 */

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
    swap1 = 0; swap2 = 0;	/* big -> program */
#endif
  }

  return 0;
}  


int DICOM::read_single_op(int fd, char *buf)
{
  int	group, element;

  if (read_int1(fd, &opcode)) {
    cout<<"bad read opcode: "<<opcode<<endl;
    return(-2);		/* err */
  }

  group = opcode & 0xffff;
  element = opcode >> 16;

  read_int2(fd, &oplen);

  if (opcode == OP_PIXEL_DATA)
    return(-1);

  if(oplen>MAX_PRIVBUFLEN)
    {
      cout<<"bad read oplen: "<<oplen<<endl;
      return(-2);		/* err */
    }

  read(fd, buf, oplen); 
  buf[oplen] = 0;
 
  if ((group & 0xff00) == 0x5000)
    opcode &= 0xffffff00;

  return 0;
}
