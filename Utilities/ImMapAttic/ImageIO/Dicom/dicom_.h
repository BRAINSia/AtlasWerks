#ifndef _DICOM_H_
#define _DICOM_H_

// This class define a DICOM image. 
// We derived this class in dicomIMAGE and we use this one

#define NIMAGES          3000      /* number of image files that can be examined at once */
#ifndef TRUE
#define TRUE (1)
#endif
#ifndef FALSE
#define FALSE (0)
#endif
#ifndef PI
#define PI (3.141592654)
#endif

#define PLAN_MIN(a,b)		(((a) < (b)) ? (a) : (b))
#define PLAN_MAX(a,b)		(((a) > (b)) ? (a) : (b))
#define PLAN_ABS(a)		(((a) < 0) ? -(a) : (a))
#define PLAN_FABS(a)		(((a) < 0.0) ? -(a) : (a))
#define PLAN_CLOSE_TO(a,b,t)	(PLAN_ABS((a) - (b)) <= (t))
#define PLAN_NITEMS(array)      (sizeof(array) / sizeof(*(array)))

/*      KEY              	  EEEEGGGG  E=element G=group */
#define OP_DATE		 	0x00200008
#define OP_MODALITY	 	0x00600008
#define OP_SCANNER	 	0x10900008
#define OP_STUDY_TIME	 	0x00300008
#define OP_PAT_NAME	 	0x00100010
#define OP_PAT_ID	 	0x00200010
#define OP_ORIENTATION	 	0x51000018
#define OP_AQ_NUM	 	0x00120020
#define OP_IM_NUM	 	0x00130020  /* within acquisition */
#define OP_OLD_ORIENT	 	0x00200020
#define OP_Z_POSITION	 	0x00320020  /* use the 3rd value, eg, x/y/z */
#define OP_Z_POSITION2	 	0x00300020
#define OP_Z_POSITION3	 	0x00500020  /* maybe not the best element for this? */
#define OP_ROWS		 	0x00100028
#define OP_COLS		 	0x00110028
#define OP_PIXEL_SIZE	 	0x00300028
#define OP_PIXEL_MIN	 	0x01040028  /* dropped in dicom3? */
#define OP_PIXEL_MAX	 	0x01050028  /* dropped in dicom3? */
#define OP_OLD_HEADER	 	0x10107003
#define OP_IM_TYPE	 	0x00080008  /* image type: multivalue field (see below) */
#define OP_PIXEL_DATA    	0x00107fe0  /* the actual pixels */
#define OP_STUDY_DESC	 	0x10300008
#define OP_PIXEL_ALLOC	 	0x01000028  /* bits allocated */
#define OP_PIXEL_STORED	 	0x01010028  /* bits stored */
#define OP_PIXEL_HIGHBIT 	0x01020028  /* high bit (zero rel) */
#define OP_SLICE_COUNT   	0x00080028  /* multiple scans per image */
#define RELSLICELOCATION 	0x10410020 
#define CURVEDIMENSIONS	 	0x00055000
#define CURVENUMBEROFPOINTS	0x00105000
#define CURVETYPEOFDATA	 	0x00205000
#define CURVEDESCRIPTION	0x00225000
#define CURVEDATAVALUEREPRESENTATION	0x01035000
#define CURVEDATA   	 	0x30005000  

//#define 	PLAN_BIG_ENDIAN
#define 	MAX_PRIVBUFLEN  (64*1024) /* no oplen in file should exceed */

typedef unsigned char BOOLEAN;

//static float pixsize;

class DICOM{
  public :
    DICOM();
  ~DICOM();
  void my_swab(char *, char *, int);
  void strReplace(char *, int, char);
  void expand_filename(char*);
  int read_int1(int, int *);
  int read_int2(int, int *);
  int get_short(short *);
  float get_float(float *);
  int get_fileList (char*);
  int guess_swap(int);
  int read_single_op(int, char *);
  protected :
    int	swap1;	/* 4-byte values:0x1: swap bytes  0x2: swap shorts */
  int	swap2;  /* 2-byte values:1: swap bytes  (may be packed in 4-byte values)*/
  char	*basename;		
  char 	*fileList;
  char 	**infile;
  int 	num_files;
  int	opcode,oplen;
};



#endif // DICOM_H

