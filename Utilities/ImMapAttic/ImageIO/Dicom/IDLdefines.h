#ifndef _IDLdefines_h_
#define _IDLdefines_h_

enum ItXECode { ItXSuccess, 
		ItXError, 
		ItXInvalidFile,
		ItXShortRead,
		ItXShortWrite,
		ItXInvalidImage,
		ItXInvalidDataType,
		ItXInvalidImageType,
		ItXInvalidDimensions,
		ItXNoMemory,
		ItXInvalidParms,
	      };
#define MAX(a,b) ((a) > (b)? (a) : (b))
#define MIN(a,b) ((a) < (b)? (a) : (b))
#define MIN3(a,b,c) (MIN(MIN((a),(b)),(c)))
#define MAX3(a,b,c) (MAX(MAX((a),(b)),(c)))

// Check for appropriate location of Endian.h

// Solaris hack
//#ifdef GNU_COMPILER
//#include <endian.h>
//#else
//#include <sys/endian.h>
/*typedef unsigned char u_char;*/
//#endif
typedef unsigned char u_char;

#ifndef BYTE_ORDER
#define BYTE_ORDER BIG_ENDIAN
#endif

#define big_endian_char(a) (a) // when swapping a char, nothing happens

// Check to see which byte order. If big endian, do nothing. else swap.

#if BYTE_ORDER==BIG_ENDIAN 

#define big_endian_int(a) (a)
#define big_endian_float(a) (a)
#define big_endian_short(a) (a)
#define big_endian_double(a) (a)

#else // Swap
//#define big_endian_int(a) ( ((a) << 24) | 
inline int big_endian_int(int a)
{ return 	(((a) << 24) | \
		  (((a) << 8) & 0x00ff0000) | \
		  (((a) >> 8) & 0x0000ff00) | \
		  ((unsigned long)(a) >>24));
}

// #define big_endian_short(a) ( ((a & 0xff) << 8) | ((unsigned short)(a) >> 8) )
inline short big_endian_short(short a) { return (((a & 0xff) << 8) | ((unsigned short)(a) >> 8)); }
inline float big_endian_float(float a) { 
  float b; 
  float c =a; 
  char *ptr1 = (char *)&b; 
  char *ptr2 = (char *)&c; 
  (ptr1[0]) = (ptr2[3]);  
  (ptr1[1]) = (ptr2[2]); 
  (ptr1[2]) = (ptr2[1]); 
  (ptr1[3]) = (ptr2[0]); 
  return(b); 
}

inline double big_endian_double(double a) { 
  double b; 
  double c =(a); 
  char *ptr1 = (char *)&b; 
  char *ptr2 = (char *)&c; 
  (ptr1[0]) = (ptr2[7]);  
  (ptr1[1]) = (ptr2[6]); 
  (ptr1[2]) = (ptr2[5]); 
  (ptr1[3]) = (ptr2[4]); 
  (ptr1[4]) = (ptr2[3]); 
  (ptr1[5]) = (ptr2[2]); 
  (ptr1[6]) = (ptr2[1]); 
  (ptr1[7]) = (ptr2[0]); 
  return(b); 
}
#endif //end of endian if

inline int swab_int(int a)
{ return 	(((a) << 24) | \
		  (((a) << 8) & 0x00ff0000) | \
		  (((a) >> 8) & 0x0000ff00) | \
		  ((unsigned long)(a) >>24));
}

// #define big_endian_short(a) ( ((a & 0xff) << 8) | ((unsigned short)(a) >> 8) )
inline short swab_short(short a) { return (((a & 0xff) << 8) | ((unsigned short)(a) >> 8)); }
inline float swab_float(float a) { 
  float b; 
  float c =a; 
  char *ptr1 = (char *)&b; 
  char *ptr2 = (char *)&c; 
  (ptr1[0]) = (ptr2[3]);  
  (ptr1[1]) = (ptr2[2]); 
  (ptr1[2]) = (ptr2[1]); 
  (ptr1[3]) = (ptr2[0]); 
  return(b); 
}

inline double swab_double(double a) { 
  double b; 
  double c =(a); 
  char *ptr1 = (char *)&b; 
  char *ptr2 = (char *)&c; 
  (ptr1[0]) = (ptr2[7]);  
  (ptr1[1]) = (ptr2[6]); 
  (ptr1[2]) = (ptr2[5]); 
  (ptr1[3]) = (ptr2[4]); 
  (ptr1[4]) = (ptr2[3]); 
  (ptr1[5]) = (ptr2[2]); 
  (ptr1[6]) = (ptr2[1]); 
  (ptr1[7]) = (ptr2[0]); 
  return(b); 
}


#endif
