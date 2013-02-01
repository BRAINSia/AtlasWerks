/* created by gaudenz danuser
   for use in his libraries & programs
*/

#ifndef __MACROS_H__
#define __MACROS_H__

#include <math.h>

#include "const.h"

/*******************************************************
 general mathematical definitions
 *******************************************************/
#define DIGIT2STRING(x) x+48

#define Grad2Rad(x) x*PI/200.0
#define Rad2Grad(x) x*200.0/PI
#define Deg2Rad(x) x*PI/180.0
#define Rad2Deg(x) x*180.0/PI

#define DROUND(x) (int)floor(x+(double)0.5)
#define DNONINTPART(x) fabs((double)DROUND(x)-x)

#ifdef __sun
#define FROUND(x) (int)floor((double)x+(double)0.5)
#elif __sgi
#define FROUND(x) (int)ffloor(x+(float)0.5)
#endif /* __sgi */

#define FNONINTPART(x) fabsf((float)FROUND(x)-x)

#define ABS(x) (((x) > 0) ? (x) : (-1*(x)))
#define CLIP(x) (((x) > 0) ? (x) : 0.0)
#define IROUND(x) (((x) > 0) ? (int)((x) + 0.5) : (int)((x) - 0.5))
//#define MAX(x,y) ((x) > (y) ? (x) : (y))
//#define MIN(x,y) ((x) > (y) ? (y) : (x))
#define SGN(a)   (((a)<0) ? -1 : 1)
#define SQR(x) ((x)*(x)) /* square of a number */
#define XOR(a,b) ((a||b)&&(!(a&&b)))
#define XY(x,y,nx) ((x) + (y)*(nx))

#define FATAN(x)((float) atan ((double) x))
#define FATAN2(x,y) ((float) atan2((double)(x),(double)(y)))
#define FCEIL(x) ((float) ceil((double)x))
#define FCOS(x) ((float) cos ((double)(x)))
#define FEXP(x) ((float) exp ((double)(x)))
#define FERF(x) ((float) erf ((double)(x)))
#define FERFC(x) ((float) erfc ((double)(x)))
#define FFABS(x) ((float) fabs ((double) (x)))
#define FFLOOR(x) ((float) floor((double)x))
#define FLOG(x) ((float) log ((double)(x)))
#define FMODF(x,y) ((float) fmod((double)(x),(double)(y)))
#define FPOW(x,y) ((float) pow ((double) (x), (double) (y)))
#define FSIN(x) ((float) sin ((double)(x)))
#define FSQRT(x) ((float) sqrt((double)(x)))
#define FTAN(x) ((float) tan ((double) x))

/*******************************************************
 emulation of special functions
 *******************************************************/
#define ALLOC(p,s,t) ((p==NULL) ? (p=(t*)malloc(s)) :(p=(t*)realloc(p,s)))
/* special allocation which checks the existance of the pointer p;
   dependant on the result the macro works with malloc or with
   realloc. This macro emulates the realloc function on all non
   solaris systems. Solaris makes the NULL pointer check itself */

#endif
