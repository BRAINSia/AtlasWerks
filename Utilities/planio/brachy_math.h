
/* brachy_math.h */

#include <math.h>

#define TERMS 5
#define POLY(x, a, b, c, d) ((((d) * (x) + (c)) * (x) + (b)) * (x) + (a))
#define DIST(a, b) ((float) sqrt((double) ((a) * (a) + (b) * (b))))
#define SIN(x) (((x) < 0.0001) ? (x) : (float) sin((double) (x)))
/*
#define COS(x) (((x) > (PI / 2.0 - 0.0001)) ? (0.0001) : (float) cos((double) (x)))
*/
#define COS(x) (((x) > (1.5706963)) ? (0.0001) : (float) cos((double) (x)))
#define LOG(x) ((float) log((double) (x)))
/*
#define COS_88 ((float) cos((double) (88.0 * PI / 180.0)))
*/
#define COS_88 (0.0348995)
#define SQRT(x) ((float) sqrt((double) (x)))

#define ACOS(x) \
    ((x < -1.0) ? 3.141592 : (x > 1.0) ? 0.0 : acos((double)(x)))

#define ATAN(x) ((float) atan((double) (x)))
#define ABS(x) ((float) fabs((double) (x)))
#ifdef MIN
#undef MIN
#endif
#define MIN(x, y) ((x) < (y) ? (x) : (y))
