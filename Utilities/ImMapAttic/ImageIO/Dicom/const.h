#ifndef __CONST_H__
#define __CONST_H__

#if defined (__STDC__) && (__sun) && !defined(__cplusplus)
/* sun cc in ansi compliant mode does not define the following consts */
const double M_E         = 2.718281828459045235360287471352;
const double M_LOG2E     = 1.442695040888963407359924681002;
const double M_LOG10E    = 0.434294481903251827651128918916;
const double M_LN2       = 0.693147180559945309417232121458;
const double M_LN10      = 2.302585092994045684017991454684;
const double M_PI        = 3.141592653589793238462643383276;
const double M_PI_2      = 1.570796326794896619231321691638;
const double M_PI_4      = 0.785398163397448309615660845819;
const double M_1_PI      = 0.318309886183790671537767526745;
const double M_2_PI      = 0.636619772367581343075535053490;
const double M_2_SQRTPI  = 1.128379167095512573896158903122;
const double M_SQRT2     = 1.414213562373095048801688724209;
const double M_SQRT1_2   = 0.707106781186547524400844362104;

#endif

const double SQRT2PI     = 2.506628274631000502415765284808;
const double M_1_SQRT2PI = 0.398942280401432677939946059934;
const double NEARLYZERO  = 1.0e-37;
const double VERYBIG     = 1.0e+37;
const double VERYSMALL   =-1.0e+37;

/* misc useful consts */
const int MAXLINELENGTH = 200;

#endif
