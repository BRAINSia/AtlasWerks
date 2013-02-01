
/* plan_strings.h */


#include "plan_config.h"

/*
  There are two commonly-available variations on the strings library.
*/

#ifdef PLAN_BSD4_3
#include <strings.h>
#endif

#ifdef PLAN_ULTRIX2_0
#include <strings.h>
#endif

#ifdef PLAN_STELLIX1_5
#include <string.h>
#include <sys/types.h>
#define index strchr
#define rindex strrchr
#endif

#ifdef PLAN_SOLARIS2_4
#include <string.h>
#define index strchr
#define rindex strrchr
#endif

#ifdef PLAN_DARWIN
#include <string.h>
#define index strchr
#define rindex strrchr
#endif

#ifdef PLAN_IRIX
#include <strings.h>
#endif

#ifdef PLAN_LINUX
#include <strings.h>
#include <string.h>
#endif

#ifdef PLAN_WINNT
#include <string.h>
#endif
