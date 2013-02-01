/* plan_sys.h */

/*
 ---------------------------------------------------------------------------
 
   NAME
 	plan_sys.h
 
   DESCRIPTION
	Defines regarding miscellaneous OS-dependent stuff.
 
   SEE ALSO
   	plan_config.h
	plan_file_io.h
	plan_term_io.h
	plan_strings.h

 ---------------------------------------------------------------------------
*/

#include "plan_config.h"

/*
  If the OS in question does not provide the library function
  getopt(3), you may ask for it here by defining PLAN_NEED_GETOPT.
  A public-domain version of getopt will be included in libplan.
*/

/*
#ifdef PLAN_OS_FOO
#define PLAN_NEED_GETOPT
#endif
*/

/*
  If the OS in question does not provide the bstring library,
  including functions bcopy(3), bzero(3), and bcmp(3), you may ask for
  it here by defining PLAN_NEED_BSTRING.  In that case,
  imitations of bcopy, bzero, and bcmp will be included in libplan.
  If you have the SYSV (and POSIX?) routines memcpy, memcmp, and
  memset, define PLAN_NEED_BSTRING and PLAN_HAVE_MEMCPY for more efficient
  imitations.  The include statement for memory.h should also be
  provided here.
*/

#ifdef PLAN_STELLIX1_5
#define PLAN_NEED_BSTRING
#define PLAN_HAVE_MEMCPY
#include <memory.h>
#endif

#ifdef PLAN_SOLARIS2_4
#define PLAN_NEED_BSTRING
#define PLAN_HAVE_MEMCPY
#include <memory.h>
#endif

#ifdef PLAN_DARWIN
#define PLAN_NEED_BSTRING
#define PLAN_HAVE_MEMCPY
#include <memory.h>
#endif

/*
  How do we change run priority?  Some number-crunching daemons want
  to do this.  Berkeley provides setpriority, SYSV has nice.  Please
  provide a macro here to set priority given an absolute nice value
  between -20 and 20.
*/

#ifdef PLAN_BSD4_3
#define PLAN_SET_PRIORITY(p) setpriority (0, getpid (), (p))
#endif

#ifdef PLAN_ULTRIX2_0
#define PLAN_SET_PRIORITY(p) setpriority (0, getpid (), (p))
#endif

#ifdef PLAN_STELLIX1_5
#define PLAN_SET_PRIORITY(p) nice ((p) - nice (0))
#endif

#ifdef PLAN_SOLARIS2_4
#define PLAN_SET_PRIORITY(p) nice ((p) - nice (0))
#endif

#ifdef PLAN_DARWIN
#define PLAN_SET_PRIORITY(p) nice ((p) - nice (0))
#endif

#ifdef PLAN_IRIX
#define PLAN_SET_PRIORITY(p) setpriority (0, getpid (), (p))
#endif

#ifdef PLAN_LINUX
#define PLAN_SET_PRIORITY(p) setpriority (0, getpid (), (p))
#endif

/*
  Byte order.  Are bytes stored MSB first (big-endian) or LSB first
  (little-endian)?  Sun is PLAN_BIG_ENDIAN, VAX is not,
  for instance.
*/

#ifdef PLAN_STELLIX1_5
#define PLAN_BIG_ENDIAN
#endif

#ifdef PLAN_SOLARIS2_4
#define PLAN_BIG_ENDIAN
#endif

#ifdef PLAN_DARWIN
#define PLAN_BIG_ENDIAN
#endif

#ifdef PLAN_SUNOS4_0
#define PLAN_BIG_ENDIAN
#endif

#ifdef PLAN_IRIX
#define PLAN_BIG_ENDIAN
#endif

#ifdef PLAN_AIX
#define PLAN_BIG_ENDIAN
#endif

/*
  Environment variable that contains the user name.  BSD uses USER,
  SYSV appears to use LOGNAME.
*/

#ifdef PLAN_BSD4_3
#define PLAN_USER "USER"
#endif

#ifdef PLAN_ULTRIX2_0
#define PLAN_USER "USER"
#endif

#ifdef PLAN_STELLIX1_5
#define PLAN_USER "LOGNAME"
#endif

#ifdef PLAN_SOLARIS2_4
#define PLAN_USER "USER"
#endif

#ifdef PLAN_DARWIN
#define PLAN_USER "USER"
#endif

#ifdef PLAN_IRIX
#define PLAN_USER "USER"
#endif

#ifdef PLAN_LINUX
#define PLAN_USER "USER"
#endif

#ifdef PLAN_WINNT
#define PLAN_USER "USER"
#endif


/*
  Does gettimeofday have 2 params (BSD) or 1 (SYSV)?
*/

#ifdef PLAN_BSD4_3
#define GETTIMEOFDAY(T1,T2) gettimeofday((T1),(T2))
#endif

#ifdef PLAN_ULTRIX2_0
#define GETTIMEOFDAY(T1,T2) gettimeofday((T1),(T2))
#endif

#ifdef PLAN_STELLIX1_5
#define GETTIMEOFDAY(T1,T2) gettimeofday(T1)
#endif

/* 2.6 does not seem to be upward compatible with 2.4 for this */
#ifdef PLAN_SOLARIS2_4
#  ifdef PLAN_SOLARIS2_6
#    define GETTIMEOFDAY(T1,T2) gettimeofday(T1,(void *)(T2))
#  else
#    define GETTIMEOFDAY(T1,T2) gettimeofday(T1)
#  endif
#endif

#ifdef PLAN_SOLARIS2_4
#define GETTIMEOFDAY(T1,T2) gettimeofday((T1),(T2))
#endif

#ifdef PLAN_DARWIN
#define GETTIMEOFDAY(T1,T2) gettimeofday((T1),(T2))
#endif

#ifdef PLAN_IRIX
#define GETTIMEOFDAY(T1,T2) gettimeofday(T1)
#endif

#ifdef PLAN_LINUX
#define GETTIMEOFDAY(T1,T2) gettimeofday((T1),(T2))
#endif

