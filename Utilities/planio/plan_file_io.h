
/* plan_file_io.h */

/*
 ---------------------------------------------------------------------------
 
   NAME
 	plan_file_io.h
 
   DESCRIPTION
	Defines regarding file IO in PLUNC.
 
   SEE ALSO
   	plan_config.h
	plan_term_io.h
	plan_strings.h
	plan_sys.h

 ---------------------------------------------------------------------------
*/

#include "plan_config.h"

/*
  Where are the macro definitions for open(2) and lseek(2) located?
*/

#ifdef PLAN_BSD4_3
#include <sys/file.h>
#endif

#ifdef PLAN_ULTRIX2_0
#include <sys/file.h>
#endif

#ifdef PLAN_STELLIX1_5
#include <fcntl.h>
#define L_SET (0)
#define L_INCR (1)
#define L_XTND (2)
#endif

#ifdef PLAN_SOLARIS2_4
#include <sys/file.h>
#include <unistd.h>
#include <fcntl.h>
#endif

#ifdef PLAN_DARWIN
#include <sys/file.h>
#include <unistd.h>
#include <fcntl.h>
#endif

#ifdef PLAN_IRIX
#include <sys/file.h>
#include <fcntl.h>
#include <unistd.h>
#endif

#ifdef PLAN_LINUX
#include <sys/file.h>
#include <unistd.h>
#endif

#ifdef PLAN_OSF1_3
#include <unistd.h>
#endif

#ifdef PLAN_WINNT
#include <io.h>
#include <fcntl.h>
#endif


