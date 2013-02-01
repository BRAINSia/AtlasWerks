
/* plan_config.h */

/*
 ---------------------------------------------------------------------------
 
   NAME
 	plan_config.h
 
   DESCRIPTION
	This file contains configuration information for the
	PLUNC system.

	In an effort to avoid nasty dependence tangles, please include
	plan_config.h in other plan_foo.h files, but DO NOT include
	other plan_foo.h files in each other.  Plan_config.h sets the
	state and the others react.  This strategy may lead to a
	proliferation of header files in some heavily inclusive
	applications but that is preferable to a lot of superfluous
	implicit includes in simple applications.
 
   SEE ALSO
   	plan_file_io.h
	plan_term_io.h
	plan_strings.h
	plan_sys.h

 ---------------------------------------------------------------------------
*/

#ifndef _PLAN_CONFIG
#define _PLAN_CONFIG
/*
  First, what operating system are we dealing with?

  Currently supported choices are:
  	PLAN_BSD4_3
	PLAN_ULTRIX2_0
	PLAN_STELLIX1_5
	PLAN_SOLARIS2_4
	PLAN_SOLARIS2_6
	PLAN_SUNOS4_0
	PLAN_IRIX
*/

#endif

/* A bit of a hack, but it seems to work */
#ifdef PLAN_AIX
#define PLAN_ULTRIX2_0
#endif

#ifdef PLAN_HPUX
#define PLAN_IRIX
#endif

#ifdef PLAN_OSF1_3
#define PLAN_ULTRIX2_0
#endif

#ifdef PLAN_MACHTEN_4_0_1
#define PLAN_SUNOS4_0 
#endif  

#ifdef PLAN_SOLARIS2_6
#define PLAN_SOLARIS2_4
#endif

