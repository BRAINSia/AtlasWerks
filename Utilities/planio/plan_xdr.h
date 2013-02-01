
/* plan_xdr.h */

/*
 ---------------------------------------------------------------------------
 
   NAME
 	plan_xdr.h
 
   DESCRIPTION
	Defines regarding file IO.
 
   SEE ALSO
   	plan_config.h
	plan_file_io.h
	plan_term_io.h
	plan_strings.h
	plan_sys.h

 ---------------------------------------------------------------------------
*/

#include "plan_config.h"

/*
  Data files are written to disk as binary XDR files.
*/

/*
   PLAN_XDR_LIB should be set to the absolute path of the system
   library containing XDR/RPC. This MUST BE DEFINED here before
   compiling PLUNC if you want XDRified code.  The contents of
   the named library are copied into libplanio when it is built.  On
   systems where XDR/RPC stuff is part of a library which will be
   included anyway (like libc.a under Ultrix 2.x) *DO NOT* define
   PLAN_XDR_LIB.

   In this section the xdr include files are also defined. If they
   reside in a directory other than the one specified here - please
   change this file!
*/


/*
  note that this is for the Mt. Xinu version of BSD4.3 
*/

#ifndef _PLUNC_HAVE_PLAN_XDR_H
#define _PLUNC_HAVE_PLAN_XDR_H

#ifdef PLAN_BSD4_3
#include <rpc/rpctypes.h>
#include <rpc/xdr.h>
#endif

#ifdef PLAN_ULTRIX2_0
#include <rpc/types.h>
#include <rpc/xdr.h>
#endif

#ifdef PLAN_STELLIX1_5
#include <sys/rpc/types.h>
#include <sys/rpc/xdr.h>
#define PLAN_XDR_LIB "/usr/lib/librpcsvc.a"
#endif

#ifdef PLAN_SOLARIS2_4
#include <rpc/types.h>
#include <rpc/xdr.h>
#endif

#ifdef PLAN_DARWIN
#include <rpc/types.h>
#include <rpc/xdr.h>
#endif

#ifdef PLAN_IRIX
#include <rpc/types.h>
#include <rpc/xdr.h>
#endif

#ifdef PLAN_LINUX
#include <rpc/xdr.h>
#endif

#ifdef PLAN_WINNT
#include "types.h"
#include "xdr.h"
#endif

#include "xdr_ll_planio.h"

#endif

