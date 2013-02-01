/*

  get_plunc_root_dir.c

  function to retrieve a string value for the directory holding the
  root directory of the plunc distribution.

  order of precedence
  1. environment variable UP 
  2. compile time macro UP (note: if not set defaults to ".")

  Note: use of static string storage means this is NOT multi-thread safe

*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "plan_strings.h"
#include "gen.h"
#include "libplanio.h"

#ifdef PLAN_WINNT
#define MAXPATHLEN 100
#else
#include <sys/param.h>
#endif

#ifndef UP 
#define UP "."
#endif


char *
get_plunc_root_dir()
{
    char *d;
    static char pdir[MAXPATHLEN];

    d = getenv("UP");

    if (d != NULL) {
	strcpy(pdir, d);
    }
    else {
	strcpy(pdir, UP);
    }

    return (pdir);
}

#ifdef PLAN_WINNT
int geteuid()
{
    return(0);
}

struct passwd *
getpwuid(int dummy_uid)
{   static char *home;
    static struct passwd pwd;

    home = getenv("HOME");
    if (home == NULL) {
	home = "c:";
    }
    pwd.pw_dir = home;
    return(&pwd);
}
#endif
