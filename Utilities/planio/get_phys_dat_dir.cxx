/* get_phys_dat_dir.c - ret dir of the PHYSical machine DATa files.

  Replace the evaluation of the compile-time #define PHYS_DAT_DIR
  embedded in each appl with a call to this routine.

  Order of precedence:
  1. run-time environment variable PHYS_DAT_DIR
  2. compile-time #define PHYS_DAT_DIR
  3. "./"  (if PHYS_DAT_DIR is not set at compile-time)

*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "plan_strings.h"

#ifndef PLAN_WINNT
#include <sys/param.h>
#endif

#ifndef MAXPATHLEN
#define MAXPATHLEN 256
#endif

#ifndef PHYS_DAT_DIR
#define PHYS_DAT_DIR "."
#endif


char *
get_phys_dat_dir()
{
    char *d;
    static char pdir[MAXPATHLEN];

    d = getenv("PHYS_DAT_DIR");

    if (d != NULL)
    {
	strcpy(pdir, d);
    }
    else
    {
	strcpy(pdir, PHYS_DAT_DIR);
    }

    return (pdir);
}

