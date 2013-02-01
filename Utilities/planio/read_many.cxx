
/*
  22-jun-97 gst: copied from xvsim and beefed up to print useful
                 messages on errors. ANSI C. longer filenames
                 allowed. Return codes actually indicate errors. 
*/

#include <stdio.h>
#include <stdlib.h>
#ifndef WIN32
#include <unistd.h>
#else
#include <io.h>
#endif
#include <fcntl.h>

#include "plan_file_io.h"
#include "gen.h"
#include "extbm.h"
#include "libplanio.h"


/* ---------------------------------------------------------------------------
 
   NAME
 	read_filters, read_units, read_trays
 
   SYNOPSIS
 
 
   DESCRIPTION

   	These routines all read an entire file of objects of the
        uniform type, one of: filters, units or trays. They malloc
        space for the objects and read the objects' descriptions to
        determine the number of objects in the file, and then read all
        the objects.

 
   RETURN VALUE
	 -1 on any error, else count of elements found in file.
 
   FILES
 	XDR Filter/unit/tray files in phys_dat dir
        Dir may be overridden at run-time by env var "PHYS_DAT_DIR".
   
 --------------------------------------------------------------------------- */

int
read_filters(int *count, FILTER	**f)
{
    int		fdes,
		loop;
    int		lcount;

    char	file_name[1000];

    *count = 0;
    *f = NULL;

    sprintf(file_name, "%s/filter", get_phys_dat_dir());
    fdes = open(file_name, O_RDONLY, 0);
    
    if (fdes < 0)
    {
	printf("file: %s\n", file_name);
	perror("read_filters: open error");
	return -1;
    }

    if (read_int(fdes, &lcount))
    {
	printf("file: %s\n", file_name);
	perror("read_filters: read error");
	return -1;
    }
    *count = (int) lcount;

    if (*count == 0)
	return *count;

    *f = (FILTER *)malloc (*count * sizeof (FILTER));
    if (*f == NULL)
    {
	printf("file: %s\n", file_name);
	fprintf(stderr, "filter malloc failed");
	exit(0);
    }

    for (loop = 0 ; loop < *count; loop++) {
	(*f)[loop].profiles.contours = NULL;
	if (read_filter(fdes, *f + loop))
	{
	    printf("file: %s\n", file_name);
	    perror("read_filters: read error");
	    return -1;
	}
    }

    close(fdes);

    return *count;
}

int
read_trays(int *count, TRAY **t)
{
    int		fdes,
		loop;
    int		lcount;

    char	file_name[1000];

    *count = 0;

    sprintf(file_name, "%s/tray", get_phys_dat_dir());
    fdes = open(file_name, O_RDONLY, 0);
    
    if (fdes < 0)
    {
	printf("file: %s\n", file_name);
	perror("read_trays: open error");
	return -1;
    }

    if (read_int(fdes, &lcount))
    {
	printf("file: %s\n", file_name);
	perror("read_trays: read error");
	return -1;
    }

    *count = lcount;

    if (*count == 0)
	return 0;

    *t = (TRAY *)malloc(*count * sizeof(TRAY));
    if (*t == NULL)
    {
	printf("file: %s\n", file_name);
	fprintf(stderr, "tray malloc failed");
	exit(0);
    }

    for (loop = 0 ; loop < *count; loop++)
    {
	(*t)[loop].hole_list = NULL;
	if (read_tray(fdes, *t + loop))
	{
	    printf("file: %s\n", file_name);
	    perror("read_trays: read error");
	    return -1;
	}
    }

    close(fdes);

    return(0);
}

int
read_units(int *count, UNIT **u)
{
    int		fdes,
		loop;
    int		lcount;
    char	file_name[1000];

    sprintf(file_name, "%s/unit", get_phys_dat_dir());
    fdes = open(file_name, O_RDONLY, 0);
    
    if (fdes < 0)
    {
	printf("file: %s\n", file_name);
	perror("open error");
	fprintf(stderr, file_name);
	return(1);
    }

    if (read_int(fdes, &lcount))
    {
	printf("file: %s\n", file_name);
	perror("read error");
	fprintf(stderr, file_name);
	return(1);
    }
    *count = lcount;

    if (*count == 0)
	return (0);

    *u = (UNIT *)malloc(*count * sizeof(UNIT));
    if (*u == NULL)
    {
	printf("file: %s\n", file_name);
	fprintf(stderr, "unit malloc failed");
	return(1);
    }

    for (loop = 0 ; loop < *count; loop++)
    {
	if (read_unit(fdes, *u + loop))
	{
	    printf("file: %s\n", file_name);
	    perror("read error");
	    fprintf(stderr, file_name);
	    return(1);
	}
    }

    close(fdes);

    return(0);
}

