
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


typedef struct {
    int		id;
    int		offset;
} ENTRY;


int
get_filter(char *phys_dat_dir, int filter_id, FILTER *filter)
{
    int		fdes,
      loop;
    // unused var // int cloop;
    int		count;
    ENTRY	*entry;
    char	index_file_name[150],
    		filter_file_name[150];

    sprintf(index_file_name, "%s/filter.index", phys_dat_dir);
    sprintf(filter_file_name, "%s/filter", phys_dat_dir);

/*
  Open index file.
*/
    fdes = open(index_file_name, O_RDONLY, 0);
    if (fdes < 0)
	return(-1);
/*
  Get filter count from index file, malloc space for index file
  contents, and read in the index file.  Close it.
*/
    if (read_int(fdes, &count))
	return(-1);

/*
  Loop through index entries looking for requested filter id.
  When you find it, open the filter file, seek the filter in question
  and read it.  We first read the FILTER which includes an ANASTRUCT.
  The ANASTRUCT references a dynamic array of CONTOUR which is
  malloced by read_filter.  The pointer needs to be initialized to
  NULL or by malloc.
*/
    entry = (ENTRY *)malloc(count*sizeof(ENTRY));
    if (entry == NULL) goto exit1;

    if (read_ints(fdes, (int *)entry, 2*count)) goto exit1;
    close(fdes);
    for (loop = 0; loop < count; loop++) {
	if (entry[loop].id == filter_id) {
	    fdes = open(filter_file_name, O_RDONLY, 0);
	    if (fdes < 0) goto exit1;

	    if (lseek(fdes, entry[loop].offset, SEEK_SET) == -1)
		goto exit1;

	    if (read_filter(fdes, filter)) goto exit1;
	    close(fdes);
	    free(entry);
	    return(0);
	}
    }

 exit1:
    close(fdes);
    if (entry) free(entry);
    return(-1);
}

