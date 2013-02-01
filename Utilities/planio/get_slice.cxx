

#include <stdio.h>
#include "gen.h"
#include  "plan_im.h"
#include "plan_file_io.h"
#include "plan_sys.h"
#include "libplanio.h"

int
get_slice(
    int			fd,
    plan_im_header	*im_header,
    int			start_slice,
    int			end_slice,
    PIXELTYPE		*buffer)
/*
 ---------------------------------------------------------------------------
 
   NAME
 	get_slice - read a subset of slices from a plan_im file
 
   SYNOPSIS
 
 
   DESCRIPTION
 
 
   RETURN VALUE
 
 
   DIAGNOSTICS
 
 
   FILES
 
 
   SEE ALSO
 
 
   BUGS
 
 
   AUTHOR
 
 
 ---------------------------------------------------------------------------
*/

{
    int		slice_size,
    		index,
		slice;

    slice_size = sizeof(PIXELTYPE)*
		 im_header->x_dim*
		 im_header->y_dim;

    /* read in images a slice at a time */
    index = 0;
    for (slice = start_slice; slice <= end_slice; slice++) {
	if (read_scan_xy(fd, &(buffer[index]),
		       im_header->per_scan[slice].offset_ptrs,
		       im_header->x_dim, im_header->y_dim)) {
	    perror("get_slice - cannot read slice");
	    return(-1);
	}
	index += slice_size/(sizeof (PIXELTYPE));
    }

    return (0);
}

int
put_slice(
   int			fd,
   plan_im_header	*im_header,
   int			start_slice,
   int			end_slice,
   PIXELTYPE		*buffer)
{
    int		slice_size,
    		index,
		slice;

    slice_size = sizeof(PIXELTYPE)*
		 im_header->x_dim*
		 im_header->y_dim;

    /* read in images a slice at a time */
    index = 0;
    for (slice = start_slice; slice <= end_slice; slice++) {
	if (write_scan_xy(fd, &(buffer[index]),
		       im_header->per_scan[slice].offset_ptrs,
		       im_header->x_dim, im_header->y_dim)) {
	    perror("put_slice - cannot write slice");
	    return(-1);
	}
	index += slice_size/(sizeof (PIXELTYPE));
    }

    return (0);
}


