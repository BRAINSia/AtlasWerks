
#include <stdio.h>
#include <stdlib.h>
#include "plan_file_io.h"
#include "plan_xdr.h"
#include "gen.h"
#include "libplan.h"
#include "libplanio.h"

#define ABS(a) (((a) < 0) ? (-a) : (a))

/*
 ---------------------------------------------------------------------------
 
   NAME
 	read_anastruct.c
 
   SYNOPSIS
        top level functions for performing ANASTRUCT file read i/o
 
   DESCRIPTION
       read_anastruct(fdes, anastruct_desc)
       int fdes;
       ANASTRUCT *anastruct_desc;
 
   RETURN VALUE
       0  -> ok
       -1 -> error
 
   SEE ALSO
       read_contour.c, plan_xdr_defs.c
 
 ---------------------------------------------------------------------------
*/
int
XDR_Anastruct(int fdes, ANASTRUCT *an, char mode)
{
    int status = XDR_NO_ERROR;
    XDR_fdes *xdr_fdes;

    xdr_fdes = xdr_ll_open_stream(fdes, mode);
    if (xdr_fdes == NULL) {
	XDR_ReportError("Cannot open XDR_fdes");
	return(XDR_ERROR);
    }

    if (! xdr_ANASTRUCT(xdr_fdes->xdrs, an)) {
	XDR_ReportError("Cannot read anastruct");
	return(XDR_ERROR);
    }

    xdr_ll_close_stream(xdr_fdes);

    return(status);
}

int
read_anastruct(int fdes, ANASTRUCT *anastruct_desc)
{
    register int loop;

    if (XDR_Anastruct(fdes, anastruct_desc, XDR_READ_MODE) == -1)
	return(-1);

    if (anastruct_desc->contour_count > 0) {
	anastruct_desc->contours = (CONTOUR *)
	    calloc(anastruct_desc->contour_count, sizeof(CONTOUR));
	if (anastruct_desc->contours == NULL)
	    return(-1);
    }

    for (loop = 0; loop < anastruct_desc->contour_count; loop++) {
	if (read_contour(fdes, anastruct_desc->contours + loop))
	    return(-1);
    }

    /* Some programs (such as MASK) produce adjacent contours
       that have gaps between cont[i].max.z and cont[i+1].min.z
       This will patch those gaps.
    */
    /*
    for (loop = 0; loop < anastruct_desc->contour_count; loop++) {
	int	i;
	CONTOUR	*c1, *c2;
	float	delta;
	c1 = anastruct_desc->contours + loop;
	for (i = loop+1; i < anastruct_desc->contour_count; i++) {
	    c2 = anastruct_desc->contours + i;
	    if (c1->slice_number == c2->slice_number) continue;
	    delta = c1->max.z - c2->min.z;
	    if (delta == 0.0) continue;
	    if (ABS(delta) < 0.00001) c2->min.z = c1->max.z;
	}
    }
    */

    return(0);
}

int
write_anastruct(int fdes, ANASTRUCT *anastruct_desc)
{   int		loop;

    if (XDR_Anastruct(fdes, anastruct_desc, XDR_WRITE_MODE) == -1)
	return(-1);
    for (loop = 0; loop < anastruct_desc->contour_count; loop++) {
	if (write_contour(fdes, anastruct_desc->contours + loop))
	    return (-1);
    }
    return(0);
}

