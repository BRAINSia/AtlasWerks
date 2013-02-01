
#include <stdio.h>
#include <stdlib.h>
#include "plan_file_io.h"
#include "plan_xdr.h"
#include "gen.h"
#include "libplanio.h"

int read_write_points(int fdes, CALC_POINTS *p, int desc_only, char mode);

/*
 ---------------------------------------------------------------------------
 
   NAME
 	read_points.c
 
   SYNOPSIS
        read_points(fdes, points_desc, desc_only)
	int fdes;
	CALC_POINTS *points_desc;
	Boolean desc_only;
 
 
   DESCRIPTION
        read a CALC_POINTS structure from a file
 
   RETURN VALUE
        0  -> ok
	-1 -> error
 
   SEE ALSO
        plan_xdr_defs.c
 
   AUTHOR
 	Jesse Thorn
 	Radiation Oncology
 	North Carolina Memorial Hospital
 	University of North Carolina
 	27 September 1988
 
 
 
 ---------------------------------------------------------------------------
*/

int
read_points(int fdes, CALC_POINTS *p, int desc_only)
{
    return(read_write_points(fdes, p, desc_only, XDR_READ_MODE));
}

int
write_points(int fdes, CALC_POINTS *p, int desc_only)
{
    return(read_write_points(fdes, p, desc_only, XDR_WRITE_MODE));
}

int
read_write_points(int fdes, CALC_POINTS *p, int desc_only, char mode)
{
    char *cp;
    unsigned int lp;
    int status = XDR_NO_ERROR;
    XDR_fdes *xdr_fdes;
    // unused var//int i;

    xdr_fdes = xdr_ll_open_stream(fdes, mode);
    if (xdr_fdes == NULL) {
	XDR_ReportError("Cannot open xdr_fdes");
	return(XDR_ERROR);
    }

    if (! xdr_int(xdr_fdes->xdrs, &p->count)) {
	XDR_ReportError("Cannot read calc points count");
	xdr_ll_close_stream(xdr_fdes);
	return(XDR_ERROR);
    }

    if (mode == XDR_READ_MODE) {
	p->points = (CALC_POINT *) malloc(sizeof(CALC_POINT) * p->count);
	if (p->points == NULL) {
	    XDR_ReportError("Cannot malloc calc points point list");
	    xdr_ll_close_stream(xdr_fdes);
	    return(XDR_ERROR);
	}
	p->dose = (float *) malloc(sizeof(float) * p->count);
	if (p->dose == NULL) {
	    XDR_ReportError("Cannot malloc calc points dose list");
	    xdr_ll_close_stream(xdr_fdes);
	    return(XDR_ERROR);
	}
    }


    cp = (char *) p->points;
    lp = p->count;
    if (! xdr_array(xdr_fdes->xdrs, &cp, &lp, p->count, sizeof(CALC_POINT),
		    (xdrproc_t)xdr_CALC_POINT)) {
	XDR_ReportError("Cannot read calc points point list");
	xdr_ll_close_stream(xdr_fdes);
	return(XDR_ERROR);
    }

    if (! desc_only) {
	cp = (char *)p->dose;
	lp = p->count;
	if (! xdr_array(xdr_fdes->xdrs, &cp, &lp, p->count, sizeof(float),
			(xdrproc_t)xdr_float)) {
	    XDR_ReportError("Cannot read calc points dose list");
	    status = XDR_ERROR;
	}
    }

    xdr_ll_close_stream(xdr_fdes);
    return(status);
}

