
#include <stdio.h>
#include <stdlib.h>
#include "plan_file_io.h"
#include "plan_xdr.h"
#include "gen.h"
#include "libplanio.h"

/*
 ---------------------------------------------------------------------------
 
   NAME
 	read_contour.c
 
   SYNOPSIS
        read CONTOUR structures
 
   DESCRIPTION
        read_contour(fdes, contour_desc)
	int fdes;
	CONTOUR *contour_desc;
 
   RETURN VALUE
        0  -> ok
	-1 -> error
 
   SEE ALSO
        plan_xdr_defs.c
 
 
 
 ---------------------------------------------------------------------------
*/

int
read_write_contour(int fdes, CONTOUR *c, char mode)
{
    XDR_fdes *xdr_fdes;

    xdr_fdes = xdr_ll_open_stream(fdes, mode);
    if (xdr_fdes == NULL) {
	XDR_ReportError("Cannot open XDR_fdes");
	return(XDR_ERROR);
    }

    if (! xdr_CONTOUR(xdr_fdes->xdrs, c)) {
	XDR_ReportError("Cannot read contour header");
	xdr_ll_close_stream(xdr_fdes);
	return(XDR_ERROR);
    }

    if (mode == XDR_READ_MODE && c->vertex_count > 0) {
	c->x  = (float *) malloc(c->vertex_count * sizeof(float));
	c->y  = (float *) malloc(c->vertex_count * sizeof(float));
    }

    if (! xdr_CONTOUR_X_Y(xdr_fdes->xdrs, c->x, c->vertex_count)) {
	XDR_ReportError("Cannot read contour x");
	xdr_ll_close_stream(xdr_fdes);
	return(XDR_ERROR);
    }

    if (! xdr_CONTOUR_X_Y(xdr_fdes->xdrs, c->y, c->vertex_count)) {
	XDR_ReportError("Cannot read contour y");
	xdr_ll_close_stream(xdr_fdes);
	return(XDR_ERROR);
    }
    xdr_ll_close_stream(xdr_fdes);
    return(0);
}

int
read_contour(int fdes, CONTOUR *c)
{
    return(read_write_contour(fdes, c, XDR_READ_MODE));
}

int
write_contour(int fdes, CONTOUR *c)
{
    return(read_write_contour(fdes, c, XDR_WRITE_MODE));
}


