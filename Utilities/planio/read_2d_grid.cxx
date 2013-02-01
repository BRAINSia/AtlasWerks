
#include <stdio.h>
#include <stdlib.h>
#include "plan_file_io.h"
#include "plan_xdr.h"
#include "gen.h"
#include "libplanio.h"

int read_write_2d_grid(int fdes, TWOD_GRID *g, int desc_only, char mode);

/*
 ---------------------------------------------------------------------------
 
   NAME
 	read_2d_grid(fdes, grid_desc, desc_only)
 
   SYNOPSIS
        read_2d_grid(fdes, grid_desc, desc_only)()
	int fdes;
	TWOD_GRID *grid_desc;
	Boolean desc_only;
 
 
   DESCRIPTION
        read a TWOD_GRID structure from file
 
   RETURN VALUE
        0  -> ok
	-1 -> error
 
   SEE ALSO
        plan_xdr_defs.c
 
 ---------------------------------------------------------------------------
*/
int
read_2d_grid(int fdes, TWOD_GRID *g, int desc_only)
{
    return(read_write_2d_grid(fdes, g, desc_only, XDR_READ_MODE));
}

int
write_2d_grid(int fdes, TWOD_GRID *g, int desc_only)
{
    return(read_write_2d_grid(fdes, g, desc_only, XDR_READ_MODE));
}

int
read_write_2d_grid(int fdes, TWOD_GRID *g, int desc_only, char mode)
{
    char *cp;
    XDR_fdes *xdr_fdes;
    int status = XDR_NO_ERROR;
    unsigned int lp;

    xdr_fdes = xdr_ll_open_stream(fdes, mode);
    if (xdr_fdes == NULL) {
	XDR_ReportError("Cannot open XDR_fdes");
	return(XDR_ERROR);
    }

    if (! xdr_TWOD_GRID(xdr_fdes->xdrs, g)) {
	XDR_ReportError("Cannot read twod grid");
	xdr_ll_close_stream(xdr_fdes);
	return(XDR_ERROR);
    }

    if (mode == XDR_READ_MODE) {
	g->matrix = (float *)malloc(sizeof(float)*g->x_count*g->y_count);
	if (g->matrix == NULL) {
	    XDR_ReportError("Cannot malloc twod grid matix");
	    xdr_ll_close_stream(xdr_fdes);
	    return(XDR_ERROR);
	}
    }

    if (! desc_only) {
	cp = (char *)g->matrix;
	if (! xdr_array(xdr_fdes->xdrs, &cp, &lp, g->x_count * g->y_count,
			sizeof(float), (xdrproc_t)xdr_float)) {
	    XDR_ReportError("Cannot read twod grid matrix");
	    status = XDR_ERROR;
	}
    }

    xdr_ll_close_stream(xdr_fdes);
    return(status);
}


