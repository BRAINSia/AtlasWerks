
#include <stdio.h>
#include <stdlib.h>
#include "plan_file_io.h"
#include "plan_xdr.h"
#include "gen.h"
#include "libplanio.h"

int
read_grid(int fdes, GRID *g, int desc_only)
{
    XDR_fdes		*xdr_fdes;
    int			status = XDR_NO_ERROR;
    int			num;
    int			z;

    xdr_fdes = xdr_ll_open_stream(fdes, XDR_READ_MODE);
    if (xdr_fdes == NULL) {
	XDR_ReportError("Cannot open XDR_fdes");
	return(XDR_ERROR);
    }

    if (! xdr_GRID(xdr_fdes->xdrs, g)) {
	XDR_ReportError("Cannot read  grid");
	xdr_ll_close_stream(xdr_fdes);
	return(XDR_ERROR);
    }

    num = g->x_count*g->y_count*g->z_count;
    g->matrix = (float *) calloc(sizeof(float), num);
    if (g->matrix == NULL) {
	XDR_ReportError("Cannot malloc grid matrix");
	xdr_ll_close_stream(xdr_fdes);
	return(XDR_ERROR);
    }

    if (! desc_only) {
	
	if (is_socket(fdes)) {
	    xdrfd_get_bytes(xdr_fdes->xdrs, (caddr_t)g->matrix,
	    		    num*sizeof(float));
	    xdr_ll_close_stream(xdr_fdes);
	    return(status);
	}

	for (z = 0; z < num; z++) {
	    if (! xdr_float(xdr_fdes->xdrs, (g->matrix + z))) {
		XDR_ReportError("Cannot read grid matrix");
		status = XDR_ERROR;
		break;
	    }
	}
    }

    xdr_ll_close_stream(xdr_fdes);
    return(status);
}

int
write_grid(int fdes, GRID *g, int desc_only)
{
    XDR_fdes	*xdr_fdes;
    int		status = XDR_NO_ERROR;
    int		num;
    int		z;

    xdr_fdes = xdr_ll_open_stream(fdes, XDR_WRITE_MODE);
    if (xdr_fdes == NULL) {
	XDR_ReportError("Cannot open XDR_fdes");
	return(XDR_ERROR);
    }

    if (! xdr_GRID(xdr_fdes->xdrs, g)) {
	XDR_ReportError("Cannot write  grid");
	xdr_ll_close_stream(xdr_fdes);
	return(XDR_ERROR);
    }

    if (! desc_only) {
	num = g->x_count*g->y_count*g->z_count;
	if (is_socket(fdes)) {
	    xdrfd_put_bytes(xdr_fdes->xdrs, (caddr_t)g->matrix,
	    		    num*sizeof(float));
	    xdr_ll_close_stream(xdr_fdes);
	    return(status);
	}
	for (z = 0; z < num; z++) {
	    if (! xdr_float(xdr_fdes->xdrs, (g->matrix + z))) {
		XDR_ReportError("Cannot write grid matrix");
		status = XDR_ERROR;
		break;
	    }
	}
    }

    xdr_ll_close_stream(xdr_fdes);
    return(status);
}

