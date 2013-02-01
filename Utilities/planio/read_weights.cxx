
#include <stdio.h>
#include <stdlib.h>
#include "plan_file_io.h"
#include "plan_xdr.h"
#include "gen.h"
#include "libplanio.h"

int read_write_weights(int fdes, WEIGHTS *w, char mode);

/*
 ---------------------------------------------------------------------------
 
   NAME
 	read_weights.c
 
   SYNOPSIS
        read_weights(fdes, weights_desc)
	int fdes;
	WEIGHTS *weights_desc;
 
 
   DESCRIPTION
        read a WEIGHTS structure from a file
 
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
read_weights(int fdes, WEIGHTS *w)
{
    return(read_write_weights(fdes, w, XDR_READ_MODE));
}

int
write_weights(int fdes, WEIGHTS *w)
{
    return(read_write_weights(fdes, w, XDR_WRITE_MODE));
}

int
read_write_weights(int fdes, WEIGHTS *w, char mode)
{
    char *cp;
    XDR_fdes *xdr_fdes;
    int status = XDR_NO_ERROR;
    unsigned int lp;

    xdr_fdes = xdr_ll_open_stream(fdes, mode);
    if (xdr_fdes == NULL) {
	XDR_ReportError("Cannot open xdr_fdes");
	return(XDR_ERROR);
    }

    if (! xdr_int(xdr_fdes->xdrs, &(w->count))) {
	XDR_ReportError("Cannot read weights");
	xdr_ll_close_stream(xdr_fdes);
	return(XDR_ERROR);
    }

    if (mode == XDR_READ_MODE) {
	w->weights = (WEIGHT *) malloc(w->count * sizeof(WEIGHT));
	if (w->weights == NULL) {
	    XDR_ReportError("Cannot malloc weights desc weight list");
	    xdr_ll_close_stream(xdr_fdes);
	    return(XDR_ERROR);
	}
    }

    cp = (char *) &(w->weights[0]);
    lp = w->count;
    if (! xdr_array(xdr_fdes->xdrs, &cp, &lp, w->count, sizeof(WEIGHT),
		    (xdrproc_t)xdr_WEIGHT)) {
	XDR_ReportError("Cannot read weights desc weight list");
	status = XDR_ERROR;
    }

    xdr_ll_close_stream(xdr_fdes);
    return(status);
}

