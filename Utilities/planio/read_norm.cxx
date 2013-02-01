
#include <stdio.h>
#include <stdlib.h>
#include "plan_file_io.h"
#include "plan_xdr.h"
#include "gen.h"
#include "libplanio.h"

int read_write_norm(int fdes, NORM *norm, char mode);

/*
 ---------------------------------------------------------------------------
 
   NAME
 	read_norm.c
 
   SYNOPSIS
        read_norm(fdes, norm)
	int fdes;
	NORM	*norm;
 
 
   DESCRIPTION
        read a NORM structure from a file
 
   RETURN VALUE
        0  -> ok
	-1 -> error
 
   SEE ALSO
        plan_xdr_defs.c
 
   AUTHOR
 	Timothy J. Cullip
 	Radiation Oncology
 	North Carolina Memorial Hospital
 	University of North Carolina
 	23 Oct 1996
 ---------------------------------------------------------------------------
*/

int
read_norm(int fdes, NORM *norm)
{
    return(read_write_norm(fdes, norm, XDR_READ_MODE));
}

int
write_norm(int fdes, NORM *norm)
{
    return(read_write_norm(fdes, norm, XDR_WRITE_MODE));
}

int
read_write_norm(int fdes, NORM *norm, char mode)
{
    char *cp;
    unsigned int lp;
    XDR_fdes *xfdes;
    int i;

    xfdes = xdr_ll_open_stream(fdes, mode);
    if (xfdes == NULL) {
	XDR_ReportError("Cannot open xfdes");
	return(XDR_ERROR);
    }

    if (!xdr_int(xfdes->xdrs, &norm->num)) {
	XDR_ReportError("Cannot read/write norm num");
	xdr_ll_close_stream(xfdes);
	return(XDR_ERROR);
    }

    if (!xdr_int(xfdes->xdrs, &norm->current)) {
	XDR_ReportError("Cannot read/write norm current");
	xdr_ll_close_stream(xfdes);
	return(XDR_ERROR);
    }

    if (mode == XDR_READ_MODE) {
	norm->point = (int *) malloc(sizeof(int)*norm->num);
	if (norm->point == NULL) {
	    XDR_ReportError("Cannot malloc norm point list");
	    xdr_ll_close_stream(xfdes);
	    return(XDR_ERROR);
	}
	norm->points = (int *) malloc(sizeof(int)*norm->num);
	if (norm->points == NULL) {
	    XDR_ReportError("Cannot malloc norm points list");
	    xdr_ll_close_stream(xfdes);
	    return(XDR_ERROR);
	}
	norm->beam = (int *) malloc(sizeof(int)*norm->num);
	if (norm->beam == NULL) {
	    XDR_ReportError("Cannot malloc norm beam list");
	    xdr_ll_close_stream(xfdes);
	    return(XDR_ERROR);
	}
	norm->dose = (float *) malloc(sizeof(float)*norm->num);
	if (norm->dose == NULL) {
	    XDR_ReportError("Cannot malloc norm dose list");
	    xdr_ll_close_stream(xfdes);
	    return(XDR_ERROR);
	}
	norm->fraction = (float *) malloc(sizeof(float)*norm->num);
	if (norm->fraction == NULL) {
	    XDR_ReportError("Cannot malloc norm fraction list");
	    xdr_ll_close_stream(xfdes);
	    return(XDR_ERROR);
	}
	norm->percent = (float *) malloc(sizeof(float)*norm->num);
	if (norm->percent == NULL) {
	    XDR_ReportError("Cannot malloc norm percent list");
	    xdr_ll_close_stream(xfdes);
	    return(XDR_ERROR);
	}
	for (i = 0; i < norm->num; i++) {
	    norm->percent[i] = 100.0;
	}
	norm->label = (char **) malloc(sizeof(char *)*norm->num);
	if (norm->label == NULL) {
	    XDR_ReportError("Cannot malloc norm label list");
	    xdr_ll_close_stream(xfdes);
	    return(XDR_ERROR);
	}
	for (i = 0; i < norm->num; i++) {
	    norm->label[i] = (char *)calloc(100, sizeof(char));
	}
    }
    cp = (char *) norm->point;
    lp = norm->num;
    if (!xdr_array(xfdes->xdrs, &cp, &lp, norm->num, sizeof(int),
    		   (xdrproc_t)xdr_int)) {
	XDR_ReportError("Cannot read/write norm point list");
	xdr_ll_close_stream(xfdes);
	return(XDR_ERROR);
    }
    cp = (char *) norm->points;
    lp = norm->num;
    if (!xdr_array(xfdes->xdrs, &cp, &lp, norm->num, sizeof(int), 
    		   (xdrproc_t)xdr_int)) {
	XDR_ReportError("Cannot read/write norm point list");
	xdr_ll_close_stream(xfdes);
	return(XDR_ERROR);
    }
    cp = (char *) norm->beam;
    lp = norm->num;
    if (!xdr_array(xfdes->xdrs, &cp, &lp, norm->num, sizeof(int),
    		   (xdrproc_t)xdr_int)) {
	XDR_ReportError("Cannot read/write norm beam list");
	xdr_ll_close_stream(xfdes);
	return(XDR_ERROR);
    }
    cp = (char *) norm->dose;
    lp = norm->num;
    if (!xdr_array(xfdes->xdrs, &cp, &lp, norm->num, sizeof(float),
    		   (xdrproc_t)xdr_float)) {
	XDR_ReportError("Cannot read/write norm dose list");
	xdr_ll_close_stream(xfdes);
	return(XDR_ERROR);
    }
    cp = (char *) norm->fraction;
    lp = norm->num;
    if (!xdr_array(xfdes->xdrs, &cp, &lp, norm->num, sizeof(float),
    		   (xdrproc_t)xdr_float)) {
	XDR_ReportError("Cannot read/write norm fraction list");
	xdr_ll_close_stream(xfdes);
	return(XDR_ERROR);
    }
    cp = (char *) norm->percent;
    lp = norm->num;
    if (!xdr_array(xfdes->xdrs, &cp, &lp, norm->num, sizeof(float),
    		   (xdrproc_t)xdr_float)) {
	/*
	XDR_ReportError("Cannot read/write norm percent list");
	*/
	xdr_ll_close_stream(xfdes);
	return(XDR_ERROR);
    }
    for (i = 0; i < norm->num; i++) {
	cp = norm->label[i];
	if (!xdr_string(xfdes->xdrs, &cp, 100)) {
		XDR_ReportError("Cannot read/write norm label list");
		xdr_ll_close_stream(xfdes);
		return(XDR_ERROR);
	}
    }

    xdr_ll_close_stream(xfdes);
    if (mode == XDR_READ_MODE) {
	for (i = 1; i < norm->num; i++) {
	    if (norm->beam[i] == norm->beam[i-1]) {
		printf("deleting unused norm groups\n");
		norm->num = i;
		norm->current = i-1;
	    }
	}
    }
    return(XDR_NO_ERROR);
}

