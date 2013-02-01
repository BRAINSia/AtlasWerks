
#include <stdio.h>
#include <stdlib.h>
#include "plan_file_io.h"
#include "plan_xdr.h"
#include "gen.h"
#include "libplanio.h"

int read_write_plan(int fdes, PLAN *p, char mode);

/*
 ---------------------------------------------------------------------------
 
   NAME
 	read_plan.c
 
   SYNOPSIS
        read_plan(fdes, plan_desc)
	int fdes;
	PLAN *plan_desc;
 
 
   DESCRIPTION
        reads a PLAN structure from a file
 
   RETURN VALUE
        0  -> ok
	-1 -> error
 
   SEE ALSO
        plan_xdr_defs.c
 
 
 ---------------------------------------------------------------------------
*/

int
read_plan(int fdes, PLAN *p)
{
    return(read_write_plan(fdes, p, XDR_READ_MODE));
}

int
write_plan(int fdes, PLAN *p)
{
    return(read_write_plan(fdes, p, XDR_WRITE_MODE));
}

int
read_write_plan(int fdes, PLAN *p, char mode)
{
    char	*cp;
    unsigned
    int		lp;
    int		status = XDR_NO_ERROR;
    XDR_fdes	*xdr_fdes;

    xdr_fdes = xdr_ll_open_stream(fdes, mode);
    if (xdr_fdes == NULL) {
	XDR_ReportError("Cannot open xdr_fdes");
	return(XDR_ERROR);
    }

    if (! xdr_PLAN(xdr_fdes->xdrs, p)) {
	XDR_ReportError("Cannot read PLAN");
	xdr_ll_close_stream(xdr_fdes);
	return(XDR_ERROR);
    }

    if (mode == XDR_READ_MODE) {
	p->isodoses = (float *)malloc(sizeof(float)*p->isodose_count);
	if (p->isodoses == NULL) {
	    XDR_ReportError("Cannot malloc PLAN isodes");
	    xdr_ll_close_stream(xdr_fdes);
	    return(XDR_ERROR);
	}
    }

    cp = (char *)&(p->isodoses[0]);
    lp = p->isodose_count;
    if (! xdr_array(xdr_fdes->xdrs, &cp, &lp, p->isodose_count,
		    sizeof(float), (xdrproc_t)xdr_float)) {
	XDR_ReportError("Cannot read PLAN isodose list");
	status = XDR_ERROR;
    }

    xdr_ll_close_stream(xdr_fdes);
    return(status);
}

int
read_plans(int fdes, PLAN *plan_desc)
{   int		loop;
    int		status;
    XDR_fdes	*xdr_fdes;

    loop = 0;
    while(1) {
	xdr_fdes = xdr_ll_open_stream(fdes, XDR_READ_MODE);
	if (xdr_eof(fdes)) break;
	xdr_ll_close_stream(xdr_fdes);
	status = read_plan(fdes, &plan_desc[loop]);
	if (status) break;
	loop++;
    }
    return(loop);
}

int
write_plans(int fdes, PLAN *plan_desc, int num)
{   int		loop;
    int		status=0;

    for (loop = 0; loop < num; loop++) {
	status = write_plan(fdes, &plan_desc[loop]);
	if (status) break;
    }
    return(status);
}


