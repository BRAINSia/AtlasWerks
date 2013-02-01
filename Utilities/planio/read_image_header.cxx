
#include <stdio.h>
#include "plan_file_io.h"
#include "plan_xdr.h"
#include "gen.h"

#define PLAN_IM_EXTERN
#include "plan_im.h"

#include "libplanio.h"

/*
 ---------------------------------------------------------------------------
 
   NAME
 	read_image_header(fdes, header_desc)
 
   SYNOPSIS
        read_image_header(fdes, header_desc)
	int fdes;
	plan_im_header *header_desc;
 
 
   DESCRIPTION
        reads a plan_im_header structure from a file
 
   RETURN VALUE
        0  -> ok
	-1 -> error

   SEE ALSO
        plan_xdr_defs.c
 
 ---------------------------------------------------------------------------
*/

int
read_image_header(int fdes, plan_im_header *header)
{
    XDR_fdes *xdr_fdes;
    int status = XDR_NO_ERROR;

    xdr_fdes = xdr_ll_open_stream(fdes, XDR_READ_MODE);
    if (xdr_fdes == NULL) {
	XDR_ReportError("Cannot open xdr_fdes for plan_im_header");
	return(XDR_ERROR);
    }
    if (! xdr_PLAN_IM_HEADER(xdr_fdes->xdrs, header, FALSE))
	status = XDR_ERROR;

    xdr_ll_close_stream(xdr_fdes);
    return(status);
}

int
write_image_header(int fdes, plan_im_header *header)
{
    XDR_fdes *xdr_fdes;
    int status = XDR_NO_ERROR;

    xdr_fdes = xdr_ll_open_stream(fdes, XDR_WRITE_MODE);
    if (xdr_fdes == NULL) {
	XDR_ReportError("Cannot open xdr_fdes for plan_im_header");
	return(XDR_ERROR);
    }
    if (! xdr_PLAN_IM_HEADER(xdr_fdes->xdrs, header, TRUE))
	status = XDR_ERROR;

    xdr_ll_close_stream(xdr_fdes);
    return(status);
}

