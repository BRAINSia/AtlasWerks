
#include <stdio.h>
#include "plan_file_io.h"
#include "plan_xdr.h"
#include "gen.h"
#include "brachy.h"
#include "libplanio.h"

int XDR_WriteObjects(int fdes, BRACHY_OBJECTS *o);


int
write_objects(int fdes, BRACHY_OBJECTS *objects_desc)
/*
 ---------------------------------------------------------------------------
 
   NAME
 	write_objects(fdes, objects_desc)
 
   SYNOPSIS
        write_objects(fdes, objects_desc)
	int fdes;
	BRACHY_OBJECTS *objects_desc;
 
 
   DESCRIPTION
        write a BRACHY_OBJECTS structure to a file
 
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

{
    return(XDR_WriteObjects(fdes, objects_desc));
}


int
XDR_WriteObjects(int fdes, BRACHY_OBJECTS *o)
{
    int status = XDR_NO_ERROR;
    XDR_fdes *xdr_fdes;
    register int i;

    xdr_fdes = xdr_ll_open_stream(fdes, XDR_WRITE_MODE);
    if (xdr_fdes == NULL) {
	XDR_ReportError("Cannot open xdr_fdes");
	return(XDR_ERROR);
    }

    if (! xdr_int(xdr_fdes->xdrs, &o->count)) {
	XDR_ReportError("Cannot write brachy objects");
	status = XDR_ERROR;
	goto cleanup;
    }

    xdr_ll_flush(xdr_fdes);

    for (i=0; i<o->count; i++)
	if (xdr_ll_brachy_object(xdr_fdes, &(o->objects[i]),
				 FALSE) == XDR_ERROR) {
	    XDR_ReportError("Cannot read brachy object");
	    status = XDR_ERROR;
	}

 cleanup:
    xdr_ll_close_stream(xdr_fdes);
    return(status);
}

