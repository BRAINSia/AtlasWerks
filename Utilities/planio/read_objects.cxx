
#include <stdio.h>
#include <stdlib.h>
#include "plan_file_io.h"
#include "plan_xdr.h"
#include "gen.h"
#include "brachy.h"
#include "libplanio.h"

int XDR_ReadObjects(int fdes, BRACHY_OBJECTS *o);

/*
 ---------------------------------------------------------------------------
 
   NAME
 	read_objects(fdes, objects_desc)
 
   SYNOPSIS
        read_objects(fdes, objects_desc)
	int fdes;
	BRACHY_OBJECTS *objects_desc;
 
 
   DESCRIPTION
        read a BRACHY_OBJECTS structure from a file
 
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
read_objects(int fdes, BRACHY_OBJECTS *objects_desc)
{
    return(XDR_ReadObjects(fdes, objects_desc));
}

int
XDR_ReadObjects(int fdes, BRACHY_OBJECTS *o)
{
    int status = XDR_NO_ERROR;
    XDR_fdes *xdr_fdes;
    unsigned int size;
    register int i;

    xdr_fdes = xdr_ll_open_stream(fdes, XDR_READ_MODE);
    if (xdr_fdes == NULL) {
	XDR_ReportError("Cannot open xdr_fdes");
	return(XDR_ERROR);
    }

    if (! xdr_int(xdr_fdes->xdrs, &o->count)) {
	XDR_ReportError("Cannot read brachy objects");
	status = XDR_ERROR;
	goto cleanup;
    }
    
    size = o->count * sizeof(BRACHY_OBJECT);
    o->objects = (BRACHY_OBJECT *) malloc(size);
    if (o->objects == NULL) {
	XDR_ReportError("Cannot malloc brachy objects object list");
	status = XDR_ERROR;
	goto cleanup;
    }


    for (i=0; i<o->count; i++)
	if (xdr_ll_brachy_object(xdr_fdes, &(o->objects[i]),
				 TRUE) ==XDR_ERROR) {
	    XDR_ReportError("Cannot read brachy object");
	    status = XDR_ERROR;
	}
 cleanup:
    xdr_ll_close_stream(xdr_fdes);
    return(status);
}

