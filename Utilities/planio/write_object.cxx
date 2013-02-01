
#include <stdio.h>
#include "plan_file_io.h"
#include "plan_xdr.h"
#include "gen.h"
#include "brachy.h"
#include "libplanio.h"

int XDR_WriteObject(int fdes, BRACHY_OBJECT *object_desc);


int
write_object(int fdes, BRACHY_OBJECT *object_desc)
/*
 ---------------------------------------------------------------------------
 
   NAME
 	write_object(fdes, object_desc)
 
   SYNOPSIS
        write_object(fdes, object_desc)
	int fdes;
	BRACHY_OBJECT *object_desc;
 
 
   DESCRIPTION
        write a BRACHY_OBJECT structure to a file
 
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
    return(XDR_WriteObject(fdes, object_desc));
}



int
XDR_WriteObject(int fdes, BRACHY_OBJECT *object_desc)
{
    XDR_fdes *xdr_fdes;
    int status = XDR_NO_ERROR;

    xdr_fdes = xdr_ll_open_stream(fdes, XDR_WRITE_MODE);
    if (xdr_fdes == NULL)
    {
	XDR_ReportError("Cannot open brachy object xdr_fdes");
	status = XDR_ERROR;
	goto cleanup;
    }

    status = xdr_ll_brachy_object(xdr_fdes, object_desc, FALSE);

 cleanup:
    xdr_ll_close_stream(xdr_fdes);
    return(status);
}

