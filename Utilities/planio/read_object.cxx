
#include <stdio.h>
#include "plan_file_io.h"
#include "plan_xdr.h"
#include "gen.h"
#include "brachy.h"
#include "libplanio.h"

int XDR_ReadObject(int fdes, BRACHY_OBJECT *object_desc);

/*FUNCTION : read_object ************************************************

PURPOSE
Read a BRACHY_OBJECT structure from a file

COMMENTS

AUTHORS
Jesse Thorn

HEADERS

KEYWORDS

****************************************************************************/
/*
 ---------------------------------------------------------------------------
 
   NAME
 	read_object(fdes, object_desc)
 
   SYNOPSIS
        read_object(fdes, object_desc)
	int fdes;
	BRACHY_OBJECT *object_desc;
 
 
   DESCRIPTION
        read a BRACHY_OBJECT structure from a file
 
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
read_object(int fdes, BRACHY_OBJECT *object_desc)
{
    return(XDR_ReadObject(fdes, object_desc));
}

int
XDR_ReadObject(int fdes, BRACHY_OBJECT *object_desc)
{
    XDR_fdes *xdr_fdes;
    int status = XDR_NO_ERROR;

    xdr_fdes = xdr_ll_open_stream(fdes, XDR_READ_MODE);
    if (xdr_fdes == NULL)
    {
	XDR_ReportError("Cannot open brachy object xdr_fdes");
	status = XDR_ERROR;
	goto cleanup;
    }

    status = xdr_ll_brachy_object(xdr_fdes, object_desc, TRUE);

 cleanup:
    xdr_ll_close_stream(xdr_fdes);
    return(status);
}

