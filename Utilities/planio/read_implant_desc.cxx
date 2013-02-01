
#include <stdio.h>
#include <stdlib.h>
#include "plan_file_io.h"
#include "plan_xdr.h"
#include "gen.h"
#include "brachy.h"
#include "libplanio.h"

int XDR_ReadImplantDesc(int fdes, IMPLANT_DESC *id);

/*FUNCTION : read_implant_desc *********************************************

PURPOSE
Read an IMPLANT_DESC structure to file

COMMENTS

AUTHORS
Jesse Thorn

HEADERS

KEYWORDS

****************************************************************************/
int
read_implant_desc(int fdes, IMPLANT_DESC *implant_desc)
/*
 ---------------------------------------------------------------------------
 
   NAME
 	read_implant_desc.c
 
   SYNOPSIS
        read_implant_desc(fdes, implant_desc)
	int fdes;
	IMPLANT_DESC *implant_desc;
 
   DESCRIPTION
        read an IMPLANT_DESC structure to file
 
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
    return(XDR_ReadImplantDesc(fdes, implant_desc));
}


int
XDR_ReadImplantDesc(int fdes, IMPLANT_DESC *id)
{
    char *cp;
    XDR_fdes *xdr_fdes;
    int status = XDR_NO_ERROR;
    unsigned int lp;
    bool_t xdr_SEED(), xdr_SOURCE();

    xdr_fdes = xdr_ll_open_stream(fdes, XDR_READ_MODE);
    if (xdr_fdes == NULL) {
	XDR_ReportError("Cannot open XDR_fdes");
	return(XDR_ERROR);
    }

    if (! xdr_IMPLANT_DESC(xdr_fdes->xdrs, id)) {
	XDR_ReportError("Cannot read implant desc header");
	status = XDR_ERROR;
	goto cleanup;
    }

    if (id->source_count) {
	id->sources = (SOURCE *) malloc(id->source_count * sizeof(SOURCE));
	if (id->sources == NULL) {
	    XDR_ReportError("Cannot malloc implant sources");
	    status = XDR_ERROR;
	    goto cleanup;
	}

	cp = (char *)id->sources;
	if (! xdr_array(xdr_fdes->xdrs, &cp, &lp,
			id->source_count, sizeof(SOURCE),
			(xdrproc_t)xdr_SOURCE)) {
	    XDR_ReportError("Cannot read implant sources");
	    status = XDR_ERROR;
	    goto cleanup;
	}
    }
    else
	id->sources = (SOURCE *) NULL;

    if (id->seed_count) {
	id->seeds = (SEED *) malloc(id->seed_count * sizeof(SEED));
	if (id->seeds == NULL) {
	    XDR_ReportError("Cannot malloc implant seeds");
	    status = XDR_ERROR;
	    goto cleanup;
	}

	cp = (char *)id->seeds;
	if (! xdr_array(xdr_fdes->xdrs, &cp, &lp,
			id->seed_count, sizeof(SEED),
			(xdrproc_t)xdr_SEED)) {
	    XDR_ReportError("Cannot read implant seeds");
	    status = XDR_ERROR;
	    goto cleanup;
	}
    }
    else
	id->seeds = (SEED *) NULL;

	
 cleanup:
    xdr_ll_close_stream(xdr_fdes);
    return(status);
}

