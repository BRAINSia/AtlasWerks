
#include <stdio.h>
#include "plan_file_io.h"
#include "plan_xdr.h"
#include "gen.h"
#include "extbm.h"
#include "libplanio.h"
#include "libplan.h"

/*
   ***NOTE***:
   The version variable has been moved from plan_xdr_defs.cxx's
   xdr_EXT_BEAM() to here in order to modify version 6 of the
   EXT_BEAM structure to be able to be forward compatible with
   the upcoming version 7.
   This was only done because of a mistake that put a copy of
   version 7 beams into the clinic that the current version 6
   had to be able to read.
*/

/*
 ---------------------------------------------------------------------------
 
   NAME
 	read_beam.c
 
   SYNOPSIS
        read_beam(fdes, ext_beam_desc)
	int fdes;
	EXT_BEAM *ext_beam_desc;

        write_beam(fdes, ext_beam_desc)
	int fdes;
	EXT_BEAM *ext_beam_desc;

 
   DESCRIPTION
        read or write an EXT_BEAM structure to file
 
   RETURN VALUE
        0  -> ok
	-1 -> error
 
   SEE ALSO
        plan_xdr_defs.c, read_contour.c
  
 ---------------------------------------------------------------------------
*/

int
read_beam(int fdes, EXT_BEAM *eb)
{
    XDR_fdes	*xdr_fdes;
    int		version;
    CONTOUR	dummy_con;

    xdr_fdes = xdr_ll_open_stream(fdes, XDR_READ_MODE);
    if (xdr_fdes == NULL) {
	XDR_ReportError("Cannot open XDR_fdes");
	return(XDR_ERROR);
    }

    if (! xdr_EXT_BEAM(xdr_fdes->xdrs, eb, &version)) {
	XDR_ReportError("Cannot read external beam");
	return(XDR_ERROR);
    }

    xdr_ll_close_stream(xdr_fdes); 

    if (read_contour(fdes, &(eb->beam_outline)))
	return(XDR_ERROR);
    if (read_contour(fdes, &(eb->collimated_beam_outline)))
	return(XDR_ERROR);

/* ***NOTE*** This version 7 contour needs to be read and thown away */
if (version <= -7) read_contour(fdes, &dummy_con);

    if (eb->filter_type[0] == COMPENSATOR_FILTER) {
	if (read_anastruct(fdes, &(eb->custom_filter[0])))
	    return(XDR_ERROR);
    }
    else {
	eb->custom_filter[0].contours = NULL;
	eb->custom_filter[0].contour_count = 0;
    }
    if (eb->filter_type[1] == BOLUS_FILTER) {
	if (read_anastruct(fdes, &(eb->custom_filter[1])))
	    return(XDR_ERROR);
    }
    else {
    	eb->custom_filter[1].contours = NULL;
    	eb->custom_filter[1].contour_count = 0;
    }

    return(0);
}

int
write_beam(int fdes, EXT_BEAM *eb)
{
    XDR_fdes *xdr_fdes;
    int status = XDR_NO_ERROR;
    int version;

    xdr_fdes = xdr_ll_open_stream(fdes, XDR_WRITE_MODE);
    if (xdr_fdes == NULL) {
	XDR_ReportError("Cannot open XDR_fdes");
	return(XDR_ERROR);
    }

    if (! xdr_EXT_BEAM(xdr_fdes->xdrs, eb, &version)) {
	XDR_ReportError("Cannot write external beam");
	status = XDR_ERROR;
	return(status);
    }

    xdr_ll_close_stream(xdr_fdes);

    if (write_contour(fdes, &(eb->beam_outline)))
	return(XDR_ERROR);

    if (write_contour(fdes, &(eb->collimated_beam_outline)))
	return(XDR_ERROR);

    if (eb->filter_type[0] == COMPENSATOR_FILTER) {
	if (write_anastruct(fdes, &(eb->custom_filter[0])))
	    return(XDR_ERROR);
    }
    if (eb->filter_type[1] == BOLUS_FILTER) {
	if (write_anastruct(fdes, &(eb->custom_filter[1])))
	    return(XDR_ERROR);
    }

    return(0);
}

int
write_old_beam(int fdes, EXT_BEAM *eb)
{
    XDR_fdes *xdr_fdes;
    int status = XDR_NO_ERROR;

    xdr_fdes = xdr_ll_open_stream(fdes, XDR_WRITE_MODE);
    if (xdr_fdes == NULL) {
	XDR_ReportError("Cannot open XDR_fdes");
	return(XDR_ERROR);
    }
    if (eb->filter_type[0] == COMPENSATOR_FILTER)
	eb->filter_type[0] = NO_FILTER;
    if (eb->filter_type[0] == BOLUS_FILTER)
	eb->filter_type[0] = NO_FILTER;

    if (! xdr_EXT_OLD_BEAM(xdr_fdes->xdrs, eb)) {
	XDR_ReportError("Cannot write external beam");
	status = XDR_ERROR;
	return(status);
    }

    xdr_ll_close_stream(xdr_fdes);

    if (write_contour(fdes, &(eb->beam_outline)))
	return(XDR_ERROR);

    if (write_contour(fdes, &(eb->collimated_beam_outline)))
	return(XDR_ERROR);

    return(0);
}

