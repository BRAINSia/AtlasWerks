
#include <stdio.h>
#include <stdlib.h>
#include "plan_strings.h"
#include "plan_file_io.h"
#include "plan_xdr.h"
#include "gen.h"
#include "extbm.h"
#include "libplanio.h"

int filter_version_0(XDR_fdes *xdr_fdes, FILTER *filter, int mode, int fdes);
int filter_version_1(XDR_fdes *xdr_fdes, FILTER *filter, int mode, int fdes);
int filter_version_4(XDR_fdes *xdr_fdes, FILTER *filter, int mode, int fdes);

/*
 ---------------------------------------------------------------------------
 
   NAME
 	read_filter - read filter from file
 
   SYNOPSIS
 
 
   DESCRIPTION
 	The contour space will be malloced.
 
   RETURN VALUE
 	0 if OK, -1 if not 
 
   DIAGNOSTICS
 
 
   BUGS
 
 
 ---------------------------------------------------------------------------
*/

int
read_write_filter(int fdes, FILTER *filter, int mode)
{   int		ret = 0;
    XDR_fdes	*xdr_fdes;
    XDR		*xdr;

    xdr_fdes = xdr_ll_open_stream(fdes, mode);
    if (xdr_fdes == NULL) {
	XDR_ReportError("Cannot open XDR_fdes");
	return(XDR_ERROR);
    }
    xdr = xdr_fdes->xdrs;

    if (!(xdr_int(xdr, &filter->version))) {
	XDR_ReportError("Cannot read filter");
	return(XDR_ERROR);
    }
    if (filter->version < 0) {
	if (!(xdr_int(xdr, &filter->unit_id))) {
	    XDR_ReportError("Cannot read filter");
	    return(XDR_ERROR);
	}
	filter->version = -filter->version;
    }
    else {
	filter->unit_id = filter->version;
	filter->version = 0;
    }
    switch (filter->version) {
	case 0:
	    ret = filter_version_0(xdr_fdes, filter, mode, fdes);
	    break;
	case 4:
	    ret = filter_version_4(xdr_fdes, filter, mode, fdes);
	    break;
    }
    return(ret);
}

int
read_filter(int fdes, FILTER *filter)
{
    return(read_write_filter(fdes, filter, XDR_READ_MODE));
}

int
write_filter(int fdes, FILTER *filter)
{
    return(read_write_filter(fdes, filter, XDR_WRITE_MODE));
}

int
filter_version_0(XDR_fdes *xdr_fdes, FILTER *filter, int mode, int fdes)
{   int		ret;
    char	*ver;

    ver = getenv("PHYS_DAT_VERSION");
    if (ver == NULL) {
	return(filter_version_4(xdr_fdes, filter, mode, fdes));
    }
    else {
	if (strncmp(ver, "2", 1) == 0) filter->version = 1;
	else if (strncmp(ver, "3", 1) == 0) filter->version = 2;
    }
    switch (filter->version) {
	case 1:
	    ret = filter_version_1(xdr_fdes, filter, mode, fdes);
	    break;
	case 2:
	    ret = filter_version_1(xdr_fdes, filter, mode, fdes);
	    break;
	case 4:
	    default:
	    ret = filter_version_4(xdr_fdes, filter, mode, fdes);
	break;
    }
    return(ret);
}

int
filter_version_1(XDR_fdes *xdr_fdes, FILTER *filter, int mode, int fdes)
{
    int		i, j;
    char	*cp = &(filter->name[0]);
    XDR		*xdr = xdr_fdes->xdrs;

    filter->type = REAL_FILTER;
    if (!(xdr_int(xdr, &filter->filter_id) &&
	  xdr_int(xdr, &filter->mirror_id) &&
	  xdr_int(xdr, &filter->next_id) &&
	  xdr_int(xdr, &filter->prev_id) &&
	  xdr_string(xdr, &cp, NAME_LENGTH) &&
	  xdr_int(xdr, &filter->pointy_end))) {
	XDR_ReportError("Cannot read filter");
	return(XDR_ERROR);
    }
    for (i = 0; i < 2; i++) {
	JAW	*jaw;
	jaw = filter->jaw_limits + i;
	if (!(xdr_int(xdr, &jaw->independent) &&
	      xdr_float(xdr, &jaw->min_1) &&
	      xdr_float(xdr, &jaw->max_1) &&
	      xdr_float(xdr, &jaw->min_2) &&
	      xdr_float(xdr, &jaw->max_2) &&
	      xdr_float(xdr, &jaw->min_w) &&
	      xdr_float(xdr, &jaw->max_w))) {
	    XDR_ReportError("Cannot read jaws");
	    return(XDR_ERROR);
	}
    }

    for (j = 0; j < 4; j++) {
	for (i = 0; i < 4; i++) {
	    if (!(xdr_float(xdr, &filter->T_filter_to_beam[j][i]))) {
		XDR_ReportError("Cannot read filter");
		return(XDR_ERROR);
	    }
	}
    }
    for (j = 0; j < 4; j++) {
	for (i = 0; i < 4; i++) {
	    if (!(xdr_float(xdr, &filter->T_beam_to_filter[j][i]))) {
		XDR_ReportError("Cannot read filter");
		return(XDR_ERROR);
	    }
	}
    }
    for (i = 0; i < MAX_CHUNK_COUNT; i++) {
	if (!(xdr_float(xdr, &filter->mu[i]) &&
	      xdr_float(xdr, &filter->mu_dx[i]) &&
	      xdr_float(xdr, &filter->mu_dr[i]) &&
	      xdr_float(xdr, &filter->hvl_slope[i]))) {
	    XDR_ReportError("Cannot read filter");
	    return(XDR_ERROR);
	}
    }
    xdr_ll_close_stream(xdr_fdes);

    if (mode == XDR_READ_MODE) read_anastruct(fdes, &filter->profiles);
    else write_anastruct(fdes, &filter->profiles);
    return(0);
}

int
filter_version_4(XDR_fdes *xdr_fdes, FILTER *filter, int mode, int fdes)
{
    int		i, j;
    char	*cp = &(filter->name[0]);
    XDR		*xdr = xdr_fdes->xdrs;

    if (!(xdr_int(xdr, &filter->type) &&
	  xdr_int(xdr, &filter->filter_id) &&
	  xdr_int(xdr, &filter->mirror_id) &&
	  xdr_int(xdr, &filter->next_id) &&
	  xdr_int(xdr, &filter->prev_id) &&
	  xdr_string(xdr, &cp, NAME_LENGTH) &&
	  xdr_int(xdr, &filter->pointy_end))) {
	XDR_ReportError("Cannot read filter");
	return(XDR_ERROR);
    }
    for (i = 0; i < 2; i++) {
	JAW	*jaw;
	jaw = filter->jaw_limits + i;
	if (!(xdr_int(xdr, &jaw->independent) &&
	      xdr_float(xdr, &jaw->min_1) &&
	      xdr_float(xdr, &jaw->max_1) &&
	      xdr_float(xdr, &jaw->min_2) &&
	      xdr_float(xdr, &jaw->max_2) &&
	      xdr_float(xdr, &jaw->min_w) &&
	      xdr_float(xdr, &jaw->max_w))) {
	    XDR_ReportError("Cannot read jaws");
	    return(XDR_ERROR);
	}
    }

    for (j = 0; j < 4; j++) {
	for (i = 0; i < 4; i++) {
	    if (!(xdr_float(xdr, &filter->T_filter_to_beam[j][i]))) {
		XDR_ReportError("Cannot read filter");
		return(XDR_ERROR);
	    }
	}
    }
    for (j = 0; j < 4; j++) {
	for (i = 0; i < 4; i++) {
	    if (!(xdr_float(xdr, &filter->T_beam_to_filter[j][i]))) {
		XDR_ReportError("Cannot read filter");
		return(XDR_ERROR);
	    }
	}
    }
    for (i = 0; i < MAX_CHUNK_COUNT; i++) {
	if (!(xdr_float(xdr, &filter->mu[i]) &&
	      xdr_float(xdr, &filter->mu_dx[i]) &&
	      xdr_float(xdr, &filter->mu_dr[i]) &&
	      xdr_float(xdr, &filter->hvl_slope[i]))) {
	    XDR_ReportError("Cannot read filter");
	    return(XDR_ERROR);
	}
    }
    if (filter->type == DYNAMIC_FILTER) {
	if (!xdr_float(xdr, &filter->wfo_ao)) return(XDR_ERROR);
	if (!xdr_float(xdr, &filter->dwfo_da)) return(XDR_ERROR);
	if (!xdr_float(xdr, &filter->dwf_dfs_ao)) return(XDR_ERROR);
	if (!xdr_float(xdr, &filter->dwf_dfs_da)) return(XDR_ERROR);
    }
    xdr_ll_close_stream(xdr_fdes);

    if (mode == XDR_READ_MODE) read_anastruct(fdes, &filter->profiles);
    else write_anastruct(fdes, &filter->profiles);
    return(0);
}


