
#include <stdio.h>
#include <stdlib.h>
#include "plan_file_io.h"
#include "plan_xdr.h"
#include "gen.h"
#include "extbm.h"
#include "libplanio.h"

int read_write_tray(int fdes, TRAY *tray, char mode);
int tray_version_0(XDR_fdes *xdr_fdes, TRAY *tray, char mode);
int tray_version_4(XDR_fdes *xdr_fdes, TRAY *tray, char mode);

/*
 ---------------------------------------------------------------------------
 
   NAME
 	read_tray - read a tray description from file
 
   SYNOPSIS
 
 
   DESCRIPTION
 	The hole_list space will be malloced;
 
   RETURN VALUE
 	0 if OK, -1 otherwise.
 
 ---------------------------------------------------------------------------
*/

int
read_tray(int fdes, TRAY *tray)
{
    return(read_write_tray(fdes, tray, XDR_READ_MODE));
}

int
write_tray(int fdes, TRAY *tray)
{
    return(read_write_tray(fdes, tray, XDR_WRITE_MODE));
}

int
read_write_tray(int fdes, TRAY *tray, char mode)
{   int		ret = 0;
    XDR_fdes    *xdr_fdes;
    XDR         *xdr;

    xdr_fdes = xdr_ll_open_stream(fdes, mode);
    if (xdr_fdes == NULL) {
	XDR_ReportError("Cannot open XDR_fdes");
	return(XDR_ERROR);
    }
    xdr = xdr_fdes->xdrs;
    if (!(xdr_int(xdr, &tray->version))) {
	XDR_ReportError("Cannot read tray");
	return(XDR_ERROR);
    }
    if (tray->version < 0) {
	if (!(xdr_int(xdr, &tray->unit_id))) {
	    XDR_ReportError("Cannot read tray");
	    return(XDR_ERROR);
	}
	tray->version = -tray->version;
    }
    else {
	tray->unit_id = tray->version;
	tray->version = 0;
    }
    switch (tray->version) {
	case 0:
	    ret = tray_version_0(xdr_fdes, tray, mode);
	    break;
	case 4:
	    ret = tray_version_4(xdr_fdes, tray, mode);
	    break;
    }
    return(ret);
}

int
tray_version_0(XDR_fdes *xdr_fdes, TRAY *tray, char mode)
{
    return(tray_version_4(xdr_fdes, tray, mode));
}

int
tray_version_4(XDR_fdes *xdr_fdes, TRAY *tray, char mode)
{
    int		loop;
    TRAY_HOLE	*hole;
    char	*cp;
    XDR		*xdr = xdr_fdes->xdrs;

    cp = &(tray->name[0]);
    if (!(xdr_int(xdr, &tray->tray_id) &&
	  xdr_string(xdr, &cp, NAME_LENGTH) &&
	  xdr_float(xdr, &tray->tray_factor) &&
	  xdr_float(xdr, &tray->xmin) &&
	  xdr_float(xdr, &tray->xmax) &&
	  xdr_float(xdr, &tray->ymin) &&
	  xdr_float(xdr, &tray->ymax) &&
	  xdr_float(xdr, &tray->tray_dist) &&
	  xdr_float(xdr, &tray->block_dist) &&
	  xdr_int(xdr, &tray->hole_count))) {
	XDR_ReportError("Cannot read tray");
	return(XDR_ERROR);
    }

    if (tray->hole_count == 0) {
	xdr_ll_close_stream(xdr_fdes);
	return(0);
    }
    if (mode == XDR_READ_MODE) {
	tray->hole_list = (TRAY_HOLE *)
	    malloc(tray->hole_count*sizeof(TRAY_HOLE));
	if (tray->hole_list == NULL) {
	    fprintf(stderr, "read_tray: Memory allocation failed.\n");
	    return(-1);
	}
    }
    for (loop = 0; loop < tray->hole_count; loop++) {
	hole = tray->hole_list + loop;
	if (!(xdr_int(xdr, &hole->shape) &&
	     xdr_float(xdr, &hole->x) &&
	     xdr_float(xdr, &hole->y) &&
	     xdr_float(xdr, &hole->height) &&
	     xdr_float(xdr, &hole->width))) {
	    perror("read_tray");
	    return(-1);
	}
    }
    xdr_ll_close_stream(xdr_fdes);
    return(0);
}

