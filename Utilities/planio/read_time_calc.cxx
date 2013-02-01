
#include <stdio.h>
#include "plan_file_io.h"
#include "plan_xdr.h"
#include "gen.h"
#include "extbm.h"
#include "libplanio.h"

int read_write_time_calc(int fdes, TIME_CALC_PARAMETERS *tc, char mode);
int
time_calc_version_0(XDR_fdes *xdr_fdes, TIME_CALC_PARAMETERS *time_calc);
int
time_calc_version_4(XDR_fdes *xdr_fdes, TIME_CALC_PARAMETERS *time_calc);


/*
 ---------------------------------------------------------------------------
 
   NAME
 	read_time_calc - read time_calc descriptor from file
 
   SYNOPSIS
 
 
   DESCRIPTION
 
 
   RETURN VALUE
 	0 if OK, -1 otherwise
 
 
 ---------------------------------------------------------------------------
*/

int
read_time_calc(int fdes, TIME_CALC_PARAMETERS *time_calc)
{
    return(read_write_time_calc(fdes, time_calc, XDR_READ_MODE));
}

int
write_time_calc(int fdes, TIME_CALC_PARAMETERS *time_calc)
{
    return(read_write_time_calc(fdes, time_calc, XDR_WRITE_MODE));
}

int
read_write_time_calc(int fdes, TIME_CALC_PARAMETERS *time_calc, char mode)
{   int		ret=0;
    XDR_fdes    *xdr_fdes;
    XDR         *xdr;

    xdr_fdes = xdr_ll_open_stream(fdes, mode);
    if (xdr_fdes == NULL) {
	XDR_ReportError("Cannot open XDR_fdes");
	return(XDR_ERROR);
    }
    xdr = xdr_fdes->xdrs;
    if (!(xdr_int(xdr, &time_calc->version))) {
	XDR_ReportError("Cannot read time_calc");
	return(XDR_ERROR);
    }
    if (time_calc->version < 0) {
	if (!(xdr_int(xdr, &time_calc->unit_id))) {
	    XDR_ReportError("Cannot read time_calc");
	    return(XDR_ERROR);
	}
	time_calc->version = -time_calc->version;
    }
    else {
	time_calc->unit_id = time_calc->version;
	time_calc->version = 0;
    }
    switch (time_calc->version) {
	case 0:
	    ret = time_calc_version_0(xdr_fdes, time_calc);
	    break;
	case 4:
	    ret = time_calc_version_4(xdr_fdes, time_calc);
	    break;
    }
    return(ret);
}

int
time_calc_version_0(XDR_fdes *xdr_fdes, TIME_CALC_PARAMETERS *time_calc)
{
    return(time_calc_version_4(xdr_fdes, time_calc));
}

int
time_calc_version_4(XDR_fdes *xdr_fdes, TIME_CALC_PARAMETERS *time_calc)
{   int		xloop, yloop;
    XDR		*xdr = xdr_fdes->xdrs;

    if (!(xdr_float(xdr, &time_calc->dose_rate) &&
	  xdr_int(xdr, &time_calc->calibration_date.day) &&
	  xdr_int(xdr, &time_calc->calibration_date.month) &&
	  xdr_int(xdr, &time_calc->calibration_date.year) &&
	  xdr_int(xdr, &time_calc->calibration_date.dow) &&
	  xdr_float(xdr, &time_calc->decay_constant) &&
	  xdr_float(xdr, &time_calc->cal_dist) &&
	  xdr_float(xdr, &time_calc->cal_depth) &&
	  xdr_float(xdr, &time_calc->end_effect) &&
	  xdr_int(xdr, &time_calc->time_units) &&
	  xdr_int(xdr, &time_calc->Sc_x_count) &&
	  xdr_int(xdr, &time_calc->Sc_y_count))) {
	XDR_ReportError("Cannot read time_calc");
	return(XDR_ERROR);
    }
    for (xloop = 0; xloop < time_calc->Sc_x_count; xloop++) {
	if (!(xdr_float(xdr, &time_calc->Sc_x_positions[xloop]))) {
	    XDR_ReportError("Cannot read time_calc Sc_x_positions");
	    return(XDR_ERROR);
	}
    }
    for (yloop = 0; yloop < time_calc->Sc_y_count; yloop++) {
	if (!(xdr_float(xdr, &time_calc->Sc_y_positions[yloop]))) {
	    XDR_ReportError("Cannot read time_calc Sc_y_positions");
	    return(XDR_ERROR);
	}
    }
    for (yloop = 0; yloop < time_calc->Sc_y_count; yloop++) {
	for (xloop = 0; xloop < time_calc->Sc_x_count; xloop++) {
	    if (!(xdr_float(xdr, &time_calc->Sc[yloop][xloop]))) {
		XDR_ReportError("Cannot read time_calc Sc");
		return(XDR_ERROR);
	    }
	}
    }
    xdr_ll_close_stream(xdr_fdes);
    return(0);
}

