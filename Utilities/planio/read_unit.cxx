
#include <stdio.h>
#include <stdlib.h>
#include "plan_strings.h"
#include "plan_file_io.h"
#include "plan_xdr.h"
#include "gen.h"
#include "extbm.h"
#include "libplanio.h"

int read_write_unit(int fdes, UNIT *unit, char mode);
int unit_version_0(XDR_fdes *xdr_fdes, UNIT *unit);
int unit_version_1(XDR_fdes *xdr_fdes, UNIT *unit);
int unit_version_2(XDR_fdes *xdr_fdes, UNIT *unit);
int unit_version_4(XDR_fdes *xdr_fdes, UNIT *unit);

int
read_unit(int fdes, UNIT *unit)
{
    return(read_write_unit(fdes, unit, XDR_READ_MODE));
}

int
write_unit(int fdes, UNIT *unit)
{
    return(read_write_unit(fdes, unit, XDR_WRITE_MODE));
}

int
read_write_unit(int fdes, UNIT *unit, char mode)
{   int		ret = 0;
    XDR_fdes	*xdr_fdes;
    XDR		*xdr;

    xdr_fdes = xdr_ll_open_stream(fdes, mode);
    if (xdr_fdes == NULL) {
	XDR_ReportError("Cannot open XDR_fdes");
	return(XDR_ERROR);
    }
    xdr = xdr_fdes->xdrs;
    if (!(xdr_int(xdr, &unit->version))) {
	XDR_ReportError("Cannot read unit");
	return(XDR_ERROR);
    }
    if (unit->version < 0) {
	if (!(xdr_int(xdr, &unit->unit_id))) {
	    XDR_ReportError("Cannot read unit");
	    return(XDR_ERROR);
	}
	unit->version = -unit->version;
    }
    else {
	unit->unit_id = unit->version;
	unit->version = 0;
    }
    switch (unit->version) {
	case 0:
	    ret = unit_version_0(xdr_fdes, unit);
	    break;
	case 4:
	    ret = unit_version_4(xdr_fdes, unit);
	    break;
    }
    return(ret);
}

int
unit_version_0(XDR_fdes *xdr_fdes, UNIT *unit)
{   int		ret;
    char	*ver;

    ver = getenv("PHYS_DAT_VERSION");
    if (ver == NULL) {
	return(unit_version_4(xdr_fdes, unit));
    }
    else {
	if (strncmp(ver, "2", 1) == 0) unit->version = 1;
	else if (strncmp(ver, "3", 1) == 0) unit->version = 2;
    }
    switch (unit->version) {
	case 1:
	    ret = unit_version_1(xdr_fdes, unit);
	    break;
	case 2:
	    ret = unit_version_2(xdr_fdes, unit);
	    break;
	case 4:
	default:
	    ret = unit_version_4(xdr_fdes, unit);
	    break;
    }
    return(ret);
}

int
unit_version_1(XDR_fdes *xdr_fdes, UNIT *unit)
{
    int		loop;
    char	*cp, *cp1, *cp2;
    float	SDD1, SDD2;
    JAW		*jaw;
    XDR		*xdr = xdr_fdes->xdrs;

    cp = &(unit->name[0]);
    if (!(xdr_string(xdr, &cp, NAME_LENGTH) &&
	  xdr_int(xdr, &unit->modality) &&
	  xdr_float(xdr, &unit->dmax) &&
	  xdr_float(xdr, &unit->SAD) &&
	  xdr_float(xdr, &SDD1) &&
	  xdr_float(xdr, &SDD2) &&
	  xdr_float(xdr, &unit->source_diam1) &&
	  xdr_float(xdr, &unit->source_diam2) &&
	  xdr_float(xdr, &unit->source_ratio) &&
	  xdr_int(xdr, &unit->x_jaw_count) &&
	  xdr_int(xdr, &unit->y_jaw_count))) {
	XDR_ReportError("Cannot read unit");
	return(XDR_ERROR);
    }
    unit->PSD = 7.0;
    for (loop = 0; loop < unit->x_jaw_count + unit->y_jaw_count; loop++) {
	jaw = unit->jaw + loop;
	cp1 = &(jaw->name_1[0]);
	cp2 = &(jaw->name_2[0]);
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
	if (loop < unit->x_jaw_count) jaw->SDD = SDD2;
	else jaw->SDD = SDD1;
    }

    if (!(xdr_int(xdr, &unit->gantry_sense) &&
	  xdr_float(xdr, &unit->gantry_offset) &&
	  xdr_float(xdr, &unit->gantry_minimum) &&
	  xdr_float(xdr, &unit->gantry_maximum))) {
	XDR_ReportError("Cannot read unit gantry");
	return(XDR_ERROR);
    }
    if (!(xdr_int(xdr, &unit->table_sense) &&
	  xdr_float(xdr, &unit->table_offset) &&
	  xdr_float(xdr, &unit->table_minimum) &&
	  xdr_float(xdr, &unit->table_maximum))) {
	XDR_ReportError("Cannot read unit table");
	return(XDR_ERROR);
    }
    if (!(xdr_int(xdr, &unit->collimator_sense) &&
	  xdr_float(xdr, &unit->collimator_offset) &&
	  xdr_float(xdr, &unit->collimator_minimum) &&
	  xdr_float(xdr, &unit->collimator_maximum))) {
	XDR_ReportError("Cannot read unit collimator");
	return(XDR_ERROR);
    }

    xdr_ll_close_stream(xdr_fdes);
    return(0);
}

int
unit_version_2(XDR_fdes *xdr_fdes, UNIT *unit)
{
    int		loop;
    char	*cp, *cp1, *cp2;
    float	SDD1, SDD2;
    JAW		*jaw;
    XDR		*xdr = xdr_fdes->xdrs;

    cp = &(unit->name[0]);
    if (!(xdr_string(xdr, &cp, NAME_LENGTH) &&
	  xdr_int(xdr, &unit->modality) &&
	  xdr_float(xdr, &unit->dmax) &&
	  xdr_float(xdr, &unit->SAD) &&
	  xdr_float(xdr, &SDD1) &&
	  xdr_float(xdr, &SDD2) &&
	  xdr_float(xdr, &unit->source_diam1) &&
	  xdr_float(xdr, &unit->source_diam2) &&
	  xdr_float(xdr, &unit->source_ratio) &&
	  xdr_int(xdr, &unit->x_jaw_count) &&
	  xdr_int(xdr, &unit->y_jaw_count))) {
	XDR_ReportError("Cannot read unit");
	return(XDR_ERROR);
    }
    unit->PSD = 7.0;
    for (loop = 0; loop < unit->x_jaw_count + unit->y_jaw_count; loop++) {
	jaw = unit->jaw + loop;
	cp1 = &(jaw->name_1[0]);
	cp2 = &(jaw->name_2[0]);
	if (!(xdr_int(xdr, &jaw->independent) &&
	      xdr_float(xdr, &jaw->min_1) &&
	      xdr_float(xdr, &jaw->max_1) &&
	      xdr_float(xdr, &jaw->min_2) &&
	      xdr_float(xdr, &jaw->max_2) &&
	      xdr_float(xdr, &jaw->min_w) &&
	      xdr_float(xdr, &jaw->max_w) &&
	      xdr_string(xdr, &cp1, NAME_LENGTH) &&
	      xdr_string(xdr, &cp2, NAME_LENGTH))) {
	    XDR_ReportError("Cannot read jaws");
	    return(XDR_ERROR);
	}
	if (loop < unit->x_jaw_count) jaw->SDD = SDD2;
	else jaw->SDD = SDD1;
    }

    if (!(xdr_int(xdr, &unit->gantry_sense) &&
	  xdr_float(xdr, &unit->gantry_offset) &&
	  xdr_float(xdr, &unit->gantry_minimum) &&
	  xdr_float(xdr, &unit->gantry_maximum))) {
	XDR_ReportError("Cannot read unit gantry");
	return(XDR_ERROR);
    }
    if (!(xdr_int(xdr, &unit->table_sense) &&
	  xdr_float(xdr, &unit->table_offset) &&
	  xdr_float(xdr, &unit->table_minimum) &&
	  xdr_float(xdr, &unit->table_maximum))) {
	XDR_ReportError("Cannot read unit table");
	return(XDR_ERROR);
    }
    if (!(xdr_int(xdr, &unit->collimator_sense) &&
	  xdr_float(xdr, &unit->collimator_offset) &&
	  xdr_float(xdr, &unit->collimator_minimum) &&
	  xdr_float(xdr, &unit->collimator_maximum))) {
	XDR_ReportError("Cannot read unit collimator");
	return(XDR_ERROR);
    }

    xdr_ll_close_stream(xdr_fdes);
    return(0);
}

int
unit_version_4(XDR_fdes *xdr_fdes, UNIT *unit)
{
    int		loop;
    char	*cp, *cp1, *cp2;
    JAW		*jaw;
    XDR		*xdr = xdr_fdes->xdrs;

    cp = &(unit->name[0]);
    if (!(xdr_string(xdr, &cp, NAME_LENGTH) &&
	  xdr_int(xdr, &unit->modality) &&
	  xdr_float(xdr, &unit->dmax) &&
	  xdr_float(xdr, &unit->SAD) &&
	  xdr_float(xdr, &unit->PSD) &&
	  xdr_float(xdr, &unit->source_diam1) &&
	  xdr_float(xdr, &unit->source_diam2) &&
	  xdr_float(xdr, &unit->source_ratio) &&
	  xdr_int(xdr, &unit->x_jaw_count) &&
	  xdr_int(xdr, &unit->y_jaw_count))) {
	XDR_ReportError("Cannot read unit");
	return(XDR_ERROR);
    }
    for (loop = 0; loop < unit->x_jaw_count + unit->y_jaw_count; loop++) {
	jaw = unit->jaw + loop;
	cp1 = &(jaw->name_1[0]);
	cp2 = &(jaw->name_2[0]);
	if (!(xdr_int(xdr, &jaw->independent) &&
	      xdr_float(xdr, &jaw->SDD) &&
	      xdr_float(xdr, &jaw->min_1) &&
	      xdr_float(xdr, &jaw->max_1) &&
	      xdr_float(xdr, &jaw->min_2) &&
	      xdr_float(xdr, &jaw->max_2) &&
	      xdr_float(xdr, &jaw->min_w) &&
	      xdr_float(xdr, &jaw->max_w) &&
	      xdr_string(xdr, &cp1, NAME_LENGTH) &&
	      xdr_string(xdr, &cp2, NAME_LENGTH))) {
	    XDR_ReportError("Cannot read jaws");
	    return(XDR_ERROR);
	}
    }

    if (!(xdr_int(xdr, &unit->gantry_sense) &&
	  xdr_float(xdr, &unit->gantry_offset) &&
	  xdr_float(xdr, &unit->gantry_minimum) &&
	  xdr_float(xdr, &unit->gantry_maximum))) {
	XDR_ReportError("Cannot read unit gantry");
	return(XDR_ERROR);
    }
    if (!(xdr_int(xdr, &unit->table_sense) &&
	  xdr_float(xdr, &unit->table_offset) &&
	  xdr_float(xdr, &unit->table_minimum) &&
	  xdr_float(xdr, &unit->table_maximum))) {
	XDR_ReportError("Cannot read unit table");
	return(XDR_ERROR);
    }
    if (!(xdr_int(xdr, &unit->collimator_sense) &&
	  xdr_float(xdr, &unit->collimator_offset) &&
	  xdr_float(xdr, &unit->collimator_minimum) &&
	  xdr_float(xdr, &unit->collimator_maximum))) {
	XDR_ReportError("Cannot read unit collimator");
	return(XDR_ERROR);
    }

    xdr_ll_close_stream(xdr_fdes);
    return(0);
}

