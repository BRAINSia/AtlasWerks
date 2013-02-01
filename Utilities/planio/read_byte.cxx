
#include <stdio.h>
#include "plan_file_io.h"
#include "plan_xdr.h"
#include "gen.h"
#include "extbm.h"
#include "libplanio.h"

int
read_byte(int fdes, char *ptr)
{
    return(read_write_byte(fdes, ptr, XDR_READ_MODE));
}

int
write_byte(int fdes, char *ptr)
{
    return(read_write_byte(fdes, ptr, XDR_WRITE_MODE));
}

int
read_bytes(int fdes, char *ptr, int num)
{

    return(read_byte_array(fdes, ptr, num));
}

int
write_bytes(int fdes, char *ptr, int num)
{

    return(write_byte_array(fdes, ptr, num));
}

int
read_write_byte(int fdes, char *ptr, char mode)
{
    XDR_fdes *xdr_fdes;
    xdr_fdes = xdr_ll_open_stream(fdes, mode);
    if (xdr_fdes == NULL) {
	XDR_ReportError("Cannot open XDR_fdes");
	return(XDR_ERROR);
    }

    if (! xdr_char(xdr_fdes->xdrs, ptr)) {
	XDR_ReportError("Cannot read/write byte");
	return(XDR_ERROR);
    }

    xdr_ll_close_stream(xdr_fdes); 
    return(XDR_NO_ERROR);
}

int
read_int(int fdes, int * ptr)
{
    return(read_write_ints(fdes, ptr, 1, XDR_READ_MODE));
}

int
write_int(int fdes, int *ptr)
{
    return(read_write_ints(fdes, ptr, 1, XDR_WRITE_MODE));
}

int
read_ints(int fdes, int *ptr, int num)
{
    return(read_write_ints(fdes, ptr, num, XDR_READ_MODE));
}

int
write_ints(int fdes, int *ptr, int num)
{   
  // unused var//int		loop;

    return(read_write_ints(fdes, ptr, num, XDR_WRITE_MODE));
}

int
read_write_ints(int fdes, int *ptr, int num, char mode)
{   int		i;

    XDR_fdes *xdr_fdes;

    xdr_fdes = xdr_ll_open_stream(fdes, mode);
    if (xdr_fdes == NULL) {
	XDR_ReportError("Cannot open XDR_fdes");
	return(XDR_ERROR);
    }

    for (i = 0; i < num; i++) {
    	if (! xdr_int(xdr_fdes->xdrs, ptr+i)) {
	    XDR_ReportError("Cannot read/write int");
	    return(XDR_ERROR);
	}
    }

    xdr_ll_close_stream(xdr_fdes); 
    return(XDR_NO_ERROR);
}

int
read_short(int fdes, short *ptr)
{
    return(read_write_short(fdes, ptr, XDR_READ_MODE));
}

int
write_short(int fdes, short *ptr)
{
    return(read_write_short(fdes, ptr, XDR_WRITE_MODE));
}

int
read_shorts(int fdes, short *ptr, int num)
{   int		loop;

    for (loop = 0; loop < num; loop++) {
	read_short(fdes, ptr+loop);
    }
    return(XDR_NO_ERROR);
}

int
write_shorts(int fdes, short *ptr, int num)
{   int		loop;

    for (loop = 0; loop < num; loop++) {
	write_short(fdes, ptr+loop);
    }
    return(XDR_NO_ERROR);
}

int
read_write_short(int fdes, short *ptr, char mode)
{
    XDR_fdes *xdr_fdes;

    xdr_fdes = xdr_ll_open_stream(fdes, mode);
    if (xdr_fdes == NULL) {
	XDR_ReportError("Cannot open XDR_fdes");
	return(XDR_ERROR);
    }

    if (! xdr_short(xdr_fdes->xdrs, ptr)) {
	XDR_ReportError("Cannot read/write int");
	return(XDR_ERROR);
    }

    xdr_ll_close_stream(xdr_fdes); 
    return(XDR_NO_ERROR);
}

int
read_float(int fdes, float *ptr)
{
    return(read_write_floats(fdes, ptr, 1, XDR_READ_MODE));
}

int
write_float(int fdes, float *ptr)
{
    return(read_write_floats(fdes, ptr, 1, XDR_WRITE_MODE));
}

int
read_floats(int fdes, float *ptr, int num)
{
    return(read_write_floats(fdes, ptr, num, XDR_READ_MODE));
}

int
write_floats(int fdes, float *ptr, int num)
{
    return(read_write_floats(fdes, ptr, num, XDR_WRITE_MODE));
}

int
read_write_floats(int fdes, float *ptr, int num, char mode)
{   int		i;
    XDR_fdes *xdr_fdes;

    xdr_fdes = xdr_ll_open_stream(fdes, mode);
    if (xdr_fdes == NULL) {
	XDR_ReportError("Cannot open XDR_fdes");
	return(XDR_ERROR);
    }

    for (i = 0; i < num; i++) {
	if (! xdr_float(xdr_fdes->xdrs, ptr+i)) {
	    XDR_ReportError("Cannot read/write int");
	    return(XDR_ERROR);
	}
    }

    xdr_ll_close_stream(xdr_fdes); 
    return(XDR_NO_ERROR);
}

int
read_byte_array(int fdes, char *ptr, int num)
{
    XDR_fdes *xdr_fdes;

    xdr_fdes = xdr_ll_open_stream(fdes, XDR_READ_MODE);
    if (xdr_fdes == NULL) {
	XDR_ReportError("Cannot open XDR_fdes");
	return(XDR_ERROR);
    }

    if (! xdrfd_get_bytes(xdr_fdes->xdrs, ptr, num)) {
	XDR_ReportError("cannot read_write_byte_array");
	return(XDR_ERROR);
    }
    xdr_ll_close_stream(xdr_fdes); 
    return(XDR_NO_ERROR);
}

int
write_byte_array(int fdes, char *ptr, int num)
{
    XDR_fdes *xdr_fdes;

    xdr_fdes = xdr_ll_open_stream(fdes, XDR_WRITE_MODE);
    if (xdr_fdes == NULL) {
	XDR_ReportError("Cannot open XDR_fdes");
	return(XDR_ERROR);
    }

    if (! xdrfd_put_bytes(xdr_fdes->xdrs, ptr, num)) {
	XDR_ReportError("cannot read_write_byte_array");
	return(XDR_ERROR);
    }
    xdr_ll_close_stream(xdr_fdes); 
    return(XDR_NO_ERROR);
}

