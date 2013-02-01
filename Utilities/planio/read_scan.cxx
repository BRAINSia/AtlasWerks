
#include <stdio.h>
#include <stdlib.h>
#include "plan_file_io.h"
#include "plan_xdr.h"
#include "plan_sys.h"
#include "gen.h"
#include "libplanio.h"

#define ABS(x) (((x) < 0) ? -(x) : x)

/*
 ---------------------------------------------------------------------------
 
   NAME
 	read_scan.c
 
   SYNOPSIS
        read_scan(fdes, scan_desc, offset, resolution)
	int fdes;
	short *scan_desc;
	int offset;
	int resolution;
 
 
   DESCRIPTION
        read a scan of the specified resolution from the file
	starting at the given offset
 
   RETURN VALUE
        0  -> ok
	-1 -> error
 
   SEE ALSO
        plan_xdr_defs.c
 
 
 ---------------------------------------------------------------------------
*/

int
read_scan(int fdes, short *scan_desc, int offset, int res)
{
    return(read_scan_xy(fdes, scan_desc, offset, res, res));
}

int
read_scan_xy(int fdes, short *scan_desc, int offset, int x_dim, int y_dim)
{
    XDR_fdes *xdr_fdes;
    int status = XDR_NO_ERROR;
    int nbytes;
    int ret;

    xdr_fdes = xdr_ll_open_stream(fdes, XDR_READ_MODE);
    if (xdr_fdes == NULL) {
	XDR_ReportError("sock_read_scan_1 : Cannot open scan xdr_fdes");
	return(XDR_ERROR);
    }

    if (!is_socket(fdes)) {
#ifndef PLAN_DARWIN
    ret = xdr_setpos(xdr_fdes->xdrs, offset);
#endif
	if (! ret) {
	    XDR_ReportError("sock_read_scan_1 : cannot setpos");
	    status = XDR_ERROR;
	    xdr_ll_close_stream(xdr_fdes);
	    return(status);
	}
    }

    nbytes = x_dim*y_dim*sizeof(short);
    if (! xdrfd_get_bytes(xdr_fdes->xdrs, (caddr_t)scan_desc, nbytes)) {
	XDR_ReportError("sock_read_scan_1 : cannot read scan");
	status = XDR_ERROR;
    }
    
#ifndef PLAN_BIG_ENDIAN
    /* swap bytes - we are running on a little-endian machine */
    if (status != XDR_ERROR)
    	my_swab((char *)scan_desc, (char *)scan_desc, nbytes);
#endif
/*
    check_compression(scan_desc, x_dim, y_dim);
*/
    xdr_ll_close_stream(xdr_fdes);
    return(status);
}

int
write_scan(int fdes, short *scan_desc, int offset, int res)
{
    return(write_scan_xy(fdes, scan_desc, offset, res, res));
}

int
write_scan_xy(int fdes, short *scan_desc, int offset, int x_dim, int y_dim)
{
    XDR_fdes *xdr_fdes;
    int status = XDR_NO_ERROR;
    int nbytes;
    int ret;
    char *tmp;

    xdr_fdes = xdr_ll_open_stream(fdes, XDR_WRITE_MODE);
    if (xdr_fdes == NULL) {
	XDR_ReportError("sock_write_scan_1 : Cannot open scan xdr_fdes");
	return(XDR_ERROR);
    }

    nbytes = x_dim*y_dim*sizeof(short);

#ifndef PLAN_BIG_ENDIAN
    /* we are running on a little-endian machine */
    /* malloc space for swap buffer */
    tmp = (char *)malloc(nbytes);
    if (tmp == NULL) {
	perror("Cannot malloc for scan write");
	return(-1);
    }
    /* swap bytes */
    my_swab((char *)scan_desc, tmp, nbytes);
#else
    tmp = (char *)scan_desc;
#endif

    if (!is_socket(fdes)) {
#ifndef PLAN_DARWIN
    ret = xdr_setpos(xdr_fdes->xdrs, offset);
#endif
	if (! ret) {
	    XDR_ReportError("sock_write_scan_1 : cannot setpos");
	    status = XDR_ERROR;
	}
    }
    if (status == XDR_NO_ERROR) {
	if (! xdrfd_put_bytes(xdr_fdes->xdrs, tmp, nbytes)) {
	    XDR_ReportError("sock_write_scan_1 : cannot read scan");
	    status = XDR_ERROR;
	}
    }

#ifndef PLAN_BIG_ENDIAN
    free(tmp);
#endif

    xdr_ll_close_stream(xdr_fdes);
    return(status);

}

void
my_swab(char *src, char *dest, int num)
{   int		i;
    char	temp;

    for (i = 0; i < num; i += 2) {
	temp = src[i];
	dest[i] = src[i+1];
	dest[i+1] = temp;
    }
}

void
check_compression(short *buff, int x_dim, int y_dim)
{   int		i, j;
    int		index;
    int		end;
    int		num;
    int		size, count;
    int		diff, val;

    num = 2*x_dim*y_dim;

    /* 4 byte int indicating total number of bytes in compressed slice */
    size = 4;

    for (j = 0; j < y_dim; j++) {
	index = j*x_dim;
	val = buff[index];
	count = 0;
	for (i = 0; i < x_dim>>1; i++) {
	    if (buff[index+i] != val) break;
	    if (buff[index+x_dim-1-i] != val) break;
	    count++;
	    if (count == 255) break;
	}
	end = index + x_dim - 1 - i;
	index += i;
	/* 1 byte count, 2 byte short val */
	size += 3;
	/* 2 byte short val */
	size += 2;
	val = buff[index];
	index++;
	for (; index <= end; index++) {
	    diff = buff[index] - val;
	    if (ABS(diff) < 128) {
		val += diff;
		/* 1 byte signed difference value */
		size++;
	    }
	    else {
		/* 1 byte = 0, 2 byte short value */
		size += 3;
		val = buff[index];
	    }
	}
    }
    printf("size: %5d %5d: %.2f\n", num, size, ((float)size)/num);
}


