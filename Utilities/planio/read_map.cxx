

#include <stdio.h>
#include <stdlib.h>
#ifndef WIN32
#include <unistd.h>
#else
#include <io.h>
#endif
#include <fcntl.h>
#include "gen.h"
#include "plan_file_io.h"
#include "plan_xdr.h"
#include "libplanio.h"

int
XDR_WriteMap(int fdes, THREED_MAP *map, int slice, int num_slices, char mode);

int
write_map(char *name, THREED_MAP *map, int slice, int num_slices)
{  int		fdes;

    fdes = open(name, O_RDWR | O_TRUNC | O_CREAT, 0666);
    if (fdes < 0) {
	fprintf(stderr, "Can't open 2D map file for output\n");
	exit(0);
    }
    return(XDR_WriteMap(fdes, map, slice, num_slices, XDR_WRITE_MODE));
}

int
read_map(char *name, THREED_MAP *map, int slice, int num_slices)
{  int fdes;

    fdes = open(name, O_RDONLY, 0);
    if (fdes < 0) {
	fprintf(stderr, "Can't open 2D map file for input %s\n", name);
	exit(0);
    }
    return(XDR_WriteMap(fdes, map, slice, num_slices, XDR_READ_MODE));
}


int
XDR_WriteMap(int fdes, THREED_MAP *map, int slice, int num_slices, char mode)
{
  // unused var//char		*cp;
   XDR_fdes	*xdr_fdes;
   int		status = XDR_NO_ERROR;
   // unused var//unsigned int	lp;
   int		i;
   int		slicecnt;
   int		pixcnt;

   xdr_fdes = xdr_ll_open_stream(fdes, mode);
   if (xdr_fdes == NULL) {
      XDR_ReportError("Cannot open XDR_fdes");
      return(XDR_ERROR);
   }
   if (! xdr_int(xdr_fdes->xdrs, &(map->x_dim))) {
      XDR_ReportError("Cannot write x dim");
      status = XDR_ERROR;
   }
   if (! xdr_float(xdr_fdes->xdrs, &(map->x_start))) {
      XDR_ReportError("Cannot write x start");
      status = XDR_ERROR;
   }
   if (! xdr_float(xdr_fdes->xdrs, &(map->x_end))) {
      XDR_ReportError("Cannot write x end");
      status = XDR_ERROR;
   }
   if (! xdr_float(xdr_fdes->xdrs, &(map->x_scale))) {
      XDR_ReportError("Cannot write x scale");
      status = XDR_ERROR;
   }
   if (! xdr_float(xdr_fdes->xdrs, &(map->inv_x_scale))) {
      XDR_ReportError("Cannot write inv x scale");
      status = XDR_ERROR;
   }
   if (! xdr_int(xdr_fdes->xdrs, &(map->y_dim))) {
      XDR_ReportError("Cannot write y dim");
      status = XDR_ERROR;
   }
   if (! xdr_float(xdr_fdes->xdrs, &(map->y_start))) {
      XDR_ReportError("Cannot write y start");
      status = XDR_ERROR;
   }
   if (! xdr_float(xdr_fdes->xdrs, &(map->y_end))) {
      XDR_ReportError("Cannot write y end");
      status = XDR_ERROR;
   }
   if (! xdr_float(xdr_fdes->xdrs, &(map->y_scale))) {
      XDR_ReportError("Cannot write y scale");
      status = XDR_ERROR;
   }
   if (! xdr_float(xdr_fdes->xdrs, &(map->inv_y_scale))) {
      XDR_ReportError("Cannot write y inv scale");
      status = XDR_ERROR;
   }

   if (! xdr_int(xdr_fdes->xdrs, &(num_slices))) {
      XDR_ReportError("Cannot write z dim");
      status = XDR_ERROR;
   }
   if (mode == XDR_READ_MODE) {
	map->z_dim = num_slices;
   }

   slicecnt = map->x_dim*map->y_dim;
   map->slicecnt = slicecnt;
   pixcnt = slicecnt*num_slices;
   if (mode == XDR_READ_MODE) {	
	map->data = (float *)malloc(pixcnt*sizeof(float));
   }
   slice *= slicecnt;
   for (i = 0; i < pixcnt; i++) {
      if (! xdr_float(xdr_fdes->xdrs, map->data+i+slice)) {
         XDR_ReportError("Cannot read/write map matrix");
         status = XDR_ERROR;
         break;
      }
   }
   xdr_ll_close_stream(xdr_fdes);
   close(fdes);
   return(status);
}


