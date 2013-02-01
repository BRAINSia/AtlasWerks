/*---------------------------------------------------------------------- 

  xdr_ll_planio.c
  8/31/88 jst

  low level XDR/planio I/O support routines.

  ---------------------------------------------------------------------- */


#ifndef FILE
#include <stdio.h>
#endif

typedef struct
{
    FILE *file;    /* XDR data stream FILE */
    XDR *xdrs;     /* XDR stream */
    int fdes;      /* duplicate file hanlde for *file */
    int o_fdes;    /* original file handle */
} XDR_fdes;


#define XDR_ERROR -1
#define XDR_NO_ERROR 0

#define XDR_READ_MODE 'r'
#define XDR_WRITE_MODE 'a'

XDR_fdes *xdr_ll_open_stream(int fdes, char mode);
void xdr_ll_close_stream(XDR_fdes *);

void XDR_ReportError(const char *);


