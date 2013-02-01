

#include <stdio.h>
#include <stdlib.h>
#ifndef WIN32
#include <unistd.h>
#else
#include <io.h>
#endif
#include "plan_file_io.h"
#include <errno.h>
#include <sys/types.h>
#include <sys/stat.h>

#include "gen.h"
#include "plan_xdr.h"
#include "libplanio.h"

static XDR		*ll_xdrs[200];
static XDR_fdes		*ll_xdr_fdes[200];
static char		ll_op[200];


/* ---------- XDR_LL_OPEN_STREAM ----------*/


XDR_fdes *
xdr_ll_open_stream(int fdes, char mode)
{
    enum	xdr_op op;
    // unused var//off_t	pos;

    if (ll_xdr_fdes[fdes] != NULL && ll_op[fdes] != mode) {
	xdr_close(ll_xdr_fdes[fdes]->fdes);
    }
    if (ll_xdr_fdes[fdes] == NULL || ll_op[fdes] != mode) {

	ll_op[fdes] = mode;
	ll_xdrs[fdes] = (XDR *) malloc(sizeof(XDR));
	ll_xdr_fdes[fdes] = (XDR_fdes *) malloc(sizeof(XDR_fdes));
	if (mode ==  'r') op = XDR_DECODE;
	else op = XDR_ENCODE;
	xdrfd_create(ll_xdrs[fdes], fdes, op);
	ll_xdr_fdes[fdes]->xdrs = ll_xdrs[fdes];
	ll_xdr_fdes[fdes]->fdes = fdes;
    }

    return(ll_xdr_fdes[fdes]);
}

/* ---------- XDR_LL_CLOSE_STREAM ----------*/

void
xdr_ll_close_stream(XDR_fdes *xdr_fdes)
{

    if (ll_xdr_fdes[xdr_fdes->fdes] == NULL) {
	fprintf(stderr, "\07 trying to xdr_ll_close a closed stream\n");
	return;
    }
    if (xdr_fdes->xdrs->x_op == XDR_ENCODE) xdr_ll_flush(xdr_fdes);
    if (!is_socket(xdr_fdes->fdes)) {
	xdr_close(xdr_fdes->fdes);
    }
}

void
xdr_close(int fdes)
{
#ifndef PLAN_DARWIN
    xdr_destroy(ll_xdr_fdes[fdes]->xdrs);
#endif
    free((char *)ll_xdr_fdes[fdes]);
    free(ll_xdrs[fdes]);
    ll_xdrs[fdes] = NULL;
    ll_xdr_fdes[fdes] = NULL;
}

/* ---------- XDR_REPORT_ERROR ----------*/

void
XDR_ReportError(const char *str)
{
    perror(str);
    fflush(stderr);
}

void
xdr_ll_flush(XDR_fdes *xdr_fdes)
{
    xdrfd_flush_buff(xdr_fdes->xdrs);
}


