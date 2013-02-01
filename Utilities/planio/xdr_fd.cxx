
#include <stdio.h>
#include <stdlib.h>
#ifndef WIN32
#include <unistd.h>
#else
#include <io.h>
#endif
#include "gen.h"
#include "plan_file_io.h"
#include "plan_xdr.h"
#include "libplanio.h"

#ifdef PLAN_WINNT
#include <winsock2.h>
#endif

#ifdef PLAN_OSF1_3
#define CONST1
#define Int int
#define Int32 int
#define Uint int
#else
#ifdef PLAN_LINUX
#define CONST1 const
#define Int long
#define Int32 int32_t
#define Uint u_int
#else
#ifdef PLAN_IRIX
#include <netinet/in.h>
#define CONST1
#define Int long
#define Int32 long
#define Uint unsigned int
#else
#define CONST1
#define Int long 
#define Int32 long
#define Uint int
#endif
#endif
#endif

static bool_t	xdrfd_getlong(XDR *,  Int *);
static bool_t	xdrfd_putlong(XDR *,  CONST1 Int *);
static bool_t	xdrfd_getbytes(XDR *, caddr_t, Uint);
static bool_t	xdrfd_putbytes(XDR *, CONST1 char *, Uint);
static u_int	xdrfd_getpos(CONST1 XDR *);
static bool_t	xdrfd_setpos(XDR *, u_int);
static Int32 *	xdrfd_inline(XDR *, unsigned int);
static void	xdrfd_destroy(XDR *);


/*
 * Buffer management stuff
*/
#define XDRFD_BUFF_SIZE (4096)

typedef struct
{
    int fd;
/*
  These are not redundant.  bytes_in_buff is needed by read ops to
  know when to stop.  bytes_used points to next usable byte.
  For writes they should stay in sync.
*/
    int bytes_in_buff;
    int bytes_used;
    unsigned char buff[XDRFD_BUFF_SIZE];
} XDRFD_BUFF;

bool_t
xdrfd_flush_buff(XDR *xdrs)
{
    XDRFD_BUFF *ptr = (XDRFD_BUFF *) xdrs->x_private;

    if (sock_write(ptr->fd, (char *)ptr->buff, ptr->bytes_used)) {
	fprintf(stderr, "bad write: %d %p %d\n", 
		ptr->fd, ptr->buff, ptr->bytes_used);
	perror("xdrfd_flush_buff : sock_write ");
	return (FALSE);
    }

    ptr->bytes_in_buff = 0;
    ptr->bytes_used = 0;
    return (TRUE);
}

bool_t
xdrfd_fill_buff(XDR *xdrs, int count)
{   // unused var//int         pos;

    XDRFD_BUFF *ptr = (XDRFD_BUFF *) xdrs->x_private;

    if (is_socket(ptr->fd)) {
    	if (count > XDRFD_BUFF_SIZE) count = XDRFD_BUFF_SIZE;
	sock_read(ptr->fd, (char *)ptr->buff, count);
	ptr->bytes_in_buff = count;
    }
    else {
	ptr->bytes_in_buff = read(ptr->fd, ptr->buff, XDRFD_BUFF_SIZE);
    }

    if (ptr->bytes_in_buff == -1) {
	fprintf (stderr, "xdrfd_fill_buff: read fail\n");
	perror ("to wit");
	return (FALSE);
    }

/*
  0 byte count means EOF - where should it be special-cased?
*/
    if (ptr->bytes_in_buff == 0) {
	fprintf(stderr, "xdrfd_fill_buff: EOF on read");
	perror("to wit");
	return(FALSE);
    }

    ptr->bytes_used = 0;
    return (TRUE);
}

bool_t
xdrfd_get_bytes(XDR *xdrs, caddr_t dest, int count)
{
    XDRFD_BUFF *ptr = (XDRFD_BUFF *) xdrs->x_private;
    register int so_far;

    if (count <= 0) return(TRUE);

    for (so_far = 0; so_far < count; ) {
	if (ptr->bytes_used == ptr->bytes_in_buff) {
	    if (! xdrfd_fill_buff (xdrs, count - so_far)) {
		if (is_socket(ptr->fd)) {
		    fprintf(stderr, "\nsocket broken - exiting\n");
		    exit(-1);
		}
		return(FALSE);
	    }
	}
	dest[so_far++] = ptr->buff[ptr->bytes_used++];
    }

    return(TRUE);
}

bool_t
xdr_eof(int fdes)
{   int		pos, fend;
    XDR_fdes	*xdr_fdes;
    XDRFD_BUFF *ptr;
    

    xdr_fdes = xdr_ll_open_stream(fdes, XDR_READ_MODE);
    ptr = (XDRFD_BUFF *)xdr_fdes->xdrs->x_private;

    if (ptr->bytes_used == ptr->bytes_in_buff) {
	pos = lseek(fdes, 0, SEEK_CUR);
	fend = lseek(fdes, 0, SEEK_END);
	if (pos == fend) {
	    return(TRUE);
	}
	lseek(fdes, pos, SEEK_SET);
    }
    return(FALSE);
}

bool_t
xdrfd_put_bytes(XDR *xdrs, caddr_t src, int count)
{
    XDRFD_BUFF *ptr = (XDRFD_BUFF *) xdrs->x_private;
    register int so_far;

    if (count <= 0)
	return(TRUE);

    for (so_far = 0; so_far < count; )
    {
	if (ptr->bytes_used == XDRFD_BUFF_SIZE)
	    if (! xdrfd_flush_buff (xdrs))
		return(FALSE);
	ptr->buff[ptr->bytes_used++] = src[so_far++];
    }
    return(TRUE);
}
	
/*
 * Destroy a stdio xdr stream.
 * Cleans up the xdr stream handle xdrs previously set up by xdrfd_create.
 */
static void
xdrfd_destroy(XDR *xdrs)
{
    XDRFD_BUFF *ptr = (XDRFD_BUFF *) xdrs->x_private;


    if (xdrs->x_op == XDR_ENCODE)
	xdrfd_flush_buff (xdrs);


    if (!is_socket(ptr->fd)) {
	if (lseek(ptr->fd, ptr->bytes_used-ptr->bytes_in_buff, SEEK_CUR) < 0) {
	}
    }

    /*
    if (!is_socket(ptr->fd)) {
	xdrs->x_private = (caddr_t) NULL;
    }
    */
    xdrs->x_private = (caddr_t) NULL;
    free (ptr);
}

static bool_t
xdrfd_getlong(XDR *xdrs, Int *lp)
{

    if (! xdrfd_get_bytes(xdrs, (caddr_t)lp, 4))
	return (FALSE);

#ifndef mc68000
    *lp = ntohl(*lp);
#endif
    return (TRUE);
}

static bool_t
xdrfd_putlong(XDR *xdrs, CONST1 Int *lp)
{

#ifndef mc68000
    Int mycopy = htonl(*lp);
    lp = &mycopy;
#endif
    if (! xdrfd_put_bytes(xdrs, (caddr_t)lp, 4))
	return (FALSE);
    return (TRUE);
}

static bool_t
xdrfd_getbytes(XDR *xdrs, caddr_t addr, Uint len)
{

    if ((len != 0) && (! xdrfd_get_bytes(xdrs, addr, len)))
	return (FALSE);
    return (TRUE);
}

static bool_t
xdrfd_putbytes(XDR *xdrs, CONST1 char *addr, Uint len)
{

    if ((len != 0) && (! xdrfd_put_bytes(xdrs, (char *)addr, len)))
	return (FALSE);
    return (TRUE);
}

static u_int
xdrfd_getpos(CONST1 XDR *xdrs)
{   u_int	pos;
    XDRFD_BUFF *ptr = (XDRFD_BUFF *) xdrs->x_private;

    pos = lseek(ptr->fd, 0 , SEEK_CUR) -
		ptr->bytes_in_buff + ptr->bytes_used;
    return(pos);
}

static bool_t
xdrfd_setpos(XDR *xdrs, u_int pos) 
{ 
    XDRFD_BUFF *ptr = (XDRFD_BUFF *) xdrs->x_private;

    if (is_socket(ptr->fd)) return(FALSE);

    /* if xdr stream is open for writing then flush buffer to file */
    if (xdrs->x_op == XDR_ENCODE)
	xdrfd_flush_buff (xdrs);

    if (lseek(ptr->fd, pos, SEEK_SET) < 0)

    ptr->bytes_in_buff = 0;
    ptr->bytes_used = 0;

    return (TRUE);
}

//unsigned int len
static Int32 *
xdrfd_inline(XDR *xdrs, unsigned int len)
{

	/*
	 * Must do some work to implement this: must insure
	 * enough data in the underlying stdio buffer,
	 * that the buffer is aligned so that we can indirect through a
	 * long *, and stuff this pointer in xdrs->x_buf.  Doing
	 * a fread or fwrite to a scratch buffer would defeat
	 * most of the gains to be had here and require storage
	 * management on this buffer, so we don't do this.
	 */
	return (NULL);
}

/*
 * Initialize a stdio xdr stream.
 * Sets the xdr stream handle xdrs for use on the stream file.
 * Operation flag is set to op.
 */

void
xdrfd_create(XDR *xdrs, int fd, enum xdr_op op)
{
    XDRFD_BUFF *ptr;

    ptr = (XDRFD_BUFF *) malloc(sizeof(XDRFD_BUFF));
    if (ptr == NULL)
    {
	fprintf (stderr, "xdrfd_create: memory allocation fail.\n");
	exit (1);
    }
    else
    {
	ptr->fd = fd;
	ptr->bytes_in_buff = 0;
	ptr->bytes_used = 0;
    }

    xdrs->x_op = op;
#if defined(PLAN_IRIX) || defined(PLAN_SOLARIS2_4)
    xdrs->x_ops = (struct xdr_ops *)malloc(sizeof(struct xdr_ops));
#else
#ifdef __cplusplus
    xdrs->x_ops = (XDR::xdr_ops *)malloc(sizeof(XDR::xdr_ops));
#else
    xdrs->x_ops = (struct xdr_ops *)malloc(sizeof(struct xdr_ops));
#endif
#endif
#ifndef PLAN_DARWIN
    xdrs->x_ops->x_getlong = xdrfd_getlong;
    xdrs->x_ops->x_putlong = xdrfd_putlong;
    xdrs->x_ops->x_getbytes = xdrfd_getbytes;
    xdrs->x_ops->x_putbytes = xdrfd_putbytes;
    xdrs->x_ops->x_getpostn = xdrfd_getpos;
    xdrs->x_ops->x_setpostn = xdrfd_setpos;
    xdrs->x_ops->x_inline = xdrfd_inline;
    xdrs->x_ops->x_destroy = xdrfd_destroy;

    xdrs->x_private = (caddr_t) ptr;
    xdrs->x_handy = 0;
    xdrs->x_base = 0;
#endif
}


int
xdr_pos(XDR *xdrs)
{
    return(xdrfd_getpos(xdrs));
}

#ifndef PLAN_WINNT

#ifndef PLAN_DARWIN
#include <poll.h>
#endif

#ifdef PLAN_LINUX
#define POLLNORM POLLIN
#endif

int
fd_poll(int fd, int msec)
{   int			ret;
#ifndef PLAN_DARWIN
    struct pollfd	fds;

    fds.fd = fd;
    fds.events = POLLERR | POLLHUP | POLLNORM;
    fds.revents = 0;
    ret = poll(&fds, 1, msec);

    if (ret == 0) {
	return(0);
    }
    if (fds.revents & (POLLERR | POLLHUP)) return(-1);
#endif
    return(ret);
}

#else

int
fd_poll(int fd, int msec)
{   fd_set		fds;
#ifndef PLAN_DARWIN
    struct timeval	timeout;

    FD_ZERO(&fds);
    FD_SET(fd, &fds);
    timeout.tv_sec = 0;
    while (msec > 1000) {
	timeout.tv_sec++;
	msec -= 1000;
    }
    timeout.tv_usec = 1000*msec;
    if (select(FD_SETSIZE, &fds, NULL, NULL, &timeout) < 0) {
	fprintf(stderr, "fd_poll(%d): bad select() call\n", fd);
	return(0);
    }
#endif
    return(FD_ISSET(fd, &fds));
}

#endif

