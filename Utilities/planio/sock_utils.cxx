
#include <stdio.h>

#include "plan_file_io.h"

#ifdef WIN32
#include <winsock2.h>
#include <io.h>
#else
#include <sys/socket.h>
#include <unistd.h>
#endif

#include "gen.h"
#include "plan_xdr.h"
#include "libplanio.h"

/* Socket indication routines */

static char is_sock[1024];

void
set_socket(int fd)
{
    is_sock[fd] = 1;
}

void
clear_socket(int fd)
{
    is_sock[fd] = 0;
}

int
is_socket(int fd)
{
    return(is_sock[fd]);
}

int
close_socket(int fd)
{
    if (!is_socket(fd))
	fprintf(stderr, "Error: close_socket on non-socket: %d\n", fd);
    is_sock[fd] = 0;
    clear_socket(fd);
    return(close(fd));
}

/* Socket read/write routines */

int
sock_read(int fdes, char *buff, unsigned int buff_size)
{   int		num;


    while (buff_size) {
#ifdef PLAN_WINNT
	int is_sock = is_socket(fdes);
	if (is_sock) num = recvfrom(fdes, buff, buff_size, 0, NULL, 0);
	else num = read(fdes, buff, buff_size);
#else
	num = read(fdes, buff, buff_size);
#endif
	if (num == -1) {
	    return(-1);
	}
	else {
	    if (num == 0) {
		return(-1);
	    }
	}
	buff_size -= num;
	buff += num;
    }
    return(0);
}

int
sock_write(int fdes, char *buff, unsigned int buff_size)
{   int		num;

    while (buff_size) {
#ifdef PLAN_WINNT
	int is_sock = is_socket(fdes);
	if (is_sock) num = sendto(fdes, buff, buff_size, 0, NULL, 0);
	else num = write(fdes, buff, buff_size);
#else
	num = write(fdes, buff, buff_size);
#endif
	if (num == -1) {
	    return(-1);
	}
	buff_size -= num;
	buff += num;
    }
    return(0);
}


