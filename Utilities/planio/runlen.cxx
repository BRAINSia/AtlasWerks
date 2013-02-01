
#include <fcntl.h>
#include "stdio.h"
#include "stdlib.h"
#include "gen.h"
#include "plan_file_io.h"
#include "plan_strings.h"
#include "libplanio.h"

#define MAGIC                   0x22873

typedef struct {
    int	size;
    int	magic;
} HEADER;

typedef struct 
{
    unsigned char	count;
    unsigned char	red;
    unsigned char	green;
    unsigned char	blue;
} CODE;

int
runlen_encode(char *filename, RGB_PIXEL *buf, int xdim, int ydim)
{   int		fd;
    int		res;
    HEADER	magic;
    CODE	*out;
    int		runstart;
    int		i;
    int         nruns;

    if (xdim != ydim) {
	fprintf(stderr, "Can encode non-square image\n");
	return(1);
    }
    res = xdim;
    /* Open File */
    if ((fd = open(filename, O_WRONLY | O_TRUNC | O_CREAT, 0666)) < 0) {
	fprintf(stderr,"runlen_encode: open error for %s.\n", filename);
	perror("open");
	return(-1);
    }

    /* Write Magic Header */
    magic.size = res;
    magic.magic = MAGIC;
    if (write(fd, &magic, sizeof(HEADER)) != sizeof(HEADER)) {
	fprintf(stderr,"runlen_encode: magic number write error.\n");  
	perror("write");
	return(-1);
    }

    /* Create Buffer */
    out = (CODE *) malloc(res*res*sizeof(CODE));
    if (out == NULL) {
	fprintf(stderr,"runlen_encode: malloc error for %d %ld %ld.\n",
		res, sizeof(CODE), res*res*sizeof(CODE));
	return(-1);
    }

    nruns = 0;
    runstart = 0;
    for (i = 0; i < res*res; i++) {
	if (memcmp(&buf[i], &buf[i+1], sizeof(RGB_PIXEL)) ||
	    ((i+1) % res == 0) || (i-runstart == 255)) {
	    /* Write out Run */
	    out[nruns].count = i - runstart;
	    out[nruns].red = buf[i].red;
	    out[nruns].green = buf[i].green;
	    out[nruns].blue= buf[i].blue;
	    nruns++;
	    runstart = i+1;
	}
    }

    if (write(fd, out, nruns*sizeof(CODE) ) != int(nruns*sizeof(CODE))) {
	fprintf(stderr,"runlen_encode: write error.\n");
	perror("write");
	return(-1);
    }
  
    /* Close Output file */
    close(fd);
    free(out);
    return(0);
}


