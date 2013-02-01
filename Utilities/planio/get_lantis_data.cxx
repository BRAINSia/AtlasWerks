
#include <stdio.h>
#include <stdlib.h>
#ifndef WIN32
#include <unistd.h>
#else
#include <io.h>
#endif
#include <string.h>
#include "gen.h"
#include "plan_strings.h"
#include "libplanio.h"

int read_line(FILE *fp, LANTIS_DATA *lantis);

int
get_lantis_data(LANTIS_DATA *lantis)
{
    char	lantis_file[200];
    FILE	*fp;

    lantis->tolerance_photon = 2;
    lantis->tolerance_electron = 1;
    sprintf(lantis_file, "%s/site/lantis_data", get_plunc_root_dir());
    fp = fopen(lantis_file, "r");
    if (fp != NULL) {
    	while (!read_line(fp, lantis)) {
	}
    	fclose(fp);
	return(0);
    }
    return(1);
}

int
ival(char *buf)
{   int		val;
    char	dummy[200];

    sscanf(buf, "%s %d", dummy, &val);
    return(val);
}

int
read_line(FILE *fp, LANTIS_DATA *lantis)
{   int		i, num;
    char	buf[200];

    buf[0] = 0;
    for (i = 0; i < 200; i++) {
	num = fscanf(fp, "%c", buf+i);
	if (num <= 0) return(1);
	if (buf[i] == '\n') {
	    buf[i] = 0;
	    break;
	}
    }
    if (buf[0] == '#') return(0);

    if (strncmp(buf, "TOLERANCE_PHOTON", 16) == 0)
	lantis->tolerance_photon = ival(buf);
    if (strncmp(buf, "TOLERANCE_ELECTRON", 18) == 0)
	lantis->tolerance_electron = ival(buf);

    return(0);
}

