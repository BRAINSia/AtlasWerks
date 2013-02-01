
#include <stdio.h>
#ifndef WIN32
#include <unistd.h>
#else
#include <io.h>
#endif
#include <fcntl.h>
#include "plan_file_io.h"
#include "gen.h"
#include "extbm.h"
#include "libplanio.h"

int
get_unit(char *phys_dat_dir, int unit_id, UNIT *unit)
{
    int		fdes,
		loop;
    int		count;
    char	unit_file_name[150];

    sprintf(unit_file_name, "%s/unit", phys_dat_dir);
    fdes = open(unit_file_name, O_RDONLY, 0);
    if (fdes < 0)
	return(-1);

    if (read_int(fdes, &count))
	return(-1);

    for (loop = 0; loop < count; loop++) {
	if (read_unit(fdes, unit))
	    return(-1);

	if (unit->unit_id == unit_id) {
	    close(fdes);
	    return(0);
	}
    }

    close(fdes);
    return(-1);
}

