
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
get_time_calc(
    char			*phys_dat_dir,
    int				unit_id,
    TIME_CALC_PARAMETERS	*time_calc)
{
    int		fdes,
		loop;
    int		count;
    char	time_calc_file_name[150];

    sprintf(time_calc_file_name, "%s/time_calc", phys_dat_dir);
    fdes = open(time_calc_file_name, O_RDONLY, 0);
    if (fdes < 0)
	return(-1);

    if (read_int(fdes, &count)) {
	close(fdes);
	return(-1);
    }

    for (loop = 0; loop < count; loop++) {
	if (read_time_calc(fdes, time_calc)) {
	    close(fdes);
	    return(-1);
	}

	if (time_calc->unit_id == unit_id) {
	    close(fdes);
	    return(0);
	}
    }

    close(fdes);
    return(-1);
}

