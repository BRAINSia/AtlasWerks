
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
get_tray(
    char	*phys_dat_dir,
    int		tray_id,
    TRAY	*tray)
{
    int		fdes,
		loop;
    int		count;
    char	tray_file_name[150];
    char	*malloc();

    sprintf(tray_file_name, "%s/tray", phys_dat_dir);
    fdes = open(tray_file_name, O_RDONLY, 0);
    if (fdes < 0)
	return(-1);

    if (read_int(fdes, &count)) {
	close(fdes);
	return(-1);
    }

    for (loop = 0; loop < count; loop++) {
/*
  Make sure tray->tray_list is initialized to either NULL or some
  malloc space.
*/
	if (read_tray(fdes, tray)) {
	    close(fdes);
	    return(-1);
	}

	if (tray->tray_id == tray_id) {
	    close(fdes);
	    return(0);
	}
    }

    close(fdes);
    return(-1);
}

