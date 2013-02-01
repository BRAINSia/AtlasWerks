
#include <stdio.h>
#include <stdlib.h>
#include "plan_xdr.h"


int main ()
{
#ifdef PLAN_XDR_LIB
    printf (PLAN_XDR_LIB);
#else
    printf ("empty_archive");
#endif
    exit (0);
}
