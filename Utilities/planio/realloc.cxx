
#include <stdio.h>
#include <stdlib.h>

void *
Realloc(void *ptr, int size)
{

    if (size == 0) size = sizeof(int);

    if (ptr == NULL) ptr = calloc(size,1);
    else ptr = realloc(ptr, size);

    return(ptr);
}


