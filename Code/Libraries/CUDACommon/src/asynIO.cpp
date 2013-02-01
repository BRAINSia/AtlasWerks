#include <aio.h>
#include <stdio.h>
#include <strings.h>
#include <errno.h>
#include <unistd.h>

template<class T>
void readArrayFromRawFile(T* h_a, const char* fname, unsigned int n)
{
    unsigned int nbytes = n * sizeof(T);
    int fd, ret;
    fd = open( fname, O_RDONLY );
    if (fd < 0) perror("open");
    ret = read(fd, h_a, nbytes);
    close(fd);
}

template<class T>
void writeArrayToRawFile(T* h_a, const char* fname, unsigned int n)
{
    unsigned int nbytes = n * sizeof(T);
    int fd, ret;
    fprintf(stderr, "Writing file %s ",  fname);
    fd = open( fname, O_WRONLY | O_CREAT | O_TRUNC);
    ret = write(fd, h_a, nbytes);
    fprintf(stderr, "writing %d byte from %d byte ", ret, nbytes);
    close(fd);
    fprintf(stderr, "done \n");
}

template void writeArrayToRawFile(float* h_a, const char* fname, unsigned int n);
template void writeArrayToRawFile(int* h_a, const char* fname, unsigned int n);
template void writeArrayToRawFile(unsigned int* h_a, const char* fname, unsigned int n);

template<class T>
void readArrayFromRawFileAsyn(T* h_a, const char* fname, unsigned int n){
    unsigned int nbytes = n * sizeof(T);
    int fd, ret;
    struct aiocb my_aiocb;
    fd = open( fname, O_RDONLY | O_DIRECT);
    if (fd < 0) perror("open");

    /* Zero out the aiocb structure (recommended) */
    bzero( (char *)&my_aiocb, sizeof(struct aiocb) );

    /* Allocate a data buffer for the aiocb request */
    my_aiocb.aio_buf = h_a;
    if (!my_aiocb.aio_buf)
        perror("NULL input");

    /* Initialize the necessary fields in the aiocb */
    my_aiocb.aio_fildes = fd;
    my_aiocb.aio_nbytes = nbytes;
    my_aiocb.aio_offset = 0;

    ret = aio_read( &my_aiocb );
    if (ret < 0)
        perror("aio_read");

    while ( aio_error( &my_aiocb ) == EINPROGRESS ) ;

    if ((ret = aio_return( &my_aiocb )) > 0) {
        /* got ret bytes on the read */
    } else {
        /* read failed, consult errno */
    }
    close(fd);
}

template void readArrayFromRawFileAsyn(float* h_a, const char* fname, unsigned int n);
template void readArrayFromRawFileAsyn(int* h_a, const char* fname, unsigned int n);
template void readArrayFromRawFileAsyn(unsigned int* h_a, const char* fname, unsigned int n);



template<class T>
void writeArrayToRawFileAsyn(T* h_a, const char* fname, unsigned int n){
    unsigned int nbytes = n * sizeof(T);
    int fd, ret;
    struct aiocb my_aiocb;
    fd = open( fname, O_WRONLY | O_CREAT | O_TRUNC);

    fprintf(stderr, "Writing file %s ",  fname);
        
    /* Zero out the aiocb structure (recommended) */
    bzero( (char *)&my_aiocb, sizeof(struct aiocb) );

    /* Allocate a data buffer for the aiocb request */
    my_aiocb.aio_buf = h_a;
    if (!my_aiocb.aio_buf)
        perror("NULL input");

    /* Initialize the necessary fields in the aiocb */
    my_aiocb.aio_fildes = fd;
    my_aiocb.aio_nbytes = nbytes;
    my_aiocb.aio_offset = 0;

    ret = aio_write( &my_aiocb );
    if (ret < 0)
        perror("aio_write");
    
    while ( aio_error( &my_aiocb ) == EINPROGRESS ) ;
    if ((ret = aio_return( &my_aiocb )) > 0) {
        /* got ret bytes on the read */
        fprintf(stderr, "writing %d byte from %d byte ", ret, nbytes);
    } else {
        /* read failed, consult errno */
    }
    close(fd);
    fprintf(stderr, "done \n");
}

template void writeArrayToRawFileAsyn(float* h_a, const char* fname, unsigned int n);
template void writeArrayToRawFileAsyn(int* h_a, const char* fname, unsigned int n);
template void writeArrayToRawFileAsyn(unsigned int* h_a, const char* fname, unsigned int n);

