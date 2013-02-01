#include <cudaStream.h>
#include <cuda_runtime_api.h>
#include <assert.h>

int           streamCnt = 0;
cudaStream_t* streams = NULL;

void streamCreate(int n)
{
    streamCnt = n;
    streams   = new cudaStream_t [n];
    for(int i = 0; i < n; i++)
        cudaStreamCreate(&(streams[i]));
}

cudaStream_t getStream(int id)
{
    if (id == 0 && streamCnt == 0)
        return cudaStream_t (NULL);
    else {
        assert(id < streamCnt);
        return streams[id];
    }
}

void streamDestroy()
{
    for (int i=0; i< streamCnt; ++i)
        cudaStreamDestroy(streams[i]);
    delete []streams;
    streamCnt = 0;
    streams = NULL;
}
