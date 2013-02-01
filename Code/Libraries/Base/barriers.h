//
// simple implementation of pthread_barrier
#ifndef barriers_h
#define barriers_h
#include <pthread.h>

#if defined(__APPLE__)
const int PTHREAD_BARRIER_SERIAL_THREAD(1);
struct osx_barrier_t
{
  //number of nodes to synchronise
  int nodes;
  //two counts to avoid race conditions
  int count[2];
  //which count to use
  int whichcount;
  //mutex to lock
  pthread_mutex_t lock;
  //condition to lock
  pthread_cond_t cv;
};
typedef struct osx_barrierattr_t
{
};
//initialize a barrier
inline int osx_barrier_init(struct osx_barrier_t* PB, 
                             osx_barrierattr_t *,
                             int nodes)
{
  PB->nodes = nodes;
  PB->count[0] = 0;
  PB->count[1] = 0;
  PB->whichcount = 0;
  pthread_mutex_init(&PB->lock, NULL);
  pthread_cond_init(&PB->cv, NULL);
  return 0;
}

//destroy a barrier
inline void osx_barrier_destroy(struct osx_barrier_t* PB)
{
  pthread_mutex_destroy(&PB->lock);
  pthread_cond_destroy(&PB->cv);
}

//wait for a barrier
inline int osx_barrier_wait(struct osx_barrier_t* PB)
{
  int WhichCount, Count;
  int rval = -1;

  //which counter variable to use
  WhichCount = PB->whichcount;

  //lock the mutex
  pthread_mutex_lock(&PB->lock);

  //get the count of nodes
  Count = ++PB->count[WhichCount];

  //test whether it should block
  if (Count < PB->nodes)
    {
    pthread_cond_wait(&PB->cv, &PB->lock);
    rval = 0;
    }
  else
    {
    //reset the counter
    PB->count[WhichCount] = 0;
    PB->whichcount = 1 - WhichCount;

    //release the wait
    pthread_cond_broadcast(&PB->cv);
    return PTHREAD_BARRIER_SERIAL_THREAD;
    }

  //unlock the threads
  pthread_mutex_unlock(&PB->lock);
  return rval;
}

typedef osx_barrier_t pthread_barrier_t;
typedef osx_barrierattr_t pthread_barrierattr_t;

#define pthread_barrier_init osx_barrier_init
#define pthread_barrier_destroy osx_barrier_destroy
#define pthread_barrier_wait osx_barrier_wait
#endif

#endif
