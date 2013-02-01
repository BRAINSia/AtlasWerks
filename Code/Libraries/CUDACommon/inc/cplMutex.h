/* ================================================================
 *
 * AtlasWerks Project
 *
 * Copyright (c) Sarang C. Joshi, Bradley C. Davis, J. Samuel Preston,
 * Linh K. Ha. All rights reserved.  See Copyright.txt or for details.
 *
 * This software is distributed WITHOUT ANY WARRANTY; without even the
 * implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 * PURPOSE.  See the above copyright notice for more information.
 *
 * ================================================================ */

#include <cplException.h>

class mutex {
private:
    mutex(mutex const&);
    mutex& operator=(mutex const&);        
    pthread_mutex_t m;
public:
    mutex(){
        int const res=pthread_mutex_init(&m,NULL);
        if(res) {
            cplException::throw_exception(thread_resource_error());
        }
    }
    ~mutex() {
        pthread_mutex_destroy(&m);
    }
        
    void lock(){
        int const res=pthread_mutex_lock(&m);
        if(res){
            cplException::throw_exception(lock_error(res));
        }
    }

    void unlock(){
        pthread_mutex_unlock(&m);
    }
        
    bool try_lock()   {
        int const res=pthread_mutex_trylock(&m);
        if(res && (res!=EBUSY))         {
            cplException::throw_exception(lock_error(res));
        }
        return !res;
    }

    pthread_mutex_t* get_handle() {
        return &m;
    }
};

class timed_mutex {
private:
    timed_mutex(timed_mutex const&);
    timed_mutex& operator=(timed_mutex const&);        
private:
    pthread_mutex_t m;
public:
    timed_mutex()  {
        int const res=pthread_mutex_init(&m,NULL);
        if(res) {
            cplException::throw_exception(thread_resource_error());
        }
    }

    ~timed_mutex() {
        pthread_mutex_destroy(&m);
    }

    void lock() {
        pthread_mutex_lock(&m);
    }

    void unlock() {
        pthread_mutex_unlock(&m);
    }
    
    bool try_lock()  {
        int const res=pthread_mutex_trylock(&m);
        if(res && (res!=EBUSY))         {
            cplException::throw_exception(lock_error(res));
        }
        return !res;
    }
    
    bool timed_lock(struct timespec const & abs_timeout) {
        int const res=pthread_mutex_timedlock(&m,&abs_timeout);
        assert(!res || res==ETIMEDOUT);
        return !res;
    }
};
