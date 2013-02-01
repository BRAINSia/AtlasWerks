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

#include <cplMutex.h>
#include <pthread.h>

class pthread_mutex_scoped_lock{
    pthread_mutex_t* m;
public:
    explicit pthread_mutex_scoped_lock(pthread_mutex_t* m_): m(m_) {
        assert(!pthread_mutex_lock(m));
    }
    ~pthread_mutex_scoped_lock() {
        assert(!pthread_mutex_unlock(m));
    }
};

class pthread_mutex_scoped_unlock{
    pthread_mutex_t* m;
public:
    explicit pthread_mutex_scoped_unlock(pthread_mutex_t* m_): m(m_) {
        assert(!pthread_mutex_unlock(m));
    }
    ~pthread_mutex_scoped_unlock() {
        assert(!pthread_mutex_lock(m));
    }
};

template<typename Mutex>
class lock_guard
{
private:
    Mutex& m;
    explicit lock_guard(lock_guard&);
    lock_guard& operator=(lock_guard&);
public:
    explicit lock_guard(Mutex& m_): m(m_)     {
        m.lock();
    }
    ~lock_guard()  {
        m.unlock();
    }
};
