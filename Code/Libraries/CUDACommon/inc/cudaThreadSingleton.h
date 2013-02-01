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

#ifndef __GPU_THREAD_SINGLETON_H
#define __GPU_THREAD_SINGLETON_H

#define MAX_NUMBER_GPU 16

template <class T>
class cplThreadSingleton
{
public:
    static T& Instance(int id);
    static void DestroyInstance(int id);
private:
    cplThreadSingleton();           // ctor hidden  
    ~cplThreadSingleton();          // dtor hidden
    cplThreadSingleton(cplThreadSingleton const&);    // copy ctor hidden
    cplThreadSingleton& operator=(cplThreadSingleton const&);  // assign op hidden
    static T* insA[MAX_NUMBER_GPU];
};

#endif
