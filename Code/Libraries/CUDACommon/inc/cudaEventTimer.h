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

#ifndef __CUDA_EVENT_TIMER_H
#define __CUDA_EVENT_TIMER_H

#include <cuda_runtime.h>

class cudaEventTimer {
public:
    cudaEventTimer();
    ~cudaEventTimer();
    
    void start();
    void stop();
    
    void reset();
    float getTime();
    void printTime(const char* name, int niters = 1);
    
private:
    cudaEvent_t m_start, m_stop;
    float m_time;
};

#endif 
