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

#pragma once
#ifndef __HPC_TIMER_H
#define __HPC_TIMER_H

#ifdef _WIN32
#define PLATFORM_FOUND
#include<windows.h>

inline void QueryHPCTimer(__int64 *time)
{
	LARGE_INTEGER winTimer;
	QueryPerformanceCounter(&winTimer);
	*time = winTimer.QuadPart;
};

inline void QueryHPCFrequency(__int64 *freq)
{
	LARGE_INTEGER winFreq;
	QueryPerformanceFrequency(&winFreq);
	*freq = winFreq.QuadPart;
};
#endif
#ifdef __linux__
#define PLATFORM_FOUND
#include <sys/time.h>
#include <stdio.h>
#include <stdint.h>
#define TIMER_FREQUENCY 1000000
inline void QueryHPCTimer(int64_t *time)
{
	timeval linTimer;
	gettimeofday(&linTimer, 0);
	*time = linTimer.tv_sec * TIMER_FREQUENCY + linTimer.tv_usec;
};

inline void QueryHPCFrequency(int64_t *freq)
{
	*freq = TIMER_FREQUENCY;
};
#undef TIMER_FREQUENCY

class hpcTimer
{
public:
    hpcTimer():m_timer(0),m_start(0){};
    ~hpcTimer(){}

    void Start(){
        QueryHPCTimer(&m_start);
    }
    
    void Stop(){
        int64_t m_stop;
        QueryHPCTimer(&m_stop);
        m_timer += m_stop - m_start;
    }
        
    void Reset(){
        m_timer = 0;
        Start();
    }

    double getTimeInMicroseconds(){
        return (double) m_timer;
    }
    
    double getTimeInMilliseconds(){
        return (double) m_timer / 1000;
    }

    double getTimeInSeconds(){
        int64_t freq;
        QueryHPCTimer(&freq);
        return (double) m_timer/1000000.f;
    }
    void printTime(const char* msg) {
        fprintf(stderr, "%s run time %f \n", msg, getTimeInMicroseconds());
    }
private:
    int64_t m_timer;
    int64_t m_start;
};

#endif


#ifndef PLATFORM_FOUND
#error Compilation platform not found or not supported. Define _WIN32 or _LIN to select a platform.
#endif

#undef PLATFORM_FOUND


#endif

