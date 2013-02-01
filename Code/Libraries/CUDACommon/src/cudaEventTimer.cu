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

#include <cudaEventTimer.h>
#include <stdio.h>

cudaEventTimer::cudaEventTimer(){
    cudaEventCreate(&m_start);
    cudaEventCreate(&m_stop);
    m_time = 0;
}

void cudaEventTimer::start(){
    cudaEventRecord(m_start, 0);
}

void cudaEventTimer::reset()
{
    m_time = 0;
    start();
}

void cudaEventTimer::stop(){
    cudaEventRecord(m_stop, 0);
    cudaEventSynchronize(m_stop);
    float time;
    cudaEventElapsedTime(&time, m_start, m_stop);
    m_time += time;
}

float cudaEventTimer::getTime(){
    return m_time;
}

void cudaEventTimer::printTime(const char* name, int niters){
    fprintf(stderr, "Run time of %s is %f ms\n", name, m_time/niters);
}

cudaEventTimer::~cudaEventTimer(){
    cudaEventDestroy(m_start);
    cudaEventDestroy(m_stop);
}
