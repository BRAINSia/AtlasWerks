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

#ifndef __CUDA_SORT_UTILS_H
#define __CUDA_SORT_UTILS_H

class cplReduce;
template<class T2>
bool checkHostPairOrder(T2* h_a, unsigned int n);

template<class T2>
bool checkDevicePairOrder(T2* d_a, unsigned int n);

template<class T2>
bool checkHostPairOrder(T2* h_a, unsigned int n, const char* name);

template<class T2>
bool checkDevicePairOrder(T2* d_a, unsigned int n, const char* name);


template<class T>
bool checkHostOrder(T* h_a, unsigned int n);

template<class T>
bool checkDeviceOrder(T* d_a, unsigned int n);

template<class T>
bool checkHostOrder(T* h_a, unsigned int n, const char* name);

template<class T>
bool checkDeviceOrder(T* d_a, unsigned int n, const char* name);


////////////////////////////////////////////////////////////////////////////////
//
////////////////////////////////////////////////////////////////////////////////
class checkOrderPlan{
public :
    checkOrderPlan():m_rd(0){
    }
    checkOrderPlan(cplReduce *rd):m_rd(rd){
    }
    void SetReducePlan(cplReduce *rd) { m_rd = rd; };
    
    template<class T> int checkOrder(T* d_i, int n, int* d_temp, int dir);
    template<class T> int fastCheckOrder(T* d_i, int n, int* d_temp, int dir);
private:
    cplReduce* m_rd;
};

void testCheckOrder(int n);
void testCheckOrderPerformance(int n);

#endif

