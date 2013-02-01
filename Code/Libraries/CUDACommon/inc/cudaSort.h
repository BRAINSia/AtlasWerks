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

#ifndef __CUDA_SORT_H
#define __CUDA_SORT_H

#include <cudaReduce.h>
#include "cudpp/cudpp.h"

class cplSort{
public:
    cplSort(int maxSize, cplReduce* rd);
    ~cplSort(){clean();};
    
    void setCapacity(int newSize);
    
    void init();
    void clean();

    //Sorting with AOS format 
    void sort(float2* d_data, int n);
    void sort(int2* d_data,   int n, int nBits=0);
    void sort(uint2* d_data,  int n, int nBits=0);

    void sort(float2* d_o, float2* d_i, int n);
    void sort(uint2* d_o , uint2* d_i , int n, int nBits=0);
    void sort(int2* d_o  , int2* d_i  , int n, int nBits=0);

    //sorting with SOA format 
    void sort(float* d_data, uint* d_index, int n);
    void sort(int* d_data,   uint* d_index, int n, int nBits=0);
    void sort(uint* d_data,  uint* d_index, int n, int nBits=0);

    void sort(int* d_o  , uint* d_oi, int* d_i  , uint* d_ii, int n, int nBits=0);
    void sort(uint* d_o , uint* d_oi, uint* d_i , uint* d_ii, int n, int nBits=0);
    void sort(float* d_o, uint* d_oi, float* d_i, uint* d_ii, int n);

    void EnableAppSorting() { m_appSort = true;};
    void DisableAppSorting(){ m_appSort = false;};
private:
    void sortSupport(uint* d_o, uint* d_oi, uint* d_i, uint* d_ii, int n, int flip, int nBits);
    void sortSupport(uint2* d_o, uint2* d_i, int n, int flip, int nBits);

    template<class T>
    void getInputRange(T& maxValue, T& minValue, T* d_i, unsigned int n);
        
    void implSort(uint2* d_o, uint2* d_i, int n, int nBits);

    void load_C1   (uint2* d_o, uint * d_i, int n, int nAlign, int flip);
    void loadAOS_C2(uint2* d_o, uint2* d_i, int n, int nAlign, int flip);
    void loadSOA_C2(uint2* d_o, uint* d_i, uint* d_ii, int n, int nAlign, int flip);

    void loadSOA_offset(uint2* d_o, int* d_i, uint* d_ii, int minV, int n, int nAlign);
    void loadSOA_offset(uint2* d_o, uint* d_i, uint* d_ii, uint minV, int n, int nAlign);
    void sortReturnSOA_offset(int* d_o, uint* d_oi, uint2* d_i, int minV, int n);
    void sortReturnSOA_offset(uint* d_o, uint* d_oi, uint2* d_i, uint minV, int n);

    void loadAOSInt(uint2* d_o, uint2* d_i, int n, int nAlign);
    void sortReturnSOAInt(uint* d_o, uint* d_oi, uint2* d_i, int n);
    void sortReturnAOSInt(uint2* d_o, uint2* d_i, int n);
    
        
    void sort_small(uint2* d_o, uint2* d_i, int n);
    void radixSortSingleBlock(uint2* d_o, uint2* d_i, int n, int nBits);

    template<int flip>
    void radixSortSingleBlock(uint* d_o, uint* d_oi, uint* d_i, uint* d_ii, int n, int nBits);

    void sortReturn(uint2* d_o, uint2* d_i, int n, int flip);
    void sortReturn(uint* d_o, uint* d_oi, uint2* d_i, int n, int flip);

private:
    cplReduce*  m_rd;
    CUDPPHandle  m_scanPlan;        // CUDPP plan handle for prefix sum

    uint2* d_ping;
    uint2* d_pong;
    
    int* d_off; // store the offset for each radix digit
    int* d_cnt; // store the count number of each radix digit in the block
    int* d_sum; // store the segmented scan for the count

    int nc;
    int flip;   // floating point or integer

    bool m_appSort;
};

void testSortAOS(int n, int nBits);
void testSortSOA(int n, int nBits);

#endif
