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

#ifndef __CUDA__SCAN__H
#define __CUDA__SCAN__H

#include <cudpp/cudpp.h>
#include <cutil_comfunc.h>

template<class T>
inline CUDPPDatatype getCUDPPType(T* data){
    return CUDPP_INT;
}

template<>
inline CUDPPDatatype getCUDPPType(int* data){
    return CUDPP_INT;
}

template<>
inline CUDPPDatatype getCUDPPType(uint* data){
    return CUDPP_UINT;
}

template<>
inline CUDPPDatatype getCUDPPType(float* data){
    return CUDPP_FLOAT;
}

class cudaAddScanPlan{
public:
    cudaAddScanPlan(){
    };

    cudaAddScanPlan(uint nMax){
        allocate(nMax);
    };
    
    ~cudaAddScanPlan(){
        clean();
    }
    
    void allocate(uint nMax);
    void clean();

    template<typename T>
    T* L1_buffer() const{
        return (T*) d_buffer;
    }

    template<typename T>
    T* L2_buffer() const{
        return (T*) d_buffer + L0;
    }

    template<typename T>
    T* total_buffer() const{
        return (T*) d_buffer + L0 + L1;
    }
    
    template<class T>
    void scan(T *d_odata, T *d_idata, uint n);

    void test(int n);
private:

    template<class T>
    void biScan(T *d_odata, T *d_idata, uint n);

    template<class T>
    uint getDataType(T* data);

    CUDPPDatatype datatype;
    uint nMax;

    uint nMaxBlocks;
    uint nMaxSubBlocks;
    void *d_buffer;
    uint L0;
    uint L1;

    CUDPPConfiguration scanConfig;
    CUDPPHandle        mScanPlan;        // CUDPP plan handle for prefix sum
};

template<class T> void addScanCPU(T* h_odata, T* h_idata, int n);

/*
   Implementation of scan using CUDPP
*/
class scanImpl{
public:
    scanImpl(CUDPPDatatype type, bool inclusive);

    void add_scan_init(uint);
    void min_scan_init(uint);
    void max_scan_init(uint);
    void mul_scan_init(uint);
    
    template<class T>
    void scan(T* d_odata, T* d_idata, uint n);
        
    ~scanImpl();

private:
    CUDPPDatatype type;
    CUDPPConfiguration config;
    CUDPPHandle scanPlan;
};

class segScanImpl{
public:
    segScanImpl(CUDPPDatatype type, bool inclusive);
    void add_scan_init(uint);
    void min_scan_init(uint);
    void max_scan_init(uint);
    void mul_scan_init(uint);
    
    template<class T>
    void scan(T* d_odata, T* d_idata, uint* d_iflag, uint n);
    
    ~segScanImpl();

private:
    CUDPPDatatype type;
    CUDPPConfiguration config;
    CUDPPHandle scanPlan;
};


//Segmented scan function
template<class T, int inclusive>
void segScan(T* g_scan, T* g_iData, uint* g_iFlags, int n);
    
template<class T>
void segScan(T* g_lScan, T* g_iData, uint* g_iFlags, int n);

template<class T>
void segmentedSum(T* d_iData, uint* d_iFlags, uint n);

template<class T>
void segScan_block_cpu(T* scan, T* idata, uint* iflags, int n, uint* blockSum);

template<class T>
void segScan_cpu(T* scan, T* idata, uint* iflags, int n);

void segCount(uint* g_count, uint* g_iFlags, int n);

void findLastPos(uint* g_pos, uint4* g_iData, uint n);
void buildLastPosFlags(uint* g_flags, uint4* g_iData, uint n);

void testSegScan(int n, int s, int inclusive);
void testSegScan_block(int n, int s);
void testSegScan_inclusive_block(int n, int s);

void testSegScan_int2(int n, int s, int inclusive);
void testSegScan_float2(int n, int s, int inclusive);
void testSegScanDouble(int n, int s, int inclusive);

#endif
