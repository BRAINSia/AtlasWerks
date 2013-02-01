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

#ifndef __CUDA__REDUCE__H
#define __CUDA__REDUCE__H

#include <cutil_math.h>
#include <Vector3D.h>
#include <cudaThreadSingleton.h>
class cplReduce{
public:
    cplReduce() { init(); };
    ~cplReduce(){ clean();};

    void init();
    void clean();

    template<typename T>
    T Max(const T* pSrc, unsigned int n);

    template<typename T>
    T Min(const T* pSrc, unsigned int n);

    template<typename T>
    T Sum(const T* pSrc, unsigned int n);

    template<typename T>
    T MaxAbs(const T* pSrc, unsigned int n);

    template<typename T>
    T Sum2(const T* pSrc, unsigned int n);

    template<typename T>
    T SumAbs(const T* pSrc, unsigned int n);

    template<class T>
    void MaxMin(T& maxV, T&  minV, const T* data, unsigned int n);

    template<class T>
    void MaxSum(T& maxV, T&  minV, const T* data, unsigned int n);

    template<typename T>
    T Dot(const T* d_i, const T* d_i1,  unsigned int n);

    template<typename T>
    T SumAdd(const T* d_i, const T* d_i1,  unsigned int n);

    template<typename T>
    T MaxAdd(const T* d_i, const T* d_i1,  unsigned int n);

    template<typename T>
    T MinAdd(const T* d_i, const T* d_i1,  unsigned int n);

    bool selfTest(int n);
    void timeTest(int n);
    
protected:
    template <class T, class trait>
    T cplSOPReduce(const T* d_i, int n);

    template <class T, class traits, class traits1>
    T cplBOPReduce(const T* pSrc, int n);

    template <class T, class oper, class oper1>
    void biReduce(T& rd0, T& rd1, const T*g_idata, unsigned int n);

    template <class T, class traits, class traits1>
    T reduceProduct(const T*g_idata, const T*g_idata1, unsigned int n);

    void* d_temp;
    void* h_temp;

    bool m_zeroCopy;
};

class cplVector3DArray;
void  MaxMin(cplReduce& rdPlan, Vector3Df& maxV, Vector3Df& minV, const cplVector3DArray& d_i, uint n);
float     MaxAbsAll(cplReduce& rdPlan, const cplVector3DArray& d_i, uint n);
Vector3Df MaxAbs(cplReduce& rd, const cplVector3DArray& d_i, uint n);
Vector3Df Sum(cplReduce& rd, const cplVector3DArray& d_i, uint n);
Vector3Df Sum2(cplReduce& rd, const cplVector3DArray& d_i, uint n);

//typedef Singleton<cplReduce> cplReduceSingleton;
typedef cplThreadSingleton<cplReduce> cplReduceTSingleton;

#endif
