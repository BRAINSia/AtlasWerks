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

#include "cudaAverageDef.h"
#include <cudaInterface.h>

template
void cplAverage(float* d_odata, float** d_idata, int nImg, int nP, void* temp, cudaStream_t stream);

template
void cplAverage(float* d_odata, float* d_idata, unsigned int nP, unsigned int pitch,
                 unsigned int nImgs, cudaStream_t stream);


template<typename T>
void cpuAverage(T* h_Avg, T** h_Imgs, int w, int h, int l, int nImgs)
{
    int n = w * h * l;
    
    for (int i=0; i <n; ++i)
        h_Avg[i] = 0;

    for (int ii=0; ii <nImgs; ++ii)
        for (int i=0; i <n; ++i)
            h_Avg[i] += h_Imgs[ii][i];

    for (int i=0; i <n; ++i)
        h_Avg[i] /= nImgs;
}

void testAverage(int w, int h, int l, int nImgs)
{
    float** h_Imgs = new float* [nImgs];

    int n = w * h * l;
    for (int i=0; i< nImgs; ++i)
        h_Imgs[i] = new float [n];

    float* h_o     = new float [n];
    for (int ii=0; ii< nImgs; ++ii)
        for (int i=0; i <n; ++i)
            h_Imgs[ii][i] = (float) rand() / RAND_MAX;
    cpuAverage(h_o, h_Imgs, w, h, l, nImgs);


    int nAlign = iAlignUp(n, 256);
    float** d_Imgs = new float* [nImgs];
    dmemAlloc(d_Imgs[0], nAlign * nImgs);
    for (int i=1; i< nImgs; ++i)
        d_Imgs[i] = d_Imgs[0] + i * nAlign;
    
    float* d_o;
    dmemAlloc(d_o, n);
    
    for (int i=0; i< nImgs; ++i)
        copyArrayToDevice(d_Imgs[i], h_Imgs[i], n);

    cplAverage(d_o, d_Imgs, nImgs, n, (void*)NULL, NULL);
    testError(h_o, d_o, 1e-5, n, "Average base on pointer array");

    cplVectorOpers::SetMem(d_o, 0.f, n);
    cplAverage(d_o, d_Imgs[0], n, nAlign, nImgs, NULL);
    testError(h_o, d_o, 1e-5, n, "Average base on pitch array");
    
    
    for (int i=0; i< nImgs; ++i)
        delete []h_Imgs[i];

    dmemFree(d_Imgs[0]);
    dmemFree(d_o);
    delete []h_o;
    delete []h_Imgs;
    delete []d_Imgs;
    
}
