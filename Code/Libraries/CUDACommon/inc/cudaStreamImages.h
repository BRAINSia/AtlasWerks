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

#ifndef CUDA_STREAM_IMAGES_H
#define CUDA_STREAM_IMAGES_H

#include <vector>

class cplReduceS;
namespace cplMVOpers{
    template<typename T>
    void Max(T* d_o, std::vector<T* >& h_Imgs, int n, int nImgs,          
             cplReduceS* rd, T* d_scratchI[]);

    template<typename T>
    void Min(T* d_o, std::vector<T* >& h_Imgs, int n, int nImgs,          
                 cplReduceS* rd, T* d_scratchI[]);
    template<typename T>
    void Sum(T* d_o, std::vector<T* >& h_Imgs, int n, int nImgs,          
                 cplReduceS* rd, T* d_scratchI[]);

    template<typename T>
    void MulC_I(std::vector<T*>& h_Imgs, T c, int n, int nImgs, T* d_scratchI[]);
    template<typename T>
    void MulC_I(std::vector<T* >& h_Imgs, T c, int n, int nImgs);
    template<typename T>
    void MulC_I(std::vector<T* >& h_Imgs, std::vector<T* >& d_Imgs, T c, int n, int nImgs);
};


#endif
