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

#ifndef __CUDA_FFT_WRAPPER
#define __CUDA_FFT_WRAPPER

#include <cufft.h>
#include <cudaComplex.h>
#include <Vector3D.h>

class cplFFT3DWrapper{
public:
    cplFFT3DWrapper(): m_size(0,0,0), hasPlan(false) {
        
    }
    
    explicit cplFFT3DWrapper(const Vector3Di& size):m_size(size){
        createPlan();
    }
    
    ~cplFFT3DWrapper(){
        clear();
    }

    void setSize(const Vector3Di& size) {
        if (hasPlan) {
            if (checkSize(size))
                return;
            else 
                clear();
        }
        m_size = size;
        createPlan();
    };
    
    const Vector3Di& getSize() const { return m_size; }
    
    void forwardFFT(cplComplex* d_o, cplComplex* d_i) const;
    void forwardFFT(cplComplex* d_o, float* d_i)  const;
    void backwardFFT(cplComplex* d_o, cplComplex* d_i)  const;
    void backwardFFT(float* d_o, cplComplex* d_i)  const;
    bool checkSize(const Vector3Di& size) const;
    
private:
    void createPlan();
    void clear();


    Vector3Di m_size;
    cufftHandle  pR2C, pC2R, pC2C;
    bool hasPlan;
};

class cplFFT3DConvolutionWrapper{
public:
    cplFFT3DConvolutionWrapper():d_iC(NULL){
    }
    ~cplFFT3DConvolutionWrapper(){
        clear();
    }
    void setSize(const Vector3Di& size);
    void convolve(float* d_o, float* d_i, cplComplex* d_kernelC, const Vector3Di& size) const;
    
    const Vector3Di& getSize() const { return m_fft.getSize();}
    const cplFFT3DWrapper& getFFTWrapper() { return m_fft; };
private:
    void convolve(float* d_o, float* d_i, cplComplex* d_kernelC) const;
    void clear();
    
    cplComplex* d_iC;
    cplFFT3DWrapper m_fft;
};

class cplFFTGaussianFilter {
public:
    cplFFTGaussianFilter();

    void init(const Vector3Di& size, float sigma);
    void filter(float* d_o, float* d_i, const Vector3Di& size) const;
private:
    cplComplex* d_KernelC;
    cplFFT3DConvolutionWrapper m_fftFilter;
};
#endif
