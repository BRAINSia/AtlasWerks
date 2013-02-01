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

#ifndef __DIFF_OPER_GPU_H__
#define __DIFF_OPER_GPU_H__

#include <vector>
#include "AtlasWerksTypes.h"
#include "DiffOper.h"
#include <cudaVector3DArray.h>
#include <cudaComplex.h>
#include <VectorMath.h>
#include <cudaInterface.h>
#include <cufft.h>

class LookupTable3D{
public:
    LookupTable3D():mCosWX(NULL), mCosWY(NULL), mCosWZ(NULL),
                    mSinWX(NULL), mSinWY(NULL), mSinWZ(NULL)
        {
            Allocate();
        };
    
    ~LookupTable3D(){
        Clear();
    }
  
    void SetSize(const Vector3Di& size, const Vector3Df& sp=Vector3Df(1.f, 1.f, 1.f));

protected:
  
    void Allocate();
    void InitTable();
    void Upload();
    void Clear();
  
    // host data 
    float* mCosWX,* mCosWY,* mCosWZ;
    float* mSinWX,* mSinWY,* mSinWZ;
    
    Vector3Di mSize;
    Vector3Df mSp;
};

class DiffOperGPU {
  
public:  
  
    // ===== Constructors =====
  
    /**
     * Construct a fully-initialized (Initialize() is called internally)
     * DiffOper of the given logicalSize and spacing.
     */
    DiffOperGPU(const Vector3Di &maxLogicalSize);

    ~DiffOperGPU();
  
    // ===== Public Members =====

    void SetSize(const Vector3Di &logicalSize, const Vector3Df &spacing);
  
    /** Set all parameters via DiffOperParam */
    void SetParams(const DiffOperParam &param);
    /** Get all parameters in a DiffOperParam */
    DiffOperParam GetParams();
    /** Set the \f$\alpha\f$ parameter.  Controls fluid viscosity. */
    void SetAlpha(Real alpha);
    /** Get the \f$\alpha\f$ parameter.  Controls fluid viscosity. */
    Real GetAlpha();
    /** Set the \f$\beta\f$ parameter.  Controls fluid viscosity. */
    void SetBeta(Real beta);
    /** Get the \f$\beta\f$ parameter.  Controls fluid viscosity. */
    Real GetBeta();
    /** Set the \f$\gamma\f$ parameter.  Usually << 1, maintains invertability. */
    void SetGamma(Real gamma);
    /** Get the \f$\gamma\f$ parameter.  Usually << 1, maintains invertability. */
    Real GetGamma();
    /** Set the power of L.  One by default */
    void SetLPow(Real p);
    /** Get the power of L */
    Real GetLPow();
    /** Set whether to perform precomputation to gain speed at the expense of memory */
    void SetUseEigenLUT(bool b);
    /** Get whether precomputation is performed to gain speed at the expense of memory */
    bool GetUseEigenLUT();

    /**
     * If SetDivergenceFree is set to true, incompressibility of the
     * fluid transformation will be enforced by projecting each
     * deformation step to the 'nearest' divergence-free deformation
     * step.
     */
    void SetDivergenceFree(bool df);
    /**
     * See SetDivergenceFree()
     */
    bool GetDivergenceFree();
  
    /**
     * Copy the data from vf into the internal FFTWVectorField.  vf must
     * be the same size as the logical dimensions of the internal array.
     */
    void CopyIn(const float *dVf);

    /**
     * Copy the data from the internal FFTWVectorField into vf.  vf must
     * be the same size as the logical dimensions of the internal array.
     */
    void CopyOut(float *dVf);

    /**
     * f = Lv
     * 
     * v field is overwritten in this operation (holds f).
     */
    void ApplyOperator(float* dVx,
                       float* dVy,
                       float* dVz);
  
    /**
     * f = Lv
     * 
     * v field is overwritten in this operation (holds f).
     */
    void ApplyOperator(cplVector3DArray& dV);
  
    /**
     * v = Kf
     * 
     * f field is overwritten in this operation (holds v).
     */
    void ApplyInverseOperator(float* dFx,
                              float* dFy,
                              float* dFz);

    /**
     * v = Kf
     * 
     * f field is overwritten in this operation (holds v).
     */
    void ApplyInverseOperator(cplVector3DArray& dF);

protected:

    // ===== Protected Member Functions =====

    /**
     * 
     */
    void
    Initialize(const Vector3Di &maxSize);
  
    void Delete();

    void Alloc();
    void DeAlloc();
    
    void Apply(float* dDataX,
               float* dDdataY,
               float* dDdataZ,
               bool inverseOp);

    // ===== Member Data =====
  
    Vector3Di mSize;
    Vector3Df mSpacing;
    Vector3Di mComplexSize;
    unsigned int mMaxComplexSize;

    // Lookup table
    LookupTable3D mLookupTable;
  
    // CUDA FFT plans
    cufftHandle mPlanR2C, mPlanC2R;

    // device FFT array pointer
    cplComplex* mdFFTArrayX;
    cplComplex* mdFFTArrayY;
    cplComplex* mdFFTArrayZ;
    unsigned int mAllocateSize;
};


#endif //__DIFF_OPER_GPU_H__
