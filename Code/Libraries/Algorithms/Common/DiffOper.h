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

#ifndef __DIFF_OPER_H__
#define __DIFF_OPER_H__

#include "AtlasWerksTypes.h"
#include "KernelInterface.h"
#include "DiffOperParam.h"

#ifndef SWIG

#include <fftw3.h>
#include <pthread.h>
#include <vector>

#include "Array3D.h"

#endif // ! SWIG

/**
 * DiffOperFFTWrapper is set up to allow templating the FFTW methods
 * of DiffOper over the precision desired (right now just float or
 * double).  The generic template is never implemented, only the
 * specialized versions for float and double
 */
template<class T>
class DiffOperFFTWWrapper {
public:
  DiffOperFFTWWrapper()
  {
    throw AtlasWerksException(__FILE__,__LINE__,"Error, template should not be automatically generated");
  }
  
  void Initialize(SizeType logicalSize, 
		  Array3D<Vector3D< T > > array, 
		  int nThreads,
		  bool measure){}

  void ExecuteForward(){}
  
  void ExecuteBackward(){}
  
  void Delete(){}

};

/**
 * single-precision specialization of the wrapper, calls fftwf_
 * version of classes
 */  
template<>
class DiffOperFFTWWrapper<float> {
public:
  DiffOperFFTWWrapper();
  void Initialize(SizeType logicalSize, 
	     Array3D<Vector3D< float > >* array, 
	     int nThreads,
	     bool measure);
  void Delete();
  void ExecuteForward();
  void ExecuteBackward();
  
protected:
  fftwf_plan mFFTWForwardPlan;
  fftwf_plan mFFTWBackwardPlan;
};

/**
 * double-precision specialization of the wrapper, calls fftwf_
 * version of classes
 */  
template<>
class DiffOperFFTWWrapper<double> {
public:
  DiffOperFFTWWrapper();
  void Initialize(SizeType logicalSize, 
	     Array3D<Vector3D< double > >* array, 
	     int nThreads,
	     bool measure);
  void Delete();
  void ExecuteForward();
  void ExecuteBackward();
  
protected:
  fftw_plan mFFTWForwardPlan;
  fftw_plan mFFTWBackwardPlan;
};

/**
 * DiffOper implements the differential operator:
 *
 * \f[
 * \mathbf{L} = -\alpha\nabla^2 - \beta\nabla(\nabla\cdot) + \gamma\mathbf{I}
 * \f]
 *
 * (where \f$\mathbf{I}\f$ is the identity matrix) used in fluid
 * registration.
 *
 * This operator comes from fluid mechanics, specifically a
 * simplification of the Navier-Stokes equation for compressible
 * fluids with a very low Reynolds number. The parameters \f$\alpha\f$
 * and \f$\beta\f$ control the viscosity of the fluid, while
 * \f$\gamma\f$ ensures that \f$\mathbf{L}\f$ is invertable.
 * Furthermore, the SetLPow() function allows the operator
 * \f$\mathbf{L}^p\f$ to be used for values of p other than one --
 * theoretically this will cause greater smoothing of the deformation
 * fields.  If eigenvalue/vector lookup tables are used,
 * precomputation is done (SetUseEigenLUT()) gaining speed at the
 * expense of memory.
 *
 * The Initialize() function should be called once all parameters
 * (alpha, beta, whether to use an eigenvector/value lookup table,
 * FFTW params, etc.) are set, but before data is copied into the
 * internal FFTWVectorFeild array (Initialize will overwrite any data
 * in this array).  Data can then be copied in and out, and the
 * ApplyOperator() or ApplyInverseOperator() functions executed.
 *
 * This class is not safe for use with multple threads.
 *
 * A sample usage would be something like:
 *
 <pre>
 .
 .
 .
 DiffOper op(size, spacing);
 op.SetAlpha(alpha);
 op.SetBeta(beta);
 op.SetGamma(gamma);
 op.SetUseEigenLUT(true);
 op.SetLPow(lPow);
 op.Initialize();
 
 Array3D<Vector3D<float> > newVF(size);
 op.CopyIn(vf);
 op.ApplyInverseOperator();
 op.CopyOut(newVF);
 .
 .
 .
 </pre>
 * 
 */
template<class T>
class DiffOperT : 
  public KernelInterfaceT<T>
{
  
public:  
  
  // ===== Static Members =====
  
  /**
   * FFTWVectorField is just a vector field, the renaming is just a
   * reminder that it has been resized to allow extra space for fftw's
   * fourier transform.  FFTWVectorFields should *always* be initialized
   * by InitializeFFTWVecField().
   */
  typedef typename KernelInterfaceT<T>::KernelInternalVF FFTWVectorField;

  /**
   * Create a new vector field suitable for use with this class.
   * imSize is the logical size of the vector field (the size of the
   * image the vector field will be associated with).
   *
   * IMPORTANT: Since FFT calculation requires a slightly larger array
   * for holding complex-vauled data, this VectorField is padded with
   * extra x-dimension elements. This means that elements of this
   * array should ONLY be accessed using the (x,y,z) operator, NOT
   * pointer arithmatic.
   */
  static
  FFTWVectorField * 
  NewFFTWVecField(const SizeType imSize){
    unsigned xSizeFFT = 2 * (imSize.x / 2 + 1);
    return (new FFTWVectorField(xSizeFFT, imSize.y, imSize.z));
  }

  /**
   * Static mutex to synchronize initialization of multiple DiffOpers
   * -- FFTW plan initialization must be synchronized
   */
  static pthread_mutex_t mFFTWPlanInitializationMutex;

  // ===== Constructors =====

  /**
   * Construct a fully-initialized (Initialize() is called internally)
   * DiffOper of the given logicalSize and spacing.
   */
  DiffOperT(const SizeType &logicalSize, 
	   const SpacingType &spacing,
	   const DiffOperParam &params);

  /**
   * DiffOper constructs an internal FFTWVectorField of the given
   * logical size.  This can be accessed through
   * GetInternalFFTWVectorField().
   */
  DiffOperT(const SizeType &logicalSize, 
	   const SpacingType &spacing = SpacingType(1.0, 1.0, 1.0));

  /**
   * The FFTWVectorField should have been initialized by
   * InitializeFFTWVecField().  This vector field holds the
   * input and output of the differential equation solution.
   */
  DiffOperT(FFTWVectorField *vf, 
	   const SizeType &logicalSize, 
	   const SpacingType &spacing);

  ~DiffOperT();
  
  // ===== Public Members =====

  /**
   * Get the logical size of the vector field passed to the
   * constructor.
   */
  const SizeType &GetVectorFieldLogicalSize();

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

  

  //
  // fftw interface
  //
  /** Set the number of threads used by the FFTW library */
  void         SetFFTWNumberOfThreads(unsigned int numThreads);
  /** Get the number of threads used by the FFTW library */
  unsigned int GetFFTWNumberOfThreads() const;
  
  /** Tells the FFTW library whether to set parameters by explicitly testing performance, or just estimating */
  void         SetFFTWMeasureOn();
  /** Tells the FFTW library whether to set parameters by explicitly testing performance, or just estimating */
  void         SetFFTWMeasureOff();
  /** Tells the FFTW library whether to set parameters by explicitly testing performance, or just estimating */
  void         SetFFTWMeasure(bool b);
  /** Tells the FFTW library whether to set parameters by explicitly testing performance, or just estimating */
  bool         GetFFTWMeasure() const;

  /**
   * Return the internal FFTWVectorField
   */
  FFTWVectorField *GetInternalFFTWVectorField(){
    return this->mFFTWVectorField;
  }

  /**
   * Copy the data from vf into the internal FFTWVectorField.  vf must
   * be the same size as the logical dimensions of the internal array.
   */
  void CopyIn(const VectorField &vf);

  /**
   * Copy the data from the internal FFTWVectorField into vf.  vf must
   * be the same size as the logical dimensions of the internal array.
   */
  void CopyOut(VectorField &vf);

  /**
   * Note: this call overwrites the current contents of the internal
   * FFTWVectorField.  This should be called before any data is put
   * there, and before ApplyOperator() or ApplyInverseOperator() is
   * called.
   */
  void
  Initialize();

  /**
   * Apply L operator to internal FFTWVectorField.
   * InitializeFFTWPlans() should be called prior to invoking this
   * method for the first time.
   */
  void
  ApplyOperator();

  /**
   * Apply inverse L operator to internal FFTWVectorField.
   * InitializeFFTWPlans() should be called prior to invoking this
   * method for the first time.
   */
  void
  ApplyInverseOperator();
  
  /**
   * Pointwise multiply internal vector field by rhs.  Standard
   * pointwise multiply routines are not safe for FFTWVectorField
   */
  void pointwiseMultiplyBy_FFTW_Safe(const Array3D<Real> &rhs);

protected:

  // ===== Protected Member Functions =====

  /**
   * Setup and initialize data objects.  Require mLogicalSize and
   * mFFTWVectorField be set.
   */
  void InitializeNewObject();

  void Delete();

  /**
   * Note: this call overwrites the current contents of the internal
   * FFTWVectorField.  This should be called before any data is put
   * there, and before ApplyOperator() or ApplyInverseOperator() is
   * called.
   */
  void
  InitializeFFTWPlans();

  void
  ProjectIncomp(T* complexPtr, 
		unsigned int x, 
		unsigned int y, 
		unsigned int z);

  void 
  InitializeOperatorLookupTable();

  void 
  InitializeEigenLookup();

  /**
   * Apply L operator to internal FFTWVectorField.
   * InitializeFFTWPlans() should be called prior to invoking this
   * method for the first time.
   */
  void 
  ApplyOperatorOnTheFly();

  /**
   * Apply inverse L operator to internal FFTWVectorField.
   * InitializeFFTWPlans() should be called prior to invoking this
   * method for the first time.
   */
  void 
  ApplyInverseOperatorOnTheFly();

  void 
  ApplyOperatorPrecomputedEigs(bool inverse);

  void
  InverseOperatorMultiply(T* complexPtr,
			  T& L00,
			  T& L10, T& L11,
			  T& L20, T& L21, T& L22);

  void
  OperatorMultiply(T* complexPtr,
		   T& L00,
		   T& L10, T& L11,
		   T& L20, T& L21, T& L22);

  void
  OperatorPowMultiply(unsigned int index, bool inverse);

  void
  PowL(T& L00,
       T& L10, T& L11, 
       T& L20, T& L21, T& L22, 
       unsigned int index, short invFlag);
    
  // ===== Member Data =====
  
  //
  // scratch field and logical size
  FFTWVectorField *mFFTWVectorField;
  bool mOwnFFTWVectorField;
  SizeType mLogicalSize;
  SpacingType mSpacing;
  unsigned int mXFFTMax;

  DiffOperFFTWWrapper<T> mFFTWWrapper;

  //
  // parameters for calculating L
  Real mAlpha;
  Real mBeta;
  Real mGamma;
  Real mLPow;
  
  // enforce incompressability?
  bool mDivergenceFree;

  // precompute eigenvalues/vectors of transform?
  bool mEigenLUTNeedsUpdate;
  bool mUseEigenLUT;

  // FFT parameters
  bool         mFFTWPlansNeedUpdate;
  bool         mFFTWMeasure;
  unsigned int mFFTWNumberOfThreads;

  /**
   * look up table to for Linv computation
   */
  struct LUT
  {
    std::vector<T> cosWX, cosWY, cosWZ;
    std::vector<T> sinWX, sinWY, sinWZ;
    Array3D<T> nsq;
    
    LUT()
      : cosWX(0), cosWY(0), cosWZ(0),
	sinWX(0), sinWY(0), sinWZ(0)
    {}
    
    LUT(unsigned int xSize, 
	unsigned int ySize, 
	unsigned int zSize)
      : cosWX(xSize / 2 + 1), cosWY(ySize), cosWZ(zSize),
	sinWX(xSize / 2 + 1), sinWY(ySize), sinWZ(zSize), nsq(xSize,ySize,zSize)
    {}
  };

  //
  // Lookup table 
  LUT mOperatorLookupTable;

  /**
   * eigenvalue/vector lookup table to for Linv computation
   */
  struct LUTEIGEN
  {
    std::vector<T> eigenValues, eigenVectors;
    LUTEIGEN()
      : eigenValues(0), eigenVectors(0) 
    {}

    LUTEIGEN(SizeType size)
      : eigenValues(3*(size.x / 2 + 1)*size.y*size.z), eigenVectors(9*(size.x / 2 + 1)*size.y*size.z) 
    {}
  };
  
  LUTEIGEN mEigenLookupTable;
  
};

typedef DiffOperT<Real> DiffOper;

#endif //__DIFF_OPER_H__
