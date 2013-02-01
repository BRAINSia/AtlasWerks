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


#include "DiffOper.h"
#include "ApplicationUtils.h"
#include "Matrix3D.h"

/* ================================================================
 * Float Specialization
 ================================================================ */

DiffOperFFTWWrapper<float>::
DiffOperFFTWWrapper() :
  mFFTWForwardPlan(NULL),
  mFFTWBackwardPlan(NULL)
{
}

void
DiffOperFFTWWrapper<float>::
Initialize(SizeType logicalSize, 
	   Array3D<Vector3D< float > > *array, 
	   int nThreads,
	   bool measure)
{
  // create the plans
  int rank = 3;
  int logicalSizeParam[3];
  logicalSizeParam[0] = logicalSize.z;
  logicalSizeParam[1] = logicalSize.y;
  logicalSizeParam[2] = logicalSize.x;
  
  int howMany = 3;
  int stride  = 3;
  int dist    = 1;
  float *dataPtr = 
    (float*) &(array->getDataPointer()->x);
  
  fftwf_plan_with_nthreads(nThreads);
    
  this->mFFTWForwardPlan = 
    fftwf_plan_many_dft_r2c(rank, logicalSizeParam, howMany, 
			    dataPtr, 
			    0, stride, dist, 
			    (fftwf_complex*) dataPtr, 
			    0, stride, dist,
			    measure ? FFTW_MEASURE : FFTW_ESTIMATE);
  
  this->mFFTWBackwardPlan = 
    fftwf_plan_many_dft_c2r(rank, logicalSizeParam, howMany, 
			    (fftwf_complex*) dataPtr,
			    0, stride, dist, 
			    dataPtr,
			    0, stride, dist,
			    measure ? FFTW_MEASURE : FFTW_ESTIMATE);
  
  if (!this->mFFTWForwardPlan)
    {
      throw AtlasWerksException(__FILE__, __LINE__, "FFTW forward plan failed to initialize");
    }
  if (!this->mFFTWBackwardPlan)
    {
      throw AtlasWerksException(__FILE__, __LINE__, "FFTW backward plan failed to initialize");
    }
  
}

void
DiffOperFFTWWrapper<float>::
Delete()
{
  if(this->mFFTWForwardPlan)
    fftwf_destroy_plan(this->mFFTWForwardPlan);
  if(this->mFFTWBackwardPlan)
    fftwf_destroy_plan(this->mFFTWBackwardPlan);
}

void
DiffOperFFTWWrapper<float>::
ExecuteForward(){
  fftwf_execute(this->mFFTWForwardPlan);
}

void
DiffOperFFTWWrapper<float>::
ExecuteBackward(){
  fftwf_execute(this->mFFTWBackwardPlan);
}

/* ================================================================
 * Double Specialization
 ================================================================ */

DiffOperFFTWWrapper<double>::
DiffOperFFTWWrapper() :
  mFFTWForwardPlan(NULL),
  mFFTWBackwardPlan(NULL)
{
}

void
DiffOperFFTWWrapper<double>::
Initialize(SizeType logicalSize, 
	   Array3D<Vector3D< double > > *array, 
	   int nThreads,
	   bool measure)
{
  // create the plans
  int rank = 3;
  int logicalSizeParam[3];
  logicalSizeParam[0] = logicalSize.z;
  logicalSizeParam[1] = logicalSize.y;
  logicalSizeParam[2] = logicalSize.x;
  
  int howMany = 3;
  int stride  = 3;
  int dist    = 1;
  double *dataPtr = 
    (double*) &(array->getDataPointer()->x);
  
  fftw_plan_with_nthreads(nThreads);
    
  this->mFFTWForwardPlan = 
    fftw_plan_many_dft_r2c(rank, logicalSizeParam, howMany, 
			   dataPtr, 
			   0, stride, dist, 
			   (fftw_complex*) dataPtr, 
			   0, stride, dist,
			   measure ? FFTW_MEASURE : FFTW_ESTIMATE);
  
  this->mFFTWBackwardPlan = 
    fftw_plan_many_dft_c2r(rank, logicalSizeParam, howMany, 
			   (fftw_complex*) dataPtr,
			   0, stride, dist, 
			   dataPtr,
			   0, stride, dist,
			   measure ? FFTW_MEASURE : FFTW_ESTIMATE);
  
  if (!this->mFFTWForwardPlan)
    {
      throw AtlasWerksException(__FILE__, __LINE__, "FFTW forward plan failed to initialize");
    }
  if (!this->mFFTWBackwardPlan)
    {
      throw AtlasWerksException(__FILE__, __LINE__, "FFTW backward plan failed to initialize");
    }  
  
}

void
DiffOperFFTWWrapper<double>::
Delete()
{
  if(this->mFFTWForwardPlan)
    fftw_destroy_plan(this->mFFTWForwardPlan);
  if(this->mFFTWBackwardPlan)
    fftw_destroy_plan(this->mFFTWBackwardPlan);
}

void
DiffOperFFTWWrapper<double>::
ExecuteForward(){
  fftw_execute(this->mFFTWForwardPlan);
}

void
DiffOperFFTWWrapper<double>::
ExecuteBackward(){
  fftw_execute(this->mFFTWBackwardPlan);
}

/* ================================================================
 * DiffOperT
 ================================================================ */

// ======== Public Members ======== //

template<class T> pthread_mutex_t DiffOperT<T>::mFFTWPlanInitializationMutex = PTHREAD_MUTEX_INITIALIZER;

template<class T>
DiffOperT<T>::
DiffOperT(const SizeType &logicalSize, 
	 const SpacingType &spacing,
	 const DiffOperParam &params)
{
  this->mLogicalSize = logicalSize;
  this->mSpacing = spacing;
  this->mFFTWVectorField = DiffOperT<T>::NewFFTWVecField(mLogicalSize);
  this->mOwnFFTWVectorField = true;
  this->InitializeNewObject();
  this->SetParams(params);
  this->Initialize();
}

template<class T>
DiffOperT<T>::
DiffOperT(const SizeType &logicalSize,
	 const SpacingType &spacing)
{
  this->mLogicalSize = logicalSize;
  this->mSpacing = spacing;
  this->mFFTWVectorField = DiffOperT<T>::NewFFTWVecField(mLogicalSize);
  this->mOwnFFTWVectorField = true;
  this->InitializeNewObject();
  // use default values in DiffOperParam
  this->SetParams(DiffOperParam());
}

template<class T>
DiffOperT<T>::
DiffOperT(FFTWVectorField *vf, 
	 const SizeType &logicalSize,
	 const SpacingType &spacing)
{
  this->mLogicalSize = logicalSize;
  this->mSpacing = spacing;
  this->mFFTWVectorField = vf;
  this->mOwnFFTWVectorField = false;
  this->InitializeNewObject();
}

template<class T>
DiffOperT<T>::
~DiffOperT()
{
  this->Delete();
}

template<class T>
const SizeType &
DiffOperT<T>::
GetVectorFieldLogicalSize()
{
  return this->mLogicalSize;
}

template<class T>
void 
DiffOperT<T>::
SetParams(const DiffOperParam &param)
{
  // set values from param
  this->SetAlpha(param.Alpha());
  this->SetBeta(param.Beta());
  this->SetGamma(param.Gamma());
  this->SetLPow(param.LPow());
  this->SetUseEigenLUT(param.UseEigenLUT());
  this->SetDivergenceFree(param.DivergenceFree());
  this->SetFFTWNumberOfThreads(param.FFTWNumberOfThreads());
  this->SetFFTWMeasure(param.FFTWMeasure());
}

template<class T>
DiffOperParam 
DiffOperT<T>::
GetParams()
{
  DiffOperParam param;
  param.Alpha() = this->GetAlpha();
  param.Beta() = this->GetBeta();
  param.Gamma() = this->GetGamma();
  param.LPow() = this->GetLPow();
  param.UseEigenLUT() = this->GetUseEigenLUT();
  param.DivergenceFree() = this->GetDivergenceFree();
  param.FFTWNumberOfThreads() = this->GetFFTWNumberOfThreads();
  param.FFTWMeasure() = this->GetFFTWMeasure();
  return param;
}

template<class T>
void 
DiffOperT<T>::
SetAlpha(Real alpha)
{
  if(alpha != this->mAlpha){
    this->mAlpha = alpha;
    this->mEigenLUTNeedsUpdate = true;
  }
}

template<class T>
Real 
DiffOperT<T>::
GetAlpha(){
  return this->mAlpha;
}

template<class T>
void 
DiffOperT<T>::
SetBeta(Real beta){
  if(beta != this->mBeta){
    this->mBeta = beta;
    this->mEigenLUTNeedsUpdate = true;
  }
}

template<class T>
Real 
DiffOperT<T>::
GetBeta(){
  return this->mBeta;
}

template<class T>
void 
DiffOperT<T>::
SetGamma(Real gamma){
  if(gamma != this->mGamma){
    this->mGamma = gamma;
    this->mEigenLUTNeedsUpdate = true;
  }
}

template<class T>
Real 
DiffOperT<T>::
GetGamma(){
  return this->mGamma;
}

template<class T>
void 
DiffOperT<T>::
SetDivergenceFree(bool df){
  this->mDivergenceFree = df;
}

template<class T>
bool 
DiffOperT<T>::
GetDivergenceFree(){
  return this->mDivergenceFree;
}

template<class T>
void 
DiffOperT<T>::
SetLPow(Real p){
  mLPow = p;
  if(p != 1.0) SetUseEigenLUT(true);
}

template<class T>
Real 
DiffOperT<T>::
GetLPow(){
  return mLPow;
}

template<class T>
void 
DiffOperT<T>::
SetUseEigenLUT(bool b){
  mUseEigenLUT = b; 
}

template<class T>
bool 
DiffOperT<T>::
GetUseEigenLUT(){
  return mUseEigenLUT;
}

template<class T>
void
DiffOperT<T>::
SetFFTWNumberOfThreads(unsigned int numThreads)
{
  if(numThreads != this->mFFTWNumberOfThreads){
    this->mFFTWNumberOfThreads = numThreads;
    this->mFFTWPlansNeedUpdate = true;
  }
}

template<class T>
unsigned int 
DiffOperT<T>::
GetFFTWNumberOfThreads() const
{
  return this->mFFTWNumberOfThreads;
}

template<class T>
void 
DiffOperT<T>::
SetFFTWMeasure(bool b)
{
  if(b != this->mFFTWMeasure){
    this->mFFTWMeasure = b;
    this->mFFTWPlansNeedUpdate = true;
  }
}

template<class T>
void 
DiffOperT<T>::
SetFFTWMeasureOn()
{
  this->SetFFTWMeasure(true);
}

template<class T>
void 
DiffOperT<T>::
SetFFTWMeasureOff()
{
  this->SetFFTWMeasure(false);
}

template<class T>
bool 
DiffOperT<T>::
GetFFTWMeasure() const
{
  return this->mFFTWMeasure;
}

template<class T>
void
DiffOperT<T>::
Initialize()
{
  this->InitializeFFTWPlans();
  if(mUseEigenLUT){
    this->InitializeEigenLookup();
  }
}

template<class T>
void 
DiffOperT<T>::
CopyIn(const VectorField &vf)
{
  for(unsigned int z = 0; z < this->mLogicalSize.z; z++){
    for(unsigned int y = 0; y < this->mLogicalSize.y; y++){
      for(unsigned int x = 0; x < this->mLogicalSize.x; x++){
	(*this->mFFTWVectorField)(x,y,z) = vf(x,y,z);
      }
    }
  }
}

template<class T>
void 
DiffOperT<T>::
CopyOut(VectorField &vf)
{
  for(unsigned int z = 0; z < this->mLogicalSize.z; z++){
    for(unsigned int y = 0; y < this->mLogicalSize.y; y++){
      for(unsigned int x = 0; x < this->mLogicalSize.x; x++){
	vf(x,y,z) = (*this->mFFTWVectorField)(x,y,z);
      }
    }
  }
}

template<class T>
void
DiffOperT<T>::
ApplyOperator()
{
  if(mUseEigenLUT){
    if(mEigenLUTNeedsUpdate){
      std::cerr 
	<< "Error, eigenvalue/vector lookup table needs to be initialized " 
	<< "(or re-initialized) prior to calling ApplyOperator().  This is "
	<< "generally accomplished by calling Initialize()." 
	<< std::endl;
    }
    this->ApplyOperatorPrecomputedEigs(false);
  }else{
    if(mLPow != 1){
      std::cerr << "Error, cannot compute non-unitary powers of L operator without using precomputed eigenvals/vectors" << std::endl;
    }
    ApplyOperatorOnTheFly();
  }
}

template<class T>
void
DiffOperT<T>::
ApplyInverseOperator(){
  if(mUseEigenLUT){
    if(mEigenLUTNeedsUpdate){
      std::cerr 
	<< "Error, eigenvalue/vector lookup table needs to be initialized " 
	<< "(or re-initialized) prior to calling ApplyInverseOperator().  "
	<< "This is generally accomplished by calling Initialize()." 
	<< std::endl;
    }
    this->ApplyOperatorPrecomputedEigs(true);
  }else{
    if(mLPow != 1){
      std::cerr << "Error, cannot compute non-unitary powers of L operator without using precomputed eigenvals/vectors" << std::endl;
    }
    ApplyInverseOperatorOnTheFly();
  }
}

template<class T>
void 
DiffOperT<T>::
pointwiseMultiplyBy_FFTW_Safe(const Array3D<Real> &rhs)
{
  Vector3D<unsigned int> size = rhs.getSize();
  for (unsigned int z = 0; z < size.z; ++z) {
    for (unsigned int y = 0; y < size.y; ++y) {
      for (unsigned int x = 0; x < size.x; ++x) {
	(*mFFTWVectorField)(x,y,z) *= rhs(x,y,z);
      }
    }
  }
}

// ======== Protected Members ======== //

template<class T>
void 
DiffOperT<T>::
InitializeNewObject()
{
  // set up initial variable vals
  this->mXFFTMax = this->mLogicalSize.x/2 + 1;
  this->mDivergenceFree = false;
  this->mFFTWMeasure = true;
  this->mFFTWNumberOfThreads = 1;
  this->mFFTWPlansNeedUpdate = true;
  this->mEigenLUTNeedsUpdate = true;
  this->mUseEigenLUT = false;
  this->mLPow = 1;
  // allocate and precompute
  this->InitializeOperatorLookupTable();
}

template<class T>
void 
DiffOperT<T>::
Delete()
{
  mFFTWWrapper.Delete();
  if(this->mOwnFFTWVectorField){
    delete this->mFFTWVectorField;
  }
}

template<class T>
void
DiffOperT<T>::
InitializeFFTWPlans()
{

  // don't update unless we need to
  if(!this->mFFTWPlansNeedUpdate){
    return;
  }

  // this only deallocates if already allocated
  mFFTWWrapper.Delete();

  // need to synchronize fftw plan creation
  pthread_mutex_lock(&mFFTWPlanInitializationMutex);

  mFFTWWrapper.Initialize(mLogicalSize,
			  mFFTWVectorField,
			  mFFTWNumberOfThreads,
			  mFFTWMeasure);
  
  // need to synchronize fftw plan creation
  pthread_mutex_unlock(&mFFTWPlanInitializationMutex);
  
  this->mFFTWPlansNeedUpdate = false;
  
}

template<class T>
void 
DiffOperT<T>::
InitializeOperatorLookupTable()
{
  typename DiffOperT<T>::LUT lut(this->mLogicalSize.x, 
			  this->mLogicalSize.y, 
			  this->mLogicalSize.z);

  //
  // precompute some values
  //
//   double sX = this->mSpacing.x * 2.0 * M_PI / this->mLogicalSize.x; 
//   double sY = this->mSpacing.y * 2.0 * M_PI / this->mLogicalSize.y; 
//   double sZ = this->mSpacing.z * 2.0 * M_PI / this->mLogicalSize.z; 
  double sX = 2.0 * M_PI / this->mLogicalSize.x; 
  double sY = 2.0 * M_PI / this->mLogicalSize.y; 
  double sZ = 2.0 * M_PI / this->mLogicalSize.z; 

  SpacingType spacingSqr = this->mSpacing * this->mSpacing;

  //
  // fill in luts
  //
  for (unsigned int x = 0; x < lut.cosWX.size(); ++x) 
    {
      lut.cosWX[x] = (2.0 * cos(sX * static_cast<T>(x)) - 2.0) / spacingSqr.x;
      lut.sinWX[x] = sin(sX * static_cast<T>(x)) / this->mSpacing.x;
    }
  for (unsigned int y = 0; y < lut.cosWY.size(); ++y)
    {
      lut.cosWY[y] = (2.0 * cos(sY * static_cast<T>(y)) - 2.0) / spacingSqr.y;
      lut.sinWY[y] = sin(sY * static_cast<T>(y)) / this->mSpacing.y;
    }
  for (unsigned int z = 0; z < lut.cosWZ.size(); ++z)
    {
      lut.cosWZ[z] = (2.0 * cos(sZ * static_cast<T>(z)) - 2.0) / spacingSqr.z;
      lut.sinWZ[z] = sin(sZ * static_cast<T>(z)) / this->mSpacing.z;
    }  

  for (unsigned int x = 0; x < lut.cosWX.size(); ++x)
    for (unsigned int y = 0; y < lut.cosWY.size(); ++y)
      for (unsigned int z = 0; z < lut.cosWZ.size(); ++z)
        lut.nsq(x,y,z) = lut.sinWX[x]*lut.sinWX[x]
          + lut.sinWY[y]*lut.sinWY[y]
          + lut.sinWZ[z]*lut.sinWZ[z]; // norm squared of projection
                                       // vector for incompressible flow

  //
  // copy values to the ivar
  //
  this->mOperatorLookupTable = lut;
}

template<class T>
void 
DiffOperT<T>::
InitializeEigenLookup()
{  

  // only recompute this if needed
  if(!this->mEigenLUTNeedsUpdate){
    return;
  }

  // input array
  Matrix3D<T> L;
  // output
  T eigenValues[3];
  Matrix3D<T> eigenVectors;

  //
  // fill in luteigens
  //
  mEigenLookupTable =  LUTEIGEN(mLogicalSize);
  T lambda;
  T L00;
  T L10, L11;
  T L20, L21, L22;
  int i;
  unsigned int x, y, z;
  unsigned int index = 0;
  
  // TEST
  // std::ofstream myfile;
  // myfile.open ("LUTEIGEN_New.txt");
  // END TEST

  for ( z = 0; z < this->mLogicalSize.z; ++z)
  {
    for ( y = 0; y < this->mLogicalSize.y; ++y)
    {
      for ( x = 0; x < this->mXFFTMax; ++x)
      {

	lambda = - this->mAlpha 
	  * (this->mOperatorLookupTable.cosWX[x] + 
	     this->mOperatorLookupTable.cosWY[y] + 
	     this->mOperatorLookupTable.cosWZ[z]) 
	  + this->mGamma;	      

	L00 = lambda - 
	  this->mBeta * this->mOperatorLookupTable.cosWX[x];
	L11 = lambda - 
	  this->mBeta * this->mOperatorLookupTable.cosWY[y];
	L22 = lambda - 
	  this->mBeta * this->mOperatorLookupTable.cosWZ[z];
	L10 = this->mBeta * this->mOperatorLookupTable.sinWX[x] * 
	  this->mOperatorLookupTable.sinWY[y];
	L20 = this->mBeta * this->mOperatorLookupTable.sinWX[x] * 
	  this->mOperatorLookupTable.sinWZ[z];
	L21 = this->mBeta * this->mOperatorLookupTable.sinWY[y] * 
	  this->mOperatorLookupTable.sinWZ[z];

	// fill in lower triangular matrix
	L(0,0) = L00;
	L(1,0) = L10;
	L(2,0) = L20;
	L(1,1) = L11;
	L(2,1) = L21;
	L(2,2) = L22;
	
	//Compute EigenDecomposition and store it in luteigen
	Matrix3D<T>::factorEVsym(L.a, eigenValues, eigenVectors.a, false);
	for(i=0;i<3;i++){
	  mEigenLookupTable.eigenValues[index+i] = eigenValues[i]; 
	}
	for(i=0;i<9;i++){
	  mEigenLookupTable.eigenVectors[3*index+i] = eigenVectors(i); 
	}

	// TEST
	// myfile <<"*********************"<<std::endl;
	// myfile <<"mat:"<<std::endl;
	// myfile<<L00<<" "<<L10<<" "<<L20<<std::endl;
	// myfile<<L10<<" "<<L11<<" "<<L21<<std::endl;
	// myfile<<L20<<" "<<L21<<" "<<L22<<std::endl;
	// myfile <<"vals:"<<std::endl;
	// myfile<<mEigenLookupTable.eigenValues[index]<<" "<<mEigenLookupTable.eigenValues[index+1]<<" "<<mEigenLookupTable.eigenValues[index+2]<<std::endl;
	// myfile <<"vecs:"<<std::endl;
	// myfile<<eigenVectors(0)<<" "<<eigenVectors(1)<<" "<<eigenVectors(2)<<std::endl;
	// myfile<<eigenVectors(3)<<" "<<eigenVectors(4)<<" "<<eigenVectors(5)<<std::endl;
	// myfile<<eigenVectors(6)<<" "<<eigenVectors(7)<<" "<<eigenVectors(8)<<std::endl;	
	// myfile<<"*********************"<<std::endl;
	// END TEST

	index += 3;
      }
    }
  }
  std::cout << "LUTEIGENs done" << std::endl;

  // TEST
  // myfile.close();
  // END TEST

  this->mEigenLUTNeedsUpdate = false;

}

template<class T>
void 
DiffOperT<T>::
ApplyOperatorOnTheFly()
{
  if(this->mFFTWPlansNeedUpdate){
    std::cerr << "Error, InitializeFFTWPlans() needs to be called after changing FFTW params and before calling ApplyOperator()" << std::endl;
    return;
  }

  // forward fft (scale array, then compute fft)
  this->mFFTWVectorField->
    scale(1.0 / this->mLogicalSize.productOfElements());
  mFFTWWrapper.ExecuteForward();

  // apply operator
  double lambda;
  T L00;
  T L10, L11;
  T L20, L21, L22;

  for (unsigned int z = 0; z < this->mLogicalSize.z; ++z)
    {
      for (unsigned int y = 0; y < this->mLogicalSize.y; ++y)
	{
	  for (unsigned int x = 0; x < this->mXFFTMax; ++x)
	    {
	      //
	      // compute L (it is symmetric, only need lower triangular part)
	      //
	      
	      // maybe lambda should be stored in a lut
	      // it would reduce computation but may cause cache misses
	      lambda = - this->mAlpha 
		* (this->mOperatorLookupTable.cosWX[x] + 
                   this->mOperatorLookupTable.cosWY[y] + 
                   this->mOperatorLookupTable.cosWZ[z]) 
		+ this->mGamma;	      
	      
	      L00 = lambda - 
		this->mBeta * this->mOperatorLookupTable.cosWX[x];
	      L11 = lambda - 
		this->mBeta * this->mOperatorLookupTable.cosWY[y];
	      L22 = lambda - 
		this->mBeta * this->mOperatorLookupTable.cosWZ[z];
	      L10 = this->mBeta * this->mOperatorLookupTable.sinWX[x] * 
                this->mOperatorLookupTable.sinWY[y];
	      L20 = this->mBeta * this->mOperatorLookupTable.sinWX[x] * 
                this->mOperatorLookupTable.sinWZ[z];
	      L21 = this->mBeta * this->mOperatorLookupTable.sinWY[y] * 
                this->mOperatorLookupTable.sinWZ[z];

	      //
	      // compute F = LV (for real and imaginary parts)
	      //
              T* complexPtr =
                &(*this->mFFTWVectorField)(x * 2, y, z).x; 
	      this->OperatorMultiply(complexPtr,
				      L00,
				      L10, L11,
				      L20, L21, L22);
	      
              if (this->mDivergenceFree && (x | y | z))
                {
		  // Project onto incompressible field
		  this->ProjectIncomp(complexPtr,x,y,z);
                }
	    }
	}
    }
  
  // backward fft
  mFFTWWrapper.ExecuteBackward();
}

template<class T>
void 
DiffOperT<T>::
ApplyInverseOperatorOnTheFly()
{

  if(this->mFFTWPlansNeedUpdate){
    std::cerr << "Error, InitializeFFTWPlans() needs to be called after changing FFTW params and before calling ApplyInverseOperator()" << std::endl;
    return;
  }

#ifdef __DEBUG__
  std::cout << "Force field max " << getMax(*this->mFFTWVectorField) << std::endl;
  std::cout << "Force sum " << getSum(*this->mFFTWVectorField) << std::endl;
#endif

  // forward fft (scale array, then compute fft)
  this->mFFTWVectorField->
    scale(1.0 / this->mLogicalSize.productOfElements());
  mFFTWWrapper.ExecuteForward();

  double lambda;
  T L00;
  T L10, L11;
  T L20, L21, L22;

  for (unsigned int z = 0; z < this->mLogicalSize.z; ++z)
    {
      for (unsigned int y = 0; y < this->mLogicalSize.y; ++y)
	{
	  for (unsigned int x = 0; x < this->mXFFTMax; ++x)
	    {
	      //
	      // compute L (it is symmetric, only need lower triangular part)
	      //
	      
	      // maybe lambda should be stored in a lut
	      // it would reduce computation but may cause cache misses
	      lambda = - mAlpha 
		* (this->mOperatorLookupTable.cosWX[x] + 
                   this->mOperatorLookupTable.cosWY[y] + 
                   this->mOperatorLookupTable.cosWZ[z]) 
		+ this->mGamma;	      
	      
	      L00 = lambda - 
		this->mBeta * this->mOperatorLookupTable.cosWX[x];
	      L11 = lambda - 
		this->mBeta * this->mOperatorLookupTable.cosWY[y];
	      L22 = lambda - 
		this->mBeta * this->mOperatorLookupTable.cosWZ[z];
	      L10 = this->mBeta * this->mOperatorLookupTable.sinWX[x] * 
                this->mOperatorLookupTable.sinWY[y];
	      L20 = this->mBeta * this->mOperatorLookupTable.sinWX[x] * 
                this->mOperatorLookupTable.sinWZ[z];
	      L21 = this->mBeta * this->mOperatorLookupTable.sinWY[y] * 
                this->mOperatorLookupTable.sinWZ[z];

	      //
	      // compute V = Linv F (for real and imaginary parts)
	      //
              T* complexPtr =
                &(*this->mFFTWVectorField)(x * 2, y, z).x; 
	      this->InverseOperatorMultiply(complexPtr,
					    L00,
					    L10, L11,
					    L20, L21, L22);
	      
              if (this->mDivergenceFree && (x | y | z))
                {
		  // Project onto incompressible field
		  this->ProjectIncomp(complexPtr,x,y,z);
                }
	    }
	}
    }
  
  // backward fft
  mFFTWWrapper.ExecuteBackward();

#ifdef __DEBUG__
  std::cout << "Velocity  field max " << getMax(*this->mFFTWVectorField) << std::endl;
#endif

}


template<class T>
void 
DiffOperT<T>::
ApplyOperatorPrecomputedEigs(bool inverse)
{

  if(this->mFFTWPlansNeedUpdate){
    std::cerr << "Error, InitializeFFTWPlans() needs to be called after changing FFTW params and before calling ApplyInverseOperator()" << std::endl;
    return;
  }

  // forward fft (scale array, then compute fft)
  this->mFFTWVectorField->
    scale(1.0 / this->mLogicalSize.productOfElements());
  mFFTWWrapper.ExecuteForward();

  unsigned int index = 0;
  for (unsigned int z = 0; z < this->mLogicalSize.z; ++z)
    {
      for (unsigned int y = 0; y < this->mLogicalSize.y; ++y)
	{
	  for (unsigned int x = 0; x < this->mXFFTMax; ++x)
	    {
	      
	      this->OperatorPowMultiply(index, inverse);
	      index += 3;
	      
              if (this->mDivergenceFree && (x | y | z) )
                {
		  // Project onto incompressible field
		  T* complexPtr =
		    &(*this->mFFTWVectorField)(x * 2, y, z).x; 
		  this->ProjectIncomp(complexPtr,x,y,z);
                }
	    }
	}
    }
  
  // backward fft
  mFFTWWrapper.ExecuteBackward();
  
}

template<class T>
void
DiffOperT<T>::
ProjectIncomp(T* complexPtr, unsigned int x, unsigned int y, unsigned int z)
{
  // in Fourier space we project onto (-i*sin(u),-i*sin(v),-i*sin(w)) and remove that component
  // 2008 jdh
  
  T bRealX = complexPtr[0];
  T bRealY = complexPtr[2];
  T bRealZ = complexPtr[4];
  
  T bImagX = complexPtr[1];
  T bImagY = complexPtr[3];
  T bImagZ = complexPtr[5];
  
  T& vRealX = complexPtr[0];
  T& vRealY = complexPtr[2];
  T& vRealZ = complexPtr[4];
  
  T& vImagX = complexPtr[1];
  T& vImagY = complexPtr[3];
  T& vImagZ = complexPtr[5];
  
  typename DiffOperT<T>::LUT *lut = &this->mOperatorLookupTable;
  
  // This is now in LUT
  //   nsq = lut->sinWX[x]*lut->sinWX[x]
  //       + lut->sinWY[y]*lut->sinWY[y]
  //       + lut->sinWZ[z]*lut->sinWZ[z]; // norm squared of projection vector
  
  // S=(sinwx,sinwy,sinwz)
  // Real part of S dot V in Fourier
  double ReSdotV = ( bRealX*lut->sinWX[x]
                     +bRealY*lut->sinWY[y]
                     +bRealZ*lut->sinWZ[z]);
  // Imag part of S dot V in Fourier
  double ImSdotV = ( bImagX*lut->sinWX[x]
                     +bImagY*lut->sinWY[y]
                     +bImagZ*lut->sinWZ[z]);
  
  // Subtract S dot V (normalizing S)
  vRealX = bRealX - ReSdotV*lut->sinWX[x]/lut->nsq(x,y,z);
  vRealY = bRealY - ReSdotV*lut->sinWY[y]/lut->nsq(x,y,z);
  vRealZ = bRealZ - ReSdotV*lut->sinWZ[z]/lut->nsq(x,y,z);
  
  vImagX = bImagX - ImSdotV*lut->sinWX[x]/lut->nsq(x,y,z);
  vImagY = bImagY - ImSdotV*lut->sinWY[y]/lut->nsq(x,y,z);
  vImagZ = bImagZ - ImSdotV*lut->sinWZ[z]/lut->nsq(x,y,z);
}

template<class T>
void
DiffOperT<T>::
InverseOperatorMultiply(T* complexPtr,
                        T& L00,
                        T& L10, T& L11,
                        T& L20, T& L21, T& L22)
{
  T G00;
  T G10, G11;
  T G20, G21, G22;
  T y0, y1, y2;
  //
  // Given that A is pos-def symetric matrix, solve Ax=b by finding
  // cholesky decomposition GG'=A
  // and then performing 2 back-solves, Gy=b and then G'x=y to get x.
  // 
	   
  // 1. find cholesky decomposition by finding G such that GG'=A.
  //    A must be positive definite symetric (we assume that here)
  //    G is then lower triangular, see algorithm 4.2.1 p142-3
  //    in Golub and VanLoan
  // Note: these are in matlab notation 1:3
  // [ G(1,1)   0      0    ]   [ G(1,1) G(2,1) G(3,1) ]   
  // [ G(2,1) G(2,2)   0    ] * [   0    G(2,2) G(3,2) ] = Amatrix
  // [ G(3,1) G(3,2) G(3,3) ]   [   0      0    G(3,3) ]

  T bRealX = complexPtr[0];
  T bRealY = complexPtr[2];
  T bRealZ = complexPtr[4];

  T bImagX = complexPtr[1];
  T bImagY = complexPtr[3];
  T bImagZ = complexPtr[5];

  T& vRealX = complexPtr[0];
  T& vRealY = complexPtr[2];
  T& vRealZ = complexPtr[4];

  T& vImagX = complexPtr[1];
  T& vImagY = complexPtr[3];
  T& vImagZ = complexPtr[5];

  G00 = sqrt(L00);
  G10 = L10 / G00;
  G20 = L20 / G00;

  G11 = L11 - G10 * G10;
  G21 = L21 - G20 * G10;
  G11 = sqrt(G11);
  G21 = G21 / G11;

  G22 = L22 - (G20*G20 + G21*G21);
  G22 = sqrt(G22);

  // back-solve Gy=b to get a temporary vector y
  // back-solve G'x=y to get answer in x
  //
  // Note: these are in matlab notation 1:3
  // [ G(1,1)   0      0    ]   [ y(1) ] = b(1)
  // [ G(2,1) G(2,2)   0    ] * [ y(2) ] = b(2)
  // [ G(3,1) G(3,2) G(3,3) ]   [ y(3) ] = b(3)
  //
  // [ G(1,1) G(2,1) G(3,1) ]   [ x(1) ] = y(1)
  // [   0    G(2,2) G(3,2) ] * [ x(2) ] = y(2)
  // [   0      0    G(3,3) ]   [ x(3) ] = y(3)
  y0 = bRealX / G00;
  y1 = (bRealY - G10*y0) / G11;
  y2 = (bRealZ - G20*y0 - G21*y1) / G22;

  vRealZ = y2 / G22;
  vRealY = (y1 - G21*vRealZ) / G11;
  vRealX = (y0 - G10*vRealY - G20*vRealZ) / G00;

  y0 = bImagX / G00;
  y1 = (bImagY - G10*y0) / G11;
  y2 = (bImagZ - G20*y0 - G21*y1) / G22;

  vImagZ = y2 / G22;
  vImagY = (y1 - G21*vImagZ) / G11;
  vImagX = (y0 - G10*vImagY - G20*vImagZ) / G00;
}

template<class T>
void
DiffOperT<T>::
OperatorMultiply(T* complexPtr,
		 T& L00,
		 T& L10, T& L11,
		 T& L20, T& L21, T& L22)
{
  T bRealX = complexPtr[0];
  T bRealY = complexPtr[2];
  T bRealZ = complexPtr[4];

  T bImagX = complexPtr[1];
  T bImagY = complexPtr[3];
  T bImagZ = complexPtr[5];

  T& vRealX = complexPtr[0];
  T& vRealY = complexPtr[2];
  T& vRealZ = complexPtr[4];

  T& vImagX = complexPtr[1];
  T& vImagY = complexPtr[3];
  T& vImagZ = complexPtr[5];

  vRealX = L00*bRealX + L10*bRealY + L20*bRealZ;
  vRealY = L10*bRealX + L11*bRealY + L21*bRealZ;
  vRealZ = L20*bRealX + L21*bRealY + L22*bRealZ;
  
  vImagX = L00*bImagX + L10*bImagY + L20*bImagZ;
  vImagY = L10*bImagX + L11*bImagY + L21*bImagZ;
  vImagZ = L20*bImagX + L21*bImagY + L22*bImagZ;	
}

template<class T>
void
DiffOperT<T>::
OperatorPowMultiply(unsigned int index, bool inverse)
{
  
  T* dataPtr = &(this->mFFTWVectorField->getDataPointer()->x);
  
  T bRealX = dataPtr[2*index+0];
  T bRealY = dataPtr[2*index+2];
  T bRealZ = dataPtr[2*index+4];
  
  T bImagX = dataPtr[2*index+1];
  T bImagY = dataPtr[2*index+3];
  T bImagZ = dataPtr[2*index+5];

  T& vRealX = dataPtr[2*index+0];
  T& vRealY = dataPtr[2*index+2];
  T& vRealZ = dataPtr[2*index+4];

  T& vImagX = dataPtr[2*index+1];
  T& vImagY = dataPtr[2*index+3];
  T& vImagZ = dataPtr[2*index+5];

  T L00;
  T L10, L11;
  T L20, L21, L22;

  short invFlag = 1;
  if(inverse) invFlag = -1;

  this->PowL(L00, 
	     L10, L11,
	     L20, L21, L22,
	     index, invFlag);

  vRealX = L00*bRealX + L10*bRealY + L20*bRealZ;
  vRealY = L10*bRealX + L11*bRealY + L21*bRealZ;
  vRealZ = L20*bRealX + L21*bRealY + L22*bRealZ;
  
  vImagX = L00*bImagX + L10*bImagY + L20*bImagZ;
  vImagY = L10*bImagX + L11*bImagY + L21*bImagZ;
  vImagZ = L20*bImagX + L21*bImagY + L22*bImagZ;	
}

/**
 * If we have precomputed the eigenvectors and eigenvalues of the
 * function, this function can quickly compute powers of L, including
 * negative (inverse) powers.
 *
 * Given L = Q*V*Q^-1, where V has the eigenvalues and Q is the matrix
 * containing the corresponding eigenvectors in its columns (an
 * orthogonal matrix if L is SPD), then L^p = Q*V^p*Q^-1
 *
 * invFlag should be either 1 or -1, and denotes whether L^epsilon or
 * L^-epsilon
 */

template<class T>
void  
DiffOperT<T>::
PowL(T& L00,
     T& L10, T& L11, 
     T& L20, T& L21, T& L22, 
     unsigned int index, short invFlag)
{
  T Q00 = mEigenLookupTable.eigenVectors[3*index];
  T Q01 = mEigenLookupTable.eigenVectors[3*index+1];
  T Q02 = mEigenLookupTable.eigenVectors[3*index+2];
  T Q10 = mEigenLookupTable.eigenVectors[3*index+3];
  T Q11 = mEigenLookupTable.eigenVectors[3*index+4];
  T Q12 = mEigenLookupTable.eigenVectors[3*index+5];
  T Q20 = mEigenLookupTable.eigenVectors[3*index+6];
  T Q21 = mEigenLookupTable.eigenVectors[3*index+7];
  T Q22 = mEigenLookupTable.eigenVectors[3*index+8];

  T e0 = mEigenLookupTable.eigenValues[index];
  T e1 = mEigenLookupTable.eigenValues[index+1];
  T e2 = mEigenLookupTable.eigenValues[index+2];

  if(invFlag == -1)
    {
      e0 = pow(e0,static_cast<T>(-1.0)*mLPow);
      e1 = pow(e1,static_cast<T>(-1.0)*mLPow);
      e2 = pow(e2,static_cast<T>(-1.0)*mLPow);
    }
  else
    {
      e0 = pow(e0,mLPow);
      e1 = pow(e1,mLPow);
      e2 = pow(e2,mLPow);
    } 
  
  L00 = pow(Q00,2)* e0 + pow(Q01,2)* e1 + pow(Q02,2)* e2;
  L11 = pow(Q10,2)* e0 + pow(Q11,2)* e1 + pow(Q12,2)* e2;
  L22 = pow(Q20,2)* e0 + pow(Q21,2)* e1 + pow(Q22,2)* e2;		
  
  L10 = Q00*Q10*e0 + Q01*Q11* e1 + Q02*Q12* e2; 
  L20 = Q00*Q20*e0 + Q01*Q21* e1 + Q02*Q22* e2; 
  L21 = Q10*Q20*e0 + Q11*Q21* e1 + Q12*Q22* e2;
}

template class DiffOperT<float>;
template class DiffOperT<double>;

