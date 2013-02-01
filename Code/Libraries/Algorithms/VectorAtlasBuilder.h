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


// Vector version of AtlasBuilder
// With choice of KL metric and normalized geometric mean strategy
//
// Vector interpretation:
// euclidean -> arith mean, body force is euclidean diff
// model centric pdf w/ KL -> normalized geom mean, body force is ratio
// population centric pdf w/ KL -> arith mean, body force is log ratio


// prastawa@sci.utah.edu 8/2008

#ifndef VectorAtlasBuilder_h
#define VectorAtlasBuilder_h

//#include "itkVariableLengthVector.h"
#include "vnl/vnl_vector.h"

#include <Array3D.h>
#include <Image.h>

#include <FluidWarpParameters.h>
#include <Timer.h>

#include <fftw3.h>

#include <ostream>
#include <map>
#include <vector>

template <class VoxelType>
class MeanComputationStrategyBase
{
public:
  virtual ~MeanComputationStrategyBase(){}
  virtual VoxelType ComputeMean(unsigned int numValues, VoxelType* values)=0;
  void SetNumberOfComponents(unsigned int n)
  {
    this->_numComponents = n;
  }
protected:
  unsigned int _numComponents;
};

template <class VoxelType>
class WeightedMeanComputationStrategyBase:
  public MeanComputationStrategyBase<VoxelType>
{
public:
  void SetNumberOfElements(unsigned int n)
    {
      this->_weights.resize(n);
      this->SetWeightsToEqual();
    }

  unsigned int GetNumberOfElements() const
    {
      return this->_weights.size();
    }

  void SetNthWeight(unsigned int n, double weight)
    {
      this->_weights[n] = weight;
    }

  double GetNthWeight(unsigned int n) const
    {
      return this->_weights[n];
    }
  void SetWeightsToEqual()
    {
      std::fill(this->_weights.begin(), this->_weights.end(),
                1.0/this->_weights.size());
    }

protected:
  std::vector<double> _weights;
};

template <class VoxelType>
class ArithmeticMeanComputationStrategy:
  public WeightedMeanComputationStrategyBase<VoxelType>
{
public:
  virtual VoxelType ComputeMean(unsigned int numValues, VoxelType* values) 
  {
//std::cout << "PP ComputeMean comp = " << this->_numComponents << " w = " << this->_weights[0] << ", " << this->_weights[1] << std::endl;
    VoxelType mean(this->_numComponents, 0.0);
    if (numValues == 0)
      return mean;
    for (unsigned int i = 0; i < numValues; ++i)
    {
      mean += values[i] * this->_weights[i];
    }
//std::cout << "PP ComputeMean " << numValues << " values = " << values[0] << ", " << values[1] << " mean = " << mean << std::endl;
    return mean;
  } 

  virtual ~ArithmeticMeanComputationStrategy() { }
};

template <class VoxelType>
class NormalizedGeometricMeanComputationStrategy:
  public WeightedMeanComputationStrategyBase<VoxelType>
{
public:
  virtual VoxelType ComputeMean(unsigned int numValues, VoxelType* values) 
  {
    VoxelType mean(this->_numComponents);
    if (numValues == 0)
    {
      mean.fill(0.0);
      return mean;
    }
    mean.fill(1.0);
    for (unsigned int i = 0; i < numValues; ++i)
    {
      double w = this->_weights[i];
      VoxelType v = values[i];

      for (unsigned int k = 0; k < v.size(); k++)
      {
        mean[k] *= pow(v[k], w);
      }
    }

    double sumMean = 1e-20;
    for (unsigned int k = 0; k < mean.size(); k++)
      sumMean += mean[k];
    mean /= sumMean;

    return mean;
  } 

  virtual ~NormalizedGeometricMeanComputationStrategy() { }
};

class VectorAtlasBuilder
{
 public:
  typedef float                                  ScalarType;
  typedef vnl_vector<ScalarType>	         VoxelType;
  typedef float                                  CoordinateType;
  typedef Image<VoxelType>                       ImageType;
  typedef Vector3D<CoordinateType>               VectorType;
  typedef Array3D<Vector3D<CoordinateType> >     VectorFieldType;
  typedef WeightedMeanComputationStrategyBase<VoxelType>
    MeanComputationStrategyType;

  struct IterationData
  {
    unsigned int        IterationNumber;
    unsigned int        ImageNumber;
    double              IterationEllapsedTimeInSeconds;
    double              TotalEllapsedTimeInSeconds;
    unsigned int        ProcessingThreadID;
    double              Error;
    double              MaxL2Displacement;
    double              Delta;
  };

  //
  // constructors/destructors
  //
  VectorAtlasBuilder();
  ~VectorAtlasBuilder();

  //
  // fftw interface
  //
  void         SetFFTWNumberOfThreads(unsigned int numThreads);
  unsigned int GetFFTWNumberOfThreads() const;
  
  void         SetFFTWMeasureOn();
  void         SetFFTWMeasureOff();
  void         SetFFTWMeasure(bool b);
  bool         GetFFTWMeasure() const;

  //
  // ouput options
  //
  void         SetLogOutputStream(std::ostream& ostream);

  //
  // algorithm options
  //
  void         SetNumberOfThreads(unsigned int numThreads);
  unsigned int GetNumberOfThreads() const;

  void         SetUpdateAverageEverySubIterationOn();
  void         SetUpdateAverageEverySubIterationOff();
  void         SetUpdateAverageEverySubIteration(bool b);
  bool         GetUpdateAverageEverySubIteration() const;

  void         SetComputeInverseDeformationsOn();
  void         SetComputeInverseDeformationsOff();
  void         SetComputeInverseDeformations(bool b);
  bool         GetComputeInverseDeformations() const;

  void ZeroOrderInverseOn() { m_ZeroOrderInverse = true; }
  void ZeroOrderInverseOff() { m_ZeroOrderInverse = false; }

  void         SetFluidWarpParameters(const FluidWarpParameters& fluidParams);
  FluidWarpParameters& GetFluidWarpParameters();

  void         SetMeanComputationStrategy(MeanComputationStrategyType* s);
  MeanComputationStrategyType* GetMeanComputationStrategy() const;

  enum VectorModeType { 
    ModelCentricPDF, PopulationCentricPDF, Euclidean};

  enum DeltaSelectionType { DELTA_USE_MEAN,
                            DELTA_USE_INDIVIDUAL
  };
  void               SetDeltaSelectionToIndividual()
  { this->_DeltaSelectionMethod = DELTA_USE_INDIVIDUAL; }
  void               SetDeltaSelectionToMean()
  { this->_DeltaSelectionMethod = DELTA_USE_MEAN; }
  DeltaSelectionType GetDeltaSelectionType() const
  { return this->_DeltaSelectionMethod; }

  void SetVectorMode(VectorModeType m)
  {
    _VectorMode = m;
// NOTE: if set previously, use number of input images here
    if (m == Euclidean || m == PopulationCentricPDF)
      this->_MeanComputationStrategy =
        new ArithmeticMeanComputationStrategy<VoxelType>();
    if (m == ModelCentricPDF)
      this->_MeanComputationStrategy =
        new NormalizedGeometricMeanComputationStrategy<VoxelType>();
  }

  void SetPDFEpsilon(double e) { this->_PDFEpsilon = e; }
  double GetPDFEpsilon() const { return this->_PDFEpsilon; }

  //
  // set inputs
  //
  void             SetNumberOfInputImages(unsigned int n);
  unsigned int     GetNumberOfInputImages() const;

  void             SetNthInputImage(unsigned int n, ImageType* imagePointer);
  ImageType*       GetNthInputImage(unsigned int n) const;

  ImageType*       GetNthDeformedImage(unsigned int n) const;

  void             SetNthDeformationField(unsigned int n, 
                                          VectorFieldType* fieldPointer);
  VectorFieldType* GetNthDeformationField(unsigned int n) const;

  void             SetNthDeformationFieldInverse(unsigned int n, 
                                                 VectorFieldType* 
                                                 fieldPointer);
  VectorFieldType* GetNthDeformationFieldInverse(unsigned int n) const;

  void             SetAverageImage(ImageType* imagePointer);
  ImageType*       GetAverageImage() const;

  //
  // run it
  //
  void             GenerateAverage();

  IterationData    GetIterationData(unsigned int iteration) const;

 private:
  void             RunAlgorithm();
  static void*     ThreadedUpdateImages(void* arg);
  static void*     ThreadedUpdateAverageImage(void* arg);

  void             InitializeScratchMemory();
  void             DeleteScratchMemory();
  void             CheckInputs();
  void             InitializeFFTWPlans();
  void             DeleteFFTWPlans();
  void             InitializeOperatorLookupTable();
  
  void             UpdateAverageImage();
  void             ThreadedUpdateAverageImage();
  void             UpdateError();
  void             UpdateBodyForce        (unsigned int imageIndex, 
                                           unsigned int threadIndex);
  void             UpdateVelocityField    (unsigned int imageIndex, 
                                           unsigned int threadIndex);
  void             UpdateDeformationFields(unsigned int imageIndex, 
                                           unsigned int threadIndex);
  void             UpdateDeformedImage    (unsigned int imageIndex, 
                                           unsigned int threadIndex);
  void             UpdateDelta();

  void             LogIterationData(unsigned int imageIndex,
                                    int threadID,
                                    double iterationEllapsedSeconds);

  void             LockMeanImage();
  void             UnlockMeanImage();
  unsigned int     GetJobImageIndex();

  static void      InverseOperatorMultiply(CoordinateType* complexPtr,
                                           float& L00,
                                           float& L10, float& L11,
                                           float& L20, float& L21, float& L22);

  static pthread_t GetThreadID();
  long GetThreadIndex();

  VectorFieldType* GetInverseMap(VectorFieldType* hField);
  
  //
  // look up table to for Linv computation
  //
  struct LUT
  {
    std::vector<CoordinateType> cosWX, cosWY, cosWZ;
    std::vector<CoordinateType> sinWX, sinWY, sinWZ;

    LUT()
      : cosWX(0), cosWY(0), cosWZ(0),
	sinWX(0), sinWY(0), sinWZ(0)
    {}

    LUT(unsigned int xSize, 
	unsigned int ySize, 
	unsigned int zSize)
      : cosWX(xSize / 2 + 1), cosWY(ySize), cosWZ(zSize),
	sinWX(xSize / 2 + 1), sinWY(ySize), sinWZ(zSize)
    {}
  };

  VectorAtlasBuilder(const VectorAtlasBuilder& rhs);             // not implemented
  VectorAtlasBuilder& operator =(const VectorAtlasBuilder& rhs); // not implemented

  //
  // how many images will be processed
  unsigned int                     _NumberOfImages;

  //
  // input data managed by the user
  std::vector<ImageType*>          _ImagePointers;
  ImageType*                       _AverageImagePointer;

  std::vector<VectorFieldType*>    _DeformationFieldPointers;
  std::vector<VectorFieldType*>    _DeformationFieldInversePointers;

  // 
  // scratch data managed by this class
  std::vector<VectorFieldType*>    _ScratchVectorFieldPointers;
  std::vector<ImageType*>          _DeformedImagePointers;

  //
  // FFT parameters
  bool                             _FFTWMeasure;
  unsigned int                     _FFTWNumberOfThreads;
  std::vector<fftwf_plan>          _FFTWForwardPlans;
  std::vector<fftwf_plan>          _FFTWBackwardPlans;

  //
  // algorithm parameters
  bool                             _UpdateAfterEverySubIteration;
  bool                             _ComputeInverseDeformations;  
  MeanComputationStrategyType*     _MeanComputationStrategy;
  FluidWarpParameters              _FluidWarpParameters;
  LUT                              _OperatorLookupTable;
  
  //
  // processing current state
  std::vector<double>              _Delta;
  double                           _Error;
  unsigned int                     _Iteration;
  std::vector<IterationData>       _IterationDataLog;
  Timer                            _TotalTimer;
  std::vector<double>              _MaxL2Displacements;
  std::vector<double>              _MaxVelocityL2Displacements;

  //
  // processing options
  DeltaSelectionType               _DeltaSelectionMethod;

  //
  // used for multithreading
  unsigned int                     _NumberOfThreads;
  unsigned int                     _NextImageToProcess;
  pthread_mutex_t                  _NextImageToProcessMutex;
  pthread_mutex_t                  _AverageImageMutex;

  //
  // output settings
  std::ostream&                    _IterationDataOutputStream;

  // Vector interpretation
  VectorModeType _VectorMode;

  double _PDFEpsilon;

  bool m_ZeroOrderInverse;
};

#endif
