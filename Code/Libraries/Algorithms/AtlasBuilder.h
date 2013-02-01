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

#ifndef AtlasBuilder_h
#define AtlasBuilder_h

#include <vector>
#include <Array3D.h>
#include <MultiscaleManager.h>
#include <DiffOper.h>
#include <fftw3.h>
#include <Timer.h>
#include <ostream>
#include <map>

#ifdef MPI_ENABLED
#include <mpi.h>
#endif // MPI_ENABLED

/**
 * Base class for methods computing 'mean' of a set of values
 */
template <class VoxelType>
class MeanComputationStrategyBase
{
 public:
  virtual ~MeanComputationStrategyBase(){};
  virtual VoxelType ComputeMean(unsigned int numValues, VoxelType* values)=0;
};

/**
 * Computes the mean as a weighted average
 */
template <class VoxelType>
class ArithmeticMeanComputationStrategy:
public MeanComputationStrategyBase<VoxelType>
{
 public:

  //~ArithmeticMeanComputationStrategy(){}

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

  virtual VoxelType ComputeMean(unsigned int numValues, VoxelType* values) 
    {
      double mean = 0;
      for (unsigned int i = 0; i < numValues; ++i)
      {
        mean += values[i] * _weights[i];
      }
      return static_cast<VoxelType>(mean);
    } 

 private:
  std::vector<double> _weights;
};

class AtlasBuilderParam : public CompoundParam {
public:
  AtlasBuilderParam(const std::string& name = "AtlasBuilder", 
		    const std::string& desc = "Settings for a single scale level of greedy atlas building", 
		    ParamLevel level = PARAM_COMMON)
    : CompoundParam(name, desc, level)
  {
    this->
      AddChild(ValueParam<unsigned int>("NIterations",
					"Number of iterations to run",
					PARAM_COMMON,
					0));
    this->AddChild(DiffOperParam("DiffOper"));
    this->
      AddChild(ValueParam<Real>("MaxPert",
				"Scale factor on the maximum velocity in a given "
				"deformation for computing delta",
				PARAM_COMMON,
				0.1));
    this->
      AddChild(ValueParam<bool>("UpdateAfterSubIteration",
				"Update the average after each deformation is updated?",
				PARAM_RARE,
				false));
    this->
      AddChild(ValueParam<bool>("DeltaSelectionUseMean",
				"Use the mean maximum velocity across all images" 
				" to compute delta step for all deformations. "
				"If this is false the maximum velocity in each "
				"vector field will be used for the delta "
				"computation for that deformation",
				PARAM_RARE,
				false));
    
    this->
      AddChild(ValueParam<unsigned int>("OutputMeanEveryNIterations",
					"Write out the mean image evern N iterations "
					"at this scale level (0 means do not write)",
					PARAM_DEBUG, 0));

    this->
      AddChild(ValueParam<bool>("OutputInitialScaleLevelMean",
				"Write out the mean image at the beginning "
				"of the scale level",
				PARAM_DEBUG, 0));
    this->
      AddChild(ValueParam<bool>("OutputFinalScaleLevelMean",
				"Write out the mean image at the end "
				"of the scale level",
				PARAM_DEBUG, 0));
  }
  ValueParamAccessorMacro(unsigned int, NIterations)
  ParamAccessorMacro(DiffOperParam, DiffOper)
  ValueParamAccessorMacro(Real, MaxPert)
  ValueParamAccessorMacro(bool, UpdateAfterSubIteration)
  ValueParamAccessorMacro(bool, DeltaSelectionUseMean)
  ValueParamAccessorMacro(unsigned int, OutputMeanEveryNIterations);
  ValueParamAccessorMacro(bool, OutputInitialScaleLevelMean);
  ValueParamAccessorMacro(bool, OutputFinalScaleLevelMean);

  CopyFunctionMacro(AtlasBuilderParam)

};

/**
 * Class for creating an atlas from a set of images.  Runs multithreaded.
 */
class AtlasBuilder
{
 public:
  typedef float                                  VoxelType;
  typedef float                                  CoordinateType;
  typedef Array3D<VoxelType>                     ImageType;
  typedef Array3D<Vector3D<CoordinateType> >     VectorFieldType;
  typedef MeanComputationStrategyBase<VoxelType> MeanComputationStrategyType;
  typedef DiffOper::FFTWVectorField              FFTWVectorField;
  struct IterationData
  {
    unsigned int        IterationNumber;
    unsigned int        ImageNumber;
    double              IterationEllapsedTimeInSeconds;
    double              TotalEllapsedTimeInSeconds;
    unsigned int        ProcessingThreadID;
    double              MeanSquaredError;
    double              MaxL2Displacement;
    double              RootMeanSquaredError;
    double              Delta;
  };

  //
  // constructors/destructors
  //
  AtlasBuilder();
  ~AtlasBuilder();

  //
  // fftw interface
  //
  void           SetFFTWNumberOfThreads(unsigned int numThreads);
  unsigned int   GetFFTWNumberOfThreads() const;
  
  void           SetFFTWMeasureOn();
  void           SetFFTWMeasureOff();
  void           SetFFTWMeasure(bool b);
  bool           GetFFTWMeasure() const;

  //
  // ouput options
  //
  void           SetLogOutputStream(std::ostream& ostream);

  //
  // algorithm options
  //
  void           SetNumberOfThreads(unsigned int numThreads);
  unsigned int   GetNumberOfThreads() const;

  void           SetParams(const AtlasBuilderParam &param);
  AtlasBuilderParam GetParams();
  void           GetParams(AtlasBuilderParam &param);

  void           SetUpdateAverageEverySubIterationOn();
  void           SetUpdateAverageEverySubIterationOff();
  void           SetUpdateAverageEverySubIteration(bool b);
  bool           GetUpdateAverageEverySubIteration() const;

  void           SetComputeInverseDeformationsOn();
  void           SetComputeInverseDeformationsOff();
  void           SetComputeInverseDeformations(bool b);
  bool           GetComputeInverseDeformations() const;

  void           SetDiffOperParams(const DiffOperParam& params);
  DiffOperParam& GetDiffOperParams();

  void           SetMeanComputationStrategy(MeanComputationStrategyType* s);
  MeanComputationStrategyType* GetMeanComputationStrategy() const;

  enum DeltaSelectionType { DELTA_USE_MEAN,
                            DELTA_USE_INDIVIDUAL
  };
  void           SetDeltaSelectionToIndividual()
  { this->_DeltaSelectionMethod = DELTA_USE_INDIVIDUAL; }
  void           SetDeltaSelectionToMean()
  { this->_DeltaSelectionMethod = DELTA_USE_MEAN; }
  DeltaSelectionType GetDeltaSelectionType() const
  { return this->_DeltaSelectionMethod; }

  void           SetNumIterations(unsigned int niters);
  unsigned int   GetNumIterations();
  void           SetMaxPerturbation(double maxPert);
  double         GetMaxPerturbation();

  void           SetScaleLevel(unsigned int scaleLevel)
  { this->_ScaleLevel = scaleLevel; }

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

  void             ProjectIncomp(CoordinateType* complexPtr, unsigned int x, unsigned int y, unsigned int z);

  void             InitializeScratchMemory();
  void             DeleteScratchMemory();
  void             CheckInputs();
  void             UpdateDiffOperParams();
  
  void             UpdateAverageImage();
  void             ThreadedUpdateAverageImage();
  void             SumAcrossNodes();
  void             UpdateError();
  void             UpdateGradient         (unsigned int imageIndex, 
                                           unsigned int threadIndex);
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

  static pthread_t GetThreadID();
  long GetThreadIndex();
  
  AtlasBuilder(const AtlasBuilder& rhs);             // not implemented
  AtlasBuilder& operator =(const AtlasBuilder& rhs); // not implemented

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
  // differential operators 
  std::vector<DiffOper*>           _DiffOperPointers;
  std::vector<FFTWVectorField*>    _ScratchVectorFieldPointers;

  // defferential operator params
  bool                             _FFTWMeasure;
  unsigned int                     _FFTWNumberOfThreads;

  // 
  // scratch data managed by this class
  std::vector<ImageType*>          _DeformedImagePointers;
  // only used if compiled with MPI support
  ImageType*                       _MPIAverageImagePointer;


  //
  // output parameters
  bool                             _OutputInitialScaleLevelMean;
  bool                             _OutputFinalScaleLevelMean;
  unsigned int                     _ScaleLevel;
  
  //
  // algorithm parameters
  bool                             _UpdateAfterEverySubIteration;
  bool                             _ComputeInverseDeformations;  
  MeanComputationStrategyType*     _MeanComputationStrategy;
  DiffOperParam                    _DiffOperParams;
  unsigned int                     _NumIterations;
  double                           _MaxPerturbation;
  
  //
  // processing current state
  std::vector<double>              _Delta;
  double                           _MeanSquaredError;
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
};

#endif
