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

#ifndef DiffeoImageGrowth_h
#define DiffeoImageGrowth_h

#include <vector>
#include <Array3D.h>
#include <FluidWarpParameters.h>
#include <LDMMParameters.h>
#include <fftw3.h>
#include <Timer.h>
#include <ostream>
#include <map>

class DiffeoImageGrowth
{
 public:
  typedef float                                  VoxelType;
  typedef float                                  CoordinateType;
  typedef Array3D<VoxelType>                     ImageType;
  typedef Array3D<CoordinateType>                RealImageType;
  typedef Array3D<Vector3D<CoordinateType> >     VectorFieldType;

  struct IterationData
  {
    unsigned int        IterationNumber;
    double              Epsilon;
    double              IterationEllapsedTimeInSeconds;
    double              TotalEllapsedTimeInSeconds;
    double              RootMeanSquaredError;
    double              SumSquaredError;
    double              MaxL2Displacement;
  };

  //
  // constructors/destructors
  DiffeoImageGrowth();
  ~DiffeoImageGrowth();

  //
  // fftw interface
  void         SetFFTWNumberOfThreads(unsigned int numThreads);
  unsigned int GetFFTWNumberOfThreads() const;
  void         SetFFTWMeasureOn();
  void         SetFFTWMeasureOff();
  void         SetFFTWMeasure(bool b);
  bool         GetFFTWMeasure() const;

  //
  // ouput options
  void         SetLogOutputStream(std::ostream& ostream);
  void         SetVerbose(bool v);
  bool         GetVerbose() const;

  //
  // algorithm options
  void         SetNumberOfThreads(unsigned int numThreads);
  unsigned int GetNumberOfThreads() const;

  void         SetFluidWarpParameters(const FluidWarpParameters& fluidParams);
  FluidWarpParameters& GetFluidWarpParameters();
  void         SetLDMMParameters(const LDMMParameters& ldmmParams);
  LDMMParameters& GetLDMMParameters();

  //
  // set inputs
  void             SetNumberOfInputImages(unsigned int n);
  unsigned int     GetNumberOfInputImages() const;

  void             SetNthInputImage(unsigned int n, ImageType* imagePointer);
  ImageType*       GetNthInputImage(unsigned int n) const;

  void             SetNthVelocityField(unsigned int n, 
                                          VectorFieldType* fieldPointer);
  VectorFieldType* GetNthVelocityField(unsigned int n) const;

  //
  // run it
  void             Run();

  IterationData    GetIterationData(unsigned int iteration) const;

 private:
  //
  // the actual algorithm
  void             RunAlgorithm();
  static void*     ThreadedUpdateImages(void* arg);
  static void*     ThreadedPullbackImages(void* arg);

  //
  // clean up and tear down
  void             InitializeScratchMemory();
  void             DeleteScratchMemory();
  void             CheckInputs();
  void             InitializeFFTWPlans();
  void             DeleteFFTWPlans();
  void             InitializeOperatorLookupTable();

  unsigned int     GetJobImageIndex();

  //
  // algorithm steps
  void             ComputeJ0t();
  void             ComputeJTjTi(int j);
  void             ComputeDeformedImageGradient(unsigned int imageIndex,
                                                unsigned int threadIndex);
  void             ComputeBodyForce(unsigned int imageIndex,
                                    unsigned int threadIndex);
  void             ComputeGreensFunction(unsigned int imageIndex,
                                         unsigned int threadIndex);
  void             UpdateVelocity(unsigned int imageIndex,
                                  unsigned int threadIndex);
  void             ComputeError();
  void             ReportProgress();
  void             LogIterationData();

  static void      InverseOperatorMultiply(CoordinateType* complexPtr,
                                           float& L00,
                                           float& L10, float& L11,
                                           float& L20, float& L21, float& L22);

  //
  // look up table to for Linv computation
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

  DiffeoImageGrowth(const DiffeoImageGrowth& rhs);             // not implemented
  DiffeoImageGrowth& operator =(const DiffeoImageGrowth& rhs); // not implemented

  //
  // how many images will be processed
  unsigned int                     _NumberOfImages;

  //
  // input data managed by the user
  std::vector<ImageType*>          _ImagePointers;
  std::vector<VectorFieldType*>    _VelocityFieldPointers;

  // 
  // scratch data managed by this class
  std::vector<ImageType*>                   _J0t;
  std::vector<VectorFieldType*>             _ScratchVectorFieldPointers;
  std::vector<std::vector<RealImageType*> > _dPhiTiTj;
  std::vector<std::vector<ImageType*> >     _JTjTi;

  Vector3D<unsigned int>           _ImageSize;

  //
  // FFT parameters
  bool                             _FFTWMeasure;
  unsigned int                     _FFTWNumberOfThreads;
  std::vector<fftwf_plan>          _FFTWForwardPlans;
  std::vector<fftwf_plan>          _FFTWBackwardPlans;

  //
  // algorithm parameters
  FluidWarpParameters              _FluidWarpParameters;
  LDMMParameters                   _LDMMParameters;
  LUT                              _OperatorLookupTable;
  
  //
  // processing current state
  std::vector<double>              _Epsilon;
  double                           _RootMeanSquaredError;
  double                           _SumSquaredError;
  unsigned int                     _Iteration;
  std::vector<IterationData>       _IterationDataLog;
  Timer                            _TotalTimer;
  std::vector<double>              _MaxL2Displacements;
  bool                             _WriteDebugImages;
  bool                             _Verbose;

  //
  // used for multithreading
  unsigned int                     _NumberOfThreads;
  unsigned int                     _NextImageToProcess;
  pthread_mutex_t                  _NextImageToProcessMutex;

  //
  // output settings
  std::ostream&                    _IterationDataOutputStream;
};

#endif
