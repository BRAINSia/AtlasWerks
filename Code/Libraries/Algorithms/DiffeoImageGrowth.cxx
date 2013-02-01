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

#include <DiffeoImageGrowth.h>
#include <strstream>
#include <HField3DUtils.h>
#include <assert.h>
#include <Timer.h>
#include <iomanip>
#include <stdio.h>

struct ThreadInfo
{
  DiffeoImageGrowth* algorithmClass;
  unsigned int  threadIndex;
};

DiffeoImageGrowth::
DiffeoImageGrowth()
  : 
  _IterationDataOutputStream(std::cerr)
{
  this->_NumberOfImages               = 0;
  this->_NumberOfThreads              = 1;
  this->_FFTWNumberOfThreads          = 1;
  this->_FFTWMeasure                  = true;
  this->_WriteDebugImages             = true;
  this->_Verbose                      = false;
  pthread_mutex_init(&this->_NextImageToProcessMutex, NULL);
}

DiffeoImageGrowth::
~DiffeoImageGrowth()
{
}

void 
DiffeoImageGrowth::
SetFFTWMeasure(bool b)
{
  this->_FFTWMeasure = b;
}

void 
DiffeoImageGrowth::
SetFFTWMeasureOn()
{
  this->SetFFTWMeasure(true);
}

void 
DiffeoImageGrowth::
SetFFTWMeasureOff()
{
  this->SetFFTWMeasure(false);
}

bool 
DiffeoImageGrowth::
GetFFTWMeasure() const
{
  return this->_FFTWMeasure;
}

void
DiffeoImageGrowth::
SetNumberOfThreads(unsigned int numThreads)
{
  this->_NumberOfThreads = numThreads;
}

unsigned int 
DiffeoImageGrowth::
GetNumberOfThreads() const
{
  return this->_NumberOfThreads;
}

void
DiffeoImageGrowth::
SetVerbose(bool v)
{
  this->_Verbose = v;
}

bool
DiffeoImageGrowth::
GetVerbose() const
{
  return this->_Verbose;
}

void
DiffeoImageGrowth::
SetFFTWNumberOfThreads(unsigned int numThreads)
{
  this->_FFTWNumberOfThreads = numThreads;
}

unsigned int 
DiffeoImageGrowth::
GetFFTWNumberOfThreads() const
{
  return this->_FFTWNumberOfThreads;
}

void
DiffeoImageGrowth::
SetFluidWarpParameters(const FluidWarpParameters& fluidParams)
{
  this->_FluidWarpParameters = fluidParams;
}

FluidWarpParameters&
DiffeoImageGrowth::
GetFluidWarpParameters()
{
  return this->_FluidWarpParameters;
}

void
DiffeoImageGrowth::
SetLDMMParameters(const LDMMParameters& ldmmParams)
{
  this->_LDMMParameters = ldmmParams;
}

LDMMParameters&
DiffeoImageGrowth::
GetLDMMParameters()
{
  return this->_LDMMParameters;
}

void 
DiffeoImageGrowth::
SetNumberOfInputImages(unsigned int n)
{
  this->_NumberOfImages = n;

  this->_ImagePointers.resize(n);
  this->_VelocityFieldPointers.resize(n-1);
  this->_MaxL2Displacements.resize(n-1);
}

unsigned int
DiffeoImageGrowth::
GetNumberOfInputImages() const
{
  return this->_NumberOfImages;
}

void
DiffeoImageGrowth::
SetNthInputImage(unsigned int n, ImageType* imagePointer)
{
  this->_ImagePointers[n] = imagePointer;
}

DiffeoImageGrowth::ImageType*
DiffeoImageGrowth::
GetNthInputImage(unsigned int n) const
{
  return this->_ImagePointers[n];
}

void
DiffeoImageGrowth::
SetNthVelocityField(unsigned int n, VectorFieldType* fieldPointer)
{
  this->_VelocityFieldPointers[n] = fieldPointer;
}

DiffeoImageGrowth::VectorFieldType*
DiffeoImageGrowth::
GetNthVelocityField(unsigned int n) const
{
  return this->_VelocityFieldPointers[n];
}

DiffeoImageGrowth::IterationData
DiffeoImageGrowth::
GetIterationData(unsigned int iteration) const
{
  return this->_IterationDataLog[iteration];
}

// ^^^ setting and getting parameters
// ----------------------------------------------------------------------------
// vvv running the algorithm

void
DiffeoImageGrowth::
Run()
{
  this->_TotalTimer.restart();

  //
  // check that input data is consistent: an exception will be thrown
  // if there is a problem
  std::cerr << "Checking inputs...";
  this->CheckInputs();
  std::cerr << "DONE" << std::endl;  

  //
  // initialize memory: order is important
  std::cerr << "Initializing memory...";
  if (this->_Verbose) { std::cerr << "Scratch memory..."; }
  this->InitializeScratchMemory();
  if (this->_Verbose) { std::cerr << "FFTW Plans..."; }
  this->InitializeFFTWPlans();
  if (this->_Verbose) { std::cerr << "LUT..."; }
  this->InitializeOperatorLookupTable();
  std::cerr << "DONE" << std::endl;

  //
  // start algorithm
  //
  this->RunAlgorithm();

  if (this->_WriteDebugImages)
  {
    //
    // deform template image (at time 0) to each time point
    std::cerr << "Generating and saving deformed images...";
    this->ComputeJ0t();
    for (unsigned int i = 0; i < this->_NumberOfImages; ++i)
    {
      std::stringstream ss;
      ss << "debug/FinalDeformedImage_" << i;
      Array3DIO::writeMETAVolume(*this->_J0t[i],ss.str().c_str());      
    }
  }

  //
  // clean up memory: order is important
  //
  std::cerr << "Cleaning up temporary memory...";
  this->DeleteFFTWPlans();
  this->DeleteScratchMemory();
  std::cerr << "DONE" << std::endl;  
}

unsigned int 
DiffeoImageGrowth::
GetJobImageIndex()
{
  pthread_mutex_lock(&this->_NextImageToProcessMutex);
  unsigned int imageIndex = this->_NextImageToProcess++;
  pthread_mutex_unlock(&this->_NextImageToProcessMutex);
  return imageIndex;
}

void
DiffeoImageGrowth::
RunAlgorithm()
{
  Timer localTimer;
  Timer iterationTimer;
  Timer globalTimer;
  globalTimer.restart();
  this->_Iteration = 0;
  double intitialRMSE;
  double lastRMSE = 0;

  //
  // deform template image (at time 0) to each time point
  if (this->_Verbose) { std::cerr << "Pushing forward template image..."; }
  localTimer.restart();
  this->ComputeJ0t();
  if (this->_Verbose)
  {
    std::cerr << "DONE, " << localTimer.getSeconds() << " (sec) " << std::endl;
  }

  //
  // compute and report initial error
  std::cerr << "Computing initial error...";
  localTimer.restart();
  this->ComputeError();
  std::cerr << "DONE, " << localTimer.getSeconds() << " (sec) " 
            << ", SSE="  << this->_SumSquaredError
            << ", RMSE=" << this->_RootMeanSquaredError << std::endl;
  intitialRMSE = this->_RootMeanSquaredError;

  for (unsigned int iter = 1; iter <= this->_LDMMParameters.numIterations; ++iter)
  {
    if (this->_Verbose)
    {
      std::cerr << "Iteration " << iter << "..." << std::endl;
    }
    this->_Iteration = iter;
    //
    // start the timer
    iterationTimer.restart();

    //
    // pull back images to each time point from later time points
    // also compute the det Jacobian of each of these xforms
    if (this->_Verbose) 
      {
        std::cerr << "Pulling back images and computing det Jacobians...";
      }
    localTimer.restart();

    // create threads
    std::vector<pthread_t> threadsPullback(this->_NumberOfThreads);

    // reset job counter
    this->_NextImageToProcess = 0;

    // info passed to threads
    std::vector<ThreadInfo> threadInfoPullback(this->_NumberOfThreads);

    // create threads that will update images
    for (unsigned int threadIndex = 0; threadIndex < this->_NumberOfThreads; 
         ++threadIndex)
    {
      threadInfoPullback[threadIndex].algorithmClass = this;
      threadInfoPullback[threadIndex].threadIndex    = threadIndex;
  
      int rv = pthread_create(&threadsPullback[threadIndex], NULL,
                              &DiffeoImageGrowth::ThreadedPullbackImages, 
                              &threadInfoPullback[threadIndex]);
      if (rv != 0)
      {
        throw std::runtime_error("Error creating thread.");
      }
    }

    // join threads
    for (unsigned int threadIndex = 0; threadIndex < this->_NumberOfThreads; 
         ++threadIndex)
    {
      int rv = pthread_join(threadsPullback[threadIndex], NULL);
      if (rv != 0)
      {
        throw std::runtime_error("Error joining thread.");
      }
    }

    //this->ComputeJTjTi();


    if (this->_Verbose)
    {
      std::cerr << "DONE, " << localTimer.getSeconds() << " (sec) " << std::endl;
    }

    //
    // Threaded update of velocity
    if (this->_Verbose)
    {
      std::cerr << "Updating velocity...";
    }
    localTimer.restart();

    // create threads
    std::vector<pthread_t> threads(this->_NumberOfThreads);

    // reset job counter
    this->_NextImageToProcess = 0;

    // info passed to threads
    std::vector<ThreadInfo> threadInfo(this->_NumberOfThreads);

    // create threads that will update images
    for (unsigned int threadIndex = 0; threadIndex < this->_NumberOfThreads; 
         ++threadIndex)
    {
      threadInfo[threadIndex].algorithmClass = this;
      threadInfo[threadIndex].threadIndex    = threadIndex;
  
      int rv = pthread_create(&threads[threadIndex], NULL,
                              &DiffeoImageGrowth::ThreadedUpdateImages, 
                              &threadInfo[threadIndex]);
      if (rv != 0)
      {
        throw std::runtime_error("Error creating thread.");
      }
    }

    // join threads
    for (unsigned int threadIndex = 0; threadIndex < this->_NumberOfThreads; 
         ++threadIndex)
    {
      int rv = pthread_join(threads[threadIndex], NULL);
      if (rv != 0)
      {
        throw std::runtime_error("Error joining thread.");
      }
    }
    if (this->_Verbose)
    {
      std::cerr << "DONE, " << localTimer.getSeconds() << " (sec) " << std::endl;
    }
    //
    // deform template image (at time 0) to each time point
    if (this->_Verbose)
    {
      std::cerr << "Pushing forward template image...";
    }
    localTimer.restart();
    this->ComputeJ0t();
    if (this->_Verbose)
    {
      std::cerr << "DONE, " << localTimer.getSeconds() << " (sec) " << std::endl;
    }

    //
    // compute and report error
    if (this->_Verbose)
    {
      std::cerr << "Computing current error...";
    }
    localTimer.restart();
    this->ComputeError();
    if (this->_Verbose)
    {
      std::cerr << "DONE, " << localTimer.getSeconds() << " (sec) " 
                << ", SSE="  << this->_SumSquaredError
                << ", RMSE=" << this->_RootMeanSquaredError << std::endl;
      std::cerr << "Max L2 Norms: " << std::endl;
      for (unsigned int ti = 0; ti < this->_NumberOfImages-1; ++ti)
      {
        std::cerr << this->_MaxL2Displacements[ti] << ", ";
      }
      std::cerr << std::endl;
      
      std::cerr << "Iteration time " << iterationTimer.getSeconds() 
                << " (sec) " << std::endl;
    }

    float maxL2 = *std::max_element(this->_MaxL2Displacements.begin(),
                                    this->_MaxL2Displacements.end());

    std::cerr 
      << "[" << iter << "/" << this->_LDMMParameters.numIterations << "] "
      << "Time=" << iterationTimer.getSeconds() << "|" 
      << globalTimer.getSeconds()
      << " MaxL2=" << maxL2 << " RMSE=" << this->_RootMeanSquaredError
      << " (" << 100 * this->_RootMeanSquaredError / intitialRMSE << "%)" 
      << std::endl;
    
    if (iter > 1 && this->_RootMeanSquaredError > lastRMSE)
    {
      std::cerr << " <<--<<--<<-- RMSE Increased! --<<--<<--<<" << std::endl;
    }
    lastRMSE = this->_RootMeanSquaredError;
  }
}

void*
DiffeoImageGrowth::
ThreadedUpdateImages(void* arg)
{
  ThreadInfo* threadInfoPtr = static_cast<ThreadInfo*>(arg);
  DiffeoImageGrowth* ptr = threadInfoPtr->algorithmClass;
  unsigned int threadIndex = threadInfoPtr->threadIndex;

  // loop until all jobs are taken
  while (true)
  {
    //
    // get an image index to process
    //std::cerr << "{" << threadIndex << "} Looking for work..." << std::endl;
    unsigned int vFieldIndex = ptr->GetJobImageIndex();
    //std::cerr << "{" << threadIndex << "} Got assignment: " 
    //<< imageIndex << std::endl;    
    if (vFieldIndex >= ptr->_NumberOfImages-1)
    {
      // all the jobs are taken
      break;
    }

    //std::cerr << "{" << threadIndex << "Grad" << std::endl;
    ptr->ComputeDeformedImageGradient(vFieldIndex, threadIndex);
    //std::cerr << "{" << threadIndex << "Body" << std::endl;
    ptr->ComputeBodyForce(vFieldIndex, threadIndex);
    //std::cerr << "{" << threadIndex <<"Greens" << std::endl;
    ptr->ComputeGreensFunction(vFieldIndex, threadIndex);
    //std::cerr << "{" << threadIndex <<"Velocity" << std::endl;
    ptr->UpdateVelocity(vFieldIndex, threadIndex);
    //std::cerr << "{" << threadIndex << "Done" << std::endl;
  }

  return (void*)(0);
}

void*
DiffeoImageGrowth::
ThreadedPullbackImages(void* arg)
{
  ThreadInfo* threadInfoPtr = static_cast<ThreadInfo*>(arg);
  DiffeoImageGrowth* ptr = threadInfoPtr->algorithmClass;
  // unsigned int threadIndex = threadInfoPtr->threadIndex;

  // loop until all jobs are taken
  while (true)
  {
    //
    // get an image index to process
    //std::cerr << "{" << threadIndex << "} Looking for work..." << std::endl;
    unsigned int jIndex = ptr->GetJobImageIndex();
    //std::cerr << "{" << threadIndex << "} Got assignment: " 
    //<< imageIndex << std::endl;    
    if (jIndex >= ptr->_NumberOfImages)
    {
      // all the jobs are taken
      break;
    }

    //std::cerr << "{" << threadIndex << "Grad" << std::endl;
    ptr->ComputeJTjTi(jIndex);
  }

  return (void*)(0);
}

void
DiffeoImageGrowth::
CheckInputs()
{
  if (this->_NumberOfImages < 1)
  {
      std::strstream ss;
      ss << "No images loaded: " << this->_NumberOfImages;
      throw std::runtime_error(ss.str());    
  }

  if (this->_ImagePointers[0] == NULL)
  {
    std::strstream ss;
    ss << "Image 0 is null.";
    throw std::runtime_error(ss.str());
  }

  // use the size of the first image as the standard
  this->_ImageSize = this->_ImagePointers[0]->getSize();

  //
  // check the data for all the images
  //
  for (unsigned int i = 1; i < this->_NumberOfImages; ++i)
  {
    // check that the image is there
    if (this->_ImagePointers[i] == NULL)
    {
      std::strstream ss;
      ss << "Input image is null: " << i;
      throw std::runtime_error(ss.str());
    }

    // check that is is the right size
    if (this->_ImageSize != this->_ImagePointers[i]->getSize())
    {
      std::strstream ss;
      ss << "Incompatible image size for image " << i;
      throw std::runtime_error(ss.str());
    }
  }

  for (unsigned int i = 0; i < this->_NumberOfImages-1; ++i)
  {
    // check that the deformation field is there
    if (this->_VelocityFieldPointers[i] == NULL)
    {
      std::strstream ss;
      ss << "Velocity field is null: " << i;
      throw std::runtime_error(ss.str());
    }

    // check that the deformation field is the correct size
    if (this->_ImageSize != this->_VelocityFieldPointers[i]->getSize())
    {
      std::strstream ss;
      ss << "Incompatible velocity field size for image " << i;
      throw std::runtime_error(ss.str());
    }
  }
}

void
DiffeoImageGrowth::
InitializeScratchMemory()
{
  //
  // this function assumes that CheckInput has been called without error.
  //

  //
  // resize memory holders
  this->_J0t.resize(this->_NumberOfImages);

  this->_ScratchVectorFieldPointers.resize(this->_NumberOfThreads);
  this->_FFTWForwardPlans.resize(this->_NumberOfThreads);
  this->_FFTWBackwardPlans.resize(this->_NumberOfThreads);

  this->_J0t.resize(this->_NumberOfImages);
  for (unsigned int i = 0; i < this->_NumberOfImages; ++i)
  {
    this->_J0t[i] = new ImageType(this->_ImageSize);
  }

  this->_JTjTi.resize(this->_NumberOfImages);
  this->_dPhiTiTj.resize(this->_NumberOfImages);
  for (unsigned int j = 0; j < this->_NumberOfImages; ++j)
  {
    this->_JTjTi[j].resize(this->_NumberOfImages);
    this->_dPhiTiTj[j].resize(this->_NumberOfImages);

    for (unsigned int i = 0; i <= j; ++i)
    {
      this->_JTjTi[j][i]    = new ImageType(this->_ImageSize);
      this->_dPhiTiTj[i][j] = new RealImageType(this->_ImageSize);
    }
  }

  // this multipurpose array is used to hold gradient, body force, and
  // velocity arrays (in that order) this saves LOTS of memory
  //
  // VERY IMPORTANT: however, because it is used to hold the result of an fft
  // it needs to be padded just a little bit, see www.fftw.org
  // this wont affect its use as gradient and body force arrays
  // as long as logical size (image size) is used for this array
  // and access into this array is done **ONLY** via the (x, y, z) operator
  // and not via incremented pointers.  I REPEAT, dont access the vector
  // field via incremented pointers unless you *know* what you are doing.
  unsigned xSizeFFT = 2 * (this->_ImageSize.x / 2 + 1);

  for (unsigned int i = 0; i < this->_NumberOfThreads; ++i)
  {
    // allocate scratch vector field
    this->_ScratchVectorFieldPointers[i] = 
      new VectorFieldType(xSizeFFT, this->_ImageSize.y, this->_ImageSize.z);
  }

  for (unsigned int i = 0; i < this->_NumberOfImages; ++i)
  {
    // allocate deformed image
    this->_J0t[i] = new ImageType(this->_ImageSize);
  }
}

void
DiffeoImageGrowth::
DeleteScratchMemory()
{
  for (unsigned int i = 0; i < this->_NumberOfThreads; ++i)
  {
    if (this->_ScratchVectorFieldPointers[i])
    {
      delete this->_ScratchVectorFieldPointers[i];
      this->_ScratchVectorFieldPointers[i] = NULL;
    }
  }

  for (unsigned int i = 0; i < this->_NumberOfImages; ++i)
  {
    if (this->_J0t[i])
    {
      delete this->_J0t[i];
      this->_J0t[i] = NULL;
    }
  }

  for (unsigned int j = 0; j < this->_NumberOfImages; ++j)
  {
    for (unsigned int i = 0; i <= j; ++i)
    {
      delete this->_JTjTi[j][i];
      this->_JTjTi[j][i] = NULL;
      delete this->_dPhiTiTj[i][j];
      this->_dPhiTiTj[i][j] = NULL;
    }
  }
}

void
DiffeoImageGrowth::
InitializeFFTWPlans()
{
  for (unsigned int i = 0; i < this->_NumberOfThreads; ++i)
  {
    int rank = 3;
    int logicalSizeParam[3];
    logicalSizeParam[0] = this->_ImageSize.z;
    logicalSizeParam[1] = this->_ImageSize.y;
    logicalSizeParam[2] = this->_ImageSize.x;

    int howMany = 3;
    int stride  = 3;
    int dist    = 1;
    float *dataPtr = 
      (CoordinateType*) this->_ScratchVectorFieldPointers[i]->getDataPointer();
    
    fftwf_plan_with_nthreads(this->_FFTWNumberOfThreads);
    
    this->_FFTWForwardPlans[i] = 
      fftwf_plan_many_dft_r2c(rank, logicalSizeParam, howMany, 
                              dataPtr, 
                              0, stride, dist, 
                              (fftwf_complex*) dataPtr, 
                              0, stride, dist,
                              this->_FFTWMeasure ? FFTW_MEASURE : 
                              FFTW_ESTIMATE);

    this->_FFTWBackwardPlans[i] = 
      fftwf_plan_many_dft_c2r(rank, logicalSizeParam, howMany, 
                              (fftwf_complex*) dataPtr,
                              0, stride, dist, 
                              dataPtr,
                              0, stride, dist,
                              this->_FFTWMeasure ? FFTW_MEASURE : 
                              FFTW_ESTIMATE);
    
    if (!_FFTWForwardPlans[i])
    {
      throw std::runtime_error("FFTW forward plan failed to initialize");
    }
    if (!_FFTWBackwardPlans[i])
    {
      throw std::runtime_error("FFTW backward plan failed to initialize");
    }  
  }
}
  
void
DiffeoImageGrowth::
DeleteFFTWPlans()
{
  for (unsigned int i = 0; i < this->_NumberOfThreads; ++i)
  {
    fftwf_destroy_plan(this->_FFTWForwardPlans[i]);
    fftwf_destroy_plan(this->_FFTWBackwardPlans[i]);
  }
}

void 
DiffeoImageGrowth::
InitializeOperatorLookupTable()
{
  DiffeoImageGrowth::LUT lut(this->_ImageSize.x, 
                             this->_ImageSize.y, 
                             this->_ImageSize.z);

  // hardcode these for now
  double deltaX = 1.0;
  double deltaY = 1.0;
  double deltaZ = 1.0;

  //
  // precompute some values
  //
  double sX = deltaX * 2.0 * M_PI / this->_ImageSize.x; 
  double sY = deltaY * 2.0 * M_PI / this->_ImageSize.y; 
  double sZ = deltaZ * 2.0 * M_PI / this->_ImageSize.z; 

  double deltaXSq = deltaX * deltaX;
  double deltaYSq = deltaY * deltaY;
  double deltaZSq = deltaZ * deltaZ;

  //
  // fill in luts
  //
  for (unsigned int x = 0; x < lut.cosWX.size(); ++x) 
    {
      lut.cosWX[x] = (2.0 * cos(sX * static_cast<float>(x)) - 2.0) / deltaXSq;
      lut.sinWX[x] = sin(sX * static_cast<float>(x)) / deltaX;
    }
  for (unsigned int y = 0; y < lut.cosWY.size(); ++y)
    {
      lut.cosWY[y] = (2.0 * cos(sY * static_cast<float>(y)) - 2.0) / deltaYSq;
      lut.sinWY[y] = sin(sY * static_cast<float>(y)) / deltaY;
    }
  for (unsigned int z = 0; z < lut.cosWZ.size(); ++z)
    {
      lut.cosWZ[z] = (2.0 * cos(sZ * static_cast<float>(z)) - 2.0) / deltaZSq;
      lut.sinWZ[z] = sin(sZ * static_cast<float>(z)) / deltaZ;
    }  

  //
  // copy values to the ivar
  //
  this->_OperatorLookupTable = lut;
}

void
DiffeoImageGrowth::
ComputeJ0t()
{
  *this->_J0t[0] = *this->_ImagePointers[0];

  VectorFieldType vinv(this->_ImageSize);
  VectorFieldType h(this->_ImageSize);
  HField3DUtils::setToIdentity(h);

  for (unsigned int i = 1; i < this->_NumberOfImages; ++i)
  {
    if (this->_Verbose) { std::cerr << i << ","; }
    // right now use zeroth order inverse, need to add another method!
    // try using inverse of point refine
    vinv = *this->_VelocityFieldPointers[i-1];
    vinv.scale(-1.0);

    // get h = h (x+v(x))
    // temporarily store the result in vinv to save space
    HField3DUtils::composeHV(h, vinv, vinv);
    h = vinv;
    HField3DUtils::apply(*this->_ImagePointers[0], h, *this->_J0t[i]);
    
    // debug
//     {
//       std::stringstream ss;
//       ss << "debug/J0t_t" << i << "_k" << this->_Iteration;
//       std::cerr << "Writing debug file: " << ss.str() << std::endl;
//       Array3DIO::writeMETASliceZ(*this->_J0t[i], 
//                                  this->_ImageSize[2]/2, ss.str().c_str());
//     }
//     {
//       std::stringstream ss;
//       ss << "debug/I" << i << "_k" << this->_Iteration;
//       std::cerr << "Writing debug file: " << ss.str() << std::endl;
//       Array3DIO::writeMETASliceZ(*this->_ImagePointers[i], 
//                                  this->_ImageSize[2]/2, ss.str().c_str());
//     }
  }  
}

void
DiffeoImageGrowth::
ComputeJTjTi(int j)
{
  VectorFieldType h(this->_ImageSize);
  VectorFieldType h2(this->_ImageSize);
  RealImageType   tmpImage(this->_ImageSize);

  *this->_JTjTi[j][j] = *this->_ImagePointers[j];
  this->_dPhiTiTj[j][j]->fill(1.0);
  
  HField3DUtils::setToIdentity(h);      
  
  for (int i = j-1; i >= 0; --i)
  {
    // compute the jacobian of the incremental xform
    HField3DUtils::jacobian(*this->_VelocityFieldPointers[i], 
                            *this->_dPhiTiTj[i][j]);
    
    for (unsigned int z = 0; z < this->_ImageSize.z; ++z)
    {
      for (unsigned int y = 0; y < this->_ImageSize.y; ++y)
      {
        for (unsigned int x = 0; x < this->_ImageSize.x; ++x)
        {
          Vector3D<float> hx(x+(*this->_VelocityFieldPointers[i])(x,y,z).x,
                             y+(*this->_VelocityFieldPointers[i])(x,y,z).y,
                             z+(*this->_VelocityFieldPointers[i])(x,y,z).z);
          
          // update h2 gets updated h
          HField3DUtils::
            trilerp(h, hx.x, hx.y, hx.z,
                    h2(x,y,z).x, h2(x,y,z).y, h2(x,y,z).z);
          
          // deform image
          (*this->_JTjTi[j][i])(x,y,z) = 
            Array3DUtils::
            trilerp(*this->_ImagePointers[j],
                    h2(x,y,z).x, h2(x,y,z).y, h2(x,y,z).z, 0.0f);
          
          // pull current det jacobian
          (*this->_dPhiTiTj[i][j])(x,y,z) = 
            Array3DUtils::
            trilerp(*this->_dPhiTiTj[i+1][j], hx.x, hx.y, hx.z, 1.0f)
            * (1.0 + (*this->_dPhiTiTj[i][j])(x,y,z));
        }       
      }        
    }
    h = h2;
  }        
}

void 
DiffeoImageGrowth::
ComputeDeformedImageGradient(unsigned int vFieldIndex, 
                             unsigned int threadIndex)
{
  Array3DUtils::
    computeGradient(*this->_J0t[vFieldIndex],
                    *this->_ScratchVectorFieldPointers[threadIndex]);
}

void 
DiffeoImageGrowth::
ComputeBodyForce(unsigned int vFieldIndex, unsigned int threadIndex)
{
  ImageType weights(this->_ImageSize);
  weights.fill(0.0);

  VoxelType pushedForward;
  VoxelType pulledBack;
  float detJacobian;

  for (unsigned int j = vFieldIndex; j < this->_NumberOfImages; ++j)
  {
    for (unsigned int z = 0; z < this->_ImageSize.z; ++z) 
    {
      for (unsigned int y = 0; y < this->_ImageSize.y; ++y) 
      {
        for (unsigned int x = 0; x < this->_ImageSize.x; ++x) 
        {
          pushedForward = (*this->_J0t[vFieldIndex])(x,y,z);
          pulledBack    = (*this->_JTjTi[j][vFieldIndex])(x,y,z);
          detJacobian   = (*this->_dPhiTiTj[vFieldIndex][j])(x,y,z);
          weights(x,y,z) += detJacobian * (pushedForward - pulledBack);
        }
      }
    }
  }

  for (unsigned int z = 0; z < this->_ImageSize.z; ++z) 
  {
    for (unsigned int y = 0; y < this->_ImageSize.y; ++y) 
    {
      for (unsigned int x = 0; x < this->_ImageSize.x; ++x) 
      {
        (*this->_ScratchVectorFieldPointers[threadIndex])(x,y,z) *= 
          weights(x,y,z) / (this->_NumberOfImages - vFieldIndex);
      }
    }
  }  

  if (false && this->_WriteDebugImages)
  {
    // debug
    std::stringstream ss;
    ss << "debug/BodyForceWeights_v" << vFieldIndex << "_k" 
       << this->_Iteration;
    Array3DIO::writeMETASliceZ(weights, this->_ImageSize.z/2, ss.str().c_str());
  }
}

void 
DiffeoImageGrowth::
ComputeGreensFunction(unsigned int vFieldIndex, unsigned int threadIndex)
{
  Vector3D<unsigned int> logicalSize = this->_ImageSize;

  // forward fft (scale array, then compute fft)
  this->_ScratchVectorFieldPointers[threadIndex]->
    scale(1.0 / logicalSize.productOfElements());
  fftwf_execute(this->_FFTWForwardPlans[threadIndex]);

  // apply operator
  double lambda;
  float L00;
  float L10, L11;
  float L20, L21, L22;
  double alpha = this->_FluidWarpParameters.alpha;
  double beta  = this->_FluidWarpParameters.beta;
  double gamma = this->_FluidWarpParameters.gamma;

  unsigned int xFFTMax = logicalSize.x / 2 + 1;
  for (unsigned int z = 0; z < logicalSize.z; ++z)
    {
      for (unsigned int y = 0; y < logicalSize.y; ++y)
	{
	  for (unsigned int x = 0; x < xFFTMax; ++x)
	    {
	      //
	      // compute L (it is symmetric, only need lower triangular part)
	      //
	      
	      // maybe lambda should be stored in a lut
	      // it would reduce computation but may cause cache misses
	      lambda = - alpha 
		* (this->_OperatorLookupTable.cosWX[x] + 
                   this->_OperatorLookupTable.cosWY[y] + 
                   this->_OperatorLookupTable.cosWZ[z]) 
		+ gamma;	      
	      
	      L00 = lambda - beta * this->_OperatorLookupTable.cosWX[x];
	      L11 = lambda - beta * this->_OperatorLookupTable.cosWY[y];
	      L22 = lambda - beta * this->_OperatorLookupTable.cosWZ[z];
	      L10 = beta * this->_OperatorLookupTable.sinWX[x] * 
                this->_OperatorLookupTable.sinWY[y];
	      L20 = beta * this->_OperatorLookupTable.sinWX[x] * 
                this->_OperatorLookupTable.sinWZ[z];
	      L21 = beta * this->_OperatorLookupTable.sinWY[y] * 
                this->_OperatorLookupTable.sinWZ[z];

	      //
	      // compute V = Linv F (for real and imaginary parts)
	      //
              CoordinateType* complexPtr =
                &(*this->
                  _ScratchVectorFieldPointers[threadIndex])(x * 2, y, z).x; 
	      DiffeoImageGrowth::
                InverseOperatorMultiply(complexPtr,
                                        L00,
                                        L10, L11,
                                        L20, L21, L22);
	    }
	}
    }

  // backward fft
  fftwf_execute(this->_FFTWBackwardPlans[threadIndex]);
}

void 
DiffeoImageGrowth::
UpdateVelocity(unsigned int vFieldIndex, unsigned int threadIndex)
{
  double vmax = 0;
  Vector3D<CoordinateType> gradientEnergy;
  for (unsigned int z = 0; z < this->_ImageSize.z; ++z)
  {
    for (unsigned int y = 0; y < this->_ImageSize.y; ++y)
    {
      for (unsigned int x = 0; x < this->_ImageSize.x; ++x)
      {
        gradientEnergy = 
          2 * (*this->_VelocityFieldPointers[vFieldIndex])(x,y,z) - 
          (*this->_ScratchVectorFieldPointers[threadIndex])(x,y,z)
          * 2.0/(this->_LDMMParameters.sigma*this->_LDMMParameters.sigma);
        (*this->_VelocityFieldPointers[vFieldIndex])(x,y,z) -= 
          this->_LDMMParameters.epsilon * gradientEnergy;
        double currentL2 = 
          (*this->_VelocityFieldPointers[vFieldIndex])(x,y,z).normL2();
        if (currentL2 > vmax)
        {
          vmax = currentL2;
        }
      }
    }
  }
  this->_MaxL2Displacements[vFieldIndex] = vmax;
}

void
DiffeoImageGrowth::
ComputeError()
{
  double diff;
  double sse = 0.0;
  double iv;
  double jv;
  for (unsigned int i = 1; i < this->_NumberOfImages; ++i)
  {
    double sseImage = 0.0;
    double issv     = 0.0;
    double jssv     = 0.0;
    for (unsigned int z = 0; z < this->_ImageSize.z; ++z)
    {
      for (unsigned int y = 0; y < this->_ImageSize.y; ++y)
      {
        for (unsigned int x = 0; x < this->_ImageSize.x; ++x)
        {
          iv = (*this->_ImagePointers[i])(x,y,z);
          jv = (*this->_J0t[i])(x,y,z);
          diff = iv - jv;

          sseImage += diff*diff;
          issv     += iv*iv;
          jssv     += jv*jv;
        }
      }
    }
    sse += sseImage;
  }
  this->_SumSquaredError = sse;
  sse /= (this->_NumberOfImages * this->_ImageSize.productOfElements());
  this->_RootMeanSquaredError = sqrt(sse);
}

void
DiffeoImageGrowth::
InverseOperatorMultiply(CoordinateType* complexPtr,
                        float& L00,
                        float& L10, float& L11,
                        float& L20, float& L21, float& L22)
{
  float G00;
  float G10, G11;
  float G20, G21, G22;
  float y0, y1, y2;
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

  float bRealX = complexPtr[0];
  float bRealY = complexPtr[2];
  float bRealZ = complexPtr[4];

  float bImagX = complexPtr[1];
  float bImagY = complexPtr[3];
  float bImagZ = complexPtr[5];

  float& vRealX = complexPtr[0];
  float& vRealY = complexPtr[2];
  float& vRealZ = complexPtr[4];

  float& vImagX = complexPtr[1];
  float& vImagY = complexPtr[3];
  float& vImagZ = complexPtr[5];

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

