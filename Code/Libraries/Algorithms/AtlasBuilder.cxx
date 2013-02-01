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

// todo: 
// save memory in update h
// trade off scratch: per thread or per image, what whatever is smaller

#include <AtlasBuilder.h>
#include <strstream>
#include <HField3DUtils.h>
#include <assert.h>
#include <Timer.h>
#include <iomanip>
#include "ApplicationUtils.h"
#include "StringUtils.h"



struct ThreadInfo
{
  AtlasBuilder* atlasBuilder;
  unsigned int  threadIndex;
};

AtlasBuilder::
AtlasBuilder()
  : 
  _IterationDataOutputStream(std::cout)
{
  this->_NumberOfImages               = 0;
  this->_AverageImagePointer          = NULL;
  this->_NumberOfThreads              = 1;

  this->_FFTWNumberOfThreads          = 1;
  this->_FFTWMeasure                  = true;

  this->_OutputInitialScaleLevelMean  = false;
  this->_OutputFinalScaleLevelMean    = false;
  this->_ScaleLevel                   = 0;

  this->_DeltaSelectionMethod         = DELTA_USE_INDIVIDUAL;
  this->_UpdateAfterEverySubIteration = true;
  this->_ComputeInverseDeformations   = false;
  this->_MeanComputationStrategy      = 
    new ArithmeticMeanComputationStrategy<AtlasBuilder::VoxelType>();

  this->_MeanSquaredError             = 0;
  this->_Iteration                    = 0;
  this->_NextImageToProcess           = 0;

  pthread_mutex_init(&this->_NextImageToProcessMutex, NULL);
  pthread_mutex_init(&this->_AverageImageMutex, NULL);
}

AtlasBuilder::
~AtlasBuilder()
{
  this->DeleteScratchMemory();
}

/**
 * Set the output stream used for logging messages
 */
void
AtlasBuilder::
SetLogOutputStream(std::ostream& out)
{
  //this->_IterationDataOutputStream = out;
}

void
AtlasBuilder::
SetNumberOfThreads(unsigned int numThreads)
{
  this->_NumberOfThreads = numThreads;
}

unsigned int 
AtlasBuilder::
GetNumberOfThreads() const
{
  return this->_NumberOfThreads;
}

void
AtlasBuilder::
SetParams(const AtlasBuilderParam &param)
{
  this->SetDiffOperParams(param.DiffOper());
  this->SetNumIterations(param.NIterations());
  this->SetMaxPerturbation(param.MaxPert());
  this->SetUpdateAverageEverySubIteration(param.UpdateAfterSubIteration());
  if(param.DeltaSelectionUseMean()){
    this->SetDeltaSelectionToMean();
  }else{
    this->SetDeltaSelectionToIndividual();
  }
  if(param.OutputMeanEveryNIterations() != 0){
    throw AtlasWerksException(__FILE__, __LINE__, "Error, OutputMeanEveryNIterations not supported in CPU verison");
  }
  this->_OutputInitialScaleLevelMean = param.OutputInitialScaleLevelMean();
  this->_OutputFinalScaleLevelMean = param.OutputFinalScaleLevelMean();
}

AtlasBuilderParam
AtlasBuilder::
GetParams()
{
  AtlasBuilderParam param;
  param.DiffOper() = this->GetDiffOperParams();
  param.NIterations() = this->GetNumIterations();
  param.MaxPert() = this->GetMaxPerturbation();
  param.UpdateAfterSubIteration() = this->GetUpdateAverageEverySubIteration();
  switch(GetDeltaSelectionType()){
  case DELTA_USE_MEAN:
    param.DeltaSelectionUseMean() = true;
    break;
  case DELTA_USE_INDIVIDUAL:
    param.DeltaSelectionUseMean() = false;
    break;
  default:
    std::cerr << "Error, unknown Delta SelectionMethod" << std::endl;
    break;
  }
  return param;
}

void
AtlasBuilder::
SetFFTWNumberOfThreads(unsigned int numThreads)
{
  this->_FFTWNumberOfThreads = numThreads;
}

unsigned int 
AtlasBuilder::
GetFFTWNumberOfThreads() const
{
  return this->_FFTWNumberOfThreads;
}

void 
AtlasBuilder::
SetFFTWMeasure(bool b)
{
  this->_FFTWMeasure = b;
}

void 
AtlasBuilder::
SetFFTWMeasureOn()
{
  this->SetFFTWMeasure(true);
}

void 
AtlasBuilder::
SetFFTWMeasureOff()
{
  this->SetFFTWMeasure(false);
}

bool 
AtlasBuilder::
GetFFTWMeasure() const
{
  return this->_FFTWMeasure;
}

void 
AtlasBuilder::
SetUpdateAverageEverySubIterationOn()
{
  this->SetUpdateAverageEverySubIteration(true);
}

void 
AtlasBuilder::
SetUpdateAverageEverySubIterationOff()
{
  this->SetUpdateAverageEverySubIteration(false);
}

void 
AtlasBuilder::
SetUpdateAverageEverySubIteration(bool b)
{
  this->_UpdateAfterEverySubIteration = b;
}

bool
AtlasBuilder::
GetUpdateAverageEverySubIteration() const
{
  return this->_UpdateAfterEverySubIteration;
}

void 
AtlasBuilder::
SetComputeInverseDeformationsOn()
{
  this->SetComputeInverseDeformations(true);
}

void 
AtlasBuilder::
SetComputeInverseDeformationsOff()
{
  this->SetComputeInverseDeformations(false);
}

void 
AtlasBuilder::
SetComputeInverseDeformations(bool b)
{
  this->_ComputeInverseDeformations = b;
}

bool
AtlasBuilder::
GetComputeInverseDeformations() const
{
  return this->_ComputeInverseDeformations;
}

void
AtlasBuilder::
SetNumIterations(unsigned int niters){
  this->_NumIterations = niters;
}

unsigned int
AtlasBuilder::
GetNumIterations(){
  return this->_NumIterations;
}

void
AtlasBuilder::
SetMaxPerturbation(double maxPert){
  this->_MaxPerturbation = maxPert;
}

double
AtlasBuilder::
GetMaxPerturbation(){
  return this->_MaxPerturbation;
}

DiffOperParam&
AtlasBuilder::
GetDiffOperParams()
{
  return this->_DiffOperParams;
}

void
AtlasBuilder::
SetDiffOperParams(const DiffOperParam &params)
{
  this->_DiffOperParams = params;
}

void
AtlasBuilder::
SetMeanComputationStrategy(AtlasBuilder::MeanComputationStrategyType* s)
{
  this->_MeanComputationStrategy = s;
}

AtlasBuilder::MeanComputationStrategyType*
AtlasBuilder::
GetMeanComputationStrategy() const
{
  return this->_MeanComputationStrategy;
}

void 
AtlasBuilder::
SetNumberOfInputImages(unsigned int n)
{
  this->_NumberOfImages = n;

  this->_ImagePointers.resize(n);
  this->_DeformationFieldPointers.resize(n);
  this->_DeformationFieldInversePointers.resize(n);
  this->_Delta.assign(n, 0.0);
}

unsigned int
AtlasBuilder::
GetNumberOfInputImages() const
{
  return this->_NumberOfImages;
}

void
AtlasBuilder::
SetNthInputImage(unsigned int n, ImageType* imagePointer)
{
  this->_ImagePointers[n] = imagePointer;
}

AtlasBuilder::ImageType*
AtlasBuilder::
GetNthInputImage(unsigned int n) const
{
  return this->_ImagePointers[n];
}

AtlasBuilder::ImageType*
AtlasBuilder::
GetNthDeformedImage(unsigned int n) const
{
  if (this->_DeformedImagePointers.size() <= n)
    {
      return NULL;
    }
  return this->_DeformedImagePointers[n];
}

void
AtlasBuilder::
SetNthDeformationField(unsigned int n, VectorFieldType* fieldPointer)
{
  this->_DeformationFieldPointers[n] = fieldPointer;
}

AtlasBuilder::VectorFieldType*
AtlasBuilder::
GetNthDeformationField(unsigned int n) const
{
  return this->_DeformationFieldPointers[n];
}

void
AtlasBuilder::
SetNthDeformationFieldInverse(unsigned int n, VectorFieldType* fieldPointer)
{
  this->_DeformationFieldInversePointers[n] = fieldPointer;
}

AtlasBuilder::VectorFieldType*
AtlasBuilder::
GetNthDeformationFieldInverse(unsigned int n) const
{
  return this->_DeformationFieldInversePointers[n];
}

void
AtlasBuilder::
SetAverageImage(ImageType* imagePointer)
{
  this->_AverageImagePointer = imagePointer;
}

AtlasBuilder::ImageType*
AtlasBuilder::
GetAverageImage() const
{
  return this->_AverageImagePointer;
}

AtlasBuilder::IterationData
AtlasBuilder::
GetIterationData(unsigned int iteration) const
{
  return this->_IterationDataLog[iteration];
}

void
AtlasBuilder::
CheckInputs()
{
  //
  // make sure that a valid mean strategy is set
  //
  if (this->_MeanComputationStrategy == NULL)
  {
    std::runtime_error("Mean computation strategy is null.");    
  }

  //
  // make sure that the mean image is there there
  //
  if (this->_AverageImagePointer == NULL)
  {
    std::runtime_error("Average Image is null.");
  }

  // use the size of the average as the standard
  Vector3D<unsigned int> imageSize = this->_AverageImagePointer->getSize();

  //
  // check the data for all the images
  //
  for (unsigned int i = 0; i < this->_NumberOfImages; ++i)
  {
    // check that the image is there
    if (this->_ImagePointers[i] == NULL)
    {
      std::strstream ss;
      ss << "Input image is null: " << i;
      throw std::runtime_error(ss.str());
    }

    // check that is is the right size
    if (imageSize != this->_ImagePointers[i]->getSize())
    {
      std::strstream ss;
      ss << "Incompatible image size for image " << i;
      throw std::runtime_error(ss.str());
    }

    // check that the deformation field is there
    if (this->_DeformationFieldPointers[i] == NULL)
    {
      std::strstream ss;
      ss << "Deformation field is null: " << i;
      throw std::runtime_error(ss.str());
    }

    // check that the deformation field is the correct size
    if (imageSize != this->_DeformationFieldPointers[i]->getSize())
    {
      std::strstream ss;
      ss << "Incompatible deformation field size for image " << i;
      throw std::runtime_error(ss.str());
    }

    // check that the inverse deformation field is there
    if (this->_ComputeInverseDeformations &&
        this->_DeformationFieldInversePointers[i] == NULL)
    {
      std::strstream ss;
      ss << "Inverse deformation field is null: " << i;
      throw std::runtime_error(ss.str());
    }

    // check inverse deformation field is the correct size
    if (this->_ComputeInverseDeformations &&
        imageSize != this->_DeformationFieldInversePointers[i]->getSize())
    {
      std::strstream ss;
      ss << "Incompatible inverse deformation field size for image " << i;
      throw std::runtime_error(ss.str());
    }
  }
}

void
AtlasBuilder::
InitializeScratchMemory()
{
  //
  // this function assumes that CheckInput has been called without error.
  //

  //
  // resize memory holders
  this->_DeformedImagePointers.resize(this->_NumberOfImages);
  this->_MaxL2Displacements.resize(this->_NumberOfImages, 0.0);
  this->_MaxVelocityL2Displacements.resize(this->_NumberOfImages, 0.0);

  this->_DiffOperPointers.resize(this->_NumberOfThreads);
  this->_ScratchVectorFieldPointers.resize(this->_NumberOfThreads);


  // use the size of the average as the standard
  Vector3D<unsigned int> imageSize = this->_AverageImagePointer->getSize();
  // we assume identity spacing, but this should be changed
  Vector3D<float> imageSpacing(1.0, 1.0, 1.0);

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

  for (unsigned int i = 0; i < this->_NumberOfThreads; ++i)
  {
    // create the differential operator for this thread
    this->_DiffOperPointers[i] =
      new DiffOper(imageSize, imageSpacing);
    // allocate scratch vector field
    this->_ScratchVectorFieldPointers[i] = 
      this->_DiffOperPointers[i]->GetInternalFFTWVectorField();
  }

  for (unsigned int i = 0; i < this->_NumberOfImages; ++i)
  {
    // allocate deformed image
    this->_DeformedImagePointers[i] = new ImageType(imageSize);

    // generate the deformed image
    HField3DUtils::apply(*this->_ImagePointers[i], 
                         *this->_DeformationFieldPointers[i],
                         *this->_DeformedImagePointers[i]);
  }

#ifdef MPI_ENABLED
  // this will be used to hold result of summing across MPI nodes
  _MPIAverageImagePointer = new ImageType(imageSize);
#endif

}

void
AtlasBuilder::
DeleteScratchMemory()
{
  for (unsigned int i = 0; i < this->_NumberOfThreads; ++i)
    {
      if (this->_DiffOperPointers[i])
	{
	  delete this->_DiffOperPointers[i];
	  this->_DiffOperPointers[i] = NULL;
	}
    }
  
  for (unsigned int i = 0; i < this->_NumberOfImages; ++i)
    {
      if (this->_DeformedImagePointers[i])
	{
	  delete this->_DeformedImagePointers[i];
	  this->_DeformedImagePointers[i] = NULL;
	}
    }
}

void
AtlasBuilder::
UpdateDiffOperParams()
{
  // update params for all DiffOpers
  for (unsigned int i = 0; i < this->_NumberOfThreads; ++i)
  {
    this->_DiffOperPointers[i]->SetParams(this->_DiffOperParams);
    this->_DiffOperPointers[i]->Initialize();
  }
}

void
AtlasBuilder::
GenerateAverage()
{
  //
  // start the total timer
  this->_TotalTimer.restart();

  //
  // check that input data is consistent: an exception will be thrown
  // if there is a problem
  std::cout << "Checking inputs...";
  this->CheckInputs();
  std::cout << "DONE" << std::endl;  

  //
  // set num threads to min(num threads, num images)
  if (this->_NumberOfThreads > this->_NumberOfImages)
  {
    std::cerr << "WARNING: More threads than images." << std::endl
              << "WARNING: Setting number of threads to " 
              << this->_NumberOfImages << std::endl;
    this->_NumberOfThreads = this->_NumberOfImages;
  }

  //
  // initialize memory, order matters
  std::cout << "Initializing memory...";
  this->InitializeScratchMemory();
  std::cout << "Updating L params...";
  this->UpdateDiffOperParams();
  std::cout << "DONE" << std::endl;

  //
  // start algorithm
  //
  this->RunAlgorithm();
}

unsigned int 
AtlasBuilder::
GetJobImageIndex()
{
  pthread_mutex_lock(&this->_NextImageToProcessMutex);
  unsigned int imageIndex = this->_NextImageToProcess++;
  pthread_mutex_unlock(&this->_NextImageToProcessMutex);
  return imageIndex;
}

void 
AtlasBuilder::
LockMeanImage()
{
  pthread_mutex_lock(&this->_AverageImageMutex);
}

void 
AtlasBuilder::
UnlockMeanImage()
{
  pthread_mutex_unlock(&this->_AverageImageMutex);
}

pthread_t
AtlasBuilder::
GetThreadID()
{
  return pthread_self();
}

void*
AtlasBuilder::
ThreadedUpdateImages(void* arg)
{
  ThreadInfo* threadInfoPtr = static_cast<ThreadInfo*>(arg);
  AtlasBuilder* ptr = threadInfoPtr->atlasBuilder;
  unsigned int threadIndex = threadInfoPtr->threadIndex;

  // loop until all jobs are taken
  while (true)
  {
    //
    // get an image index to process
    unsigned int imageIndex = ptr->GetJobImageIndex();
    //std::cerr << "{" << threadIndex << " / " << ptr->GetThreadID()
    //<< "} Got assignment: " 
    //<< imageIndex << std::endl;    
    if (imageIndex >= ptr->_NumberOfImages)
    {
      // all the jobs are taken
      break;
    }

    //
    // start the timer
    Timer iterationTimer;
    iterationTimer.start();

    //
    // compute gradient of deformed image
    //std::cerr << "{" << threadIndex << " / " << ptr->GetThreadID()
    //<< "} Gradient: " 
    //<< imageIndex << std::endl;    
    ptr->UpdateGradient(imageIndex, threadIndex);

    // 
    // compute body force 
    //
    // Note: the body force computation requires the average image,
    // make sure we don't access it while it is being updated
    if (ptr->_UpdateAfterEverySubIteration) ptr->LockMeanImage();
    //std::cerr << "{" << threadIndex << " / " << ptr->GetThreadID()
    //<< "} Body force: " 
    //<< imageIndex << std::endl;    
    ptr->UpdateBodyForce(imageIndex, threadIndex);
    if (ptr->_UpdateAfterEverySubIteration) ptr->UnlockMeanImage();

    //
    // update velocity field according to Euler-Lagrange equation
    //std::cerr << "{" << threadIndex << " / " << ptr->GetThreadID()
    //<< "} Velocity: " 
    //<< imageIndex << std::endl;    
    ptr->UpdateVelocityField(imageIndex, threadIndex);

    //
    // compose velocity to generate new deformation field
    //std::cerr << "{" << threadIndex << " / " << ptr->GetThreadID()
    //<< "} Deformation fields: " 
    //<< imageIndex << std::endl;    
    ptr->UpdateDeformationFields(imageIndex, threadIndex);

    //
    // measure maximum displacement
    //std::cerr << "{" << threadIndex << " / " << ptr->GetThreadID()
    //<< "} Min/max L2: " 
    //<< imageIndex << std::endl;    
    double minL2, maxL2;
    HField3DUtils::
      minMaxDeformationL2Norm(*ptr->_DeformationFieldPointers[imageIndex], 
                              minL2, maxL2);
    //ptr->_MaxL2Displacements[imageIndex] = maxL2;
    //std::cerr << "{" << threadIndex << " / " << imageIndex
    //<< "} Min/max L2: " << maxL2 << std::endl;
    
    //
    // deform the image according to the new deformation field
    //std::cerr << "{" << threadIndex << " / " << ptr->GetThreadID()
    //<< "} Deformed image: " 
    //<< imageIndex << std::endl;    
    ptr->UpdateDeformedImage(imageIndex, threadIndex);

    //
    // if the average should be updated after each sub-iteration, do
    // that now and report the results
    //
    if (ptr->_UpdateAfterEverySubIteration)
    {
      ptr->LockMeanImage();
      //std::cerr << "{" << threadIndex << " / " << ptr->GetThreadID()
      //<< "} Average: " 
      //<< imageIndex << std::endl;    
      ptr->UpdateAverageImage();
      //std::cerr << "{" << threadIndex << " / " << ptr->GetThreadID()
      //<< "} Error: " 
      //<< imageIndex << std::endl;    
      ptr->UpdateError();
      //std::cerr << "{" << threadIndex << " / " << ptr->GetThreadID()
      //<< "} Log: " 
      //<< imageIndex << std::endl;    
      ptr->LogIterationData(imageIndex, threadIndex, 
                             iterationTimer.getSeconds());
      ptr->UnlockMeanImage();
    }
  }  

  return ((void*)0);
}

void
AtlasBuilder::
UpdateDelta()
{
  if (this->_DeltaSelectionMethod == DELTA_USE_MEAN)
  {
    double l2Sum = std::accumulate(this->_MaxVelocityL2Displacements.begin(),
                                   this->_MaxVelocityL2Displacements.end(),
                                   0.0);
    this->_Delta.assign(this->_NumberOfImages, 
                        this->_MaxPerturbation /
                        (this->_NumberOfImages * l2Sum));
  }
  else if (this->_DeltaSelectionMethod == DELTA_USE_INDIVIDUAL)
  {
    this->_Delta.resize(this->_NumberOfImages);
    for (unsigned int i = 0; i < this->_NumberOfImages; ++i)
    {
      this->_Delta[i] = this->_MaxPerturbation / 
        this->_MaxVelocityL2Displacements[i];
      std::cout<< "Delta of image "<< i << ": " <<  this->_Delta[i] << std::endl;
    }
  }
  else
  {
    std::runtime_error("Unknown delta update method.");    
  }
  // TEST
  std::cout << "Deltas/MaxVelL2 are: ";
  for (unsigned int i = 0; i < this->_NumberOfImages; ++i){
    std::cout << i << ":{" << this->_Delta[i] << " / " << this->_MaxVelocityL2Displacements[i] << "}" << std::endl; 
  }
  // END TEST
}

    
void 
AtlasBuilder::
UpdateGradient(unsigned int imageIndex, unsigned int threadIndex)
{
  Array3DUtils::
    computeGradient(*this->_DeformedImagePointers[imageIndex],
                    *this->_ScratchVectorFieldPointers[threadIndex]);
  
}

void 
AtlasBuilder::
UpdateBodyForce(unsigned int imageIndex, unsigned int threadIndex)
{
#ifdef __DEBUG__
    float maxA = getMax(*this->_AverageImagePointer);
    fprintf(stderr,"Maximumvalue of the average %f \n",maxA);
#endif
            
  Vector3D<unsigned int> size = this->_AverageImagePointer->getSize();
  double di;
  for (unsigned int z = 0; z < size.z; ++z) {
    for (unsigned int y = 0; y < size.y; ++y) {
      for (unsigned int x = 0; x < size.x; ++x) {
        di = 
            (*this->_AverageImagePointer)(x, y, z) - 
            (*this->_DeformedImagePointers[imageIndex])(x, y, z);
        (*this->_ScratchVectorFieldPointers[threadIndex])(x,y,z) *= di;
      }
    }
  }

}

void 
AtlasBuilder::
UpdateVelocityField(unsigned int imageIndex, unsigned int threadIndex)
{
  Vector3D<unsigned int> logicalSize = this->_AverageImagePointer->getSize();
  this->_DiffOperPointers[threadIndex]->ApplyInverseOperator();
  
  //
  // compute max L2 norm for delta computation
  double maxL2 = 0;
  for (unsigned int z = 0; z < logicalSize.z; ++z)
  {
    for (unsigned int y = 0; y < logicalSize.y; ++y)
    {
      for (unsigned int x = 0; x < logicalSize.x; ++x)
      {
        double L2sq = 
          (*this->_ScratchVectorFieldPointers[threadIndex])(x,y,z).
          lengthSquared();
        if (maxL2 < L2sq)
        {
          maxL2 = L2sq;
        }
      }
    }
  }
  this->_MaxVelocityL2Displacements[imageIndex] = sqrt(maxL2);
#ifdef __DEBUG__
  std::cout << "Max velocity displacements " << sqrt(maxL2) << std::endl;
#endif
}

// TODO: we could probably rearrange this to save some memory...
void 
AtlasBuilder::
UpdateDeformationFields(unsigned int imageIndex, unsigned int threadIndex)
{
  Vector3D<unsigned int> size = this->_AverageImagePointer->getSize();

  //
  // compute hIncremental(x) = x + velocity(x) * delta
  //
  VectorFieldType hIncremental(size);
  for (unsigned int z = 0; z < size.z; ++z) {
    for (unsigned int y = 0; y < size.y; ++y) {
      for (unsigned int x = 0; x < size.x; ++x) {
	hIncremental(x,y,z).x = x + 
          (*this->_ScratchVectorFieldPointers[threadIndex])(x,y,z).x * 
          this->_Delta[imageIndex];
	hIncremental(x,y,z).y = y + 
          (*this->_ScratchVectorFieldPointers[threadIndex])(x,y,z).y * 
          this->_Delta[imageIndex];
	hIncremental(x,y,z).z = z + 
          (*this->_ScratchVectorFieldPointers[threadIndex])(x,y,z).z * 
          this->_Delta[imageIndex];
      }
    }
  }

  //
  // compute h(x) = h(hIncremental(x))
  //
  VectorFieldType oldDeformation(*this->_DeformationFieldPointers[imageIndex]);
  HField3DUtils::compose(oldDeformation, hIncremental, 
                         *this->_DeformationFieldPointers[imageIndex]);

  // update inverse deformation field
  if (this->_ComputeInverseDeformations)
  {
    //
    // compute 
    // hIncrementalInv(x) = x + x - hIncremental(x)
    //
    VectorFieldType& hIncrementalInv = oldDeformation; // reuse memory here
    HField3DUtils::computeInverseZerothOrder(hIncremental, hIncrementalInv);
    
    //
    // compute hInv(x) = hIncrementalInv(hInv(x))
    //
    HField3DUtils::
      compose(hIncrementalInv, 
              *this->_DeformationFieldInversePointers[imageIndex], 
              *this->_DeformationFieldInversePointers[imageIndex]);
  }
}

void 
AtlasBuilder::
UpdateDeformedImage(unsigned int imageIndex, unsigned int threadIndex)
{
  HField3DUtils::apply(*this->_ImagePointers[imageIndex],
                       *this->_DeformationFieldPointers[imageIndex],
                       *this->_DeformedImagePointers[imageIndex]);
}

void
AtlasBuilder::
RunAlgorithm()
{
  //
  // compute average image
  this->ThreadedUpdateAverageImage();

  if(this->_OutputInitialScaleLevelMean){
    std::string fname = StringUtils::strPrintf("Scale%02dInitialMeanImage.nhdr", this->_ScaleLevel);
    Image<float> tmp(*this->_AverageImagePointer);
    ApplicationUtils::SaveImageITK(fname.c_str(), tmp);
  }

  for (unsigned int iter = 0; iter < this->_NumIterations;
       ++iter)
  {
    Timer iterationTimer;
    iterationTimer.start();

    this->_Iteration = iter;

    //
    // start threads which will update each image
    std::vector<pthread_t> threads(this->_NumberOfThreads);

    // reset job counter
    this->_NextImageToProcess = 0;

    // info passed to threads
    std::vector<ThreadInfo> threadInfo(this->_NumberOfThreads);

    // create threads that will update images
    for (unsigned int threadIndex = 0; threadIndex < this->_NumberOfThreads; 
         ++threadIndex)
    {
      threadInfo[threadIndex].atlasBuilder = this;
      threadInfo[threadIndex].threadIndex  = threadIndex;
  
      int rv = pthread_create(&threads[threadIndex], NULL,
                              &AtlasBuilder::ThreadedUpdateImages, 
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

    //
    // delta will will be average deltas for individual images.  A
    // delta for an individual image means that maxpert will be
    // realized for that image.
    if (this->_Iteration == 0)
    {
      this->UpdateDelta();
    }

    //
    // update the average if it is only done once per iteration.
    // otherwise, it will be updated by the threads.
    if (this->_UpdateAfterEverySubIteration == false)
    {
      this->ThreadedUpdateAverageImage();
      this->UpdateError();
      this->LogIterationData(this->_NumberOfImages-1, 0,
                             iterationTimer.getSeconds());
    }
  }


  if(this->_OutputFinalScaleLevelMean){
    std::string fname = StringUtils::strPrintf("Scale%02dFinalMeanImage.nhdr", this->_ScaleLevel);
    Image<float> tmp(*this->_AverageImagePointer);
    ApplicationUtils::SaveImageITK(fname.c_str(), tmp);
  }

}

void 
AtlasBuilder::
SumAcrossNodes()
{
#ifdef MPI_ENABLED
  // compute the total sum of all image on the cluster
  // use the size of the average as the standard
  int nVox = this->_AverageImagePointer->getSize().productOfElements();
  MPI_Allreduce(_AverageImagePointer->getDataPointer(0), _MPIAverageImagePointer->getDataPointer(0),
		nVox, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
  *_AverageImagePointer = *_MPIAverageImagePointer;
#endif
}

void
AtlasBuilder::
UpdateAverageImage()
{
  unsigned int numElements = this->_AverageImagePointer->getNumElements();
  std::vector<VoxelType> voxelData(this->_NumberOfImages);

  for (unsigned int i = 0; i < numElements; ++i)
  {
    // get data from each image for this voxel
    for (unsigned int j = 0; j < this->_NumberOfImages; ++j)
    {
      voxelData[j] = (*this->_DeformedImagePointers[j])(i);
    }

    // compute the mean from the list of voxels
    (*this->_AverageImagePointer)(i) = 
      this->_MeanComputationStrategy->ComputeMean(this->_NumberOfImages,
                                                  &voxelData[0]);
  }
}

void*
AtlasBuilder::
ThreadedUpdateAverageImage(void* arg)
{
  ThreadInfo* threadInfoPtr = static_cast<ThreadInfo*>(arg);
  AtlasBuilder* ptr = threadInfoPtr->atlasBuilder;
  unsigned int threadIndex = threadInfoPtr->threadIndex;
  unsigned int numberOfThreads  = ptr->_NumberOfThreads;
  unsigned int numberOfElements = ptr->_AverageImagePointer->getNumElements();
  unsigned int thisThreadBegin  = 
    threadIndex * (numberOfElements/numberOfThreads);
  unsigned int thisThreadEnd    = 
    (threadIndex + 1) * (numberOfElements/numberOfThreads);
  
  if (threadIndex == numberOfThreads-1)
  {
    thisThreadEnd = numberOfElements;
  }

  std::vector<VoxelType> voxelData(ptr->_NumberOfImages);
  // loop over voxels
  for (unsigned int i = thisThreadBegin; i < thisThreadEnd; ++i)
  {
    // get data from each image for this voxel
    for (unsigned int j = 0; j < ptr->_NumberOfImages; ++j)
    {
      voxelData[j] = (*ptr->_DeformedImagePointers[j])(i);
    }

    // compute the mean from the list of voxels
    (*ptr->_AverageImagePointer)(i) = 
      ptr->_MeanComputationStrategy->ComputeMean(ptr->_NumberOfImages,
                                                 &voxelData[0]);    
  }

  return ((void*)0);
}

void
AtlasBuilder::
ThreadedUpdateAverageImage()
{
  if (this->_AverageImagePointer->getNumElements() < this->_NumberOfThreads)
  {
    // unlikely, but possible
    this->UpdateAverageImage();
    return;
  }

  //
  // start threads which will update the average
  std::vector<pthread_t> threads(this->_NumberOfThreads);

  // info passed to threads
  std::vector<ThreadInfo> threadInfo(this->_NumberOfThreads);
  
  // create threads that will update the average
  for (unsigned int threadIndex = 0; threadIndex < this->_NumberOfThreads; 
       ++threadIndex)
  {
    threadInfo[threadIndex].atlasBuilder = this;
    threadInfo[threadIndex].threadIndex  = threadIndex;
    
    int rv = pthread_create(&threads[threadIndex], NULL,
                            &AtlasBuilder::ThreadedUpdateAverageImage, 
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

  // if using MPI, compute sum across nodes
  SumAcrossNodes();
}

void
AtlasBuilder::
UpdateError()
{
  unsigned int numElements = this->_AverageImagePointer->getNumElements();
  double MSE = 0;

  for (unsigned int i = 0; i < this->_NumberOfImages; ++i)
    {
      for (unsigned int j = 0; j < numElements; ++j)
      {
        double diff = (*this->_DeformedImagePointers[i])(j) - 
          (*this->_AverageImagePointer)(j);
        MSE += diff * diff;
      }
    }
  MSE /= (this->_NumberOfImages * numElements);
  this->_MeanSquaredError = MSE;
}

void
AtlasBuilder::
LogIterationData(unsigned int imageNumber, 
                 int threadID, 
                 double iterationTime)
{
  IterationData newData;
  newData.IterationNumber                = this->_Iteration;
  newData.ImageNumber                    = imageNumber;
  newData.IterationEllapsedTimeInSeconds = iterationTime;
  newData.TotalEllapsedTimeInSeconds     = this->_TotalTimer.getSeconds();
  newData.ProcessingThreadID             = threadID;
  newData.MeanSquaredError               = this->_MeanSquaredError;
  newData.RootMeanSquaredError           = sqrt(this->_MeanSquaredError);
  newData.Delta                          = this->_Delta[imageNumber];
  newData.MaxL2Displacement              = 
    *std::max_element(this->_MaxL2Displacements.begin(), 
                      this->_MaxL2Displacements.end());

  double MSEPercent  = 100.0;
  double RMSEPercent = 100.0;
  bool ValueDidImprove = true;

  if (newData.IterationNumber > 0)
  {
    double initialMSE = this->_IterationDataLog[0].MeanSquaredError;
    MSEPercent = 100.0 * newData.MeanSquaredError / initialMSE;

    double initialRMSE = this->_IterationDataLog[0].RootMeanSquaredError;
    RMSEPercent = 100.0 * newData.RootMeanSquaredError / initialRMSE;

    if (newData.MeanSquaredError > 
        this->_IterationDataLog.back().MeanSquaredError)
    {
      ValueDidImprove = false;
    }
  }

  this->_IterationDataLog.push_back(newData);

  this->_IterationDataOutputStream 
    << "["  << newData.IterationNumber
    << "/"  << this->_NumIterations
    << " " << newData.ImageNumber
    << "/" << this->_NumberOfImages << "] "
    << "Time=" << newData.IterationEllapsedTimeInSeconds 
    << "|" << newData.TotalEllapsedTimeInSeconds
    << " " << "Thd=" << newData.ProcessingThreadID;

  if (this->_DeltaSelectionMethod == DELTA_USE_MEAN)
  {
    this->_IterationDataOutputStream 
      << " " 
      << "D=M" << std::setprecision(4) << newData.Delta;
  }
  else
  {
    this->_IterationDataOutputStream 
      << " " << "D=IND";
  }
  this->_IterationDataOutputStream 
    << " " << "MaxL2=" << newData.MaxL2Displacement
    << " " << "RMSE=" << newData.RootMeanSquaredError 
    << " (" << RMSEPercent << "%)";
                               
  if (!ValueDidImprove)
  {
    this->_IterationDataOutputStream 
      << " <<--<<--<<-- MSE Increased! --<<--<<--<<";
  }
  this->_IterationDataOutputStream << std::endl;
}
