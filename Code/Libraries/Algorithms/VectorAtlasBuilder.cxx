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

#include <VectorAtlasBuilder.h>
#include <strstream>
#include <HField3DUtils.h>
#include <assert.h>
#include <Timer.h>
#include <iomanip>

#include "itkImage.h"
#include "itkVector.h"
#include "itkIterativeInverseDeformationFieldImageFilter.h"

// Vector version of trilerp
static VectorAtlasBuilder::VoxelType
_interpolateVector(
  VectorAtlasBuilder::ImageType* img, 
  double x, double y, double z)
{
  unsigned int vectorDim = img->get(0, 0, 0).size();

  // a faster version of the floor function
  int floorX = static_cast<int>(x);
  int floorY = static_cast<int>(y);
  int floorZ = static_cast<int>(z);
  if (x < 0 && x != static_cast<int>(x)) --floorX;
  if (y < 0 && y != static_cast<int>(y)) --floorY;
  if (z < 0 && z != static_cast<int>(z)) --floorZ;

  // this is not truly ceiling, but floor + 1, which is usually ceiling
  int ceilX = floorX + 1;
  int ceilY = floorY + 1;
  int ceilZ = floorZ + 1;

  VectorAtlasBuilder::VoxelType background(vectorDim);
  background.fill(0.0);

  VectorAtlasBuilder::VoxelType v0, v1, v2, v3, v4, v5, v6, v7;

  int sizeX = img->getSizeX();
  int sizeY = img->getSizeY();
  int sizeZ = img->getSizeZ();

  if (floorX >= 0 && ceilX < sizeX &&
      floorY >= 0 && ceilY < sizeY &&
      floorZ >= 0 && ceilZ < sizeZ)
  {
    // this is the fast path
    v0 = img->get(floorX, floorY, floorZ);
    v1 = img->get(ceilX, floorY, floorZ);
    v2 = img->get(ceilX, ceilY, floorZ);
    v3 = img->get(floorX, ceilY, floorZ);
    v4 = img->get(floorX, ceilY, ceilZ);
    v5 = img->get(ceilX, ceilY, ceilZ);
    v6 = img->get(ceilX, floorY, ceilZ);
    v7 = img->get(floorX, floorY, ceilZ);
  }
  else
  {
    bool floorXIn = floorX >= 0 && floorX < sizeX;
    bool floorYIn = floorY >= 0 && floorY < sizeY;
    bool floorZIn = floorZ >= 0 && floorZ < sizeZ;

    bool ceilXIn = ceilX >= 0 && ceilX < sizeX;
    bool ceilYIn = ceilY >= 0 && ceilY < sizeY;
    bool ceilZIn = ceilZ >= 0 && ceilZ < sizeZ;

    v0 = (floorXIn && floorYIn && floorZIn)
         ? img->get(floorX, floorY, floorZ) : background;
    v1 = (ceilXIn && floorYIn && floorZIn)
         ? img->get(ceilX, floorY, floorZ)  : background;
    v2 = (ceilXIn && ceilYIn && floorZIn)
         ? img->get(ceilX, ceilY, floorZ)   : background;
    v3 = (floorXIn && ceilYIn && floorZIn)
         ? img->get(floorX, ceilY, floorZ)  : background;
    v4 = (floorXIn && ceilYIn && ceilZIn)
         ? img->get(floorX, ceilY, ceilZ)   : background;
    v5 = (ceilXIn && ceilYIn && ceilZIn)
         ? img->get(ceilX, ceilY, ceilZ)    : background;
    v6 = (ceilXIn && floorYIn && ceilZIn)
         ? img->get(ceilX, floorY, ceilZ)   : background;
   v7 = (floorXIn && floorYIn && ceilZIn)
         ? img->get(floorX, floorY, ceilZ)  : background;
  }

  const double t = x - floorX;
  const double u = y - floorY;
  const double v = z - floorZ;
  const double oneMinusT = 1.0 - t;
  const double oneMinusU = 1.0 - u;
  const double oneMinusV = 1.0 - v;

/*
  return
    oneMinusT * (oneMinusU * (v0 * oneMinusV + v7 * v)  +
                 u         * (v3 * oneMinusV + v4 * v)) +
    t         * (oneMinusU * (v1 * oneMinusV + v6 * v)  +
                 u         * (v2 * oneMinusV + v5 * v));
*/
  return
    ((v0 * oneMinusV + v7 * v) * oneMinusU +
     (v3 * oneMinusV + v4 * v) * u) * oneMinusT +
    ((v1 * oneMinusV + v6 * v) * oneMinusU +
     (v2 * oneMinusV + v5 * v) * u) * t;

}

static void
_applyVectorDeformation(
  VectorAtlasBuilder::ImageType* outimg,
  VectorAtlasBuilder::ImageType* inimg,
  VectorAtlasBuilder::VectorFieldType* hField)
{
  Vector3D<unsigned int> size = inimg->getSize();

  for (unsigned int z = 0; z < size.z; ++z)
    for (unsigned int y = 0; y < size.y; ++y)
      for (unsigned int x = 0; x < size.x; ++x)
      {
        VectorAtlasBuilder::VectorType h = hField->get(x, y, z);
        outimg->set(x, y, z, _interpolateVector(inimg, h.x, h.y, h.z));
      }

}

VectorAtlasBuilder::VectorFieldType*
VectorAtlasBuilder
::GetInverseMap(VectorFieldType* hField)
{
  typedef itk::Vector<float, 3> DisplacementType;
  typedef itk::Image<DisplacementType, 3> DeformationFieldType;

  Vector3D<unsigned int> size = hField->getSize();
  Vector3D<double> spacing = this->_AverageImagePointer->getSpacing();
  Vector3D<double> origin = this->_AverageImagePointer->getOrigin();

  DeformationFieldType::SizeType itksize;
  itksize[0] = size.x;
  itksize[1] = size.y;
  itksize[2] = size.z;

  DeformationFieldType::SpacingType itkspacing;
  itkspacing[0] = spacing.x;
  itkspacing[1] = spacing.y;
  itkspacing[2] = spacing.z;

  DeformationFieldType::PointType itkorigin;
  itkorigin[0] = origin.x;
  itkorigin[1] = origin.y;
  itkorigin[2] = origin.z;

  DeformationFieldType::RegionType region;
  region.SetSize(itksize);

  DeformationFieldType::Pointer defImg = DeformationFieldType::New();
  defImg->SetRegions(region);
  defImg->Allocate();
  defImg->SetSpacing(itkspacing);
  defImg->SetOrigin(itkorigin);

  for (unsigned int i = 0; i < size.x; i++)
    for (unsigned int j = 0; j < size.y; j++)
      for (unsigned int k = 0; k < size.z; k++)
      {
        double x = i*spacing.x + origin.x;
        double y = j*spacing.y + origin.y;
        double z = k*spacing.z + origin.z;

        Vector3D<float> h = hField->get(i, j, k);

        DisplacementType d;
        d[0] = h.x - x;
        d[1] = h.y - y;
        d[2] = h.z - z;

        DeformationFieldType::IndexType ind;
        ind[0] = i;
        ind[1] = j;
        ind[2] = k;

        defImg->SetPixel(ind, d);
      }
  
  typedef itk::IterativeInverseDeformationFieldImageFilter<
    DeformationFieldType, DeformationFieldType>
    InverterType;
  InverterType::Pointer invf = InverterType::New();
  invf->SetNumberOfIterations(10);
  invf->SetStopValue(1e-2);
  invf->SetInput(defImg);
  invf->Update();

  defImg = invf->GetOutput();

  VectorFieldType* hFieldInv = new VectorFieldType(*hField);

  for (unsigned int i = 0; i < size.x; i++)
    for (unsigned int j = 0; j < size.y; j++)
      for (unsigned int k = 0; k < size.z; k++)
      {
        double x = i*spacing.x + origin.x;
        double y = j*spacing.y + origin.y;
        double z = k*spacing.z + origin.z;

        DeformationFieldType::IndexType ind;
        ind[0] = i;
        ind[1] = j;
        ind[2] = k;

        DisplacementType d = defImg->GetPixel(ind);

        Vector3D<float> h;
        h.x = x + d[0];
        h.y = y + d[1];
        h.z = z + d[2];

        hFieldInv->set(i, j, k, h);
      }

  return hFieldInv;

}

struct ThreadInfo
{
  VectorAtlasBuilder* atlasBuilder;
  unsigned int  threadIndex;
};

VectorAtlasBuilder::
VectorAtlasBuilder()
  : 
  _IterationDataOutputStream(std::cerr)
{
  this->_NumberOfImages               = 0;
  this->_AverageImagePointer          = NULL;
  this->_NumberOfThreads              = 1;

  this->_FFTWNumberOfThreads          = 1;
  this->_FFTWMeasure                  = true;

  this->_DeltaSelectionMethod         = DELTA_USE_INDIVIDUAL;
  this->_UpdateAfterEverySubIteration = true;
  this->_ComputeInverseDeformations   = false;
  this->_MeanComputationStrategy      = 
    new ArithmeticMeanComputationStrategy<VectorAtlasBuilder::VoxelType>();

  this->_Error                        = 0;
  this->_Iteration                    = 0;
  this->_NextImageToProcess           = 0;

  pthread_mutex_init(&this->_NextImageToProcessMutex, NULL);
  pthread_mutex_init(&this->_AverageImageMutex, NULL);

  this->SetVectorMode(Euclidean);
  this->_PDFEpsilon = 1e-5;

  //this->ZeroOrderInverseOn();
  this->ZeroOrderInverseOff();
}

VectorAtlasBuilder::
~VectorAtlasBuilder()
{
// PP: BUG with itk's variablelengthvector
  this->DeleteScratchMemory();

  delete this->_MeanComputationStrategy;
}

void
VectorAtlasBuilder::
SetLogOutputStream(std::ostream& out)
{
  //this->_IterationDataOutputStream = out;
}

void
VectorAtlasBuilder::
SetNumberOfThreads(unsigned int numThreads)
{
  this->_NumberOfThreads = numThreads;
}

unsigned int 
VectorAtlasBuilder::
GetNumberOfThreads() const
{
  return this->_NumberOfThreads;
}

void
VectorAtlasBuilder::
SetFFTWNumberOfThreads(unsigned int numThreads)
{
  this->_FFTWNumberOfThreads = numThreads;
}

unsigned int 
VectorAtlasBuilder::
GetFFTWNumberOfThreads() const
{
  return this->_FFTWNumberOfThreads;
}

void 
VectorAtlasBuilder::
SetFFTWMeasure(bool b)
{
  this->_FFTWMeasure = b;
}

void 
VectorAtlasBuilder::
SetFFTWMeasureOn()
{
  this->SetFFTWMeasure(true);
}

void 
VectorAtlasBuilder::
SetFFTWMeasureOff()
{
  this->SetFFTWMeasure(false);
}

bool 
VectorAtlasBuilder::
GetFFTWMeasure() const
{
  return this->_FFTWMeasure;
}

void 
VectorAtlasBuilder::
SetUpdateAverageEverySubIterationOn()
{
  this->SetUpdateAverageEverySubIteration(true);
}

void 
VectorAtlasBuilder::
SetUpdateAverageEverySubIterationOff()
{
  this->SetUpdateAverageEverySubIteration(false);
}

void 
VectorAtlasBuilder::
SetUpdateAverageEverySubIteration(bool b)
{
  this->_UpdateAfterEverySubIteration = b;
}

bool
VectorAtlasBuilder::
GetUpdateAverageEverySubIteration() const
{
  return this->_UpdateAfterEverySubIteration;
}

void 
VectorAtlasBuilder::
SetComputeInverseDeformationsOn()
{
  this->SetComputeInverseDeformations(true);
}

void 
VectorAtlasBuilder::
SetComputeInverseDeformationsOff()
{
  this->SetComputeInverseDeformations(false);
}

void 
VectorAtlasBuilder::
SetComputeInverseDeformations(bool b)
{
  this->_ComputeInverseDeformations = b;
}

bool
VectorAtlasBuilder::
GetComputeInverseDeformations() const
{
  return this->_ComputeInverseDeformations;
}

void
VectorAtlasBuilder::
SetFluidWarpParameters(const FluidWarpParameters& fluidParams)
{
  this->_FluidWarpParameters = fluidParams;
}

FluidWarpParameters&
VectorAtlasBuilder::
GetFluidWarpParameters()
{
  return this->_FluidWarpParameters;
}

void
VectorAtlasBuilder::
SetMeanComputationStrategy(VectorAtlasBuilder::MeanComputationStrategyType* s)
{
  this->_MeanComputationStrategy = s;
}

VectorAtlasBuilder::MeanComputationStrategyType*
VectorAtlasBuilder::
GetMeanComputationStrategy() const
{
  return this->_MeanComputationStrategy;
}

void 
VectorAtlasBuilder::
SetNumberOfInputImages(unsigned int n)
{
  this->_NumberOfImages = n;

  this->_ImagePointers.resize(n, 0);
  this->_DeformationFieldPointers.resize(n, 0);
  this->_DeformationFieldInversePointers.resize(n, 0);
  this->_Delta.assign(n, 0);
}

unsigned int
VectorAtlasBuilder::
GetNumberOfInputImages() const
{
  return this->_NumberOfImages;
}

void
VectorAtlasBuilder::
SetNthInputImage(unsigned int n, ImageType* imagePointer)
{
  this->_ImagePointers[n] = imagePointer;
}

VectorAtlasBuilder::ImageType*
VectorAtlasBuilder::
GetNthInputImage(unsigned int n) const
{
  return this->_ImagePointers[n];
}

VectorAtlasBuilder::ImageType*
VectorAtlasBuilder::
GetNthDeformedImage(unsigned int n) const
{
  if (this->_DeformedImagePointers.size() <= n)
    {
      return NULL;
    }
  return this->_DeformedImagePointers[n];
}

void
VectorAtlasBuilder::
SetNthDeformationField(unsigned int n, VectorFieldType* fieldPointer)
{
  this->_DeformationFieldPointers[n] = fieldPointer;
}

VectorAtlasBuilder::VectorFieldType*
VectorAtlasBuilder::
GetNthDeformationField(unsigned int n) const
{
  return this->_DeformationFieldPointers[n];
}

void
VectorAtlasBuilder::
SetNthDeformationFieldInverse(unsigned int n, VectorFieldType* fieldPointer)
{
  this->_DeformationFieldInversePointers[n] = fieldPointer;
}

VectorAtlasBuilder::VectorFieldType*
VectorAtlasBuilder::
GetNthDeformationFieldInverse(unsigned int n) const
{
  return this->_DeformationFieldInversePointers[n];
}

void
VectorAtlasBuilder::
SetAverageImage(ImageType* imagePointer)
{
  this->_AverageImagePointer = imagePointer;
}

VectorAtlasBuilder::ImageType*
VectorAtlasBuilder::
GetAverageImage() const
{
  return this->_AverageImagePointer;
}

VectorAtlasBuilder::IterationData
VectorAtlasBuilder::
GetIterationData(unsigned int iteration) const
{
  return this->_IterationDataLog[iteration];
}

void
VectorAtlasBuilder::
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
VectorAtlasBuilder::
InitializeScratchMemory()
{
  //
  // this function assumes that CheckInput has been called without error.
  //

  //
  // resize memory holders
  this->_DeformedImagePointers.resize(this->_NumberOfImages, 0);
  this->_MaxL2Displacements.resize(this->_NumberOfImages, 0.0);
  this->_MaxVelocityL2Displacements.resize(this->_NumberOfImages, 1e-10);

  this->_ScratchVectorFieldPointers.resize(this->_NumberOfThreads, 0);
  this->_FFTWForwardPlans.resize(this->_NumberOfThreads, 0);
  this->_FFTWBackwardPlans.resize(this->_NumberOfThreads, 0);


  // use the size of the average as the standard
  Vector3D<unsigned int> imageSize = this->_AverageImagePointer->getSize();

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
  unsigned xSizeFFT = 2 * (imageSize.x / 2 + 1);

  for (unsigned int i = 0; i < this->_NumberOfThreads; ++i)
  {
    // allocate scratch vector field
    this->_ScratchVectorFieldPointers[i] = 
      new VectorFieldType(xSizeFFT, imageSize.y, imageSize.z);
  }

  for (unsigned int i = 0; i < this->_NumberOfImages; ++i)
  {
    // allocate deformed image
    this->_DeformedImagePointers[i] = new ImageType(imageSize);

    // generate the deformed image
/*
    unsigned int vectorDim = (this->_ImagePointers[i])->get(0, 0, 0).size();

    VoxelType bg(vectorDim);
    bg.Fill(0.0);

    // can only return double, can't be used
    HField3DUtils::apply(*this->_ImagePointers[i], 
                         *this->_DeformationFieldPointers[i],
                         *this->_DeformedImagePointers[i],
                         bg);
*/

    _applyVectorDeformation(
      this->_DeformedImagePointers[i],
      this->_ImagePointers[i],
      this->_DeformationFieldPointers[i]);
  }
}

void
VectorAtlasBuilder::
DeleteScratchMemory()
{

  // order is important
  this->DeleteFFTWPlans();

  for (unsigned int i = 0; i < this->_NumberOfThreads; ++i)
  {
    if (this->_ScratchVectorFieldPointers[i])
    {
      delete this->_ScratchVectorFieldPointers[i];
      this->_ScratchVectorFieldPointers[i] = 0;
    }
  }

  for (unsigned int i = 0; i < this->_NumberOfImages; ++i)
  {
    if (this->_DeformedImagePointers[i])
    {
      delete this->_DeformedImagePointers[i];
      this->_DeformedImagePointers[i] = 0;
    }
  }
}

void
VectorAtlasBuilder::
InitializeFFTWPlans()
{
 assert(this->_AverageImagePointer != NULL);

  // use the size of the average as the standard
  Vector3D<unsigned int> imageSize = this->_AverageImagePointer->getSize();  

  for (unsigned int i = 0; i < this->_NumberOfThreads; ++i)
  {
    int rank = 3;
    int logicalSizeParam[3];
    logicalSizeParam[0] = imageSize.z;
    logicalSizeParam[1] = imageSize.y;
    logicalSizeParam[2] = imageSize.x;

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
VectorAtlasBuilder::
DeleteFFTWPlans()
{
  for (unsigned int i = 0; i < this->_NumberOfThreads; ++i)
  {
    fftwf_destroy_plan(this->_FFTWForwardPlans[i]);
    fftwf_destroy_plan(this->_FFTWBackwardPlans[i]);
  }
}

void 
VectorAtlasBuilder::
InitializeOperatorLookupTable()
{
  Vector3D<unsigned int> imageSize = this->_AverageImagePointer->getSize();
  VectorAtlasBuilder::LUT lut(imageSize.x, imageSize.y, imageSize.z);

  // hardcode these for now
  double deltaX = 1.0;
  double deltaY = 1.0;
  double deltaZ = 1.0;

  //
  // precompute some values
  //
  double sX = deltaX * 2.0 * M_PI / imageSize.x; 
  double sY = deltaY * 2.0 * M_PI / imageSize.y; 
  double sZ = deltaZ * 2.0 * M_PI / imageSize.z; 

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
VectorAtlasBuilder::
GenerateAverage()
{
  //
  // start the total timer
  this->_TotalTimer.restart();

  //
  // check that input data is consistent: an exception will be thrown
  // if there is a problem
  std::cerr << "Checking inputs...";
  this->CheckInputs();
  std::cerr << "DONE" << std::endl;  

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
  // initialize memory: order is important
  std::cerr << "Initializing memory...";
  std::cerr << "Scratch memory...";
  this->InitializeScratchMemory();
  std::cerr << "FFTW Plans...";
  this->InitializeFFTWPlans();
  std::cerr << "LUT...";
  this->InitializeOperatorLookupTable();
  std::cerr << "DONE" << std::endl;

  //
  // start algorithm
  //
  this->RunAlgorithm();
}

unsigned int 
VectorAtlasBuilder::
GetJobImageIndex()
{
  pthread_mutex_lock(&this->_NextImageToProcessMutex);
  unsigned int imageIndex = this->_NextImageToProcess++;
  pthread_mutex_unlock(&this->_NextImageToProcessMutex);
  return imageIndex;
}

void 
VectorAtlasBuilder::
LockMeanImage()
{
  pthread_mutex_lock(&this->_AverageImageMutex);
}

void 
VectorAtlasBuilder::
UnlockMeanImage()
{
  pthread_mutex_unlock(&this->_AverageImageMutex);
}

pthread_t
VectorAtlasBuilder::
GetThreadID()
{
  return pthread_self();
}

void*
VectorAtlasBuilder::
ThreadedUpdateImages(void* arg)
{
  ThreadInfo* threadInfoPtr = static_cast<ThreadInfo*>(arg);
  VectorAtlasBuilder* ptr = threadInfoPtr->atlasBuilder;
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
    ptr->_MaxL2Displacements[imageIndex] = maxL2;
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

  return (void*)0;
}

void
VectorAtlasBuilder::
UpdateDelta()
{
  if (this->_DeltaSelectionMethod == DELTA_USE_MEAN)
  {
    double l2Sum = std::accumulate(this->_MaxVelocityL2Displacements.begin(),
                                   this->_MaxVelocityL2Displacements.end(),
                                   1e-10);
    double aveDisp = l2Sum / this->_NumberOfImages;
    if (aveDisp < this->_FluidWarpParameters.maxPerturbation)
      this->_Delta.assign(this->_NumberOfImages, 1.0);
    else
      this->_Delta.assign(this->_NumberOfImages, 
                          this->_FluidWarpParameters.maxPerturbation / aveDisp);
//std::cout << "Delta mean: " << this->_Delta[0] << std::endl;
  }
  else if (this->_DeltaSelectionMethod == DELTA_USE_INDIVIDUAL)
  {
    this->_Delta.resize(this->_NumberOfImages, 1.0);
//std::cout << "Delta ind: ";
    for (unsigned int i = 0; i < this->_NumberOfImages; ++i)
    {
      double maxDisp = this->_MaxVelocityL2Displacements[i];
/*
      if (maxDisp < this->_FluidWarpParameters.maxPerturbation)
      {
        this->_Delta[i] = 1.0;
      }
      else
      {
        this->_Delta[i] = this->_FluidWarpParameters.maxPerturbation / 
          maxDisp;
      }
*/
      this->_Delta[i] = this->_FluidWarpParameters.maxPerturbation / 
        maxDisp;
//std::cout << this->_Delta[i] << " ";
    }
//std::cout << std::endl;
  }
  else
  {
    std::runtime_error("Unknown delta update method.");    
  }
}

void 
VectorAtlasBuilder::
UpdateBodyForce(unsigned int imageIndex, unsigned int threadIndex)
{
  Vector3D<unsigned int> size = this->_AverageImagePointer->getSize();

  Image<VoxelType>& img = *this->_DeformedImagePointers[imageIndex];
  Vector3D<double> spacing = img.getSpacing();

  double dx = spacing.x;
  double dy = spacing.y;
  double dz = spacing.z;

  unsigned int numChannels = (img.get(0, 0, 0)).size();

  // Zero out the vectors in scratch memory before accumulating forces
  VectorType zerov;
  zerov[0] = 0;
  zerov[1] = 0;
  zerov[2] = 0;
  this->_ScratchVectorFieldPointers[threadIndex]->fill(zerov);

  for (unsigned int z = 1; z < (size.z-1); ++z) {
    for (unsigned int y = 1; y < (size.y-1); ++y) {
      for (unsigned int x = 1; x < (size.x-1); ++x) {

        std::vector<VoxelType> gradients;
        gradients.push_back((img(x+1, y, z) - img(x-1, y, z)) / (2.0*dx));
        gradients.push_back((img(x, y+1, z) - img(x, y-1, z)) / (2.0*dy));
        gradients.push_back((img(x, y, z+1) - img(x, y, z-1)) / (2.0*dz));

        VoxelType mu = (*this->_AverageImagePointer)(x, y, z);
        VoxelType v = img(x, y, z);

        for (unsigned int chan = 0; chan < numChannels; chan++)
        {
          VectorType grad_c;
          grad_c.x = gradients[0][chan];
          grad_c.y = gradients[1][chan];
          grad_c.z = gradients[2][chan];

          double di = 0.0;
          if (_VectorMode == Euclidean)
          {
            di = mu[chan] - v[chan];
          }
          else if (_VectorMode == ModelCentricPDF)
          {
            di = 2.0 - v[chan] / mu[chan]; // Approx
            //di = -mu[chan] / v[chan]; // Actual
          }
          else if (_VectorMode == PopulationCentricPDF)
          {
            double ratio = v[chan] / mu[chan];
            di = -3.0 * (ratio*ratio - (4.0/3.0)*ratio + 1.0) / 2.0; // Approx
            //di = 1.0 + log(ratio); // Actual
          }
          else
          {
            throw std::runtime_error("Unknown vector mode");
          }

          (*this->_ScratchVectorFieldPointers[threadIndex])(x,y,z) += 
             grad_c * di;
        }
      }
    }
  }    
}

void 
VectorAtlasBuilder::
UpdateVelocityField(unsigned int imageIndex, unsigned int threadIndex)
{
  Vector3D<unsigned int> logicalSize = this->_AverageImagePointer->getSize();

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
	      VectorAtlasBuilder::
                InverseOperatorMultiply(complexPtr,
                                        L00,
                                        L10, L11,
                                        L20, L21, L22);
	    }
	}
    }

  // backward fft
  fftwf_execute(this->_FFTWBackwardPlans[threadIndex]);

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
}

// TODO: we could probably rearrange this to save some memory...
void 
VectorAtlasBuilder::
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
    if (m_ZeroOrderInverse)
    {
      VectorFieldType& hIncrementalInv = oldDeformation; // reuse memory here
      HField3DUtils::computeInverseZerothOrder(hIncremental, hIncrementalInv);
    
      //
      // compute hInv(x) = hIncrementalInv(hInv(x))
      //
      VectorFieldType oldDeformationInv(*this->_DeformationFieldInversePointers[imageIndex]);
      HField3DUtils::
        compose(hIncrementalInv, 
                oldDeformationInv,
                *this->_DeformationFieldInversePointers[imageIndex]);
    }
    else
    {
      // Do Newton
      VectorFieldType* hIncrementalInv = this->GetInverseMap(&hIncremental);
    
      //
      // compute hInv(x) = hIncrementalInv(hInv(x))
      //
      VectorFieldType oldDeformationInv(*this->_DeformationFieldInversePointers[imageIndex]);
      HField3DUtils::
        compose(*hIncrementalInv, 
                oldDeformationInv,
                *this->_DeformationFieldInversePointers[imageIndex]);

      delete hIncrementalInv;
    }
  }
}

void 
VectorAtlasBuilder::
UpdateDeformedImage(unsigned int imageIndex, unsigned int threadIndex)
{
/*
  // Scalar version
  unsigned int vectorDim =
    (this->_ImagePointers[imageIndex])->get(0, 0, 0).size();

  VoxelType bg(vectorDim);
  bg.Fill(0.0);

  HField3DUtils::apply(*this->_ImagePointers[imageIndex],
                       *this->_DeformationFieldPointers[imageIndex],
                       *this->_DeformedImagePointers[imageIndex],
                       bg);
*/

  _applyVectorDeformation(
    this->_DeformedImagePointers[imageIndex],
    this->_ImagePointers[imageIndex],
    this->_DeformationFieldPointers[imageIndex]);
}

void
VectorAtlasBuilder::
RunAlgorithm()
{
  //
  // compute average image
  this->ThreadedUpdateAverageImage();

  for (unsigned int iter = 0; iter < this->_FluidWarpParameters.numIterations;
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
                              &VectorAtlasBuilder::ThreadedUpdateImages, 
                              &threadInfo[threadIndex]);
      if (rv != 0)
      {
        throw std::runtime_error("Error creating thread.");
      }
    }

    // join threads
    for (int threadIndex = 0; threadIndex < (int)this->_NumberOfThreads; 
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
}

void
VectorAtlasBuilder::
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
VectorAtlasBuilder::
ThreadedUpdateAverageImage(void* arg)
{
  ThreadInfo* threadInfoPtr = static_cast<ThreadInfo*>(arg);
  VectorAtlasBuilder* ptr = threadInfoPtr->atlasBuilder;
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

  return (void*)0;
}

void
VectorAtlasBuilder::
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
                            &VectorAtlasBuilder::ThreadedUpdateAverageImage, 
                            &threadInfo[threadIndex]);
    if (rv != 0)
    {
      throw std::runtime_error("Error creating thread.");
    }
  }
  
  // join threads
  for (int threadIndex = 0; threadIndex < (int)this->_NumberOfThreads; 
       ++threadIndex)
  {
    int rv = pthread_join(threads[threadIndex], NULL);
    if (rv != 0)
    {
      throw std::runtime_error("Error joining thread.");
    }
  }
}

void
VectorAtlasBuilder::
UpdateError()
{
  unsigned int numElements = this->_AverageImagePointer->getNumElements();

  double error = 0;
  for (unsigned int i = 0; i < this->_NumberOfImages; ++i)
    {
      for (unsigned int j = 0; j < numElements; ++j)
      {
        if (_VectorMode == ModelCentricPDF)
        {
          VoxelType p = (*this->_DeformedImagePointers[i])(j);
          VoxelType p_mean = (*this->_AverageImagePointer)(j);
          for (unsigned int chan = 0; chan < p.size(); chan++)
          {
            if (p_mean[chan] <= _PDFEpsilon)
              continue;
            error += p_mean[chan] * log(p_mean[chan] / p[chan] + _PDFEpsilon);
          }
        }
        else if (_VectorMode == PopulationCentricPDF)
        {
          VoxelType p = (*this->_DeformedImagePointers[i])(j);
          VoxelType p_mean = (*this->_AverageImagePointer)(j);
          for (unsigned int chan = 0; chan < p.size(); chan++)
          {
            if (p[chan] <= _PDFEpsilon)
              continue;
            error += p[chan] * log(p[chan] / p_mean[chan] + _PDFEpsilon);
          }
        }
        else
        {
          VoxelType diff = (*this->_DeformedImagePointers[i])(j) - 
            (*this->_AverageImagePointer)(j);
          for (unsigned int chan = 0; chan < diff.size(); chan++)
            error += diff[chan] * diff[chan];
        }
      }
    }
  error /= (this->_NumberOfImages * numElements);
  this->_Error = error;
}

void
VectorAtlasBuilder::
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
  newData.Error                          = this->_Error;
  newData.Delta                          = this->_Delta[imageNumber];
  newData.MaxL2Displacement              = 
    *std::max_element(this->_MaxL2Displacements.begin(), 
                      this->_MaxL2Displacements.end());

  double ErrorPercent  = 100.0;
  bool ValueDidImprove = true;

  if (newData.IterationNumber > 0)
  {
    double initialError = this->_IterationDataLog[0].Error;
    ErrorPercent = 100.0 * newData.Error / initialError;

    if (newData.Error > 
        this->_IterationDataLog.back().Error)
    {
      ValueDidImprove = false;
    }
  }

  this->_IterationDataLog.push_back(newData);

  this->_IterationDataOutputStream 
    << "["  << newData.IterationNumber
    << "/"  << this->_FluidWarpParameters.numIterations
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
      << " " << "D=Ind";
  }
  this->_IterationDataOutputStream 
    << " " << "MaxL2=" << newData.MaxL2Displacement
    << " " << "Error=" << newData.Error 
    << " (" << ErrorPercent << "%)";
                               
  if (!ValueDidImprove)
  {
    this->_IterationDataOutputStream 
      << " <<--<<--<<-- Error Increased! --<<--<<--<<";
  }
  this->_IterationDataOutputStream << std::endl;
}


void
VectorAtlasBuilder::
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
