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

#include <ctime>
#include "ApplicationUtils.h"
#include "Image.h"
#include "Vector3D.h"
#include "Array3D.h"
#include "Array3DUtils.h"
#include "ImageUtils.h"
#include "HField3DUtils.h"
#include "HField3DIO.h"
#include <string>
#include <sstream>
#include <vector>
#include <iostream>
#include <exception>
#include "FluidWarp.h"
#include "FluidWarpParameters.h"
#include "HField3DUtils.h"

#include <itkMultiThreader.h>

#include <cstring>

/**
 * \page fWarp
 *
 * Generate diffeomorphic deformation between two images using
 * 'greedy' fluid warping.  Makes calls to FluidWarp class for
 * creating deformation at each scale level.  This functionality is
 * duplicated in MultiscaleFluid, should probably be merged.
 *
 * Run 'fWarp -h' for full options
 */

#define appout std::cerr
const std::string PROGRAM_NAME = "fWarp";

typedef float VoxelType;


void printUsage()
{
  appout << "Usage: " 
         << PROGRAM_NAME << " [OPTION]... fixedImage movingImage "
         << std::endl << std::endl;
  
  appout << "Options:" 
         << std::endl
         << "  -h, --help                      show this help"
         << std::endl
         << "  -v, --verbose                   verbose output"
         << std::endl
         << "  -o, --outputImagePrefix=PREFIX  image filename prefix"
         << std::endl
         << "  -h, --outputHFieldPrefix=PREFIX hfield filename prefix"
         << std::endl
         << "      --writeInverseHFields=BOOL  default is false"
         << std::endl
         << "  -n, --intensityWindowMin=VAL    intensity window min value"
         << std::endl
         << "  -x, --intensityWindowMax=VAL    intensity window max value"
         << std::endl
         << "  -s, --scaleLevel=SCALE          add scale level (1,2,4,...)"
         << std::endl << std::endl
         << "  Algorithm parameters for the current scale level:"
         << std::endl
         << "      --alpha=VAL                 "
         << std::endl
         << "      --beta=VAL                  "
         << std::endl
         << "      --gamma=VAL                 "
         << std::endl
         << "      --maxPerturbation=VAL       "
         << std::endl
         << "  -i, --numberOfIterations=VAL    "
         << std::endl
         << "  LDMM parameters for the current scale level:"
         << std::endl
         << "      --numberOfTimeSteps=VAL     (NOT IMPLEMENTED!)"
         << std::endl
         << "      --epsilon=VAL               (NOT IMPLEMENTED!)"
         << std::endl
         << "      --sigma=VAL                 (NOT IMPLEMENTED!)"
         << std::endl << std::endl
         << "  Miscellaneous parameters:       "
         << std::endl
         << "      --fftwMeasure=BOOL          default is true"
         << std::endl
         << "      --fftwNumberOfThreads=VAL   default is number of processors"
         << std::endl
         << "      --fftwPlan=FILENAME         FFTW plan (NOT IMPLEMENTED!)"
         << std::endl
         << "  Extra output:" << std::endl 
         << "      --extraOutputStride=VAL     write atlas data every VAL iterations"
         << std::endl
         << "      --writeVolume               write atlas volume every Stride iterations"
         << std::endl
         << "      --writeXSlice=SLICE         write x slice of atlas volume every Stride iterations"
         << std::endl
         << "      --writeYSlice=SLICE         write y slice of atlas volume every Stride iterations"
         << std::endl
         << "      --writeZSlice=SLICE         write z slice of atlas volume every Stride iterations"
         << std::endl;
  appout << std::endl << std::endl;
  
  appout << "Example: " << std::endl
         << PROGRAM_NAME << "                                            \\"
         << std::endl
         << " --outputImagePrefix=deformedImage_              \\"
         << std::endl
         << " --outputHFieldPrefix=deformationField_          \\"
         << std::endl
         << " --scaleLevel=4 --numberOfIterations=50          \\"
         << std::endl
         << " --scaleLevel=2 --numberOfIterations=25          \\"
         << std::endl
         << " --scaleLevel=1 --numberOfIterations=10          \\"
         << std::endl
         << " fixed.mhd moving.mhd" 
         << std::endl 
         << std::endl
         << "This downsamples twice, runs 50 iterations, upsamples to half "
         << std::endl 
         << "resolution, runs 50 iterations, and finally upsamples to full"
         << std::endl
         << "resolution and runs 25 iterations." << std::endl;
  appout << std::endl << std::endl;
  appout << "Brad Davis, Sarang Joshi (davisb@cs.unc.edu)" << std::endl;
}



int main(int argc, char **argv)
{
  //
  // default parameters
  //
  enum FluidOutputMode {FluidSilent, FluidStandard, FluidVerbose};
  FluidOutputMode outputMode = FluidStandard;

  std::string outputImagePrefix = "";
  std::string outputHFieldPrefix = "";
  bool writeInverseHFields = false;
  std::vector<std::string> inputFilenames;

  std::vector<FluidWarpParameters> fluidParams;
  std::vector<float> scaleLevels;
  FluidWarpParameters defaultFluidParams;
  defaultFluidParams.alpha = 0.01;
  defaultFluidParams.beta  = 0.01;
  defaultFluidParams.gamma = 0.001;
  defaultFluidParams.maxPerturbation = 0.5;
  defaultFluidParams.numIterations = 50;
  defaultFluidParams.numBasis = 2000;

  bool useIntensityWindow = false;
  int extraOutputStride = -1;
  bool writeVolume = true;
  int writeXSlice = -1;
  int writeYSlice = -1;
  int writeZSlice = -1;
  float iwMin, iwMax;
  iwMin = iwMax = 0; // make compiler happy
  bool fftwMeasure = true;
  int  fftwNumberOfThreads = 
    itk::MultiThreader::GetGlobalDefaultNumberOfThreads();

  //
  // parse command line
  //
  argv++; argc--;
  while (argc > 0)
  {
    std::string arg(argv[0]);

    if (arg.find("-h")==0 || arg.find("--help")==0)
    {
      // print help and exit
      printUsage();
      exit(0);
    }
    else if (arg.find("-v")==0 || arg.find("--verbose")==0)
    {
      // set output mode to verbose
      outputMode = FluidVerbose;
      ApplicationUtils::ParseOptionVoid(argc, argv);
    }
    else if (arg.find("-s")==0 || arg.find("--scaleLevel")==0)
    {
      double newScaleLevel = 
        ApplicationUtils::ParseOptionDouble(argc, argv);
      scaleLevels.push_back(newScaleLevel);
      fluidParams.push_back(defaultFluidParams);
    }
    else if (arg.find("-o")==0 || arg.find("--outputImagePrefix")==0)
    {
      outputImagePrefix = 
        ApplicationUtils::ParseOptionString(argc, argv);
    }
    else if (arg.find("-h")==0 || arg.find("--outputHFieldPrefix")==0)
    {
      outputHFieldPrefix = 
        ApplicationUtils::ParseOptionString(argc, argv);
    }
    else if (arg.find("--writeInverseHFields")==0)
    {
      writeInverseHFields = 
        ApplicationUtils::ParseOptionBool(argc, argv);
    }
    else if (arg.find("--extraOutputStride")==0)
    {
      extraOutputStride = ApplicationUtils::ParseOptionBool(argc, argv);
    }
    else if (arg.find("--writeVolumeStride")==0)
    {
      writeVolume = ApplicationUtils::ParseOptionBool(argc, argv);
    }
    else if (arg.find("--writeXSliceStride")==0)
    {
      writeXSlice = ApplicationUtils::ParseOptionInt(argc, argv);
    }
    else if (arg.find("--writeYSliceStride")==0)
    {
      writeYSlice = ApplicationUtils::ParseOptionInt(argc, argv);
    }
    else if (arg.find("--writeZSliceStride")==0)
    {
      writeZSlice = ApplicationUtils::ParseOptionInt(argc, argv);
    }
    else if (arg.find("--alpha")==0)
    {
      double alpha = ApplicationUtils::ParseOptionDouble(argc, argv);

      if (fluidParams.size() == 0)
      {
        fluidParams.push_back(defaultFluidParams);
        scaleLevels.push_back(1.0F);
      }
      fluidParams.back().alpha = alpha;
    }
    else if (arg.find("--beta")==0)
    {
      double beta = ApplicationUtils::ParseOptionDouble(argc, argv);

      if (fluidParams.size() == 0)
      {
        fluidParams.push_back(defaultFluidParams);
        scaleLevels.push_back(1.0F);
      }
      fluidParams.back().beta = beta;
    }
    else if (arg.find("--gamma")==0)
    {
      double gamma = ApplicationUtils::ParseOptionDouble(argc, argv);

      if (fluidParams.size() == 0)
      {
        fluidParams.push_back(defaultFluidParams);
        scaleLevels.push_back(1.0F);
      }
      fluidParams.back().gamma = gamma;
    }
    else if (arg.find("--maxPerturbation")==0)
    {
      double maxPert = ApplicationUtils::ParseOptionDouble(argc, argv);

      if (fluidParams.size() == 0)
      {
        fluidParams.push_back(defaultFluidParams);
        scaleLevels.push_back(1.0F);
      }
      fluidParams.back().maxPerturbation = maxPert;
    }
    else if (arg.find("-i")==0 || arg.find("--numberOfIterations")==0)
    {
      int numIter = ApplicationUtils::ParseOptionInt(argc, argv);

      if (fluidParams.size() == 0)
      {
        fluidParams.push_back(defaultFluidParams);
        scaleLevels.push_back(1.0F);
      }
      fluidParams.back().numIterations = numIter;
    }
    else if (arg.find("--numberOfTimeSteps")==0)
    {
      std::cerr << "numberOfTimeSteps is not currently implemented!" 
                << std::endl;
      exit(0);
    }
    else if (arg.find("--epsilon")==0)
    {
      std::cerr << "epsilon is not currently implemented!" 
                << std::endl;
      exit(0);
    }
    else if (arg.find("--sigma")==0)
    {
      std::cerr << "sigma is not currently implemented!" 
                << std::endl;
      exit(0);
    }
    else if (arg.find("-n")==0 || arg.find("--intensityWindowMin")==0)
    {
      iwMin = ApplicationUtils::ParseOptionDouble(argc, argv);
      useIntensityWindow = true;
    }
    else if (arg.find("-x")==0 || arg.find("--intensityWindowMax")==0)
    {
      iwMax = ApplicationUtils::ParseOptionDouble(argc, argv);
      useIntensityWindow = true;
    }
    else if (arg.find("--fftwMeasure")==0)
    {
      fftwMeasure = ApplicationUtils::ParseOptionBool(argc, argv);
    }
    else if (arg.find("--fftwNumberOfThreads")==0)
    {
      fftwNumberOfThreads = ApplicationUtils::ParseOptionInt(argc, argv);
    }
    else if (arg.find("--fftwPlan")==0)
    {
      std::cerr << "The fftwPlan option is currently not implemented!" 
                << std::endl;
      exit(0);
    }
    else
    {
      inputFilenames.push_back(argv[0]);
      ApplicationUtils::ParseOptionVoid(argc, argv);
    }
  }

  unsigned int numImages = inputFilenames.size();
  if (numImages != 2)
  {
    printUsage();
    return 0;
  }
  if (fluidParams.size() == 0)
  {
    fluidParams.push_back(defaultFluidParams);
    scaleLevels.push_back(1.0F);
  }

  //
  // print parameters
  //
  appout << PROGRAM_NAME << " parameters:" << std::endl;
  appout << "Output Mode           : ";
  switch(outputMode)
  {
  case (FluidVerbose):
    appout << "Verbose" << std::endl;
    break;
  case (FluidStandard):
    appout << "Standard" << std::endl;
    break;
  case (FluidSilent):
    appout << "Silent" << std::endl;
    break;
  default:
    appout << "Unknown" << std::endl;
    printUsage();
    return(0);
  }      
  appout << "Output Image Prefix   : " 
         << (outputImagePrefix.length() ? 
             outputImagePrefix : "(none)")
         << std::endl;
  appout << "Output h Field Prefix : " 
         << (outputHFieldPrefix.length() ? 
             outputHFieldPrefix : "(none)")
         << std::endl;
  appout << "Write inverse h fields: " 
         << (writeInverseHFields ? "true" : "false")
         << std::endl;
  appout << "Intensity Window      : ";
  if (useIntensityWindow)
  {
    appout << iwMin << "-" << iwMax << std::endl;
  }
  else
  {
    appout << "No Intensity Windowing" << std::endl;
  }
  appout << "FFTW Parameters       : "
         << (fftwMeasure ? "Measure plan, " : "Don't measure plan, ")
         << "Threads=" << fftwNumberOfThreads
         << std::endl;

  for (int scale = 0; scale < (int) scaleLevels.size(); ++scale)
  {
    appout << "Scale                 : " 
           << scaleLevels[scale] 
           << std::endl;
    appout << "alpha                 : " 
           << fluidParams[scale].alpha 
           << std::endl;
    appout << "beta                  : " 
           << fluidParams[scale].beta 
           << std::endl;
    appout << "gamma                 : " 
           << fluidParams[scale].gamma 
           << std::endl;
    appout << "Max. Pert.            : " 
           << fluidParams[scale].maxPerturbation 
           << std::endl;
    appout << "Num. Iterations       : " 
           << fluidParams[scale].numIterations 
           << std::endl;
  }

  //
  // load images
  //
  appout << "Loading Images..." << std::endl;
  Image<VoxelType>** images = new Image<VoxelType>*[numImages];
  for (int i = 0; i < (int) numImages; ++i)
  {
    images[i] = new Image<VoxelType>;
    ApplicationUtils::LoadImageITK(inputFilenames[i].c_str(), *images[i]);
    appout << "   Loaded: " << inputFilenames[i] << std::endl;
    appout << "   Dimensions: " << images[i]->getSize() 
           << std::endl;
    appout << "   Origin: " << images[i]->getOrigin() 
           << std::endl;
    appout << "   Spacing: " << images[i]->getSpacing() 
           << std::endl;
    VoxelType iMin, iMax;
    iMin = iMax = 0; // make compiler happy
    Array3DUtils::getMinMax(*images[i], iMin, iMax);
    appout << "   Intensity Range: " << iMin << "-" << iMax << std::endl;
    if (!useIntensityWindow)
    {
      iwMin = iMin;
      iwMax = iMax;
    }
    appout << "   Rescaling to [0,1]...";
    Array3DUtils::rescaleElements(*images[i], iwMin, iwMax, 0.0F, 1.0F);
    appout << "DONE" << std::endl;
    Array3DUtils::getMinMax(*images[i], iMin, iMax);
    appout << "   Intensity Range: " << iMin << "-" << iMax << std::endl;
  }  

  //
  // setup FluidWarp class
  //
  FluidWarp fluidWarper;
  fluidWarper.setFFTWMeasure(fftwMeasure);
  fluidWarper.setFFTWNumberOfThreads(fftwNumberOfThreads);

  switch (outputMode)
  {
  case (FluidVerbose):
    fluidWarper.setOutputMode(FluidWarp::FW_OUTPUT_MODE_VERBOSE);
    break;
  default:
    fluidWarper.setOutputMode(FluidWarp::FW_OUTPUT_MODE_NORMAL);      
  }
  if (extraOutputStride > 0) {
    fluidWarper.setFilePrefix(outputImagePrefix.c_str());
    fluidWarper.setWritePerIter(extraOutputStride);
    fluidWarper.setWriteDeformedImageFiles(true);
    if (writeVolume) {
      fluidWarper.setWriteVolumes(true);
    }
    if (writeXSlice >= 0) {
      fluidWarper.setWriteXSlices(true);
      fluidWarper.setXSlice(writeXSlice);
    }
    if (writeYSlice >= 0) {
      fluidWarper.setWriteYSlices(true);
      fluidWarper.setYSlice(writeYSlice);
    }
    if (writeZSlice >= 0) {
      fluidWarper.setWriteZSlices(true);
      fluidWarper.setZSlice(writeZSlice);
    }
  }
  
  //
  // deform image at each scale level
  //
  Image<VoxelType>** scaledImages = new Image<VoxelType>*[numImages];
  Array3D<Vector3D<float> >** h    = new Array3D<Vector3D<float> >*[numImages];
  Array3D<Vector3D<float> >** hinv = 0;
  if (writeInverseHFields)
  {
    hinv = new Array3D<Vector3D<float> >*[numImages];  
  }
  Vector3D<float> origin, spacing;
  origin = images[0]->getOrigin();
  for (int scale = 0; scale < (int) scaleLevels.size(); ++scale)
  {
    appout << "Scale: " << scaleLevels[scale] << std::endl;

    //
    // create images for this scale level
    //
    appout << "Downsampling Images...";
    int f = (int) scaleLevels[scale];
    if (f == 1) {
      appout << "Using Actual Images..." << std::endl;
    }
    else {
      appout << "sigma=" << f << "...";
    }
    for (int i = 0; i < (int) numImages; ++i)
    {
      if (f == 1) {
        scaledImages[i] = new Image<VoxelType>(*images[i]);
      }
      else {
        scaledImages[i] = new Image<VoxelType>;
        ImageUtils::gaussianDownsample(*images[i],
                                       *scaledImages[i],
                                       Vector3D<int>(f, f, f),
                                       Vector3D<double>(f, f, f),
                                       Vector3D<int>(2*f, 2*f, 2*f));
      }
    }
    Vector3D<unsigned int> scaledImageSize = scaledImages[0]->getSize();
    spacing = scaledImages[0]->getSpacing();
    appout << "DONE, size = " << scaledImageSize 
           << ", spacing = " << spacing
           << std::endl;

    //
    // create h fields for this scale level
    //
    appout << "Creating h-fields...";
    if (scale == 0)
    {
      // start with identity
      try {
        h[0] = new Array3D<Vector3D<float> >(scaledImageSize);
        HField3DUtils::setToIdentity(*h[0]);

        if (hinv)
        {
          hinv[0] = new Array3D<Vector3D<float> >(scaledImageSize);
          HField3DUtils::setToIdentity(*hinv[0]);            
        }
      } catch (std::exception& e)
      {
        appout << "Error creating h-field: " << e.what() << std::endl;
        exit(0);
      }
    }
    else
    {
      appout << "Upsampling...";
      // upsample old xforms
      Array3D<Vector3D<float> > tmph(h[0]->getSize());
      tmph = *h[0];
      HField3DUtils::resample(tmph, *h[0], scaledImageSize);

      if (hinv)
      {
        tmph = *hinv[0];
        HField3DUtils::resample(tmph, *hinv[0], scaledImageSize);          
      }
    }
    appout << "DONE" << std::endl;

    //
    // create deformed image at this scale level
    //
    appout << "Computing deformation at this scale level..." << std::endl;
    Image<VoxelType> deformedImage(*scaledImages[0]);
    fluidWarper.setImageSpacing(deformedImage.getSpacing());
    if (writeXSlice >= 0) fluidWarper.setXSlice(writeXSlice/f);
    if (writeYSlice >= 0) fluidWarper.setYSlice(writeYSlice/f);
    if (writeZSlice >= 0) fluidWarper.setZSlice(writeZSlice/f);

    if (hinv)
    {
      fluidWarper.
        computeHFieldAsymmetric(*scaledImages[0],
                                *scaledImages[1],
                                fluidParams[scale], 
                                *h[0], *hinv[0]);      
    }
    else
    {
      fluidWarper.
        computeHFieldAsymmetric(*scaledImages[0],
                                *scaledImages[1],
                                fluidParams[scale], 
                                *h[0]);      
    }
    appout << "DONE Computing deformation." << std::endl;

    //
    // write deformed image
    //
    if (outputImagePrefix != "")
    {
      HField3DUtils::apply(*scaledImages[1],
                           *h[0],
                           deformedImage);
      std::ostringstream oss;
      oss << outputImagePrefix << scale;
      appout << "Writing Deformed Image...";
      ImageUtils::writeMETA(deformedImage, oss.str().c_str());
      appout << "DONE" << std::endl;
    }

    //
    // delete scaled images
    //
    for (int i = 0; i < (int) numImages; ++i)
    {
      delete scaledImages[i];
    }          
  }

  //
  // write h field at last scale level
  //
  if (outputHFieldPrefix != "") 
  {
    std::stringstream ss;
    ss << outputHFieldPrefix;
    appout << "Writing H Field...";
    HField3DIO::writeMETA(*h[0], origin, spacing, ss.str().c_str());
    appout << "DONE" << std::endl;      

    if (writeInverseHFields) 
    {
      std::stringstream ss;
      ss << outputHFieldPrefix << "_inv";
      appout << "Writing inverse H Field...";
      HField3DIO::writeMETA(*hinv[0], origin, spacing, ss.str().c_str());
      appout << "DONE" << std::endl;      
    }    
  }

  //
  // clean up memory
  //
  for (int i = 0; i < (int) numImages;++i)
  {
    delete images[i];
  }
  delete h[0];
  delete [] h;
  if (hinv)
  {
    delete hinv[0];
    delete [] hinv;
  }
  delete [] scaledImages;
  delete [] images;
  return 0;
}


