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
#include <list>
#include <iostream>
#include <exception>
#include "FluidWarpParameters.h"
#include "LDMMParameters.h"
#include "DiffeoImageGrowth.h"
#include "Timer.h"
#include <itkMultiThreader.h>
#include <tclap/CmdLine.h>

/**
 * \page fGrowth
 *
 * frontend for DiffeoImageGrowth class.
 *
 * Performs structure deflation, see "Accommodating Bowel Gas in Large
 * Deformation Image Registration for Adaptive Radiation Therapy of
 * the Prostate", Davis, Prigent, Bechtel, Rosenman, Lovelock, and
 * Joshi
 */

#define appout std::cerr
const std::string PROGRAM_NAME = "fGrowth";

typedef float VoxelType;
bool writeDebugImages = false;

void printUsage()
{
  appout << "Usage: Read the code..." << std::endl;
}

int main(int argc, char **argv)
{
  Timer totalTimer;
  totalTimer.start();

  //
  // get input
  //
  
  //
  // main parameters
  std::vector<std::string> imageFilenames;
  std::vector<float> scaleLevels;

  std::vector<LDMMParameters> ldmmOptimizationParams;
  LDMMParameters defaultLDMMParams;
  defaultLDMMParams.epsilon         = 0.001;
  defaultLDMMParams.sigma           = 0.08;
  defaultLDMMParams.numIterations   = 10;

  std::vector<FluidWarpParameters> fluidParams;
  FluidWarpParameters defaultFluidParams;
  defaultFluidParams.alpha = 0.5;
  defaultFluidParams.beta  = 0.0;
  defaultFluidParams.gamma = 1;

  std::string outputHFieldFilenamePrefix;

  //
  // secondary parameters
  enum FluidOutputMode {FluidSilent, FluidStandard, FluidVerbose};
  //FluidOutputMode outputMode = FluidStandard;
  bool verbose = false;
  bool useIntensityWindow = false;
  bool incomp = false;
  float iwMin, iwMax;
  bool fftwMeasure = true;
  int  numberOfThreads = 
    itk::MultiThreader::GetGlobalDefaultNumberOfThreads();

  //
  // parse command line
  //
  try
  {
    appout << "Parsing command line arguments...";

    TCLAP::CmdLine cmd("fGrowth",' ',"0.1");
    
    TCLAP::SwitchArg
      verboseArg("v","verbose",
                 "Print extra output",
                 cmd, false);

    TCLAP::SwitchArg
      incompArg("I","incomp",
                 "Enforce incompressibility",
                 cmd, false);
    
    TCLAP::ValueArg<std::string>
      imageArg("i","images",
               "Input image file names (comma seperated)",
               true,"","<image1>,<image2>,<image3>,...", cmd);

    TCLAP::ValueArg<std::string>
      outputPrefixArg("","hprefix",
                    "Deformation file prefix",
                    true,"","prefix", cmd);

    TCLAP::ValueArg<int>
      numThreadsArg("t","threads",
                    "Number of threads for computation",
                    false,-1,"numThreads", cmd);

    TCLAP::ValueArg<float>
      windowMinArg("m","windowmin",
                      "Window Min",
                      false,0,"min",cmd);
    TCLAP::ValueArg<float>
      windowMaxArg("x","windowmax",
                      "Window Max",
                      false,1,"max",cmd);

    TCLAP::ValueArg<std::string>
      scaleArg("","scales",
                       "Downsample factor for each scale level (comma seperated)",
                       false,"4,2,1","<scale1>,...", cmd);

    TCLAP::ValueArg<std::string>
      numIterationsArg("","iterations",
                       "Number of iterations at each scale level (comma seperated)",
                       false,"100,50,25","<nIters1>,...", cmd);
    
    TCLAP::ValueArg<std::string>
      sigmaArg("","sigma",
               "Sigma for each scale level (comma seperated)",
               false,"0.08,0.08,0.08","<sigma1>,...", cmd);
    
    TCLAP::ValueArg<std::string>
      epsilonArg("","epsilon",
                 "Epsilon for each scale level (comma seperated)",
                 false,"0.001,0.001,0.001","<epsilon1>,...", cmd);

    TCLAP::ValueArg<std::string>
      alphaArg("","alpha",
               "Alpha for each scale level (comma seperated)",
               false,"0.5,0.5,0.5","<alpha1>,...", cmd);

    TCLAP::ValueArg<std::string>
      gammaArg("","gamma",
               "Gamma for each scale level (comma seperated)",
               true,"1.0,1.0,1.0","<gamma1>,...", cmd);
    
    cmd.parse(argc, argv);

    verbose = verboseArg.getValue();

    incomp = incompArg.getValue();
    defaultFluidParams.divergenceFree = incomp;

    iwMin = windowMinArg.getValue();
    iwMax = windowMaxArg.getValue();
    if (iwMin != 0.0 || iwMax != 1.0)
      useIntensityWindow = true;

    //
    // filenames
    outputHFieldFilenamePrefix = outputPrefixArg.getValue();
    {
      std::string s = imageArg.getValue();
      std::list<std::string> fileList;
      StringUtils::tokenize(s, ",", fileList);
      imageFilenames.resize(fileList.size());
      std::copy(fileList.begin(), fileList.end(), imageFilenames.begin());
    }

    //
    // choose the number of threads
    int numRequestedThreads = numThreadsArg.getValue();
    if (numRequestedThreads > 0)
    {
      numberOfThreads = numRequestedThreads;
    }

    //
    // algorithm parameters
    {
      std::stringstream ss;
      char c;
      ss << scaleArg.getValue();
      double scale;
      while (ss >> scale)
      {
        scaleLevels.push_back(scale);
        ldmmOptimizationParams.push_back(defaultLDMMParams);
        fluidParams.push_back(defaultFluidParams);

	ss >> c;
      }
    }        

    {
      std::stringstream ss;
      char c;
      ss << numIterationsArg.getValue();
      int iters;
      unsigned int scaleIndex = 0;
      while (ss >> iters)
      {
        if (scaleIndex >= fluidParams.size())
          {
            std::cerr << "Number of scales mismatched. Check arguments." << std::endl;
            return EXIT_FAILURE;
          }
        ldmmOptimizationParams[scaleIndex++].numIterations = iters;
	ss >> c;
      }
    }    

    {
      std::stringstream ss;
      char c;
      ss << sigmaArg.getValue();
      double sigma;
      unsigned int scaleIndex = 0;
      while (ss >> sigma)
      {
        if (scaleIndex >= fluidParams.size())
          {
            std::cerr << "Number of scales mismatched. Check arguments." << std::endl;
            return EXIT_FAILURE;
          }
        ldmmOptimizationParams[scaleIndex++].sigma = sigma;
	ss >> c;
      }
    }    

    {
      std::stringstream ss;
      char c;
      ss << epsilonArg.getValue();
      double epsilon;
      unsigned int scaleIndex = 0;
      while (ss >> epsilon)
      {
        if (scaleIndex >= fluidParams.size())
          {
            std::cerr << "Number of scales mismatched. Check arguments." << std::endl;
            return EXIT_FAILURE;
          }
        ldmmOptimizationParams[scaleIndex++].epsilon = epsilon;
	ss >> c;
      }
    }    

    {
      std::stringstream ss;
      char c;
      ss << alphaArg.getValue();
      double alpha;
      unsigned int scaleIndex = 0;
      while (ss >> alpha)
      {
        if (scaleIndex >= fluidParams.size())
          {
            std::cerr << "Number of scales mismatched. Check arguments." << std::endl;
            return EXIT_FAILURE;
          }
        fluidParams[scaleIndex++].alpha = alpha;
	ss >> c;
      }
    }    

    {
      std::stringstream ss;
      char c;
      ss << gammaArg.getValue();
      double gamma;
      unsigned int scaleIndex = 0;
      while (ss >> gamma)
      {
        if (scaleIndex >= fluidParams.size())
          {
            std::cerr << "Number of scales mismatched. Check arguments." << std::endl;
            return EXIT_FAILURE;
          }
        fluidParams[scaleIndex++].gamma = gamma;
	ss >> c;
      }
    }    

    appout << "DONE" << std::endl;
  }
  catch (TCLAP::ArgException &e)
  {
    std::cerr << "error: " << e.error() << " for arg " << e.argId()
              << std::endl;
    exit(1);
  }

    
  unsigned int numImages = imageFilenames.size();
  if (numImages < 1)
  {
    printUsage();
    return 0;
  }

  //
  // print parameters
  appout << PROGRAM_NAME << " parameters..." << std::endl;
  appout << "Num. Images           : " << numImages << std::endl;
  appout << "Output h Field Prefix : " 
         << (outputHFieldFilenamePrefix.length() ? 
             outputHFieldFilenamePrefix : "(none)")
         << std::endl;
  appout << "FFTW Parameters       : "
         << (fftwMeasure ? "Measure plan, " : "Don't measure plan, ")
         << "Threads=" << numberOfThreads
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
    appout << "sigma                 : " 
           << ldmmOptimizationParams[scale].sigma 
           << std::endl;
    appout << "epsilon               : " 
           << ldmmOptimizationParams[scale].epsilon 
           << std::endl;
    if (useIntensityWindow)
      {
        appout << "window                : "
           << iwMin << "-" << iwMax
           << std::endl;
      }
    appout << "Num. Iterations       : " 
           << ldmmOptimizationParams[scale].numIterations 
           << std::endl;
  }

  //
  // load images
  appout << "Loading Images..." << std::endl;
  Image<VoxelType>** images = new Image<VoxelType>*[numImages];
  for (int i = 0; i < (int) numImages; ++i)
  {
    images[i] = new Image<VoxelType>;
    ApplicationUtils::LoadImageITK(imageFilenames[i].c_str(), *images[i]);
    appout << "   Loaded: " << imageFilenames[i] << std::endl;
    appout << "   Dimensions: " << images[i]->getSize() 
           << std::endl;
    appout << "   Origin: " << images[i]->getOrigin() 
           << std::endl;
    appout << "   Spacing: " << images[i]->getSpacing() 
           << std::endl;
    VoxelType iMin, iMax;
    iMin = iMax = 0; // make compiler happy
    Array3DUtils::getMinMax(*images[i], iMin, iMax);
    float ssv = Array3DUtils::sumOfSquaredElements(*images[i]);
    appout << "   SSV: " << ssv << std::endl;
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
  // run growth algorithm at each scale level
  Image<VoxelType>** scaledImages = new Image<VoxelType>*[numImages];
  Array3D<Vector3D<float> >** h  = new Array3D<Vector3D<float> >*[numImages-1];
  Vector3D<float> origin, spacing;
  origin = images[0]->getOrigin();

  for (int scale = 0; scale < (int) scaleLevels.size(); ++scale)
  {
    appout << "Scale: " << scaleLevels[scale] << std::endl;

    //
    // create images for this scale level
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
      if (writeDebugImages)
      {
        // debug
        std::stringstream ss;
        ss << "debug/InputScaledImage" << i;
        Array3DIO::writeMETAVolume(*scaledImages[i],origin,scaledImages[i]->getSpacing(),ss.str().c_str());
      }
    }
    Vector3D<unsigned int> scaledImageSize = scaledImages[0]->getSize();
    spacing = scaledImages[0]->getSpacing();
    appout << "DONE, size = " << scaledImageSize 
           << ", spacing = " << spacing
           << std::endl;

    //
    // create h fields for this scale level
    appout << "Creating h-fields...";
    if (scale == 0)
    {
      // start with identity
      for (int i = 0; i < (int) numImages-1; ++i)
      {
        try {
          h[i] = new Array3D<Vector3D<float> >(scaledImageSize);
          h[i]->fill(Vector3D<float>(0.0,0.0,0.0));
        } catch (std::exception& e)
        {
          appout << "Error creating h-field: " << e.what() << std::endl;
          exit(0);
        }
      }
    }
    else
    {
      appout << "Upsampling...";
      // upsample old xforms
      Array3D<Vector3D<float> > tmph(h[0]->getSize());
      for (int i = 0; i < (int) numImages-1; ++i)
      {
        tmph = *h[i];
        HField3DUtils::resample(tmph, *h[i], scaledImageSize, 
                                HField3DUtils::BACKGROUND_STRATEGY_ZERO);
      }
    }
    appout << "DONE" << std::endl;

    //
    // run growth algorithm at this scale level
    if (ldmmOptimizationParams[scale].numIterations > 0)
    {
      appout << "Computing growth at this scale level..." << std::endl;
      Image<VoxelType> templateImage(*scaledImages[0]);
      DiffeoImageGrowth growthAlgorithm;
      growthAlgorithm.SetVerbose(verbose);
      growthAlgorithm.SetNumberOfInputImages(numImages);
      growthAlgorithm.SetNumberOfThreads(numberOfThreads);
      growthAlgorithm.SetFFTWNumberOfThreads(1);
      growthAlgorithm.SetFFTWMeasure(fftwMeasure);
      growthAlgorithm.SetFluidWarpParameters(fluidParams[scale]);
      growthAlgorithm.SetLDMMParameters(ldmmOptimizationParams[scale]);
      
      for (unsigned int imageIndex = 0; imageIndex < numImages; ++imageIndex)
      {
        growthAlgorithm.SetNthInputImage(imageIndex, scaledImages[imageIndex]);
      }
      for (unsigned int imageIndex = 0; imageIndex < numImages-1; ++imageIndex)
      {
        growthAlgorithm.SetNthVelocityField(imageIndex, h[imageIndex]);
      }
      
      growthAlgorithm.Run();
      appout << "DONE Computing Growth." << std::endl;
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
  // write hfields at last scale level
  if (outputHFieldFilenamePrefix != "") 
  {
    appout << "Writing H Fields...";
    for (int i = 0; i < (int) numImages-1; ++i) 
    {
      std::stringstream ss;
      ss << outputHFieldFilenamePrefix << i;
      HField3DIO::writeMETA(*h[i], origin, spacing, ss.str().c_str());
    }
    appout << "DONE" << std::endl;      
  }

  //
  // clean up memory
  //
  for (int i = 0; i < (int) numImages;++i)
  {
    delete images[i];
  }
  for (int i = 0; i < (int) numImages-1;++i)
  {
    delete h[i];
  }

  delete [] scaledImages;
  delete [] images;
  delete [] h;

  appout << "Total Time: " << totalTimer.getTime() << std::endl;


  return 0;
}


