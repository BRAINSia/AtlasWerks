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
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <list>
#include <iostream>
#include <exception>
#include "FluidWarp.h"
#include "FluidWarpParameters.h"
#include "LDMMParameters.h"
#include "DiffeoSpaceTimeCT.h"
#include "Timer.h"
#include <itkMultiThreader.h>
#include <tclap/CmdLine.h>
//#include "DownsampleFilter2D.h"

#include "ImageIO.h"
#include "HField3DUtils.h"


#define appout std::cerr
const std::string PROGRAM_NAME = "spacetimect";
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

  std::string outputPrefix;
  std::string baseImgFile,inputDir,indexFileName,initHFieldDir;
  //
  // secondary parameters
  // What are these? --JDH
  //enum FluidOutputMode {FluidSilent, FluidStandard, FluidVerbose};
  //FluidOutputMode outputMode = FluidStandard;
  bool verbose = false;
  bool useIntensityWindow = false;
  bool useAmpWindow = true;
  int numTimeSteps = 1;
  bool incomp = false;
  bool  givenBaseImage = false;
  unsigned int estimateImg, outputEveryNIter;
  unsigned int padXY, padZ, minX, maxX, minY, maxY;
  float iwMin, iwMax,minAmp,maxAmp,baseAmp, deltaAmp,deltaX,deltaY;
  bool fftwMeasure = true;
  int  numberOfThreads = 
    itk::MultiThreader::GetGlobalDefaultNumberOfThreads();

  //
  // parse command line
  //
  try
  {
    appout << "Parsing command line arguments...";

    TCLAP::CmdLine cmd("spacetimect",' ',"0.1");

    
    TCLAP::SwitchArg
      verboseArg("v","verbose",
                 "Print extra output",
                 cmd, false);

    TCLAP::SwitchArg
      incompArg("I","incomp",
                "Enforce incompressibility",
                cmd, false);
    
    // the following are a departure from fGrowth
    TCLAP::ValueArg<std::string>
      inputDirArg("i","inputdir",
                  "Input directory holding index.csv and ????.raw files",
                  true,"","dir", cmd);
    TCLAP::ValueArg<std::string>
      initHFieldDirArg("","initialhfielddir",
                         "Directory holding initialization of HFields (must be same scale, etc)",
                         false,"","hfielddir",cmd);
    TCLAP::ValueArg<int>
      numStepsArg("n","discretization",
                  "Number of time discretization steps",
                  false,0,"num",cmd);
    TCLAP::ValueArg<float>
      baseAmpArg("a","baseamplitude",
                 "Base amplitude",
                 true,0,"amp",cmd);
    TCLAP::ValueArg<std::string>
      baseImgArg("b","baseimg",
                 "Base image filename",
                 false,"","img",cmd);

    TCLAP::ValueArg<unsigned int>
      estImgArg("e","estimateimg",
                "Estimate image every n iterations (0 for no estimate)",
                true,1,"n",cmd);

    TCLAP::ValueArg<float>
      deltaAmpArg("d","deltaamp",
                 "Amplitude increment",
                 false,0,"deltaamp",cmd);

    TCLAP::ValueArg<float>
      minAmpArg("","minamplitude",
                "Minimum amplitude (ignore data below this)",
                false,1,"minamp",cmd); // default is greater than max so will use all data
    TCLAP::ValueArg<float>
      maxAmpArg("","maxamplitude",
                "Maximum amplitude (ignore data above this)",
                false,0,"maxamp",cmd);

    // TODO: We really should just keep track of deltaX and deltaY
    // when we set up forRecon, instead of having to set it manually
    TCLAP::ValueArg<float>
      deltaXArg("","deltax",
                 "Spacing in X direction",
                 true,1,"dx",cmd);
    TCLAP::ValueArg<float>
      deltaYArg("","deltay",
                 "Spacing in Y direction",
                 true,1,"dy",cmd);

    TCLAP::ValueArg<unsigned int>
      padXYArg("","padxy",
                "Padding in X and Y directions",
                 false,0,"n",cmd);
    TCLAP::ValueArg<unsigned int>
      padZArg("","padz",
                "Padding in Z direction",
                 false,0,"n",cmd);

    TCLAP::ValueArg<unsigned int>
      minXArg("","minx",
                "Minimum X value for cropping",
                 false,0,"n",cmd);
    TCLAP::ValueArg<unsigned int>
      maxXArg("","maxx",
                "Maximum X value for cropping",
                 false,0,"n",cmd);
    TCLAP::ValueArg<unsigned int>
      minYArg("","miny",
                "Minimum Y value for cropping",
                 false,0,"n",cmd);
    TCLAP::ValueArg<unsigned int>
      maxYArg("","maxy",
                "Maximum Y value for cropping",
                 false,0,"n",cmd);
    
    TCLAP::ValueArg<std::string>
      outputPrefixArg("","outputprefix",
                      "Output all images, etc. to this location",
                      true,"","prefix", cmd);

    TCLAP::ValueArg<unsigned int>
      outputEveryNIterArg("","outputeveryniter",
                      "Output stuff every n iterations (and also at the end always)",
                      false,0,"n", cmd);

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
      betaArg("","beta",
               "Beta for each scale level (comma seperated)",
               false,"0.1,0.1,0.1","<beta1>,...", cmd);

    TCLAP::ValueArg<std::string>
      gammaArg("","gamma",
               "Gamma for each scale level (comma seperated)",
               true,"1.0,1.0,1.0","<gamma1>,...", cmd);
    
    cmd.parse(argc, argv);

    estimateImg = estImgArg.getValue();
    baseImgFile = baseImgArg.getValue();
    if (baseImgFile == "" && estimateImg == 0)
      {
      appout << "ERROR: Must either give base image or request image to be estimated" << std::endl;
      return EXIT_FAILURE;
      }

    deltaX = deltaXArg.getValue();
    deltaY = deltaYArg.getValue();

    padXY = padXYArg.getValue();
    padZ = padZArg.getValue();
    minX = minXArg.getValue();
    maxX = maxXArg.getValue();
    minY = minYArg.getValue();
    maxY = maxYArg.getValue();

    outputEveryNIter = outputEveryNIterArg.getValue();

    verbose = verboseArg.getValue();

    initHFieldDir = initHFieldDirArg.getValue();

    incomp = incompArg.getValue();

    defaultFluidParams.divergenceFree = incomp;
    deltaAmp = deltaAmpArg.getValue();
    minAmp = minAmpArg.getValue();
    maxAmp = maxAmpArg.getValue(); // we'll check the sanity of these later...
    if (minAmp > maxAmp) useAmpWindow = false;


    numTimeSteps = numStepsArg.getValue();

    iwMin = windowMinArg.getValue();
    iwMax = windowMaxArg.getValue();
    if (iwMin != 0.0 || iwMax != 1.0)
      {
        appout << "Using intensity window" << std::endl;
        useIntensityWindow = true;
      }
    else
      {
        appout << "NOT using intensity window" << std::endl;
        useIntensityWindow = false;
      }

    //
    // filenames
    outputPrefix = outputPrefixArg.getValue();

    //{
    // std::string s = imageArg.getValue();
    //       std::list<std::string> fileList;
    //       StringUtils::tokenize(s, ",", fileList);
    //       imageFilenames.resize(fileList.size());
    //       std::copy(fileList.begin(), fileList.end(), imageFilenames.begin());
      
    //}
    inputDir = inputDirArg.getValue(); // only have the input directory itself
    indexFileName = inputDir + "/index.csv";
    baseAmp = baseAmpArg.getValue();
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
      ss << betaArg.getValue();
      double beta;
      unsigned int scaleIndex = 0;
      while (ss >> beta)
        {
          if (scaleIndex >= fluidParams.size())
            {
              std::cerr << "Number of scales mismatched. Check arguments." << std::endl;
              return EXIT_FAILURE;
            }
          fluidParams[scaleIndex++].beta = beta;
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

  

  //
  // print parameters
  appout << PROGRAM_NAME << " parameters..." << std::endl;
  //  appout << "Num. Time Steps           : " << numTimeSteps << std::endl;
  if (minAmp < maxAmp) appout << "Min/Max Amplitudes        : " << minAmp << "-" << maxAmp << std::endl;
  appout << "Output Prefix : " 
         << (outputPrefix.length() ? 
             outputPrefix : "(none)")
         << std::endl;
  appout << "FFTW Parameters       : "
         << (fftwMeasure ? "Measure plan, " : "Don't measure plan, ")
         << "Threads=" << numberOfThreads
         << std::endl;

      // Need to decide how many steps are forward, how many backward, start and end times
      // First let's see if our amplitude choices make any sense at all
      // Scan through index.csv to find min and max in file
      float minDataAmp = 1e500;
      float maxDataAmp = -1e500;
      float lineAmp;
      float zmin = 1e500;
      float zmax = -1e500;
      float zmin2 = 1e500;
      float zspacing;
      unsigned int numForwardTimeSteps, numReverseTimeSteps;
      std::string line,ampstr,zstr;
      std::string firstslicestr("");
      std::ifstream infile(indexFileName.c_str());
      while (getline(infile, line, '\n'))
        {
          // get amplitude here
          //lineAmp = row 4
          //phase = row 5 
          // also get Z max min and spacing
          std::string::size_type firstcomma = line.find(',');
          if (firstcomma == std::string::npos) continue;
          std::string::size_type lastcomma = line.find_last_of(',');
          std::string::size_type prevcomma = (line.substr(0,lastcomma-1).find_last_of(','));
          std::string::size_type prevprevcomma = (line.substr(0,prevcomma-1).find_last_of(','));
          // Format of line is NUM,DICOMFILE,Z,AMP,PHASE

          if (firstslicestr == "") firstslicestr = line.substr(0,firstcomma);
          ampstr   = line.substr(prevcomma+1,lastcomma-prevcomma-1);
          lineAmp = atof(ampstr.c_str());

          zstr = line.substr(prevprevcomma+1,prevcomma-prevprevcomma-1);
          float ztmp = atof(zstr.c_str());

          if (ztmp > zmax) zmax = ztmp;
          if (ztmp < zmin)
            {
              zmin2 = zmin;
              zmin = ztmp;
            }
          if (ztmp < zmin2 && ztmp > zmin) zmin2 = ztmp;
            
          if (lineAmp < minDataAmp) minDataAmp = lineAmp;
          if (lineAmp > maxDataAmp) maxDataAmp = lineAmp;

          //TODO: Movecodefrombelow up to here. Need min/max Z and spacing
        }

      Image<VoxelType> *baseImg = new Image<VoxelType>;
      Vector3D<unsigned int> baseImgSize;
      Vector2D<unsigned int> baseImgSliceSize;

      if (baseImgFile != "")
        {
          //
          // load base image
          appout << "Loading Base Image...";
          givenBaseImage = true;

          Image<VoxelType> *wholeImg = new Image<VoxelType>;

          ApplicationUtils::LoadImageITK(baseImgFile.c_str(), *wholeImg);
          Vector3D<double> wholeOrigin = wholeImg->getOrigin();
          Vector3D<double> wholeSpacing = wholeImg->getSpacing();
          unsigned int xdim,ydim,zdim;
          if (maxX > 0 || maxY > 0)
            { // Cropping
            if (maxX-minX+1 > wholeImg->getSizeX() || maxY-minY+1 > wholeImg->getSizeY())
              {
              std::cerr << "ERROR: Attempt to crop to size larger than base image size" << std::endl;
              return EXIT_FAILURE;
              }
            // Pad in XY directions
            xdim = maxX - minX + 1 + 2*padXY;
            ydim = maxY - minY + 1 + 2*padXY;
            
            // Pad in Z dimension as well
            // TODO: Crop in Z dimension
            zdim = wholeImg->getSizeZ() + 2*padZ;

            baseImg->resize(xdim, ydim, zdim);
            baseImg->setOrigin(wholeOrigin.x + wholeSpacing.x*(double)(minX-padXY),
                               wholeOrigin.y + wholeSpacing.y*(double)(maxX-padXY),
                               wholeOrigin.z + wholeSpacing.z*(-(double)padZ));
            baseImg->setSpacing(wholeSpacing);
            baseImg->fill(0.0f);

            for (unsigned int z=padZ; z < zdim-padZ; ++z)
              for (unsigned int y=padXY; y < ydim-padXY; ++y)
                for (unsigned int x=padXY; x < xdim-padXY; ++x)
                  baseImg->set(x,y,z,
                               wholeImg->get(x+minX-padXY,
                                             y+minY-padXY,
                                             z-padZ));

            delete wholeImg;
            }
          else
            { // No cropping (also don't pad)           
            delete baseImg;
            baseImg = wholeImg; // steal pointer
            }

          if (useIntensityWindow)
            {
              appout << "Rescaling to [0,1]...";
              Array3DUtils::rescaleElements(*baseImg, iwMin, iwMax, 0.0F, 1.0F);
            }
          appout << "DONE" << std::endl;
        }
      else
        {
          givenBaseImage = false;
          // have to build our own base image
          // TODO: Set base image size and spacing, and slice size
          zspacing = zmin2 - zmin;

          baseImg->setSpacing(deltaX,deltaY,zspacing);
          std::cout << "Spacing is set to " << baseImg->getSpacingX() << "," << baseImg->getSpacingY() << "," << baseImg->getSpacingZ() << std::endl; 

          // Determine x,y sizes
          std::ifstream firstslice((inputDir + "/" + firstslicestr + ".raw").c_str());
          if (!firstslice)
            {
            std::cerr << "Could not open first slice file: " << inputDir << "/" << firstslicestr << ".raw" << std::endl;
            return EXIT_FAILURE;
            }
          unsigned int begin = firstslice.tellg();
          firstslice.seekg(0, std::ios::end);
          unsigned int end = firstslice.tellg();
          firstslice.close();

          unsigned int numbytes = end - begin;
          float numpix = static_cast<float>(numbytes)/2.0;
          // Pixel values are 2 byte floats
          unsigned int dim = static_cast<unsigned int>(sqrt(numpix)+0.5);

          unsigned int xdim=dim, ydim=dim;
          if (maxX > 0 || maxY > 0)
            { // Cropping
            if (maxX-minX > dim || maxY-minY > dim)
              {
              std::cerr << "ERROR: Attempt to crop to size larger than slice size" << std::endl;
              return EXIT_FAILURE;
              }
            // Pad in XY directions
            xdim = maxX - minX + 1;
            ydim = maxY - minY + 1;
            }
          // Padding
          xdim += 2*padXY;
          ydim += 2*padXY;
          // Pad in Z dimension as well
          // TODO: Crop in Z dimension
          unsigned int zdim = static_cast<unsigned int>((zmax - zmin) / zspacing+0.5) + 1 + 2*padZ;

          baseImg->resize(xdim, ydim, zdim);
          baseImg->setOrigin(0,0,zmin - zspacing*padZ);
        }

      baseImgSize = baseImg->getSize();
      baseImgSliceSize = Vector2D<unsigned int>(baseImgSize[0], baseImgSize[1]);
     
      if (minAmp < minDataAmp || !useAmpWindow)
        { // makes no sense, just use data min
          appout << " Using data min amplitude: " << minDataAmp << std::endl;
          minAmp = minDataAmp;
        }
      if (maxAmp > maxDataAmp || !useAmpWindow)
        { // makes no sense, just use data max
          appout << " Using data max amplitude: " << maxDataAmp << std::endl;
          maxAmp = maxDataAmp;
        }
      if (baseAmp < minAmp || baseAmp > maxAmp)
        {
          appout << "Base amplitude (" << baseAmp << ") must be within minAmp-maxAmp (" << minAmp << ":" << maxAmp << ") range." << std::endl;
          return EXIT_FAILURE;
        }

      if (!deltaAmp) deltaAmp = (maxAmp - minAmp)/(numTimeSteps);

      if ((baseAmp - minAmp)/(maxAmp-minAmp) < 1e-2) numReverseTimeSteps = 0;
      else numReverseTimeSteps = (int)ceil((baseAmp - minAmp)/deltaAmp);
      if ((maxAmp - baseAmp)/(maxAmp-minAmp) < 1e-2) numForwardTimeSteps = 0;
      else numForwardTimeSteps = (int)ceil((maxAmp - baseAmp)/deltaAmp);

      Array3D<Vector3D<float> >** vforward = new Array3D<Vector3D<float> >*[numForwardTimeSteps];
      Array3D<Vector3D<float> >** vreverse = new Array3D<Vector3D<float> >*[numReverseTimeSteps];
      Vector3D<float> origin, spacing;

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
          appout << "incompressible        : " 
                 << (fluidParams[scale].divergenceFree ? "Yes" : "No")
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

          origin = baseImg->getOrigin();
          spacing = baseImg->getSpacing();

          appout << "Discretization Scheme..." << std::endl;
          appout << "  Min-Max Amplitudes    : " << minAmp << "," << maxAmp << std::endl;
          appout << "  Base Amplitude        : " << baseAmp << std::endl;
          appout << "  # Reverse Steps       : " << numReverseTimeSteps << std::endl;
          appout << "  # Forward Steps       : " << numForwardTimeSteps << std::endl;
          appout << "  deltaAmp              : " << deltaAmp << std::endl;

          // run growth algorithm at this scale level
          if (ldmmOptimizationParams[scale].numIterations > 0)
            {
              DiffeoSpaceTimeCT spaceTimeCT;
              // TODO: this is dumb, we waste memory to keep one image in here if scaleLevels[scale]==1.0
              Image<VoxelType> scaledBaseImg;

              Vector3D<unsigned int> scaledImageSize = baseImg->getSize();
              if (scaleLevels[scale] != 1.0)
                {
                  // Need to scale base image (ONLY IN X-Y)
                  int f = (int) scaleLevels[scale];
                  ImageUtils::gaussianDownsample(*baseImg,
                                                 scaledBaseImg,
                                                 Vector3D<int>(f, f, 1),
                                                 Vector3D<double>(f, f, 1),
                                                 Vector3D<int>(2*f, 2*f, 1)); // TODO: Should last kernelsize be zero??
                  scaledImageSize = scaledBaseImg.getSize();
                  spaceTimeCT.SetBaseImage(&scaledBaseImg);
                }
              else
                {
                  spaceTimeCT.SetBaseImage(baseImg);         
                }
              spaceTimeCT.SetBaseSliceSize(baseImgSliceSize);

            
              //
              // create h fields for this scale level
              appout << "Creating h-fields...";


              if (scale == 0)
                {
                  if (initHFieldDir != "")
                    { // Given hfields to start with
                      appout << "Loading Initial Hfields...";
                      for (unsigned int i = 0; i < numForwardTimeSteps; ++i)
                        {
                          std::stringstream ss;
                          ss << initHFieldDir << "/v" << i << "fwd.mhd";
                          vforward[i] = new Array3D< Vector3D<float> >(scaledImageSize);
                          HField3DIO::readMETA(*vforward[i], ss.str().c_str());
                        }
                      for (unsigned int i = 0; i < numReverseTimeSteps; ++i)
                        {
                          std::stringstream ss;
                          ss << initHFieldDir << "/v" << i << "rev.mhd";
                          vreverse[i] = new Array3D< Vector3D<float> >(scaledImageSize);
                          HField3DIO::readMETA(*vreverse[i], ss.str().c_str());
                        }
                      appout << "DONE" << std::endl;
                    }
                  else
                    {
                      appout << "Initializing Hfields...";
                      // start with identity
                      for (unsigned int i = 0; i < numForwardTimeSteps; ++i)
                        {
                          try {
                            vforward[i] = new Array3D<Vector3D<float> >(scaledImageSize);
                            vforward[i]->fill(Vector3D<float>(0.0,0.0,0.0));
                          } catch (std::exception& e)
                            {
                              appout << "Error creating h-field: " << e.what() << std::endl;
                              exit(0);
                            }
                        }
                      for (unsigned int i = 0; i < numReverseTimeSteps; ++i)
                        {
                          try {
                            vreverse[i] = new Array3D<Vector3D<float> >(scaledImageSize);
                            vreverse[i]->fill(Vector3D<float>(0.0,0.0,0.0));
                          } catch (std::exception& e)
                            {
                              appout << "Error creating h-field: " << e.what() << std::endl;
                              exit(0);
                            }
                        }
                      appout << "DONE" << std::endl;
                    }
                }
              else
                {
                  appout << "Upsampling...";
                  // upsample old xforms
                  Array3D<Vector3D<float> > tmph(vforward[0]->getSize());
                  for (int i = 0; i < (int) numForwardTimeSteps; ++i)
                    {
                      tmph = *vforward[i];
                      HField3DUtils::resample(tmph, *vforward[i], scaledImageSize, 
                                              HField3DUtils::BACKGROUND_STRATEGY_ZERO);
                    }
                  for (int i = 0; i < (int) numReverseTimeSteps; ++i)
                    {
                      tmph = *vreverse[i];
                      HField3DUtils::resample(tmph, *vreverse[i], scaledImageSize, 
                                              HField3DUtils::BACKGROUND_STRATEGY_ZERO);
                    }
                }
              appout << "DONE" << std::endl;

              appout << "Computing growth at this scale level..." << std::endl;
              //Image<VoxelType> templateImage(*scaledImages[0]);
              spaceTimeCT.SetVerbose(verbose);
              //spaceTimeCT.SetNumberOfInputImages(numImages);
              spaceTimeCT.SetNumberOfThreads(numberOfThreads);
              spaceTimeCT.SetFFTWNumberOfThreads(1);
              spaceTimeCT.SetFFTWMeasure(fftwMeasure);
              spaceTimeCT.SetFluidWarpParameters(fluidParams[scale]);
              spaceTimeCT.SetLDMMParameters(ldmmOptimizationParams[scale]);
      
              // spacetime specific parameters follow...
              spaceTimeCT.SetNumTimeStepsFwd(numForwardTimeSteps); // handled numsteps in a wierd way
              spaceTimeCT.SetNumTimeStepsRev(numReverseTimeSteps);

              spaceTimeCT.SetBaseAmplitude(baseAmp);
              spaceTimeCT.SetDeltaAmp(deltaAmp);
              spaceTimeCT.SetOutputDir(outputPrefix);
              spaceTimeCT.SetEstimateImg(estimateImg);
              spaceTimeCT.SetOutputEveryNIter(outputEveryNIter);
              
              spaceTimeCT.SetUseIntensityWindow(useIntensityWindow);
              spaceTimeCT.SetIWMin(iwMin);
              spaceTimeCT.SetIWMax(iwMax);

              spaceTimeCT.SetCroppingPadding(minX,maxX,minY,maxY,padXY,padZ);
                

              //
              // Read in and downsample data (redoing this at each scale level to save memory)
              std::vector<double> z,amp;
              std::vector<std::string> sliceFiles;
              unsigned int numSlices = 0;
              //float zmin = 1e500;
              //float zmax = -1e500;
              //float zspacing, zmin2;

              appout << "Reading index.csv..." << std::endl;
              std::ifstream indexFile(indexFileName.c_str());
              if (!indexFile.is_open())
                {
                  appout << "ERROR: Couldn't open index file " << indexFileName << std::endl;
                  return EXIT_FAILURE;
                }
              std::string line,slicestr,ampstr, zstr;
              while (!indexFile.eof())
                {
                  float ztmp,a;
                  
                  getline(indexFile, line);
                  // Format of line is NUM,DICOMFILE,Z,AMP,PHASE
                  std::string::size_type firstcomma = line.find(',');
                  std::string::size_type lastcomma = line.find_last_of(',');
                  if (lastcomma == std::string::npos) continue;
                  std::string::size_type prevcomma = (line.substr(0,lastcomma-1).find_last_of(','));
                  std::string::size_type prevprevcomma = (line.substr(0,prevcomma-1).find_last_of(','));
      
                  slicestr = line.substr(0,firstcomma);
                  //phasestr = line.substr(lastcomma+1);
                  ampstr   = line.substr(prevcomma+1,lastcomma-prevcomma-1);
                  zstr     = line.substr(prevprevcomma+1,prevcomma-prevprevcomma-1);

                  //appout << "Z=" << t << std::endl;
                  ztmp = atof(zstr.c_str());

                  //appout << "AMP=" << t << std::endl;
                  a = atof(ampstr.c_str());


                  if ( a >= minAmp &&  maxAmp >= a )
                    { // this slice is within range, so remember to load it later
                      sliceFiles.push_back(slicestr);
                      z.push_back(ztmp);
                      amp.push_back(a);
                      ++numSlices;
                    }
                }
              indexFile.close();
              appout << "DONE" << std::endl;
              
              //
              // Give the algorithm the Data
              spaceTimeCT.SetNumSlices(numSlices);
              appout << "Setting slice properties..." << std::endl;
              for (unsigned int i = 0; i < numSlices; ++i)
                {
                  spaceTimeCT.SetNthZValue(i,z[i]);
                  spaceTimeCT.SetNthAmplitude(i,amp[i]);

                  std::string filename = inputDir + "/" + sliceFiles[i] + ".raw";
                  spaceTimeCT.SetNthSlice(i, filename);

                  spaceTimeCT.SetSliceScale(scaleLevels[scale]);
                }
              appout << "DONE." << std::endl;

              for (unsigned int i = 0; i < numForwardTimeSteps; ++i)
                {
                  spaceTimeCT.SetNthVelocityField(i, vforward[i],1);
                }
              for (unsigned int i = 0; i < numReverseTimeSteps; ++i)
                {
                  spaceTimeCT.SetNthVelocityField(i, vreverse[i],0);
                }
      


              spaceTimeCT.Run();

              appout << "DONE Computing Growth." << std::endl;
            }
        }

      delete baseImg;

      if (numForwardTimeSteps > 0)
        for (unsigned int i = 0; i < numForwardTimeSteps-1;++i)
          {
            delete vforward[i];
          }
      if (numReverseTimeSteps > 0)
        for (unsigned int i = 0; i < numReverseTimeSteps-1;++i)
          {
            delete vreverse[i];
          }
    
      //  delete [] scaledImages;
      //   delete [] images;

      delete [] vforward;
      delete [] vreverse;

      appout << "Total Time: " << totalTimer.getTime() << std::endl;


      return 0;
}


