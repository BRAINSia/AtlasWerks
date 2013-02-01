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

#include <fstream>
#include <string>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <stdlib.h>
#include <math.h>
#include <tclap/CmdLine.h>
#include "Image.h"
#include "ImageUtils.h"

int main(int argc, char **argv)
{
  bool ampBin;
  unsigned int N;
  std::string inputDir, outputImgFile;
  float deltaZ, binVal, deltaAmp;

  try
  {
    //std::cout << "Parsing command line arguments...";

    TCLAP::CmdLine cmd("binslices",' ',"0.1");
    
    TCLAP::SwitchArg
      ampBinArg("a","ampbin",
                 "Use amplitude binning? Otherwise do phase binning",
                 cmd, false);

    TCLAP::ValueArg<float>
      deltaAmpArg("","deltaamp",
                    "How far to stray for amplitude binning",
                    false,-1,"deltaamp", cmd);
    

    TCLAP::ValueArg<int>
      sizeArg("N","ndim",
                    "Size in X and Y",
                    true,512,"N", cmd);

    TCLAP::ValueArg<std::string>
      inputDirArg("i","inputdir",
                    "Input directory (with index.csv and *.raw)",
                    true,"","inputdir", cmd);

    TCLAP::ValueArg<std::string>
      outputImgArg("o","outputimage",
                    "Output Image",
                    true,"","outputimg", cmd);

    TCLAP::ValueArg<float>
      deltaZArg("","zspacing",
                    "Z spacing",
                    false,2.5,"zspacing", cmd);

    TCLAP::ValueArg<float>
      binValArg("b","binval",
                    "Bin value (find slices closest to this value",
                    true,0,"binval", cmd);

    cmd.parse(argc, argv);

    ampBin = ampBinArg.getValue();
    N = sizeArg.getValue();
    inputDir = inputDirArg.getValue();
    outputImgFile = outputImgArg.getValue();
    deltaZ = deltaZArg.getValue();
    deltaAmp = deltaAmpArg.getValue();
    binVal = binValArg.getValue();

    //std::cout << "DONE" << std::endl;
  }
  catch (TCLAP::ArgException &e)
    {
      std::cerr << "error: " << e.error() << " for arg " << e.argId()
                << std::endl;
      exit(1);
    }

  std::string indexfilein = inputDir + "/index.csv";
  
  std::ifstream indexin(indexfilein.c_str(), std::ios::in);
  float minz = 1e500;
  float maxz = -1e500;
  std::string line;
  while (getline(indexin, line, '\n'))
    { // Read through to get max, min Z 
      int lastcomma = line.find_last_of(',');
      int prevcomma = (line.substr(0,lastcomma-1).find_last_of(','));
      int prevprevcomma = (line.substr(0,prevcomma-1).find_last_of(','));
     
      std::string zstr     = line.substr(prevprevcomma+1,prevcomma-prevprevcomma-1);
      float z = atof(zstr.c_str());

      if (z < minz || minz > maxz) minz = z;
      if (z > maxz || minz > maxz) maxz = z;
    }
  
  // Reset file pointer
  indexin.close();
  indexin.open(indexfilein.c_str(), std::ios::in);
  
  // we want to round this so add 0.5 and cast
  unsigned int Nz = static_cast<unsigned int>((maxz-minz)/deltaZ+1.5);
 
  std::vector<float> distances;
  std::vector<bool> haveSlice;
  distances.resize(Nz);
  haveSlice.resize(Nz);
  for (unsigned int i = 0; i < Nz; ++i)
    haveSlice[i] = false;

  Image<short> *img = new Image<short>;
  img->resize(N,N,Nz);
  img->fill(0.0f);

  std::vector<float> valsUsed;
  valsUsed.resize(Nz);
  while (getline(indexin, line, '\n'))
    {
      // get amplitude here
      //lineAmp = row 4
      // linePhase = row 5
      // lineZ = row 6
      std::string phasestr = line.substr(line.find_last_of(',')+1);
      int lastcomma = line.find_last_of(',');
      int prevcomma = line.substr(0,lastcomma-1).find_last_of(',');
      int prevprevcomma = line.substr(0,prevcomma-1).find_last_of(',');
      int firstcomma = line.find(',');
      int secondcomma = line.substr(firstcomma+1).find(',') + firstcomma + 1;

      std::string slicestr = line.substr(0,firstcomma);
      std::string ampstr   = line.substr(prevcomma+1,lastcomma-prevcomma-1);
      std::string zstr     = line.substr(prevprevcomma+1,prevcomma-prevprevcomma-1);
      
      float amp = atof(ampstr.c_str());
      float phase = atof(phasestr.c_str());
      float z = atof(zstr.c_str());

      unsigned int zind = static_cast<unsigned int>((z - minz)/deltaZ + 0.5);

      float dist = (ampBin ? fabs(amp - binVal) : fabs(phase - binVal));

      //std::cout << ampBin << "," << phase << "," << zind << "," << haveSlice[zind] << "," << dist << "," << distances[zind] << std::endl;

      if ( ( ampBin && // Amplitude Binning
             ((deltaAmp > 0 && dist <= deltaAmp && (dist < distances[zind] || !haveSlice[zind] )) || // given deltaAmp
              (deltaAmp < 0 && ( !haveSlice[zind] || // no given deltaAmp and don't have slice yet
                                 dist < distances[zind] )))) // slice is closer than previous
           || (!ampBin && // Phase Binning
               (!haveSlice[zind] || // don't have a slice yet
                dist < distances[zind] ))) // this is closer than previous
        {
          //std::cout << "Processing slice " << slicestr << std::endl;

          std::string slicefile = inputDir + "/" + slicestr + ".raw";
                   
          std::ifstream slicein(slicefile.c_str(), std::ios::in | std::ios::binary);
          if (!slicein)
            {
              std::cerr << "Can't open slice! " << slicefile << std::endl;
              return EXIT_FAILURE;
            }

          short buffer;
          slicein.read(reinterpret_cast<char*>(&buffer),2);
          for (unsigned int k=0; !slicein.eof(); ++k)
            {
              unsigned int x = k % N;
              unsigned int y = k / N;
              if (y >= N)
                {
                std::cerr << "ERROR: N is set incorrectly. Data is bigger than N^2" << std::endl;
                return EXIT_FAILURE;
                }
              img->set(x, y, zind, buffer);

              slicein.read(reinterpret_cast<char*>(&buffer),2);
            }
          slicein.close();
          
          // Keep track of what's been used         
          valsUsed[zind] = (ampBin ? amp : phase);
          distances[zind] = dist;
          haveSlice[zind] = true;
        }
    }


  float max = -1e500; // max phase or amplitude, to report later
  float min = 1e500;
  unsigned int numused = 0;
  for (int i = 0; i < Nz; ++i)
    {
      if (haveSlice[i])
        {
          numused++;
          float val = valsUsed[i];
          // Update min and max
          if (val < min || min > max) min = val;
          if (val > max || min > max) max = val;
        }
    } 
  std::cout << "Min:" << min << " Max:" << max << " Missing:" << Nz-numused;

  // TODO: We really need to keep track of x,y spacings in the forRecon directory... (and z too for that matter)
  img->setSpacing(Vector3D<double>(1.0f,1.0f,deltaZ));
  // TODO: Get origin X,Y values correct here
  img->setOrigin(0.0f,0.0f,minz);
  ImageUtils::writeMETA(*img, outputImgFile.c_str());

  indexin.close();
}
