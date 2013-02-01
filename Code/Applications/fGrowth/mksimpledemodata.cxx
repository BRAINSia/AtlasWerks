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

float defImg(float x, float y, float z, float amp, unsigned int N, unsigned int Nz)
{
  float radsq = (float)(N*N*Nz)/4096.0;

  float out;
//   if ( x*x *sqrt(1+amp) +  y*y *sqrt(1+amp) + z*z /(1+amp) < radsq )
//     out = x*x *sqrt(1+amp)
//       + y*y *sqrt(1+amp) 
//       + z*z /(1+amp);
//   else if (sqrt(x*x *sqrt(1+amp) +  y*y *sqrt(1+amp) + z*z /(1+amp)) < 2*sqrt(radsq))
//     {
//       float distout = 2*sqrt(radsq) -  sqrt(x*x *sqrt(1+amp) +  y*y *sqrt(1+amp) + z*z /(1+amp));
//       out = distout*distout;
//     }
//   else
//     out = 0;

  float zeff = z + N*(amp/2 + 0.5)/4;
  if ( x*x +  y*y + zeff*zeff < radsq )
    out = sqrt( x*x + y*y + zeff*zeff );
  else if (sqrt(x*x + y*y + zeff*zeff) < 2*sqrt(radsq))
    {
      float distout = 2*sqrt(radsq) -  sqrt(x*x +  y*y + zeff*zeff);
      out = distout;
    }
  else
    out = 0;
  
  return out * 64;
}

int main(int argc, char **argv)
{
  std::string outputdir("/home/sci/jacob/tmp/spacetimedemo/forRecon-simple");
  std::string imagePrefix("/home/sci/jacob/tmp/spacetimedemo/trueimgs-simple/trueimg");
  std::string indexfilein("/home/sci/jacob/Data/4DSBRTLiver/RitchieBeverley/forRecon/index.csv");
  std::string indexfileout(outputdir + "/index.csv");

  // Set limits to get only certain slices (inclusive)
//   float zmin = -79.25; // for RitchieBeverley (gives 32 slices)
//   float zmax = -1.75;
//   float  deltaz = 2.5;

  unsigned int N = 32; // Size of slices acquired
  unsigned int Nz = 32; // number of z positions
//  unsigned int numslices = 2000;

  // Amplitude range: 0 to 1, uniform

  std::ofstream indexout(indexfileout.c_str(), std::ios::out);
 //  std::ifstream indexin(indexfilein.c_str(), std::ios::in);
 
//    float maxamp = -1e500;
//   float minamp = 1e500;
//   std::string line,ampstr, phasestr, zstr;
//   unsigned int i = 0;
//   while (getline(indexin, line, '\n'))
//     {
//       int lastcomma = line.find_last_of(',');
//       int prevcomma = (line.substr(0,lastcomma-1).find_last_of(','));
//       int prevprevcomma = (line.substr(0,prevcomma-1).find_last_of(','));
      
//       phasestr = line.substr(lastcomma+1);
//       ampstr   = line.substr(prevcomma+1,lastcomma-prevcomma-1);
//       zstr     = line.substr(prevprevcomma+1,prevcomma-prevprevcomma-1);
      
//       float amp = atof(ampstr.c_str());
//       float phase = atof(phasestr.c_str());
//       float z = atof(zstr.c_str());

//       if (z < zmin || z > zmax) continue; // Only take some of the slices

//       if (amp < minamp || minamp > maxamp) minamp = amp;
//       if (amp > maxamp || minamp > maxamp) maxamp = amp;


//       indexout << std::setw(4) << std::setfill('0') << i << "," 
//                << std::setw(4) << std::setfill('0') << i << ","
//                << z << "," 
//                << amp << ","
//                << phase << std::endl;

//       std::stringstream ss;
//       ss << outputdir << "/" << std::setw(4) << std::setfill('0') << i << ".raw";

//       std::ofstream sliceout(ss.str().c_str(), std::ios::out | std::ios::binary);

//       std::stringstream ss1;
//       ss1 << outputdir << "/" << std::setw(4) << std::setfill('0') << i << ".pgm";

//       std::ofstream pgmout(ss1.str().c_str(), std::ios::binary);   

//       if (pgmout.bad())
//         {
//           std::cerr << "error opening PGM file" << std::endl;
//         }
//       pgmout << "P5\n" 
//         << N << "\n" 
//         << N << "\n"
//         << 255 << "\n";

//       for (int k=0;k < N*N;++k)
//         {
//           float x = (float)(k % N) - (float)N/2.0; // x is fast moving direction
//           float y = (float)(k / N) - (float)N/2.0;
//           float zf = (z-zmin)/deltaz - (float)Nz/2.0;
          

//           float fval = defImg(x,y,zf,amp,N,Nz);
//           short sval = static_cast<short>(fval);

//           sliceout.write(reinterpret_cast<char*>(&sval),2);

//           char valchr = static_cast<char>(fval);
//           pgmout.write((char*) &valchr, 1);
//         }

//       pgmout.close();
//       sliceout.close();

//       ++i;
//     }
 
  float numFreezeFrames = 50.0; // number of whole acquisitions represented by separate slices
  for (float amp=0; amp <=1;amp += 1.0/(numFreezeFrames-1))
    {
      for (unsigned int z=0;z < Nz;++z)
          {
            unsigned int sliceIndex = z + (unsigned int)((numFreezeFrames-1)*amp*Nz);

            indexout << std::setw(4) << std::setfill('0') << sliceIndex << "," 
                     << std::setw(4) << std::setfill('0') << sliceIndex << ","
                     << z << "," 
                     << amp << ","
                     << 6.28*amp << std::endl;

            std::stringstream ss;
            ss << outputdir << "/" << std::setw(4) << std::setfill('0') << sliceIndex << ".raw";

            std::stringstream ss1;
            ss1 << outputdir << "/" << std::setw(4) << std::setfill('0') << sliceIndex << ".pgm";

            std::ofstream pgmout(ss1.str().c_str(), std::ios::binary);   
        
            if (pgmout.bad())
              {
                std::cerr << "error opening PGM file" << std::endl;
              }
            pgmout << "P5\n" 
                   << N << "\n" 
                   << N << "\n"
                   << 255 << "\n";

            std::ofstream sliceout(ss.str().c_str(), std::ios::out | std::ios::binary);
            for (int k=0;k < N*N;++k)
              {
                float x = (float)(k % N) - (float)N/2.0; // x is fast moving direction
                float y = (float)(k / N) - (float)N/2.0;
                float zf = (float)z - (float)Nz/2.0;
          

                float fval = defImg(x,y,zf,amp,N,Nz);
                short sval = static_cast<short>(fval);

                sliceout.write(reinterpret_cast<char*>(&sval),2);

                char valchr = static_cast<char>(fval);
                pgmout.write((char*) &valchr, 1);
              }
            pgmout.close();
            sliceout.close();
          }
    }
  indexout.close();

//   std::cout << "Used Min Amplitude " << minamp << std::endl;
//   std::cout << "Used Max Amplitude " << maxamp << std::endl;

  // Write true images
//   for (float amp = minamp; amp <= maxamp; amp += (maxamp - minamp)/10.0)
  for (float amp = 0; amp < 1.1; amp += 0.1)
    {
      std::stringstream ss1;
      ss1 << imagePrefix << amp << ".raw";
      std::ofstream rawout(ss1.str().c_str(), std::ios::out | std::ios::binary);

      for (int z = 0; z < Nz;++z)
        for (int k = 0; k < N*N; ++k)
          {
            float x = (float)(k % N) - (float)N/2.0; // x is fast moving direction
            float y = (float)(k / N) - (float)N/2.0;
            float zf = (float)z - (float)(Nz)/2.0;            
        
            short value = static_cast<short>(defImg(x,y,zf,amp,N,Nz));

            rawout.write(reinterpret_cast<char*>(&value),2);
          }
      rawout.close();

      std::stringstream ss2;
      ss2 << imagePrefix << amp << ".mhd";
      std::ofstream mhdtxt(ss2.str().c_str(), std::ios::out);
      mhdtxt << "ObjectType = Image" << std::endl
             << "NDims = 3" << std::endl
             << "BinaryData = True" << std::endl
             << "BinaryDataByteOrderMSB = False" << std::endl
             << "CompressedData = False" << std::endl
             << "TransformMatrix = 1 0 0 0 1 0 0 0 1" << std::endl
//              << "Offset = 0 0 " << zmin << std::endl
             << "Offset = 0 0 0" << std::endl
             << "CenterOfRotation = 0 0 0" << std::endl
             << "AnatomicalOrientation = RAI" << std::endl
//              << "ElementSpacing = " << deltaz << " " << deltaz << " " << deltaz << std::endl
             << "ElementSpacing = 1 1 1" << std::endl
             << "DimSize = " << N << " " << N << " " << Nz << std::endl
             << "ElementType = MET_SHORT" << std::endl
             << "ElementDataFile = trueimg" << amp<< ".raw" << std::endl;
      mhdtxt.close();
    }
}
