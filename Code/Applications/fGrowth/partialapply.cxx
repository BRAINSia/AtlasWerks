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

#include <string>
#include <exception>
#include <tclap/CmdLine.h>

#include "ApplicationUtils.h"
#include "Image.h"
#include "Vector3D.h"
#include "ImageIO.h"
#include "HField3DUtils.h"
#include "HField3DIO.h"

//
// Scale a deformation's displacements by a floating point amount and apply to an image
// jdh - 8/12/08
//

int main(int argc, char **argv)
{
  //
  // parse command line
  //
  std::string inputImgFile, defFile, outputImgFile;
  bool loadAsVelocity, reverse;
  double delta;
  try
    {
      std::cout << "Parsing command line arguments...";

      TCLAP::CmdLine cmd("partialapply",' ',"0.1");

      TCLAP::ValueArg<std::string>
        inputImgArg("i","inputimg",
                    "Image to deform",
                    true,"","in.mhd", cmd);
      TCLAP::ValueArg<std::string>
        defFieldArg("f","deformation",
                    "Deformation field to apply to inputimg",
                    true,"","hfield.mhd",cmd);
      TCLAP::ValueArg<std::string>
        outputImgArg("o","outputimg",
                     "Filename to write output image to",
                     true,"","out.mhd",cmd);
      TCLAP::ValueArg<double>
        deltaArg("d","delta",
                 "Amount to scale u(x) by",
                 true,1.0,"delta",cmd);

      TCLAP::SwitchArg
        loadAsVelArg("v","loadasvelocity",
                     "Deformation is loaded as velocity instead of hfield",
                     cmd, false);
      TCLAP::SwitchArg
        reverseArg("r","reverse",
                   "Apply deformation in reverse out(x) = in(h(x)) as opposed to out(x) = in(hinv(x))",
                   cmd, false);


      cmd.parse(argc, argv);

      inputImgFile = inputImgArg.getValue();
      defFile = defFieldArg.getValue();
      outputImgFile = outputImgArg.getValue();
      delta = deltaArg.getValue();
      loadAsVelocity = loadAsVelArg.getValue();
      reverse = reverseArg.getValue();

      std::cout << "DONE" << std::endl;
    }
  catch (TCLAP::ArgException &e)
    {
      std::cerr << "error: " << e.error() << " for arg " << e.argId()
                << std::endl;
      return EXIT_FAILURE;
    }

  Image<float> inputImg;
  Array3D< Vector3D<float> > def;

  //
  // Load input image
  //
  ApplicationUtils::LoadImageITK(inputImgFile.c_str(), inputImg);

  Image<float> outputImg(inputImg);
  //
  // Load deformation
  //
  HField3DIO::readMETA(def, defFile.c_str());

  //
  // Scale deformation by delta
  //
  for (unsigned int z=0; z < def.getSizeZ(); ++z)
    for (unsigned int y=0; y < def.getSizeY(); ++y)
      for (unsigned int x=0; x < def.getSizeX(); ++x)
        {
          Vector3D<float> u = def.get(x,y,z);
          
          if (loadAsVelocity)
            { // scale and convert to hfield
              u.x = u.x*delta + x;
              u.y = u.y*delta + y;
              u.z = u.z*delta + z;
            }
          else
            {
              u.x = (u.x - x)*delta + x;
              u.y = (u.y - y)*delta + y;
              u.z = (u.z - z)*delta + z;
            }
            
          def.set(x,y,z, u);
        }

  //
  // Apply deformation
  //
  if (reverse)
    HField3DUtils::apply(inputImg, def, outputImg);
  else
    HField3DUtils::forwardApply(inputImg, def, outputImg);

  //
  // Save deformed image
  //
  ApplicationUtils::SaveImageITK(outputImgFile.c_str(), outputImg);

  return EXIT_SUCCESS;
}


