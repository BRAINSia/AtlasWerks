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


#include <cmath>

#include "Array3D.h"
#include "Array3DUtils.h"
#include "ImageUtils.h"
#include "ApplicationUtils.h"
#include "StringUtils.h"
#include "HField3DUtils.h"
#include "AtlasWerksTypes.h"

#include <tclap/CmdLine.h>


int main(int argc, char *argv[]){

  try
    {
      TCLAP::CmdLine cmd("VecUtils");

      std::vector<TCLAP::Arg*> xorArgs;
      TCLAP::SwitchArg
	splitArg("x","split",
		 "Split into component images",
		 false);
      xorArgs.push_back(&splitArg);
      
      TCLAP::SwitchArg
	addIdentArg("a","addIdentity",
		    "Add identity to the input (convert from vField to hField)",
		    false);
      xorArgs.push_back(&addIdentArg);
      
      TCLAP::SwitchArg
	subIdentArg("s","subtractIdentity",
		    "Subtract identity from the input (convert from hField to vField)",
		    false);
      xorArgs.push_back(&subIdentArg);

      TCLAP::SwitchArg
	magArg("m","mag",
	       "create magnitude image from vector field",
	       false);
      xorArgs.push_back(&magArg);

      cmd.xorAdd(xorArgs);
      
      TCLAP::ValueArg<std::string>
	inFileArg("i","inFile",
		  "input file name",
		  true,"","file", cmd);
      
      TCLAP::ValueArg<std::string>
	outFileArg("o","outFile",
		  "output file name",
		  true,"","file", cmd);
      
      cmd.parse(argc, argv);
      
      // read the input
      VectorField v;
      ApplicationUtils::LoadHFieldITK(inFileArg.getValue().c_str(), v);

      if(addIdentArg.isSet())
	{
	  HField3DUtils::velocityToH(v);
	  ApplicationUtils::SaveHFieldITK(outFileArg.getValue().c_str(), v);
	  std::cout << "Warning, spacing not respected!" << std::endl;
	}
      else if(subIdentArg.isSet())
	{
	  HField3DUtils::hToVelocity(v);
	  ApplicationUtils::SaveHFieldITK(outFileArg.getValue().c_str(), v);
	  std::cout << "Warning, spacing not respected!" << std::endl;
	}
      else if(magArg.isSet())
	{
	  RealImage out(v.getSize());
	  HField3DUtils::pointwiseL2Norm(v, out);
	  ApplicationUtils::SaveImageITK(outFileArg.getValue().c_str(), out);
	  std::cout << "Warning, spacing not respected!" << std::endl;
	}
      else if(splitArg.isSet())
	{
	    // parse a pathname
	  std::string path, nameBase, nameExt;
	  ApplicationUtils::SplitName(outFileArg.getValue().c_str(),
				      path,
				      nameBase,
				      nameExt);
	  SizeType size = v.getSize();
	  RealImage xIm(size);
	  RealImage yIm(size);
	  RealImage zIm(size);
	  
	  unsigned int numEl = v.getNumElements();
	  for(unsigned int i=0;i<numEl;i++){
	    xIm(i) = v(i).x;
	    yIm(i) = v(i).y;
	    zIm(i) = v(i).z;
	  }

	  std::string outName = StringUtils::strPrintf("%s%s_x%s",path.c_str(), nameBase.c_str(), nameExt.c_str());
	  ApplicationUtils::SaveImageITK(outName.c_str(), xIm);
	  outName = StringUtils::strPrintf("%s%s_y%s",path.c_str(), nameBase.c_str(), nameExt.c_str());
	  ApplicationUtils::SaveImageITK(outName.c_str(), yIm);
	  outName = StringUtils::strPrintf("%s%s_z%s",path.c_str(), nameBase.c_str(), nameExt.c_str());
	  ApplicationUtils::SaveImageITK(outName.c_str(), zIm);
	}
      else
	{
	  std::cout << "Error, function not yet supported" << std::endl;
	}
    }
  catch (TCLAP::ArgException &e)
    {
      std::cerr << "error: " << e.error() << " for arg " << e.argId()
		<< std::endl;
      exit(1);
    }
  return 0;
}

