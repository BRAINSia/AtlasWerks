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

#include <iostream>

#include "Array3D.h"
#include "AffineTransform3D.h"
#include "HField3DUtils.h"
#include "HField3DIO.h"
#include "ApplicationUtils.h"

/**
 * \page txWerks
 *
 * Concatenate a series of transforms 
 */

void printUsage(const char* programName)
{
  std::cout << "Usage: "
            << programName
            << "\n\t[-base filename (base hfile giving origin, dims, spacing)]"
	    << "\n\t[-dims x y z]"
            << "\n\t[-origin x y z]"
            << "\n\t[-spacing x y z]"
	    << "\n\t-o outFile (without extension; .mhd + .raw is automatic)"
	    << "\n... followed by any number of..."
	    << "\n\t[-aff matrixFileName] (affine transformation to initialize with)"
	    << "\n\t[-affItk matrixFileName] (affine transformation to initialize with, ITK-style tranform)"
	    << "\n\t\t[-z] (use zero origin instead of image origin for this (previous) affine transform)"
	    << "\n\t\t[-i] (invert the previously specified affine transform)"
	    << "\n\t[-h hFieldFileName]"
	    << "\n\t[-t tx ty tz] (translation)"
	    << "\n\t[-m matrixFileName] (only used for translation)"
            << "\n Either [-base filename] or "
            << "[-dims x y z] must be included.\n"
            << std::endl;
}

int main(int argc, char **argv)
{
  //
  // parse first part of command line
  //
  std::string outputFilename = "";
  SizeType size;
  OriginType origin(0,0,0);
  SpacingType spacing(1,1,1);
  bool haveOutputFilename = false;
  bool haveSize = false;


  const char* programName = argv[0];
  argv++; argc--;
  while (argc > 0)
    {
      if (std::string(argv[0]) == std::string("-base"))
      {
        // make sure at least one argument follows
        if (argc < 2)
        {
          printUsage(programName);
          exit(-1);
        }
	
	ApplicationUtils::ReadHeaderITK(&*argv[1], size, origin, spacing);

        argv++; argc--;
        argv++; argc--;
        haveSize = true;
      }
      else if (std::string(argv[0]) == std::string("-o"))
	{
	  // make sure at least one argument follows
	  if (argc < 2)
	    {
	      printUsage(programName);
	      exit(-1);
	    }
	  outputFilename = argv[1];
	  argv++; argc--;
	  argv++; argc--;
	  haveOutputFilename = true;
	}
      else if (std::string(argv[0]) == std::string("-dims"))
	{
	  // make sure at least three arguments follow
	  if (argc < 4)
	    {
	      printUsage(programName);
	      exit(-1);
	    }
	  size.x = atoi(argv[1]);
	  size.y = atoi(argv[2]);
	  size.z = atoi(argv[3]);
          argv +=4; argc -= 4;
	  haveSize = true;
	}
      else if (std::string(argv[0]) == std::string("-origin"))
	{
	  // make sure at least three arguments follow
	  if (argc < 4)
	    {
	      printUsage(programName);
	      exit(-1);
	    }
	  origin.x = atof(argv[1]);
	  origin.y = atof(argv[2]);
	  origin.z = atof(argv[3]);
          argv +=4; argc -= 4;
	}
      else if (std::string(argv[0]) == std::string("-spacing"))
	{
	  // make sure at least three arguments follow
	  if (argc < 4)
	    {
	      printUsage(programName);
	      exit(-1);
	    }
	  spacing.x = atof(argv[1]);
	  spacing.y = atof(argv[2]);
	  spacing.z = atof(argv[3]);
          argv +=4; argc -= 4;
	}
      else
	{
	  break;
	}
    }
  
  if (!(haveOutputFilename && haveSize))
    {
      printUsage(programName);
      exit(-1);
    }

  //
  // print parameters
  //
  std::cout << "Transform size  : " 
	    << size.x << ", " << size.y << ", " << size.z
            << "\nTransform origin: "
            << origin.x << ", " << origin.y << ", " << origin.z
            << "\nTransform spacing: "
            << spacing.x << ", " << spacing.y << ", " << spacing.z
            << "\nOutput Filename : " << outputFilename << std::endl;

  //
  // create an identity transform
  //
  Array3D< Vector3D<float> > compositeXform(size);
  HField3DUtils::setToIdentity(compositeXform);

  //
  // concatenate transforms
  //
  while (argc > 0)
    {
      std::string thisArg = argv[0];

      // affine transformation
      if (std::string(argv[0]) == std::string("-aff") ||
	  std::string(argv[0]) == std::string("-affItk"))
	{

	  AffineTransform3D<float> aff;

	  // Coords in matrix file
	  // make sure at least one argument follows
	  if (argc < 2)
	    {
	      printUsage(programName);
	      exit(-1);
	    }
	  try
	    {
	      if(std::string(argv[0], argv[0]+strlen(argv[0])) == std::string("-affItk")){
		aff.readITKStyle(argv[1]);
	      }else{
		aff.readPLUNCStyle(argv[1]);
	      }
	    }
	  catch (...)
	    {
	      std::cerr << "Problem reading transform file." << std::endl;
	      exit(-1);
	    }
	  
	  std::cout << "Initializing transform with affine transform" << std::endl;
	  
	  argv += 2; argc -= 2;
	  
	  Vector3D<float> affOrigin = origin;
	  // process affine modifier args
	  while(argc >= 1 ){
	    if(std::string(argv[0]) == std::string("-z")){
	      std::cout << "Using zero origin for affine transform" << std::endl;
	      affOrigin = Vector3D<float>(0.f,0.f,0.f);
	      argv++;argc--;
	    }else if(std::string(argv[0]) == std::string("-i")){
	      std::cout << "inverting affine transform" << std::endl;
	      aff.invert();
	      argv++;argc--;
	    }else{
	      break;
	    }
	  }
	  
	  Array3D< Vector3D<float> > affXFrm(size);
	  Array3D< Vector3D<float> > tmp(size);
	  HField3DUtils::initializeFromAffine(affXFrm, 
					      aff,
					      affOrigin,
					      spacing);
 	  HField3DUtils::compose(affXFrm, origin, spacing, 
				 compositeXform, origin, spacing,
				 tmp, origin, spacing,
 				 HField3DUtils::BACKGROUND_STRATEGY_PARTIAL_ID);
 	  compositeXform = tmp;
	  std::cout << "DONE" << std::endl;
	}

      // concatenate a translation (-t and -m both do this)
      else if (thisArg == std::string("-t") || thisArg == std::string("-m"))
      {
        Vector3D<float> tx;

        // Two different ways of getting the translation vector.
        if (thisArg == std::string("-t"))
        {
          // Coords given on command line

          // make sure at least three arguments follow
          if (argc < 4)
          {
            printUsage(programName);
            exit(-1);
          }
          tx.x = atof(argv[1]);
          tx.y = atof(argv[2]);
          tx.z = atof(argv[3]);

          argv += 4; argc -= 4;
        }
        else
        {
          // Coords in PLUNC matrix file

          // make sure at least one argument follows
          if (argc < 2)
          {
            printUsage(programName);
            exit(-1);
          }
          AffineTransform3D<float> txAff;
          try
          {
            txAff.readPLUNCStyle(argv[1]);
          }
          catch (...)
          {
            std::cerr << "Problem reading transform file." << std::endl;
            exit(-1);
          }
          tx = txAff.vector;
          argv += 2; argc -= 2;

        }

        Vector3D<float> txSpacing;
        txSpacing.x = tx.x / spacing.x;
        txSpacing.y = tx.y / spacing.y;
        txSpacing.z = tx.z / spacing.z;

        std::cout << "Concatenating translation: " << tx << ", " 
                  << txSpacing << std::endl;
	  
        // h(x) = h(x) + tx
        HField3DUtils::composeTranslation(compositeXform, 
                                          txSpacing, 
                                          compositeXform);
      } // end of concatenating a translation

      // concatenate an h field
      else if (thisArg == std::string("-h"))
	{
	  // make sure at least one argument follows
	  if (argc < 2)
	    {
	      printUsage(programName);
	      exit(-1);
	    }
	  std::string filename(argv[1]);

	  // load hField
	  std::cout << "Loading hField: " << filename << "...";
	  Array3D< Vector3D<float> > hField;
          Vector3D<double> hFieldOrigin;
          Vector3D<double> hFieldSpacing;
	  ApplicationUtils::LoadHFieldITK(filename.c_str(), hFieldOrigin, hFieldSpacing, hField);

	  std::cout << "hField size: " << hField.getSize() << std::endl;
	  std::cout << "hField origin: " << hFieldOrigin << std::endl;
	  std::cout << "hField spacing: " << hFieldSpacing << std::endl;

	  // compose hField
	  std::cout << "Composing hField...";
	  Array3D< Vector3D<float> > tmp(size);
 	  HField3DUtils::compose(hField, hFieldOrigin, hFieldSpacing, 
				 compositeXform, origin, spacing,
				 tmp, origin, spacing,
 				 HField3DUtils::BACKGROUND_STRATEGY_PARTIAL_ID);
 	  compositeXform = tmp;

	  std::cout << "DONE" << std::endl;
	  argv += 2; argc -= 2;
	}
      else
	{
	  std::cout << "Invalid argument: " << argv[0] << std::endl;
	  printUsage(programName);
	  exit(-1);
	}
    }

  //
  // write composite transform
  //
  std::cout << "Writing composite transformation file...";

  // if no format specified...
  if(outputFilename.find(".") == std::string::npos){
    // append default .mhd format
    outputFilename.append(".mhd");
  }
  ApplicationUtils::SaveHFieldITK(outputFilename.c_str(), compositeXform, origin, spacing);
  std::cout << "DONE" << std::endl;

  return 0;
}


