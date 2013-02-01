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

#include "AtlasWerksTypes.h"
#include "Array3D.h"
#include "Array3DUtils.h"
#include "ImageUtils.h"
#include "ApplicationUtils.h"
#include "HField3DUtils.h"
#include "ImageUtils.h"


#include "cudaInterface.h"
#include "CUDAUtilities.h"
#include "cudaHField3DUtils.h"
#include "VectorMath.h"

#include <tclap/CmdLine.h>

#include <time.h>

int main(int argc, char *argv[]){

  try
    {
      TCLAP::CmdLine cmd("cudaWarpTimingTest");
      
      TCLAP::ValueArg<std::string>
	inputArg("i","input",
		 "input image",
		 true,"","file", cmd);
      
      TCLAP::ValueArg<std::string>
	outputArg("o","output",
		  "output image",
		  true,"","file", cmd);
      
      TCLAP::ValueArg<std::string>
	hfieldArg("f","hfield",
		  "hfield to deform image by",
		  true,"","file", cmd);
      
      cmd.parse(argc, argv);
      
      SizeType size;
      OriginType origin;
      SpacingType spacing;
      
      Vector3D<unsigned int> factor;
      
      RealImage inputImage;
      // Load the input image
      ApplicationUtils::LoadImageITK(inputArg.getValue().c_str(),
				     inputImage);
      
      size = inputImage.getSize();
      origin = inputImage.getOrigin();
      spacing = inputImage.getSpacing();
      unsigned int nVox = size.productOfElements();
      unsigned int memSize = nVox*sizeof(float);

      std::cout << "Input image size: " << size << std::endl;
      std::cout << "Input image origin: " << origin << std::endl;
      std::cout << "Input image spacing: " << spacing << std::endl;

      RealImage tmp(size, origin, spacing);

      VectorField h;
      // load the vector field
      ApplicationUtils::LoadHFieldITK(hfieldArg.getValue().c_str(),
				      h);

      // copy the data to the device
      float *dInput, *dDefIm;
      allocateDeviceArray((void**)&dInput, memSize);
      allocateDeviceArray((void**)&dDefIm, memSize);
      copyArrayToDevice(dInput, inputImage.getDataPointer(), nVox);
      
      cplVector3DArray dH;
      allocateDeviceVector3DArray(dH, nVox);
      CUDAUtilities::CopyVectorFieldToDevice(h, dH);

      // without apply
      clock_t begin = clock();
      clock_t end = clock();
      double elapsedMS = 0.0;

      begin = clock();

      int maxCnt = 1000;
      for(int cnt = 0; cnt < maxCnt; cnt++){
          cplVectorOpers::MulC_I(dDefIm, 1.0f, nVox);
      }
      std::cout << std::endl;
      end = clock();

      elapsedMS = 1000.0*(((double)(end-begin))/((double)CLOCKS_PER_SEC));
      elapsedMS /= ((double)maxCnt);

      copyArrayFromDevice(tmp.getDataPointer(), dDefIm, nVox);

      std::cout << "Elapsed time is " << elapsedMS << " milliseconds (total " << end-begin << " clock ticks)" 
		<< ", data 0 val is " << *tmp.getDataPointer() << std::endl;

      // with apply 

      begin = clock();
      
      for(int cnt = 0; cnt < maxCnt; cnt++){
	cudaHField3DUtils::apply(dDefIm, dInput, dH, size);
	cplVectorOpers::MulC_I(dDefIm, 1.0f, nVox);
      }
      std::cout << std::endl;
      end = clock();

      elapsedMS = 1000.0*(((double)(end-begin))/((double)CLOCKS_PER_SEC));
      elapsedMS /= ((double)maxCnt);

      copyArrayFromDevice(tmp.getDataPointer(), dDefIm, nVox);

      std::cout << "Elapsed time is " << elapsedMS << " milliseconds (total " << end-begin << " clock ticks)"
		<< ", data 0 val is " << *tmp.getDataPointer() << std::endl;

      // copy back

      begin = clock();
      RealImage defIm(size, origin, spacing);
      copyArrayFromDevice(defIm.getDataPointer(), dDefIm, nVox);
      end = clock();
      elapsedMS = 1000.0*(((double)(end-begin))/((double)CLOCKS_PER_SEC));
      elapsedMS /= ((double)maxCnt);
      std::cout << "Time to copy array back to host is " << elapsedMS << " milliseconds" << std::endl;

      ApplicationUtils::SaveImageITK(outputArg.getValue().c_str(), defIm);

    }
  catch (TCLAP::ArgException &e)
    {
      std::cerr << "error: " << e.error() << " for arg " << e.argId()
		<< std::endl;
      exit(1);
    }
}

