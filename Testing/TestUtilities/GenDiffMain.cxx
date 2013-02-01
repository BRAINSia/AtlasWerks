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
#include "HField3DUtils.h"
#include "AtlasWerksTypes.h"

#include <tclap/CmdLine.h>


int main(int argc, char *argv[]){

  try
    {
      TCLAP::CmdLine cmd("GenDiff");
      
      
      TCLAP::SwitchArg
	vecSwitchArg("v","vectorFields",
		     "inputs are vector fields, not images",
		     cmd, false);
      
      TCLAP::SwitchArg
	sqrErrArg("s","squaredError",
		  "use squared difference",
		  cmd, false);
      
      TCLAP::SwitchArg
	quietSwitchArg("q","quiet",
		       "supress output of image difference",
		       cmd, false);
      
      TCLAP::SwitchArg
	magSwitchArg("m","writeMag",
		     "write the magnitude of the vector image differences (only used if input are vector fields)",
		     cmd, false);
      
      TCLAP::ValueArg<std::string>
	outFileArg("o","outFile",
		   "file to save difference image to",
		   false,"","file", cmd);
      
      TCLAP::UnlabeledValueArg<std::string>
	testFileArg("testImage",
		    "image/vector field to test (diff computed as test-base)",
		    true,"","file",cmd);
      
      TCLAP::UnlabeledValueArg<std::string>
	baseFileArg("baseImage",
		    "image/vector field to test against (diff computed as test-base)",
		    true,"","file",cmd);
      
      cmd.parse(argc, argv);
      
      float diffSum;
      if(vecSwitchArg.getValue())
	{
	  // use vector fields
	  Array3D<Vector3D<float> > test;
	  Array3D<Vector3D<float> > base;
	  ApplicationUtils::LoadHFieldITK(testFileArg.getValue().c_str(),
					  test);
	  ApplicationUtils::LoadHFieldITK(baseFileArg.getValue().c_str(),
					  base);
	  Array3D<Vector3D<float> > diff(test);
	  diff.pointwiseSubtract(base);
	  if(sqrErrArg.getValue()){
	    diff.pointwiseMultiplyBy(diff);
	  }
	  if(!quietSwitchArg.getValue()){
	    diffSum = 0.0;
	    unsigned int numElements = diff.getNumElements();
	    for (unsigned int element = 0; element < numElements; ++element){
	      Vector3D<float> el = diff(element);
	      diffSum += fabs(el.x) + fabs(el.y) + fabs(el.z);
	    }
	    std::cout << "err: " << diffSum << std::endl;
	  }
	  // write out the difference image if requested
	  if(outFileArg.isSet()){
	    if(magSwitchArg.getValue()){
	      Image<float> diffMag(diff.getSize());
	      HField3DUtils::pointwiseL2Norm(diff, diffMag);
	      ApplicationUtils::SaveImageITK(outFileArg.getValue().c_str(), diffMag);
	    }else{
	      ApplicationUtils::SaveHFieldITK(outFileArg.getValue().c_str(), diff);
	    }
	  }
	}
      else // input are images
	{
	  // use vector fields
	  Image<float> test;
	  Image<float> base;
	  ApplicationUtils::LoadImageITK(testFileArg.getValue().c_str(),
					 test);
	  ApplicationUtils::LoadImageITK(baseFileArg.getValue().c_str(),
					 base);
	  Image<float> diff(test);
	  diff.pointwiseSubtract(base);
	  if(sqrErrArg.getValue()){
	    diff.pointwiseMultiplyBy(diff);
	  }
	  if(!quietSwitchArg.getValue()){
	    diffSum = 0.0;
	    unsigned int numElements = diff.getNumElements();
	    for (unsigned int element = 0; element < numElements; ++element){
	      Vector3D<float> el = diff(element);
	      diffSum += fabs(diff(element));
	    }
	    std::cout << "err: " << diffSum << std::endl;
	  }
	  // write out the difference image if requested
	  if(outFileArg.isSet()){
	    ApplicationUtils::SaveImageITK(outFileArg.getValue().c_str(), diff);
	  }
	}
      
    }
  catch (TCLAP::ArgException &e)
    {
      std::cerr << "error: " << e.error() << " for arg " << e.argId()
		<< std::endl;
      exit(1);
    }
}

