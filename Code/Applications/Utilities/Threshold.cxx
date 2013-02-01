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
#include "Array3DIO.h"
#include "AtlasWerksTypes.h"
#include "Array3D.h"
#include "Array3DUtils.h"
#include "ImageUtils.h"
#include "ApplicationUtils.h"
#include "HField3DUtils.h"
#include "ImageUtils.h"
#include <string>
#include <iostream>
#include <stdio.h>


#include <tclap/CmdLine.h>


int main(int argc, char *argv[]){
  using namespace std;
  
  try
    {   

     
      TCLAP::CmdLine cmd("Threshold"); 

      TCLAP::ValueArg<std::string> threshArg("t","thresholdVal","value to threshold the image at",true,"","float", cmd);

      TCLAP::UnlabeledValueArg<std::string> imageNameArg("inputImage", "image to threshold", true,"","file", cmd);

      TCLAP::UnlabeledValueArg<std::string> outputImageArg("outputImage", "finished image", true,"","file",cmd);
      
      cmd.parse( argc, argv );
     

      float thresh; 

      sscanf(threshArg.getValue().c_str(),"%f",&thresh) ; 
      
      RealImage array;


      RealImage newArray(array.getSize(), array.getOrigin(), array.getSpacing());

           
      ApplicationUtils::LoadImageITK(imageNameArg.getValue().c_str(),array);

    
      ImageUtils::threshold(array, newArray, thresh);


      ApplicationUtils::SaveImageITK(outputImageArg.getValue().c_str(), newArray);



    }
  catch (TCLAP::ArgException &e)
    {
      std::cerr << "error: " << e.error() << " for arg " << e.argId()
		<< std::endl;
      exit(1);
    }
}


