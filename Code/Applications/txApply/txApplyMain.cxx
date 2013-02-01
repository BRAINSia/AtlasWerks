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

#include "Array3D.h"
#include "Array3DIO.h"
#include "Array3DUtils.h"
#include "HField3DUtils.h"
#include <iostream>
#include <string>
#include "Image.h"
#include "Surface.h"
#include "SurfaceUtils.h"
#include "SurfaceUtilsVtk.h"
#include "HField3DUtils.h"
#include "HField3DIO.h"
#include "Anastruct.h"
#include "AnastructUtils.h"
#include "ImageUtils.h"

#include "vtkPolyData.h"
#include "vtkPolyDataReader.h"
#include "vtkPolyDataWriter.h"

#include "ApplicationUtils.h"

/**
 * \page txApply
 *
 * Apply a deformation field to an image
 *
 * Run 'txApply' with no options for usage
 */

typedef float VoxelType;

void printUsage()
{
  std::cerr << "Usage: txApply" << std::endl << std::endl
            << "Specify a transformation: " << std::endl
	    << "\t-h hFieldFileName          ---deformation field" << std::endl
	    << "\t-aff plunc-style affine file ---affine" << std::endl
	    << "\t-affItk itk-style affine file ---affine" << std::endl
	    << "\t-t tx ty tz                ---translation" << std::endl
            << std::endl
            << "Specify the direction to apply the transformation: " << std::endl
	    << "\t-f                         ---forwards (push, with arrows)"
            << std::endl
	    << "\t-b                         ---backwards (pull, against arrows)"
            << std::endl << std::endl
            << "Specify the object to transform: " << std::endl
	    << "\t-i imageFileName           ---image to be deformed"
            << std::endl
	    << "\t-s surfaceFileName         ---surface to be deformed"
            << std::endl
	    << "\t-a anastructFileName       ---anastruct to be deformed" 
            << std::endl << std::endl
            << "Specify the filename of the transformed object: " << std::endl
	    << "\t-o outFile                 ---filename of deformed object" << std::endl
	    << "\tFor image objects:" << std::endl
	    << "\t-outInfoFromHField         ---for images transformed by an hField, use the hField " << std::endl
	    << "\t                              size/origin/spacing for output image" << std::endl
	    << "\t-outSize sx sy sz          ---for images, specify the output size" << std::endl
	    << "\t-outOrigin ox oy oz        ---for images, specify the output origin" << std::endl
	    << "\t-outSpacing sx sy sz       ---for images, specify the output spacing" << std::endl
            << std::endl << std::endl
            << "Options: " << std::endl
	    << "\t-z ---for applying affine transform to an image, use zero origin instead of image origin"
            << std::endl
	    << "\t-n ---use nearest neighbor interpolation; trilerp is default"
            << std::endl
	    << "\t-c ---use cubic interpolation; trilerp is default"
            << std::endl
	    << "\t-d srcOrigin srcSpacing dstOrigin dstSpacing" << std::endl
            << "           ---applies to vertices of surfaces and anastructs only" << std::endl
            << "             -transform must be applied to vertices in hfield index coordinates" << std::endl
            << "             -used to apply transform to a surface stored in world coordinates" << std::endl
            << "             -before transform, v=(v-srcOrigin)/srcSpacing" << std::endl
            << "             -after transform, v=v*dstSpacing+dstOrigin" << std::endl
            << "             -example: -d 0 0 0 1.5 1.5 1.5 0 0 0 1.5 1.5 1.5" << std::endl
            << "             -If not specified, for an hField src and dest origin"
	    << "              and spacing area taken from hField origin and spacing"
	    << "\t-p alpha" << std::endl
            << "           ---for use with deformation fields only" << std::endl
            << "             -allows linear interpolation between the identity transform" << std::endl
            << "              and the given deformation field" << std::endl
            << "             -the interpolated deformation is applied to the object" << std::endl
            << "             -specify alpha between 0 (identity) and 1 (the full deformation)" << std::endl

            << std::endl;
}

enum TransformType {HFIELD, AFFINE, TRANSLATION};

int main(int argc, char **argv)
{
  //
  // parse first part of command line
  //
  std::string outputFilename = "";
  std::string hFieldFileName = "";
  std::string affineFileName = "";
  std::string imageFileName  = "";
  std::string surfaceFileName = "";
  std::string anastructFileName = "";

  bool useHFieldOutInfo = false;
  bool useOutSizeInfo = false;
  Vector3D<unsigned int> outSize;
  bool useOutOriginInfo = false;
  Vector3D<double> outOrigin;
  bool useOutSpacingInfo = false;
  Vector3D<double> outSpacing;

  Vector3D<double> dstOrigin(0,0,0);  
  Vector3D<double> dstSpacing(1,1,1);
  Vector3D<double> srcOrigin(0,0,0);  
  Vector3D<double> srcSpacing(1,1,1);
  bool useSrcDestInfo = false;
  bool doDeformationInterpolation = false;
  double alpha=1.0; // deformation field interpolation parameter
  bool forward = false;
  bool backward = false;
  bool nearestNeighbor = false;
  bool cubic = false;
  bool zeroAffOrigin = false;
  TransformType transformType = HFIELD;
  Vector3D<double> translation;
  AffineTransform3D<float> affine;
  bool pluncStyleAffineFile = true;
  bool inputSurfaceIsVtkPolyData = false;
  vtkPolyData *polyData = NULL;
  

  argv++; argc--;
  while (argc > 0)
    {
      if (std::string(argv[0]) == std::string("-o"))
	{
	  // outfile: make sure at least one argument follows
	  if (argc < 2)
	    {
	      printUsage();
	      return 0;
	    }
	  outputFilename = argv[1];
	  argv++; argc--;
	  argv++; argc--;
	}
      else if (std::string(argv[0]) == std::string("-outInfoFromHField"))
	{
	  useHFieldOutInfo = true;
	  argv++; argc--;
	}
      else if (std::string(argv[0]) == std::string("-outSize"))
	{
	  if (argc < 4)
	    {
	      printUsage();
	      return 0;
	    }
          outSize.x = atof(argv[1]);
          outSize.y = atof(argv[2]);
          outSize.z = atof(argv[3]);
	  
	  useOutSizeInfo = true;
	  
	  for (int i = 0; i < 4; ++i) 
            {
              argv++; argc--;
            }
	}
      else if (std::string(argv[0]) == std::string("-outOrigin"))
	{
	  if (argc < 4)
	    {
	      printUsage();
	      return 0;
	    }
          outOrigin.x = atof(argv[1]);
          outOrigin.y = atof(argv[2]);
          outOrigin.z = atof(argv[3]);
	  
	  useOutOriginInfo = true;
	  
	  for (int i = 0; i < 4; ++i) 
            {
              argv++; argc--;
            }
	}
      else if (std::string(argv[0]) == std::string("-outSpacing"))
	{
	  if (argc < 4)
	    {
	      printUsage();
	      return 0;
	    }
          outSpacing.x = atof(argv[1]);
          outSpacing.y = atof(argv[2]);
          outSpacing.z = atof(argv[3]);
	  
	  useOutSpacingInfo = true;
	  
	  for (int i = 0; i < 4; ++i) 
            {
              argv++; argc--;
            }
	}
      else if (std::string(argv[0]) == std::string("-i"))
	{
	  // image: make sure at least one argument follows
	  if (argc < 2)
	    {
	      printUsage();
	      return 0;
	    }
	  imageFileName = argv[1];
	  argv++; argc--;
	  argv++; argc--;
	}
      else if (std::string(argv[0]) == std::string("-d"))
	{
	  // src and dest origing and dimensions: make sure at least
	  // one argument follows
	  if (argc < 13)
	    {
	      printUsage();
	      return 0;
	    }
          srcOrigin.x = atof(argv[1]);
          srcOrigin.y = atof(argv[2]);
          srcOrigin.z = atof(argv[3]);
          srcSpacing.x = atof(argv[4]);
          srcSpacing.y = atof(argv[5]);
          srcSpacing.z = atof(argv[6]);

          dstOrigin.x = atof(argv[7]);
          dstOrigin.y = atof(argv[8]);
          dstOrigin.z = atof(argv[9]);
          dstSpacing.x = atof(argv[10]);
          dstSpacing.y = atof(argv[11]);
          dstSpacing.z = atof(argv[12]);

	  useSrcDestInfo = true;

	  for (int i = 0; i < 13; ++i) 
            {
              argv++; argc--;
            }
	}
      else if (std::string(argv[0]) == std::string("-s"))
	{
	  // surface: make sure at least one argument follows
	  if (argc < 2)
	    {
	      printUsage();
	      return 0;
	    }
	  surfaceFileName = argv[1];

	  for (int i = 0; i < 2; ++i) 
            {
              argv++; argc--;
            }
	}
      else if (std::string(argv[0]) == std::string("-a"))
	{
	  // anastruct: make sure at least one argument follows
	  if (argc < 2)
	    {
	      printUsage();
	      return 0;
	    }
	  anastructFileName = argv[1];

	  for (int i = 0; i < 2; ++i) 
            {
              argv++; argc--;
            }
	}
      else if (std::string(argv[0]) == std::string("-h"))
	{
	  // hField: make sure at least one argument follows
	  if (argc < 2)
	    {
	      printUsage();
	      return 0;
	    }
	  hFieldFileName = argv[1];
	  argv++; argc--;
	  argv++; argc--;
	}
      else if (std::string(argv[0]) == std::string("-t"))
	{
	  // translation: make sure at least one argument follows
	  if (argc < 4)
	    {
	      printUsage();
	      return 0;
	    }
	  translation.x = atof(argv[1]);
	  translation.y = atof(argv[2]);
	  translation.z = atof(argv[3]);
          transformType = TRANSLATION;
	  for (int i = 0; i < 4; ++i) 
            {
              argv++; argc--;
            }
	}
      else if (std::string(argv[0]) == std::string("-aff") ||
	       std::string(argv[0]) == std::string("-affItk"))
	{
	  // affine: make sure at least one argument follows
	  if (argc < 2)
	    {
	      printUsage();
	      return 0;
	    }
	  affineFileName = argv[1];
          transformType = AFFINE;
	  if(std::string(argv[0]) == std::string("-affItk")){
	    pluncStyleAffineFile = false;
	  }
	  argv++; argc--;
	  argv++; argc--;
	}
      else if (std::string(argv[0]) == std::string("-n"))
	{
          // nearest neighbor
	  nearestNeighbor = true;
	  argv++; argc--;	  
	}
      else if (std::string(argv[0]) == std::string("-c"))
	{
          // nearest neighbor
	  cubic = true;
	  argv++; argc--;	  
	}
      else if (std::string(argv[0]) == std::string("-z"))
	{
          // zero affine origin
	  zeroAffOrigin = true;
	  argv++; argc--;	  
	}
      else if (std::string(argv[0]) == std::string("-f"))
	{
          // forward
	  forward = true;
          backward = false;
	  argv++; argc--;	  
	}
      else if (std::string(argv[0]) == std::string("-b"))
	{
          // backward
	  forward = false;
	  backward = true;
	  argv++; argc--;	  
	}
      else if (std::string(argv[0]) == std::string("-p"))
	{
          // deformation field interpolation
	  if (argc < 2)
	    {
	      printUsage();
	      return 0;
	    }
          doDeformationInterpolation = true;
          alpha = atof(argv[1]);
	  argv++; argc--;	  
	  argv++; argc--;	  
	}
      else
	{
	  std::cerr << "Invalid argument: " << argv[0] << std::endl;
	  printUsage();
	  return 0;
	}
    }
  
  if ((surfaceFileName == "" && 
       imageFileName == "" && 
       anastructFileName == "") || 
      (transformType == HFIELD &&
       hFieldFileName == "") || 
      outputFilename == "")
    {
      printUsage();
      return 0;
    }

  if ((forward && backward) || !(forward || backward)) {
    std::cerr << "Must specify excatly one of -f and -b." << std::endl;
    printUsage();
    return 0;
  }

  //
  // print parameters
  //
  if (transformType == TRANSLATION)
    {
      std::cerr << "Translation : " << translation << std::endl;
    }
  else if(transformType == HFIELD)
    {
      std::cerr << "hField Filename  : " << hFieldFileName << std::endl;
    }
  else if(transformType == AFFINE)
    {
      std::cerr << "affine Filename  : " << affineFileName << std::endl;
    }
  if (imageFileName != "")
    {
      std::cerr << "Image Filename   : " << imageFileName << std::endl;
    }
  if (surfaceFileName != "")
    {
      std::cerr << "Surface Filename   : " << surfaceFileName << std::endl;
    }
  if (anastructFileName != "")
    {
      std::cerr << "Anastruct Filename   : " << anastructFileName << std::endl;
    }
  std::cerr << "Output Filename  : " << outputFilename << std::endl;

  //
  // load hField
  //
  Array3D<Vector3D<float> > hField;
  Vector3D<unsigned int> hSize;
  Vector3D<double> hOrigin, hSpacing;
  if (transformType == HFIELD)
    {
      std::cerr << "Loading hField: " << hFieldFileName << "...";
      ApplicationUtils::ReadHeaderITK(hFieldFileName.c_str(), hSize, hOrigin, hSpacing);
      ApplicationUtils::LoadHFieldITK(hFieldFileName.c_str(), hField);
      std::cerr << "DONE" << std::endl;
      std::cerr << "  Dimensions: " << hSize << std::endl;
      std::cerr << "  Origin    : " << hOrigin << std::endl;
      std::cerr << "  Spacing   : " << hSpacing << std::endl;

      // if the user hasn't specified another src/dest origin/spacing,
      // use origin/spacing from hField
      if(!useSrcDestInfo){
	srcOrigin = hOrigin;
	srcSpacing = hSpacing;
	dstOrigin = hOrigin;
	dstSpacing = hSpacing;
      }

      if (doDeformationInterpolation) {
        std::cerr << "Interpolating hField: alpha == " << alpha << "...";    
        Array3D<Vector3D<float> > eye(hField.getSize());
        HField3DUtils::setToIdentity(eye);
        eye.scale(1.0-alpha);
        hField.scale(alpha);
        Array3DUtils::sum(hField,eye);
        std::cerr << "DONE" << std::endl;
      }
    }
  else if(transformType == AFFINE)
    {
      if(pluncStyleAffineFile){
	affine.readPLUNCStyle(affineFileName);
      }else{
	affine.readITKStyle(affineFileName);
      }
    }

  if (surfaceFileName != "")
    {
      //
      // load surface
      //
      Surface surface;
      try 
        {
	  std::string ext = surfaceFileName.substr(surfaceFileName.size()-3);
	  if(ext == "vtk"){
	    std::cout << "Reading vtk file" << std::endl;
	    vtkPolyDataReader *reader = vtkPolyDataReader::New();
	    reader->SetFileName(surfaceFileName.c_str());
	    reader->Update();
	    inputSurfaceIsVtkPolyData = true;
	    polyData = vtkPolyData::New();
	    polyData->DeepCopy(reader->GetOutput());
	    VtkPolyDataToSurface(polyData, surface);
	  }else{
	    std::cout << "Reading byu file" << std::endl;
	    surface.readBYU(surfaceFileName.c_str());
	  }
        }
      catch (...)
        {
          std::cerr << "Can't load surface: " << surfaceFileName << std::endl;
          exit(0);
        }

      //
      // deform surface
      //
      if (forward)
        {
          if (transformType == TRANSLATION)
            {
              SurfaceUtils::worldToImageIndex(surface, srcOrigin, srcSpacing);
              surface.translate(translation);
              SurfaceUtils::imageIndexToWorld(surface, dstOrigin, dstSpacing);
            }
          else if (transformType == HFIELD)
            {
              // subtract origin and divide by spacing
	      SurfaceUtils::worldToImageIndex(surface, srcOrigin, srcSpacing);
              //SurfaceUtils::worldToImageIndex(surface, hOrigin, hSpacing);
              // assumption is that vertices are in image index coords
              HField3DUtils::apply(surface, hField);
              // multiply by spacing and add origin
              SurfaceUtils::imageIndexToWorld(surface, dstOrigin, dstSpacing);
              //SurfaceUtils::imageIndexToWorld(surface, hOrigin, hSpacing);
            }
	  else if (transformType == AFFINE)
	    {
	      surface.applyAffineTransform(affine);
	    }
        }
      else
        {
          if (transformType == TRANSLATION)
            {
              SurfaceUtils::worldToImageIndex(surface, srcOrigin, srcSpacing);
              surface.translate(-translation);
              SurfaceUtils::imageIndexToWorld(surface, dstOrigin, dstSpacing);
            }
          else if (transformType == HFIELD)
            {
              SurfaceUtils::worldToImageIndex(surface, srcOrigin, srcSpacing);
              // assumption is that vertices are in image index coords
              HField3DUtils::inverseApply(surface, hField);
              SurfaceUtils::imageIndexToWorld(surface, dstOrigin, dstSpacing);
            }
          else if (transformType == AFFINE)
	    {
	      if(!affine.invert()){
		std::cerr << "Error inverting affine transform!" << std::endl;
		exit(0);
	      }
	      surface.applyAffineTransform(affine);
	    }
        }

      //
      // write out surface
      //
      try
        {
	  std::string ext = outputFilename.substr(outputFilename.size()-3);
	  if(ext == "vtk"){
	    std::cout << "writing vtk file" << std::endl;
	    if(inputSurfaceIsVtkPolyData){
	      SetVtkPolyDataPoints(polyData, surface);
	    }else{
	      polyData = vtkPolyData::New();
	      SurfaceToVtkPolyData(surface, polyData);
	    }
	    vtkPolyDataWriter *writer = vtkPolyDataWriter::New();
	    writer->SetFileName(outputFilename.c_str());
	    writer->SetInput(polyData);
	    writer->Write();
	  }else{
	    std::cout << "writing byu file" << std::endl;
	    surface.writeBYU(outputFilename.c_str());      
	  }
        }
      catch (...)
        {
          std::cerr << "Can't write surface: " << outputFilename << std::endl;
          exit(0);
        }      
    }
  else if (anastructFileName != "")
    {
      //
      // load anastruct
      //
      Surface surface;
      Anastruct ana;
      try 
        {
          AnastructUtils::readPLUNCAnastruct(ana, anastructFileName.c_str());
        }
      catch (...)
        {
          std::cerr << "Can't load anastruct: " << anastructFileName 
                    << std::endl;
          exit(0);
        }

      //
      // create surface from anastruct
      //
      AnastructUtils::anastructToSurfacePowerCrust(ana, surface);

      //
      // deform surface
      //
      if (forward)
        {
          if (transformType == TRANSLATION)
            {
              SurfaceUtils::worldToImageIndex(surface, srcOrigin, srcSpacing);
              surface.translate(translation);
              SurfaceUtils::imageIndexToWorld(surface, dstOrigin, dstSpacing);
            }
          else if (transformType == HFIELD)
            {
              SurfaceUtils::worldToImageIndex(surface, srcOrigin, srcSpacing);
              HField3DUtils::apply(surface, hField);
              SurfaceUtils::imageIndexToWorld(surface, dstOrigin, dstSpacing);
            }
	  else if (transformType == AFFINE)
	    {
	      surface.applyAffineTransform(affine);
	    }
        }
      else
        {
          if (transformType == TRANSLATION)
	    {
              SurfaceUtils::worldToImageIndex(surface, srcOrigin, srcSpacing);
	      surface.translate(-translation);
              SurfaceUtils::imageIndexToWorld(surface, dstOrigin, dstSpacing);
            }
	  else if (transformType == HFIELD)
            {
              SurfaceUtils::worldToImageIndex(surface, srcOrigin, srcSpacing);
              HField3DUtils::inverseApply(surface, hField);
              SurfaceUtils::imageIndexToWorld(surface, dstOrigin, dstSpacing);
            }
	  else if (transformType == AFFINE)
	    {
	      if(!affine.invert()){
		std::cerr << "Error inverting affine transform!" << std::endl;
		exit(0);
	      }
	      surface.applyAffineTransform(affine);
	    }
        }

      //
      // write out surface
      //
      try
        {
          surface.writeBYU(outputFilename.c_str());      
        }
      catch (...)
        {
          std::cerr << "Can't write surface: " << outputFilename << std::endl;
          exit(-1);
        }      
    }
  else
    {
      //
      // load input image
      //
      Image<float> image;
      std::cerr << "Loading image: " << imageFileName << "...";
      ApplicationUtils::LoadImageITK(imageFileName.c_str(), image);
      std::cerr << "DONE" << std::endl;
      std::cerr << "  Dimensions: " << image.getSize() << std::endl;
      std::cerr << "  Origin    : " << image.getOrigin() << std::endl;
      std::cerr << "  Spacing   : " << image.getSpacing() << std::endl;

      if(useHFieldOutInfo){

	if(useOutSizeInfo || useOutOriginInfo || useOutSpacingInfo){
          std::cerr << "Cannot set output size/origin/spacing and request to "
		    << "use this info from hField!" << std::endl;
          exit(-1);
	}
	
	if(transformType == HFIELD){
	  outSize = hField.getSize();
	  outOrigin = hOrigin;
	  outSpacing = hSpacing;
	}else{
          std::cerr << "Can't use info from hField, not transforming by hfield!" << std::endl;
          exit(-1);
	}

      }else{
	if(!useOutSizeInfo){
	  outSize = image.getSize();
	}
	
	if(!useOutOriginInfo){
	  outOrigin = image.getOrigin();
	}
	
	if(!useOutSpacingInfo){
	  outSpacing = image.getSpacing();
	}
      }

      //
      // create deformed image and hfield
      //
      Image<VoxelType> def(image);
      
      if (transformType == TRANSLATION)
        {
          if (forward)
            {
              ImageUtils::translate(def, translation);
            }
          else
            {
              ImageUtils::translate(def, -translation);
            }
        }
      else if(transformType == HFIELD)
        {
          //
          // compute starting voxel of roi
          // NB: assume that spacing hfield == spacing image
          //
          if (hSpacing != image.getSpacing() && forward) {
	    
            std::cerr << "WARNING: hField and image spacing do not agree!"
                      << std::endl
                      << "       : this operation is not currently supported"
                      << std::endl;
          }
          if (forward)
            {
	      if(nearestNeighbor || cubic){
		std::cerr << "WARNING: only linear interpolation supported by forwardApply" << std::endl;
	      }
	      Vector3D<double> iOrigin  = image.getOrigin();
	      Vector3D<double> iSpacing = image.getSpacing();
	      Vector3D<double> roiStart = (hOrigin-iOrigin) / iSpacing;
	      Vector3D<int> roi((int)round(roiStart.x),
				(int)round(roiStart.y),
				(int)round(roiStart.z));
	      std::cerr << "ROI: hField (0,0,0) corresponds to image " << roi 
			<< std::endl 
			<< "     [rounded from " << roiStart << "]" << std::endl;
	      std::cerr << "Deforming image...";
              HField3DUtils::forwardApply(image, hField, def, 
                                          roi.x, roi.y, roi.z, 0.0F);
            }
          else
            {

	      def.resize(outSize);
	      def.setOrigin(outOrigin);
	      def.setSpacing(outSpacing);
	      
	      
	      if(nearestNeighbor){
		HField3DUtils::
		  applyAtNewResolution<float, float, DEFAULT_SCALAR_BACKGROUND_STRATEGY, Array3DUtils::INTERP_NN>
		  (image, hField, def,
		   hOrigin,
		   hSpacing,
		   0.0F);
	      }else if(cubic){
		HField3DUtils::
		  applyAtNewResolution<float, float, DEFAULT_SCALAR_BACKGROUND_STRATEGY, Array3DUtils::INTERP_CUBIC>
		  (image, hField, def,
		   hOrigin,
		   hSpacing,
		   0.0F);
	      }else{
		HField3DUtils::
		  applyAtNewResolution<float, float, DEFAULT_SCALAR_BACKGROUND_STRATEGY, Array3DUtils::INTERP_LINEAR>
		  (image, hField, def,
		   hOrigin,
		   hSpacing,
		   0.0F);
	      }
	      
            }
          std::cerr << "DONE" << std::endl;
        }
      else if(transformType == AFFINE)
	{
	  // invert the affine matrix if necessary
	  if (!forward)
            {
	      if(!affine.invert()){
		std::cerr << "Error inverting affine transform!" << std::endl;
		exit(0);
	      }
            }
          Vector3D<double> iOrigin  = image.getOrigin();
          Vector3D<double> iSpacing = image.getSpacing();
	  std::cout << "Initializing transform with affine transform" << std::endl;
	  hField.resize(image.getSize());
	  Vector3D<float> affOrigin(0.f, 0.f, 0.f);
	  if(!zeroAffOrigin){
	    affOrigin = iOrigin;
	  }
	  HField3DUtils::initializeFromAffine(hField, 
					      affine,
					      affOrigin,
					      iSpacing);
	  if(nearestNeighbor){
	    HField3DUtils::
	      applyAtNewResolution<float, float, DEFAULT_SCALAR_BACKGROUND_STRATEGY, Array3DUtils::INTERP_NN>
	      (image, hField, def,
	       iOrigin,
	       iSpacing,
	       0.0F);
	  }else if(cubic){
	    HField3DUtils::
	      applyAtNewResolution<float, float, DEFAULT_SCALAR_BACKGROUND_STRATEGY, Array3DUtils::INTERP_CUBIC>
	      (image, hField, def,
	       iOrigin,
	       iSpacing,
	       0.0F);
	  }else{
	    HField3DUtils::
	      applyAtNewResolution<float, float, DEFAULT_SCALAR_BACKGROUND_STRATEGY, Array3DUtils::INTERP_LINEAR>
	      (image, hField, def,
	       iOrigin,
	       iSpacing,
	       0.0F);
	  }
	  
	}
      
      //
      // write deformed image
      //
      if (outputFilename != "")
        {
	  // if no format specified...
	  if(outputFilename.find(".") == std::string::npos){
	    // append default .mhd format
	    outputFilename.append(".mhd");
	  }
          std::cerr << "Writing Deformed Image...";
	  ApplicationUtils::SaveImageITK(outputFilename.c_str(), def);
          std::cerr << "DONE" << std::endl;
        }
    }
  return 0;
}
  
  
  
