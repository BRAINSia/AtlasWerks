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



#include "Anastruct.h"
#include "Surface.h"
#include "AnastructUtils.h"
#include "ApplicationUtils.h"
#include "CmdLineParser.h"
#include "AtlasWerksTypes.h"
#include "ImageUtils.h"

Vector3D<double> calcCentroid(Anastruct &ana)
{
  Vector3D<double> centroid(0.0, 0.0, 0.0);
  unsigned int nVertices = 0;
  for(unsigned int cIdx = 0; cIdx < ana.contours.size(); cIdx++){
    Contour &curContour = ana.contours[cIdx];
    for(unsigned int pIdx = 0; pIdx < curContour.vertices.size(); pIdx++){
      centroid += curContour.vertices[pIdx];
      nVertices++;
    }
  }
  centroid /= ((double)nVertices);
  return centroid;
}

Vector3D<double> calcCentroid(Surface &surf)
{
  Vector3D<double> centroid(0.0, 0.0, 0.0);
  unsigned int nVertices = surf.vertices.size();
  for(unsigned int pIdx = 0; pIdx < nVertices; pIdx++){
    centroid += surf.vertices[pIdx];
  }
  centroid /= ((double)nVertices);
  return centroid;
}

/** Simple parameter class to hold parameters for LDMMWarp program */
class AlignCentroidsParamFile : public CompoundParam {
public:
  AlignCentroidsParamFile()
    : CompoundParam("AlignCentroidsParameterFile", "top-level node", PARAM_REQUIRED)
  {
    this->AddChild(ValueParam<std::string>("StaticImage", "static image filename", PARAM_REQUIRED, ""));
    this->AddChild(ValueParam<std::string>("MovingImage", "moving image filename", PARAM_REQUIRED, ""));
    this->AddChild(ValueParam<std::string>("StaticAnastruct", "static anastruct filename", PARAM_COMMON, ""));
    this->AddChild(ValueParam<std::string>("StaticBYU", "static anastruct filename", PARAM_COMMON, ""));
    this->AddChild(ValueParam<std::string>("MovingAnastruct", "moving anastruct filename", PARAM_COMMON, ""));
    this->AddChild(ValueParam<std::string>("MovingBYU", "moving anastruct filename", PARAM_COMMON, ""));
    this->AddChild(ValueParam<std::string>("CentroidAlignedImage", "output image filename after centroid alignment", PARAM_COMMON, ""));
    this->AddChild(ValueParam<std::string>("ResampledImage", "output image filename after alignment and resampling", PARAM_REQUIRED, "out.mhd"));
  }

  ValueParamAccessorMacro(std::string, StaticImage)
  ValueParamAccessorMacro(std::string, MovingImage)
  ValueParamAccessorMacro(std::string, StaticAnastruct)
  ValueParamAccessorMacro(std::string, StaticBYU)
  ValueParamAccessorMacro(std::string, MovingAnastruct)
  ValueParamAccessorMacro(std::string, MovingBYU)
  ValueParamAccessorMacro(std::string, CentroidAlignedImage)
  ValueParamAccessorMacro(std::string, ResampledImage)

  CopyFunctionMacro(AlignCentroidsParamFile)
};

/**
 * \page AlignCentroids
 * Align centroids based on anastruct centroids
 */
int main(int argc, char ** argv)
{

  AlignCentroidsParamFile pf;

  CmdLineParser parser(pf);
  try{
    parser.Parse(argc,argv);
  }catch(ParamException &e){
    std::cerr << "Error parsing arguments:" << std::endl;
    std::cerr << "   " << e.what() << std::endl;
    std::exit(-1);
  }

  //
  // Load images
  //
  RealImage *staticImage = new RealImage(); 
  RealImage *movingImage = new RealImage(); 
  ApplicationUtils::LoadImageITK(pf.StaticImage().c_str(),*staticImage);
  ApplicationUtils::LoadImageITK(pf.MovingImage().c_str(),*movingImage);

  Vector3D<unsigned int> staticSize = staticImage->getSize();
  Vector3D<double> staticSpacing = staticImage->getSpacing();
  Vector3D<double> staticOrigin = staticImage->getOrigin();

  Vector3D<unsigned int> movingSize = movingImage->getSize();
  Vector3D<double> movingSpacing = movingImage->getSpacing();
  Vector3D<double> movingOrigin = movingImage->getOrigin();

  bool movingCentroidSet = false;
  bool staticCentroidSet = false;
  Vector3D<double> staticCentroid;
  Vector3D<double> movingCentroid;

  // see if anastructs set

  if(pf.StaticAnastruct().size()){
    Anastruct *staticAnastruct = new Anastruct();
    AnastructUtils::readPLUNCAnastruct(*staticAnastruct, pf.StaticAnastruct().c_str());
    staticCentroid = calcCentroid(*staticAnastruct);
    staticCentroidSet = true;
  }
  if(pf.MovingAnastruct().size()){
    Anastruct *movingAnastruct = new Anastruct();
    AnastructUtils::readPLUNCAnastruct(*movingAnastruct, pf.MovingAnastruct().c_str());
    movingCentroid = calcCentroid(*movingAnastruct);
    movingCentroidSet = true;
  }

  // see if surfaces set
  if(pf.StaticBYU().size()){
    if(staticCentroidSet){
      throw AtlasWerksException(__FILE__,__LINE__,"Error, both anastruct and surface set for static centroid");
    }
    Surface *staticSurface = new Surface();
    staticSurface->readBYU(pf.StaticBYU().c_str());
    staticCentroid = calcCentroid(*staticSurface);
    staticCentroidSet = true;
  }
  if(pf.MovingBYU().size()){
    if(movingCentroidSet){
      throw AtlasWerksException(__FILE__,__LINE__,"Error, both anastruct and surface set for moving centroid");
    }
    Surface *movingSurface = new Surface();
    movingSurface->readBYU(pf.MovingBYU().c_str());
    movingCentroid = calcCentroid(*movingSurface);
    movingCentroidSet = true;
  }

  if(!staticCentroidSet || !movingCentroidSet){
    throw AtlasWerksException(__FILE__,__LINE__,"Error, moving and static centroid files must be set");
  }
  
  Vector3D<double> offset = staticCentroid - movingCentroid;
  std::cout << "centroid offset is " << offset << std::endl;

  movingOrigin += offset;
  movingImage->setOrigin(movingOrigin);

  std::cout << "new origin is " << movingOrigin << std::endl;

  if(pf.CentroidAlignedImage().size()){
    std::cout << "Writing centroid-aligned image " << pf.CentroidAlignedImage() << std::endl;
    ApplicationUtils::SaveImageITK(pf.CentroidAlignedImage().c_str(), *movingImage);
  }
  
  if(pf.ResampledImage().size()){
    std::cout << "Resampling image " << std::endl;
    ImageUtils::resampleNew<Real, 
      Array3DUtils::BACKGROUND_STRATEGY_VAL,
      DEFAULT_SCALAR_INTERP>
      (*movingImage, staticOrigin, staticSpacing, staticSize);
    
    std::cout << "Writing resampled image " << pf.ResampledImage() << std::endl;
    ApplicationUtils::SaveImageITK(pf.ResampledImage().c_str(), *movingImage);
  }
  
}

