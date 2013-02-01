
#ifdef WIN32
#pragma warning (disable: 4786) // truncated debug info
#endif

#include <iostream>
#include <stdexcept>
#include <string>
#include "Array3DUtils.h"
#include "BinaryIO.h"
#include "DownsampleFilter3D.h"
#include "HField3DIO.h"
#include "HField3DUtils.h"
#include "ImageUtils.h"
#include "MultiScaleFluidWarp.h"
#include "ROIUtils.h"
#include "Timer.h"


/////////////////
// constructor //
/////////////////

MultiScaleFluidWarp
::MultiScaleFluidWarp()
{
  _haveValidHField     = false;
  _useOriginAndSpacing = false;
  _verbose             = true;
  _atlasName           = "NOT SPECIFIED";
  _subjectName         = "NOT SPECIFIED";
}

////////////////
// destructor //
////////////////

MultiScaleFluidWarp
::~MultiScaleFluidWarp()
{
  std::cerr << "[MultiScaleFluidWarp::~MultiScaleFluidWarp]" << std::endl;
}
	

////////////////
// createWarp //
////////////////

void
MultiScaleFluidWarp
::createWarp(ImagePointer& atlas,
             ImagePointer& subject,
             const char *atlasName,
             const char *subjectName,
             const VoxelType& atlasRescaleMinThreshold,
             const VoxelType& atlasRescaleMaxThreshold,
             const VoxelType& subjectRescaleMinThreshold,
             const VoxelType& subjectRescaleMaxThreshold,            
             const ImageRegionType& roi,
             unsigned int numScaleLevels,
             FluidWarpParameters fluidParameters[],
             int reslice,
             int deformationType,
             MaskType mask)
{
  // make sure images are compatible
  if (!_checkImageCompatibility(atlas, subject))
    {
      std::cerr << "fluid warp: incompatible images." << std::endl;
      throw std::invalid_argument("atlas and subject are not compatible");
    }
	
  // we need at least one scale level
  if (numScaleLevels < 1)
    {
      throw std::invalid_argument("numScaleLevels == 0");
    }
	
  // save filenames
  if (atlasName != 0)
    {
      _atlasName = atlasName;
    }
  if (subjectName != 0)
    {
      _subjectName = subjectName;
    }
	
  // save origin and spacing
  _atlasOrigin.set(atlas->getOrigin().x,
		   atlas->getOrigin().y,
		   atlas->getOrigin().z);
  _atlasSpacing.set(atlas->getSpacing().x,
		    atlas->getSpacing().y,
		    atlas->getSpacing().z);
  _subjectOrigin.set(subject->getOrigin().x,
		     subject->getOrigin().y,
		     subject->getOrigin().z);
  _subjectSpacing.set(subject->getSpacing().x,
		      subject->getSpacing().y,
		      subject->getSpacing().z);
  // hfield origin and spacing are set below.

  std::cerr << "Fluid: input roi origin = ("
	    << roi.getStartX() << ", " 
	    << roi.getStartY() << ", "
	    << roi.getStartZ() << ")" << std::endl;
  std::cerr << "Fluid: input roi size = ("
	    << roi.getSizeX() << ", " 
	    << roi.getSizeY() << ", "
	    << roi.getSizeZ() << ")" << std::endl;

#if 0	
  //
  // make sure the roi is on the image
  //
  ImageIndexType startIndex = roi.getStart();
  ImageIndexType stopIndex  = roi.getStop();
  stopIndex = (roi.getStop() - 1);
	
  if (startIndex.x < 0) startIndex.x = 0;
  if (startIndex.y < 0) startIndex.y = 0;
  if (startIndex.z < 0) startIndex.z = 0;
	
  if (stopIndex.x >= static_cast<int>(atlas->getSizeX()))
    {
      stopIndex.x = atlas->getSizeX() - 1;
    }
  if (stopIndex.y >= static_cast<int>(atlas->getSizeY()))
    {
      stopIndex.y = atlas->getSizeY() - 1;
    }
  if (stopIndex.z >= static_cast<int>(atlas->getSizeZ()))
    {
      stopIndex.z = atlas->getSizeZ() - 1;
    }
	
  _roi.setStart(startIndex);
  _roi.setStop(stopIndex);      

  std::cerr << "Fluid: adjusted roi origin = ("
	    << _roi.getStartX() << ", " 
	    << _roi.getStartY() << ", "
	    << _roi.getStartZ() << ")" << std::endl;
  std::cerr << "Fluid: adjusted roi size = ("
	    << _roi.getSizeX() << ", " 
	    << _roi.getSizeY() << ", "
	    << _roi.getSizeZ() << ")" << std::endl;
#else
  _roi = roi;
#endif

  _hFieldOrigin = _atlasOrigin +
                  Vector3D<double>(_roi.getStart()) * _atlasSpacing;
  _hFieldSpacing = _atlasSpacing;
	
  // get num downsample levels required
  int maxDownsampleLevel = 0;
  for (unsigned int level = 1; level < numScaleLevels; level++)
    {
      unsigned int nbIterations = fluidParameters[level].numIterations;
      if (nbIterations != 0)
	{
	  maxDownsampleLevel = level;
	}
    }
	
  // create required downsampled (in size) images
  if (_verbose)
    {
      std::cerr << "Creating scaled images..." << std::endl;
    }

  Timer timer;
  timer.start();
	
  if (_verbose)
    {
      std::cerr << "\textracting subregions..." << std::endl;
    }
	
  ImagePointer *atlasROIChunk = new ImagePointer[maxDownsampleLevel + 1];
  ImagePointer *subjectROIChunk = new ImagePointer[maxDownsampleLevel + 1];

  for(int lcv = 0; lcv < maxDownsampleLevel + 1; lcv++) {
    atlasROIChunk[lcv] = new ImageType;
    subjectROIChunk[lcv] = new ImageType;
  }
  _createImageFromROI(atlas, atlasROIChunk[0]);
  _createImageFromROI(subject, subjectROIChunk[0]);

  MaskType *maskROIChunk = new MaskType[maxDownsampleLevel + 1];
  if(deformationType == 2)
  {
  ROIUtils<float>::extractROIFromArray3D(mask, maskROIChunk[0], _roi );
  }

  /*
    for(unsigned int z = 0 ; z <atlasROIChunk[0]->getSize().z ; z++){
    for(unsigned int y = 0 ; y <atlasROIChunk[0]->getSize().y ; y++){
    for(unsigned int x = 0 ; x <atlasROIChunk[0]->getSize().x ; x++){
    float atlasIntensity = atlasROIChunk[0]->get(x,y,z);
    float subjectIntensity = subjectROIChunk[0]->get(x,y,z);
    if (atlasIntensity < 818)
    {
    atlasROIChunk[0]->set(x,y,z,1050);
    }
    if (subjectIntensity < 818)
    {
    subjectROIChunk[0]->set(x,y,z,1050);
    }
    }}}
  */
	
  Vector3D<unsigned int> initialSize(atlasROIChunk[0]->getSize());
  //float NewSpacing;
  //reslice the first atlas and subject ROIchunk to have the same spacing in every direction
  //Atlas and subject have the same spacing so we can use the same value
  float spacingX = atlasROIChunk[0]->getSpacing().x;
  float spacingY = atlasROIChunk[0]->getSpacing().y;
  float spacingZ = atlasROIChunk[0]->getSpacing().z;
	
  if ((fabs(spacingX) == fabs(spacingY))&&
      (fabs(spacingX) != fabs(spacingZ))&& (reslice==1))
    {
      float NewSpacing = spacingX;
      //the new and the old spacing must have the sign 
      if ((NewSpacing * spacingZ) < 0)
	{
	  NewSpacing = -NewSpacing;
	}
      _resliceImage(atlasROIChunk[0], NewSpacing);
      _resliceImage(subjectROIChunk[0],NewSpacing);
      if(deformationType == 2)
      {
      _resliceMask(maskROIChunk[0], atlasROIChunk[0]->getSizeZ());
      }
    }

  if (_verbose)
    {
      std::cerr << "\tdownsampling..." << std::endl;
    }  
  if(deformationType == 2)
  {
    Vector3D<double> scaleFactors(2,2,2);
    Vector3D<double> sigma(scaleFactors);
    Vector3D<double> kernelSize(2*scaleFactors);
    for (int level_down = 1; level_down <= maxDownsampleLevel; level_down++)
    {
      Array3DUtils::gaussianDownsample(maskROIChunk[level_down - 1],maskROIChunk[level_down],
                                       scaleFactors, sigma, kernelSize);
    }

    
  }
  for (int level_down = 1; level_down <= maxDownsampleLevel; level_down++)
    {
      _downsampleImageBy2WithGaussianFilter(atlasROIChunk[level_down - 1],atlasROIChunk[level_down]);
      _downsampleImageBy2WithGaussianFilter(subjectROIChunk[level_down - 1],subjectROIChunk[level_down]);
    }
  
  setAtlasROIChunk(atlasROIChunk);
  setSubjectROIChunk(subjectROIChunk);
  timer.stop();
	
  if (_verbose)
    {
      std::cerr << "DONE, Time (sec) " << timer.getSeconds() << std::endl;
    }
	
	
  // run warp at each scale level, starting with coarsest scale
	
  // deformated Atlas to compute the hfield between atlas and subject
  Array3D<unsigned short> defAtlasROI3D;
  Array3D<Vector3D<float> > newHField;
  Array3D<Vector3D<float> > RoiHField, RoiHFieldInv;
  RoiHField.resize(atlasROIChunk[maxDownsampleLevel]->getSize());
  RoiHFieldInv.resize(subjectROIChunk[maxDownsampleLevel]->getSize());
  HField3DUtils::setToIdentity(RoiHField);
  FluidWarp fluidWarp;
  int level_up;
  for (level_up = maxDownsampleLevel; level_up >= 0; level_up--)
    {
      unsigned int nbIterations = fluidParameters[level_up].numIterations;
		
      // dont process this scale level_up if there are no iterations
      if (nbIterations == 0) continue;
		
      // save parameters 
      _fluidParameters.push_back(std::pair<int, FluidWarpParameters>(level_up, fluidParameters[level_up]));
		
      Timer timer;
      timer.start();
		
      //       // write out images for debug
      //       std::ostringstream atlasFileName;
      //       atlasFileName << "atlaslevel_up" << level_up << ".mhd";
      //       _saveImage(atlasROIChunk[level_up], atlasFileName.str().c_str());
      //       std::ostringstream subjectFileName;      
      //       subjectFileName << "subjectlevel_up" << level_up << ".mhd";
      //       _saveImage(subjectROIChunk[level_up], subjectFileName.str().c_str());
		
      //convert roi chunks to Array3D_uchar
      Array3D<float> atlasROI3D;
      _convertImageToArray3DUChar(atlasROIChunk[level_up], atlasROI3D,
				  atlasRescaleMinThreshold,atlasRescaleMaxThreshold);
      Array3D<float> subjectROI3D;
      _convertImageToArray3DUChar(subjectROIChunk[level_up], subjectROI3D,
				  subjectRescaleMinThreshold,subjectRescaleMaxThreshold);

      //
      // tmpdebug: write roi chunk for comparison
      //
      // std::ostringstream oss;
      // oss << "imapFixedROILevel" << level_up;
      // Array3DIO::writeMETAVolume(atlasROI3D, oss.str().c_str());
      // oss.str("");
      // oss << "imapMovingROILevel" << level_up;
      // Array3DIO::writeMETAVolume(subjectROI3D, oss.str().c_str());
      //

      // create scale arrays
      Vector3D<double> atlasScale;
      atlasScale = atlas->getSpacing() * _pow( (double)2, (double)level_up);
		
      Vector3D<double> subjectScale;
      subjectScale = subject->getSpacing() * _pow( (double)2, (double)level_up);
		
      switch(deformationType)
      {
      case 0: 
        fluidWarp.computeHFieldAsymmetric(atlasROI3D,
                                          subjectROI3D,
                                          fluidParameters[level_up],
                                          RoiHField,RoiHFieldInv);
        break;
        
      case 1:
        fluidWarp.computeHFieldElastic(atlasROI3D,
                                       subjectROI3D,
                                       fluidParameters[level_up],
                                       RoiHField);
        break;
      case 2:
        fluidWarp.computeHFieldElasticWithMask(atlasROI3D,
                                       subjectROI3D,
                                       maskROIChunk[level_up],
                                       fluidParameters[level_up],
                                       RoiHField);
        break;
      }
      if (level_up > 0){
	Vector3D<unsigned int> atlas_roiSize = 
	  atlasROIChunk[level_up-1]->getSize();

	newHField.resize(atlasROIChunk[level_up-1]->getSize());
	HField3DUtils::resample(RoiHField, newHField, atlas_roiSize );

	Vector3D<unsigned int> subject_roiSize = 
	  subjectROIChunk[level_up-1]->getSize();

	RoiHField.resize(atlasROIChunk[level_up-1]->getSize());
	RoiHField.setData(newHField);
	HField3DUtils::resample(RoiHFieldInv, newHField,
				subject_roiSize);
	RoiHFieldInv.resize(subjectROIChunk[level_up-1]->getSize());

	RoiHFieldInv.setData(newHField);
      }
    }
  _hField.resize(initialSize);
			
  HField3DUtils::resample(RoiHField, _hField,
			  initialSize);
		
  _hFieldInv.resize(initialSize);
			
  HField3DUtils::resample(RoiHField, _hFieldInv,
			  initialSize);
	

  timer.stop();
		
  if (_verbose)
    {
      std::cerr << "MultiScaleFluidWarp at level_up " << level_up 
		<< " finished, Time (sec) " 
		<< timer.getSeconds() << std::endl;
	
    }

  // clean up memory
  delete [] atlasROIChunk;
  delete [] subjectROIChunk;
	
  // save hfield from finest level
  std::cerr << "resampling hField...";
	
  std::cerr << "DONE" << std::endl;
	
  _haveValidHField = true;
}

void
MultiScaleFluidWarp
::shrink(ImagePointer& atlas,
         const char *atlasName,
         const VoxelType& atlasRescaleMinThreshold,
         const VoxelType& atlasRescaleMaxThreshold,  
         const ImageRegionType& roi,
         unsigned int numScaleLevels,
         FluidWarpParameters fluidParameters[],
         int reslice,
         int deformationType,
         MaskType mask)
{	
  // we need at least one scale level
  if (numScaleLevels < 1)
  {
    throw std::invalid_argument("numScaleLevels == 0");
  }
  
  // save filenames
  if (atlasName != 0)
  {
    _atlasName = atlasName;
  }
  
  // save origin and spacing
  _atlasOrigin.set(atlas->getOrigin().x,
    atlas->getOrigin().y,
    atlas->getOrigin().z);
  _atlasSpacing.set(atlas->getSpacing().x,
    atlas->getSpacing().y,
    atlas->getSpacing().z);
  
  //
  // make sure the roi is on the image
  //
  std::cerr << "Fluid: input roi origin = ("
    << roi.getStartX() << ", " 
    << roi.getStartY() << ", "
    << roi.getStartZ() << ")" << std::endl;
  std::cerr << "Fluid: input roi size = ("
    << roi.getSizeX() << ", " 
    << roi.getSizeY() << ", "
    << roi.getSizeZ() << ")" << std::endl;
  
  ImageIndexType startIndex = roi.getStart();
  ImageIndexType stopIndex  = roi.getStop();
  stopIndex = (roi.getStop());
  
  if (startIndex.x < 0) startIndex.x = 0;
  if (startIndex.y < 0) startIndex.y = 0;
  if (startIndex.z < 0) startIndex.z = 0;
  
  if (stopIndex.x >= static_cast<int>(atlas->getSizeX()))
  {
    stopIndex.x = atlas->getSizeX() - 1;
  }
  if (stopIndex.y >= static_cast<int>(atlas->getSizeY()))
  {
    stopIndex.y = atlas->getSizeY() - 1;
  }
  if (stopIndex.z >= static_cast<int>(atlas->getSizeZ()))
  {
    stopIndex.z = atlas->getSizeZ() - 1;
  }
  
  _roi.setStart(startIndex);
  _roi.setStop(stopIndex);      
  
  std::cerr << "Fluid: adjusted roi origin = ("
    << _roi.getStartX() << ", " 
    << _roi.getStartY() << ", "
    << _roi.getStartZ() << ")" << std::endl;
  std::cerr << "Fluid: adjusted roi size = ("
    << _roi.getSizeX() << ", " 
    << _roi.getSizeY() << ", "
    << _roi.getSizeZ() << ")" << std::endl;

  _hFieldOrigin = _atlasOrigin +
                  Vector3D<double>(_roi.getStart()) * _atlasSpacing;
  _hFieldSpacing = _atlasSpacing;
  
  // get num downsample levels required
  int maxDownsampleLevel = 0;
  for (unsigned int level = 1; level < numScaleLevels; level++)
  {
    unsigned int nbIterations = fluidParameters[level].numIterations;
    if (nbIterations != 0)
    {
      maxDownsampleLevel = level;
    }
  }
  
  // create required downsampled (in size) images
  if (_verbose)
  {
    std::cerr << "Creating scaled images..." << std::endl;
  }
  
  Timer timer;
  timer.start();
  
  if (_verbose)
  {
    std::cerr << "\textracting subregions..." << std::endl;
  }
  
  ImagePointer *atlasROIChunk = new ImagePointer[maxDownsampleLevel + 1];
  
  //extracting subregions of the atlas
  for(int lcv = 0; lcv < maxDownsampleLevel + 1; lcv++) {
    atlasROIChunk[lcv] = new ImageType;
  }
  _createImageFromROI(atlas, atlasROIChunk[0]);
  
  //extracting subregions of the mask
  MaskType *maskROIChunk = new MaskType[maxDownsampleLevel + 1];
  if(deformationType == 4)
  {
    ROIUtils<float>::extractROIFromArray3D(mask, maskROIChunk[0], _roi );
  }
  
  Vector3D<unsigned int> initialSize(atlasROIChunk[0]->getSize());
  
  // reslice the first atlas ROIchunk to have the same spacing in
  // every direction
  float spacingX = atlasROIChunk[0]->getSpacing().x;
  float spacingY = atlasROIChunk[0]->getSpacing().y;
  float spacingZ = atlasROIChunk[0]->getSpacing().z;
  
  if ((fabs(spacingX) == fabs(spacingY))&&
    (fabs(spacingX) != fabs(spacingZ))&& (reslice==1))
  {
    float NewSpacing = spacingX;
    //the new and the old spacing must have the sign 
    if ((NewSpacing * spacingZ) < 0)
    {
      NewSpacing = -NewSpacing;
    }
    _resliceImage(atlasROIChunk[0], NewSpacing);
  }
  //reslice the mask
  if((fabs(spacingX) == fabs(spacingY))&&
    (fabs(spacingX) != fabs(spacingZ))&& (reslice==1) && (deformationType == 4))
  {
    _resliceMask(maskROIChunk[0], atlasROIChunk[0]->getSizeZ());
  }
  
  //
  // rescale intensities 
  //
  std::cerr << "rescaling intensities" << std::endl;
  Array3DUtils::rescaleElements(*atlasROIChunk[0], 
				atlasRescaleMinThreshold,
        atlasRescaleMaxThreshold,
        0.0f, 255.0f);
  
  //
  // apply min filter to dilate dark regions
  //
  std::cerr << "dilating dark regions..." << std::endl;
  Array3DUtils::maxFilter3D(*atlasROIChunk[0]);
  Array3DUtils::maxFilter3D(*atlasROIChunk[0]);
  Array3DUtils::minFilter3D(*atlasROIChunk[0]);
  Array3DUtils::minFilter3D(*atlasROIChunk[0]);
  Array3DUtils::minFilter3D(*atlasROIChunk[0]);
  Array3DUtils::minFilter3D(*atlasROIChunk[0]);
  
  if (_verbose)
  {
    std::cerr << "\tdownsampling..." << std::endl;
  }  
  if(deformationType == 4)
  {
    Vector3D<double> scaleFactors(2,2,2);
    Vector3D<double> sigma(scaleFactors);
    Vector3D<double> kernelSize(2*scaleFactors);
    for (int level_down = 1; level_down <= maxDownsampleLevel; level_down++)
    {
      Array3DUtils::gaussianDownsample(maskROIChunk[level_down - 1],maskROIChunk[level_down],
        scaleFactors, sigma, kernelSize);
    }
    
  }
  
  for (int level_down = 1; level_down <= maxDownsampleLevel; level_down++)
  {
    _downsampleImageBy2WithGaussianFilter(atlasROIChunk[level_down - 1],atlasROIChunk[level_down]);    
  }
  
  timer.stop();
  
  if (_verbose)
  {
    std::cerr << "DONE, Time (sec) " << timer.getSeconds() << std::endl;
  }
  
  
  // run warp at each scale level, starting with coarsest scale
  
  // deformated Atlas to compute the hfield between atlas and subject
  Array3D<unsigned short> defAtlasROI3D;
  Array3D<Vector3D<float> > newHField;
  Array3D<Vector3D<float> > RoiHField, RoiHFieldInv;
  RoiHField.resize(atlasROIChunk[maxDownsampleLevel]->getSize());
  RoiHFieldInv.resize(atlasROIChunk[maxDownsampleLevel]->getSize());
  HField3DUtils::setToIdentity(RoiHField);
  HField3DUtils::setToIdentity(RoiHFieldInv);
  FluidWarp fluidWarp;
  int level_up;
  for (level_up = maxDownsampleLevel; level_up >= 0; level_up--)
  {
    unsigned int nbIterations = fluidParameters[level_up].numIterations;
    
    // dont process this scale level_up if there are no iterations
    if (nbIterations == 0) continue;
    
    // save parameters 
    _fluidParameters.push_back(std::pair<int, FluidWarpParameters>(
                                   level_up, fluidParameters[level_up]));
    
    Timer timer;
    timer.start();
    
    // create scale arrays
    Vector3D<double> atlasScale;
    atlasScale = atlas->getSpacing() * _pow( (double)2, (double)level_up);
    
    switch(deformationType)
    {
    case 3: 
        fluidWarp.shrinkRegion(*atlasROIChunk[level_up],
                               fluidParameters[level_up],
                               RoiHField,
                               RoiHFieldInv);
        break;
    case 4:
        fluidWarp.elasticShrinkRegionWithMask(*atlasROIChunk[level_up],
                                              fluidParameters[level_up],
                                              maskROIChunk[level_up],
                                              RoiHField,false);
        break;
    }
    
    if (level_up > 0){
      newHField.resize(atlasROIChunk[level_up-1]->getSize());

      HField3DUtils::resample(RoiHField, newHField,
        atlasROIChunk[level_up-1]->getSize().x,
        atlasROIChunk[level_up-1]->getSize().y,
        atlasROIChunk[level_up-1]->getSize().z);
      RoiHField.resize(atlasROIChunk[level_up-1]->getSize());
      RoiHField.setData(newHField);

      HField3DUtils::resample(RoiHFieldInv, newHField,
        atlasROIChunk[level_up-1]->getSize().x,
        atlasROIChunk[level_up-1]->getSize().y,
        atlasROIChunk[level_up-1]->getSize().z);
      RoiHFieldInv.resize(atlasROIChunk[level_up-1]->getSize());
      RoiHFieldInv.setData(newHField);
    }
  }
  _hField.resize(initialSize);

  HField3DUtils::resample(RoiHField, _hField, initialSize);
  
  HField3DUtils::resample(RoiHFieldInv, _hFieldInv, initialSize);

  timer.stop();
  
  //    //debug with jacobian
  //    
  //      Array3D<float> jacobianArray(_hField.getSize());
  //      HField3DUtils::jacobian(_hField,jacobianArray);
  //      std::string filename("F:/David prigent/unc/Matlab/image Test/jacobian.raw");  
  //      Array3DIO::writeRawVolume(jacobianArray,filename.c_str());
  //    
  //      //debug with jacobian
  //    
  //      //debug with gradient
  //    
  //      Array3D<Vector3D<float> > gradientVectorField(atlas->getSize());
  //      Array3D<float> gradient(atlas->getSize());
  //      Array3DUtils::computeGradient(*atlas,gradientVectorField);
  //      unsigned int x,y,z;
  //      for(z = 0 ; z < atlas->getSize().z ; z++){
  //        for(y = 0 ; y < atlas->getSize().y ; y++){
  //          for(x = 0 ; x < atlas->getSize().x ; x++){
  //            gradient.set(x,y,z,gradientVectorField(x,y,z).normL2());
  //          }}}
  //      std::string filename3("F:/David prigent/unc/Matlab/image Test/gradient.raw");  
  //      Array3DIO::writeRawVolume(gradient,filename3.c_str());
  //    
  //      //debug with gradiant
  //    
  //      //debug with laplacian
  //    
  //      Array3D<Vector3D<float> > laplacianVectorField(_hField.getSize());
  //      Array3D<float> laplacian(_hField.getSize());
  //      Array3DUtils::computeLaplacian(_hField,laplacianVectorField);
  //      for(z = 0 ; z < _hField.getSize().z ; z++){
  //        for(y = 0 ; y < _hField.getSize().y ; y++){
  //          for(x = 0 ; x < _hField.getSize().x ; x++){
  //            laplacian.set(x,y,z,laplacianVectorField(x,y,z).normL2());
  //          }}}
  //      std::string filename4("F:/David prigent/unc/Matlab/image Test/laplacian.raw");  
  //      Array3DIO::writeRawVolume(laplacian,filename4.c_str());
  //    
  //      //debug with laplacian
  //    
  //      //debug with displacement magnitude
  //      Array3D<Vector3D<float> >  displacement(_hField.getSize());
  //      Array3D<Vector3D<float> >  identity(_hField.getSize());
  //      Array3D<float> displacementMagnitude(_hField.getSize());
  //      HField3DUtils::setToIdentity(identity);
  //      for(z = 0 ; z < _hField.getSize().z ; z++){
  //        for(y = 0 ; y < _hField.getSize().y ; y++){
  //          for(x = 0 ; x < _hField.getSize().x ; x++){
  //            displacement.set(x,y,z,(_hField.get(x,y,z) - identity.get(x,y,z)));
  //            displacementMagnitude.set(x,y,z,displacement(x,y,z).normL2());
  //          }}}
  //      std::string filename2("F:/David prigent/unc/Matlab/image Test/displacementMagnitude.raw"); 
  //      Array3DIO::writeRawVolume(displacementMagnitude,filename2.c_str());
  //      
  //      //debug with displacement magnitude  
  if (_verbose)
  {
    std::cerr << "MultiScaleFluidWarp at level_up " << level_up 
      << " finished, Time (sec) " 
      << timer.getSeconds() << std::endl;
    
  }
  
  // clean up memory
  delete [] atlasROIChunk;
  
  // save hfield from finest level
  std::cerr << "resampling hField...";
  
  std::cerr << "DONE" << std::endl;
  
  _haveValidHField = true;
}

///////////
// apply //
///////////
MultiScaleFluidWarp::ImagePointer 
MultiScaleFluidWarp
::apply(ImagePointer& image)
{
  ImagePointer defImage = new ImageType(*image);

  if (_useOriginAndSpacing) {
    defImage = new ImageType;
    defImage->setOrigin(_atlasOrigin);
    HField3DUtils::apply(*image, _hField, *defImage, 
                         _hFieldOrigin, _hFieldSpacing);
  } else {
    defImage = new ImageType(*image);
    HField3DUtils::apply(*image, _hField, *defImage, 
                         _roi.getStartX(), _roi.getStartY(), _roi.getStartZ());
    if (defImage->getOrigin() != image->getOrigin())
      std::cout << "\n\n!!!!!!\nSomething's wrong." << std::endl;
    if (defImage->getSpacing() != image->getSpacing())
      std::cout << "\n\n!!!!!!!!!!!!!!!\nSomething's wrong." << std::endl;
  }

  return defImage;
}

///////////
// apply //
///////////
void
MultiScaleFluidWarp
::apply(Surface& surface)
{
  HField3DUtils::applyWithROI(surface,_roi,_hField);
}

//////////////
// applyInv //
//////////////
void
MultiScaleFluidWarp
::applyInv(Surface& surface)
{
  HField3DUtils::applyWithROI(surface,_roi,_hFieldInv);
}


MultiScaleFluidWarp::ImageRegionType
MultiScaleFluidWarp
::loadHFieldMETA(const char* fileName, 
                 MultiScaleFluidWarp::ImagePointer atlas,
                 MultiScaleFluidWarp::ImagePointer subject,
                 const char *atlasName,
                 const char *subjectName)
{
	
  // save filenames
  if (atlasName != 0) { _atlasName = atlasName; }
  if (subjectName != 0) { _subjectName = subjectName; }
	
  // save origin and spacing
  _atlasOrigin.set(atlas->getOrigin().x,
		   atlas->getOrigin().y,
		   atlas->getOrigin().z);
  _atlasSpacing.set(atlas->getSpacing().x,
		    atlas->getSpacing().y,
		    atlas->getSpacing().z);
  _subjectOrigin.set(subject->getOrigin().x,
		     subject->getOrigin().y,
		     subject->getOrigin().z);
  _subjectSpacing.set(subject->getSpacing().x,
		      subject->getSpacing().y,
		      subject->getSpacing().z);

  HField3DIO::readMETA(_hField, _hFieldOrigin, _hFieldSpacing, fileName);
  _useOriginAndSpacing = true;

  Vector3D<double> roiStartDouble(_hFieldOrigin);
  roiStartDouble -= _atlasOrigin;
  roiStartDouble /= _atlasSpacing;
  _roi.setStart(Vector3D<int>(roiStartDouble));

  Vector3D<double> roiSizeDouble(_hField.getSize());
  roiSizeDouble *= _hFieldSpacing / _atlasSpacing;
  _roi.setSize(Vector3D<unsigned int>(roiSizeDouble));

  return _roi;

}

////////////////
// loadHField //
////////////////

void
MultiScaleFluidWarp
::loadHField(const char* fileName)
{
  //
  // first read ascii header
  //

  std::cerr << "This hfield format is obsolete and may not work right." 
            << std::endl;
  
  std::ifstream inputASCII(fileName);
  if (inputASCII.fail())
    {
      throw std::runtime_error("failed to open file for ascii read");
    }      
	
  bool foundAtlasFilename   = false;
  bool foundSubjectFilename = false;
	
  bool foundAtlasOriginX    = false;
  bool foundAtlasOriginY    = false;
  bool foundAtlasOriginZ    = false;
  bool foundAtlasSpacingX   = false;
  bool foundAtlasSpacingY   = false;
  bool foundAtlasSpacingZ   = false;
	
  bool foundSubjectOriginX  = false;
  bool foundSubjectOriginY  = false;
  bool foundSubjectOriginZ  = false;
  bool foundSubjectSpacingX = false;
  bool foundSubjectSpacingY = false;
  bool foundSubjectSpacingZ = false;
	
  bool foundROIIndexX       = false;
  bool foundROIIndexY       = false;
  bool foundROIIndexZ       = false;
  bool foundROISizeX        = false;
  bool foundROISizeY        = false;
  bool foundROISizeZ        = false;
  // unused variable // bool foundFluidParameters = false;
  bool foundHField          = false;
	
  unsigned int roiIndexX = 0, roiIndexY = 0, roiIndexZ = 0;
  unsigned int roiSizeX = 0, roiSizeY = 0, roiSizeZ = 0;
	
  while (true)
    {
      std::string key;
      std::getline(inputASCII, key, '=');
      if (key.find('\n') != std::string::npos)
	{
	  key.erase(0, key.find('\n') + 1);
	}
		
      if (inputASCII.fail())
	{
	  std::cerr << "died on: " << key << std::endl;
	  throw std::runtime_error("ifstream failed reading header");
	}      
		
      if (key.find("H_FIELD") != string::npos )
	{
	  foundHField = true;
	  break;
	}
      else
	{
	  std::string value;
	  std::getline(inputASCII, value);
			
	  if (inputASCII.fail())
	    {
	      std::cerr << "died on: " << value << std::endl;
	      throw std::runtime_error("ifstream failed reading header");
	    }      
			
	  if (key.find("ATLAS_FILENAME") == 0)
	    {
	      _atlasName = _parseValue(value);
	      foundAtlasFilename = true;
	    }
	  else if (key.find("SUBJECT_FILENAME") == 0)
	    {
	      _subjectName = _parseValue(value);
	      foundSubjectFilename = true;
	    }
	  else if (key.find("ATLAS_ORIGIN_X") == 0)
	    {
	      _atlasOrigin.x = atof((_parseValue(value)).c_str());
	      foundAtlasOriginX = true;
	    }
	  else if (key.find("ATLAS_ORIGIN_Y") == 0)
	    {
	      _atlasOrigin.y = atof((_parseValue(value)).c_str());
	      foundAtlasOriginY = true;
	    }
	  else if (key.find("ATLAS_ORIGIN_Z") == 0)
	    {
	      _atlasOrigin.z = atof((_parseValue(value)).c_str());
	      foundAtlasOriginZ = true;
	    }
	  else if (key.find("ATLAS_SPACING_X") == 0)
	    {
	      _atlasSpacing.x = atof((_parseValue(value)).c_str());
	      foundAtlasSpacingX = true;
	    }
	  else if (key.find("ATLAS_SPACING_Y") == 0)
	    {
	      _atlasSpacing.y = atof((_parseValue(value)).c_str());
	      foundAtlasSpacingY = true;
	    }
	  else if (key.find("ATLAS_SPACING_Z") == 0)
	    {
	      _atlasSpacing.z = atof((_parseValue(value)).c_str());
	      foundAtlasSpacingZ = true;
	    }
	  else if (key.find("SUBJECT_ORIGIN_X") == 0)
	    {
	      _subjectOrigin.x = atof((_parseValue(value)).c_str());
	      foundSubjectOriginX = true;
	    }
	  else if (key.find("SUBJECT_ORIGIN_Y") == 0)
	    {
	      _subjectOrigin.y = atof((_parseValue(value)).c_str());
	      foundSubjectOriginY = true;
	    }
	  else if (key.find("SUBJECT_ORIGIN_Z") == 0)
	    {
	      _subjectOrigin.z = atof((_parseValue(value)).c_str());
	      foundSubjectOriginZ = true;
	    }
	  else if (key.find("SUBJECT_SPACING_X") == 0)
	    {
	      _subjectSpacing.x = atof((_parseValue(value)).c_str());
	      foundSubjectSpacingX = true;
	    }
	  else if (key.find("SUBJECT_SPACING_Y") == 0)
	    {
	      _subjectSpacing.y = atof((_parseValue(value)).c_str());
	      foundSubjectSpacingY = true;
	    }
	  else if (key.find("SUBJECT_SPACING_Z") == 0)
	    {
	      _subjectSpacing.z = atof((_parseValue(value)).c_str());
	      foundSubjectSpacingZ = true;
	    }
	  else if (key.find("ROI_INDEX_X") == 0)
	    {
	      roiIndexX = atoi((_parseValue(value)).c_str());
	      foundROIIndexX = true;
	    }
	  else if (key.find("ROI_INDEX_Y") == 0)
	    {
	      roiIndexY = atoi((_parseValue(value)).c_str());
	      foundROIIndexY = true;
	    }
	  else if (key.find("ROI_INDEX_Z") == 0)
	    {
	      roiIndexZ = atoi((_parseValue(value)).c_str());
	      foundROIIndexZ = true;
	    }
	  else if (key.find("ROI_SIZE_X") == 0)
	    {
	      roiSizeX = atoi((_parseValue(value)).c_str());
	      foundROISizeX = true;
	    }
	  else if (key.find("ROI_SIZE_Y") == 0)
	    {
	      roiSizeY = atoi((_parseValue(value)).c_str());
	      foundROISizeY = true;
	    }
	  else if (key.find("ROI_SIZE_Z") == 0)
	    {
	      roiSizeZ = atoi((_parseValue(value)).c_str());
	      foundROISizeZ = true;
	    }
	  else
	    {
	    }
	}
    }
	
  // check for errors and close stream
  if (inputASCII.fail())
    {
      throw std::runtime_error("ifstream failed reading header");
    }
  inputASCII.close();
	
  // make sure we got all the info we need
  bool foundAllHeaderData = true;
  if (!foundAtlasFilename)
    {
      std::cerr << "Fluid: read hField: could not find ATLAS_FILENAME" << std::endl;
      foundAllHeaderData = false;
    }
  if (!foundSubjectFilename)
    {
      std::cerr << "Fluid: read hField: could not find SUBJECT_FILENAME" << std::endl;
      foundAllHeaderData = false;
    }
  if (!foundAtlasOriginX)
    {
      std::cerr << "Fluid: read hField: could not find ATLAS_ORIGIN_X" << std::endl;
      foundAllHeaderData = false;
    }
  if (!foundAtlasOriginY)
    {
      std::cerr << "Fluid: read hField: could not find ATLAS_ORIGIN_Y" << std::endl;
      foundAllHeaderData = false;
    }
  if (!foundAtlasOriginZ)
    {
      std::cerr << "Fluid: read hField: could not find ATLAS_ORIGIN_Z" << std::endl;
      foundAllHeaderData = false;
    }
  if (!foundAtlasSpacingX)
    {
      std::cerr << "Fluid: read hField: could not find ATLAS_SPACING_X" << std::endl;
      foundAllHeaderData = false;
    }
  if (!foundAtlasSpacingY)
    {
      std::cerr << "Fluid: read hField: could not find ATLAS_SPACING_Y" << std::endl;
      foundAllHeaderData = false;
    }
  if (!foundAtlasSpacingZ)
    {
      std::cerr << "Fluid: read hField: could not find ATLAS_SPACING_Z" << std::endl;
      foundAllHeaderData = false;
    }
  if (!foundSubjectOriginX)
    {
      std::cerr << "Fluid: read hField: could not find SUBJECT_ORIGIN_X" << std::endl;
      foundAllHeaderData = false;
    }
  if (!foundSubjectOriginY)
    {
      std::cerr << "Fluid: read hField: could not find SUBJECT_ORIGIN_Y" << std::endl;
      foundAllHeaderData = false;
    }
  if (!foundSubjectOriginZ)
    {
      std::cerr << "Fluid: read hField: could not find SUBJECT_ORIGIN_Z" << std::endl;
      foundAllHeaderData = false;
    }
  if (!foundSubjectSpacingX)
    {
      std::cerr << "Fluid: read hField: could not find SUBJECT_SPACING_X" << std::endl;
      foundAllHeaderData = false;
    }
  if (!foundSubjectSpacingY)
    {
      std::cerr << "Fluid: read hField: could not find SUBJECT_SPACING_Y" << std::endl;
      foundAllHeaderData = false;
    }
  if (!foundSubjectSpacingZ)
    {
      std::cerr << "Fluid: read hField: could not find SUBJECT_SPACING_Z" << std::endl;
      foundAllHeaderData = false;
    }
  if (!foundROIIndexX)
    {
      std::cerr << "Fluid: read hField: could not find ROI_INDEX_X" << std::endl;
      foundAllHeaderData = false;
    }
  if (!foundROIIndexY)
    {
      std::cerr << "Fluid: read hField: could not find ROI_INDEX_Y" << std::endl;
      foundAllHeaderData = false;
    }
  if (!foundROIIndexZ)
    {
      std::cerr << "Fluid: read hField: could not find ROI_INDEX_Z" << std::endl;
      foundAllHeaderData = false;
    }
  if (!foundROISizeX)
    {
      std::cerr << "Fluid: read hField: could not find ROI_SIZE_X" << std::endl;
      foundAllHeaderData = false;
    }
  if (!foundROISizeY)
    {
      std::cerr << "Fluid: read hField: could not find ROI_SIZE_Y" << std::endl;
      foundAllHeaderData = false;
    }
  if (!foundROISizeZ)
    {
      std::cerr << "Fluid: read hField: could not find ROI_SIZE_Z" << std::endl;
      foundAllHeaderData = false;
    }
  if (!foundHField)
    {
      std::cerr << "Fluid: read hField: could not find H_FIELD" << std::endl;
      foundAllHeaderData = false;
    }
	
  if (!foundAllHeaderData)
    {
      throw std::runtime_error("header contains insufficient data");
    }
	
  // resize roi
  ImageIndexType roiStartIndex;
  roiStartIndex[0] = roiIndexX;
  roiStartIndex[1] = roiIndexY;
  roiStartIndex[2] = roiIndexZ;
	
  ImageSizeType roiSize;
  roiSize[0] = roiSizeX;
  roiSize[1] = roiSizeY;
  roiSize[2] = roiSizeZ;
	
  _roi.setStart(roiStartIndex);
  _roi.setSize(roiSize);  
	
  // resize array3d
  _hField.resize(_roi.getSize().x,
		 _roi.getSize().y,
		 _roi.getSize().z);
	
  std::cerr << "roiindex: " 
	    << roiIndexX << ", " 
	    << roiIndexY << ", " 
	    << roiIndexZ << std::endl;
	
  std::cerr << "roisize: " 
	    << roiSizeX << ", " 
	    << roiSizeY << ", " 
	    << roiSizeZ << std::endl;
	
  //
  // read in h field data
  //
	
  std::ifstream inputBinary(fileName, std::ios::binary);
  if (inputBinary.fail())
    {
      throw std::runtime_error("failed to open file for binary read");
    }      
	
  BinaryIO bio;
  bio.setIOEndian(BinaryIO::little_endian);
	
  // seek to h field
  // 3 for x,y,z
  unsigned int hFieldSize = 3 *
    _roi.getSizeX() *
    _roi.getSizeY() *
    _roi.getSizeZ();
  // 4 is size of float (fixed for now)
  // get length of file:
  inputBinary.seekg (0, std::ios::end);
  int fileLength = inputBinary.tellg();
  inputBinary.seekg (0, std::ios::beg);
	
  inputBinary.seekg(fileLength - (hFieldSize * 4));
	
  std::cerr << "fileLength " << fileLength << std::endl;
  std::cerr << "hFieldSize " << hFieldSize << std::endl;
	
  // read in h field
  /*Vector3D<float> *hFieldPtr = _hField.getDataPointer();  
    for (unsigned int i = 0; i < hFieldSize; i++)
    {
    *hFieldPtr++ = bio.readFloat(inputBinary);
		
    if (inputBinary.fail())
    {
    std::cerr << "io fails at: " << i << std::endl;
    break;
    }
    }*/
  Vector3D<float> hFieldVector;
  for (unsigned int z = 0; z < _hField.getSizeZ(); ++z){
    for (unsigned int y = 0; y < _hField.getSizeY(); ++y){
      for (unsigned int x = 0; x < _hField.getSizeX(); ++x){
	for (int i = 0 ; i < 3 ; i++)
	  {
	    hFieldVector[i] = bio.readFloat(inputBinary);
	    if (inputBinary.fail())
	      {
		std::cerr << "io fails at: " << i << std::endl;
		break;
	      }
	  }
	_hField.set(x,y,z,hFieldVector);
      }
    }
  }
  // check for errors and close stream
  if (inputBinary.fail())
    {
      throw std::runtime_error("ifstream failed reading hfield");
    }
  inputBinary.close();
	
  _haveValidHField = true;
}

void
MultiScaleFluidWarp
::writeHFieldMETA(const std::string& filenamePrefix)
{
  HField3DIO::writeMETA(_hField, _hFieldOrigin, _hFieldSpacing,
                        filenamePrefix);
  if (_hFieldInv.getSize() == _hField.getSize()) {
    HField3DIO::writeMETA(_hFieldInv, _hFieldOrigin, _hFieldSpacing,
                          filenamePrefix + std::string("_inv"));
  }
}

//////////////////
// writeHField  //
//////////////////

void
MultiScaleFluidWarp
::writeHField(const char* fileName)
{

  std::cerr << "WARNING: USING OLD UNSUPPORTED writeHField()" << std::endl;

  // Should remove this function.

  // need to make sure that roi size == hfield size !!!
	
  const unsigned int WRITE_VERSION = 1;
	
  // first write ascii header
  std::ofstream outputASCII(fileName);
	
  if (outputASCII.fail())
    {
      throw std::runtime_error("failed to open file for ascii write");
    }
	
  outputASCII << "## FLUID TRANSFORMATION ##\n";
	
  outputASCII << "OUTPUT_FILE_VERSION=\"" << WRITE_VERSION       << "\"\n";
	
  outputASCII << "ATLAS_FILENAME=\""      << _atlasName          << "\"\n";
  outputASCII << "SUBJECT_FILENAME=\""    << _subjectName        << "\"\n";
	
  outputASCII << "ATLAS_ORIGIN_X=\""      << _atlasOrigin.x      << "\"\n";
  outputASCII << "ATLAS_ORIGIN_Y=\""      << _atlasOrigin.y      << "\"\n";
  outputASCII << "ATLAS_ORIGIN_Z=\""      << _atlasOrigin.z      << "\"\n";
  outputASCII << "ATLAS_SPACING_X=\""     << _atlasSpacing.x     << "\"\n";
  outputASCII << "ATLAS_SPACING_Y=\""     << _atlasSpacing.y     << "\"\n";
  outputASCII << "ATLAS_SPACING_Z=\""     << _atlasSpacing.z     << "\"\n";
	
  outputASCII << "SUBJECT_ORIGIN_X=\""    << _subjectOrigin.x    << "\"\n";
  outputASCII << "SUBJECT_ORIGIN_Y=\""    << _subjectOrigin.y    << "\"\n";
  outputASCII << "SUBJECT_ORIGIN_Z=\""    << _subjectOrigin.z    << "\"\n";
  outputASCII << "SUBJECT_SPACING_X=\""   << _subjectSpacing.x   << "\"\n";
  outputASCII << "SUBJECT_SPACING_Y=\""   << _subjectSpacing.y   << "\"\n";
  outputASCII << "SUBJECT_SPACING_Z=\""   << _subjectSpacing.z   << "\"\n";
	
  outputASCII << "ROI_INDEX_X=\""         << _roi.getStart().x  << "\"\n";
  outputASCII << "ROI_INDEX_Y=\""         << _roi.getStart().y  << "\"\n";
  outputASCII << "ROI_INDEX_Z=\""         << _roi.getStart().z  << "\"\n";
  outputASCII << "ROI_SIZE_X=\""          << _roi.getSize().x   << "\"\n";
  outputASCII << "ROI_SIZE_Y=\""          << _roi.getSize().y   << "\"\n";
  outputASCII << "ROI_SIZE_Z=\""          << _roi.getSize().z   << "\"\n";
	
  outputASCII << "\n";
	
  int numUsedScaleLevels = _fluidParameters.size();
  for (int level = 0; level < numUsedScaleLevels; level++)
    {
      outputASCII << "SCALE_LEVEL=\"" << _fluidParameters[level].first << "\"\n";
      _fluidParameters[level].second.writeASCII(outputASCII);
      outputASCII << "\n";
    }
  outputASCII << "H_FIELD=";
	
  // check for errors and close stream
  if (outputASCII.fail())
    {
      throw std::runtime_error("ofstream failed writing ascii header");
    }
  outputASCII.close();
	
  // now write binary h field, append to end of file
  std::ofstream outputBinary(fileName, std::ios::binary | std::ios::app);
	
  if (outputBinary.fail())
    {
      throw std::runtime_error("failed to open file for binary write");
    }
	
  BinaryIO bio; 
  bio.setIOEndian(BinaryIO::little_endian);
  for (unsigned int z = 0; z < _hField.getSizeZ(); ++z){
    for (unsigned int y = 0; y < _hField.getSizeY(); ++y){
      for (unsigned int x = 0; x < _hField.getSizeX(); ++x){
	for (int i = 0 ; i < 3 ; i++)
	  {
	    bio.writeFloat(((_hField(x,y,z))[i]), outputBinary);
	  }
      }
    }
  }
	
  // check for errors and close stream
  if (outputBinary.fail())
    {
      throw std::runtime_error("ofstream failed writing hfield");
    }
  outputBinary.close();  
}


////////////////////////////////////
// _convertImageToArray3DUChar //
////////////////////////////////////

void
MultiScaleFluidWarp
::_convertImageToArray3DUChar(MultiScaleFluidWarp::ImagePointer image, 
			      Array3D<float>& array3DImage,
			      const VoxelType& rescaleThresholdMin,
			      const VoxelType& rescaleThresholdMax)
{
  //
  // THIS CODE IS NOT OPTIMIZED
  //
	
  //
  // rescale intensities according to 
  // [rescaleThresholdMin,rescaleThresholdMax] -> [0,255]
  // (-inf, rescaleThresholdMin)               -> 0
  // (rescaleThresholdMax, inf)                -> 255
  //
  double intensityScale = 255.0 / (rescaleThresholdMax - rescaleThresholdMin);
	
  // make array3d the proper size
  ImageSizeType imageSize(image->getSize());
  array3DImage.resize((unsigned int)imageSize.x, 
		      (unsigned int)imageSize.y, 
		      (unsigned int)imageSize.z);
	
  // fill in values
  // unused variable // ImageIndexType imageIndex;
  for (unsigned int xIndex = 0; xIndex < imageSize.x; xIndex++)
  {
    for (unsigned int yIndex = 0; yIndex < imageSize.y; yIndex++)
    {
      for (unsigned int zIndex = 0; zIndex < imageSize.z; zIndex++)
      {	      
        VoxelType pixVal = image->get(xIndex,yIndex,zIndex);
        
        // rescale intensity
        if (pixVal < rescaleThresholdMin) 
        {
          pixVal = 0;
        }
        else if (pixVal > rescaleThresholdMax)
        {
          pixVal = 255;
        }
        else
        {
          pixVal = (pixVal - rescaleThresholdMin)
            * intensityScale;
        }
        
        array3DImage(xIndex,yIndex,zIndex) = pixVal;
      }
    }
  }  
}
/////////////////////////////////
// _convertArray3DUCharToImage //
/////////////////////////////////

void
MultiScaleFluidWarp
::_convertArray3DUCharToImage(Array3D<float>& array3DImage,
			      MultiScaleFluidWarp::ImagePointer image)
{
  // set up itk image
  MultiScaleFluidWarp::ImageType::SizeType newImageSize;
  newImageSize[0] = array3DImage.getSize().x;
  newImageSize[1] = array3DImage.getSize().y;
  newImageSize[2] = array3DImage.getSize().z;
	
  ImageIndexType newImageIndex;
  newImageIndex[0] = 0;
  newImageIndex[1] = 0;
  newImageIndex[2] = 0;
	
  ImageRegionType newImageRegion;
  newImageRegion.setSize(newImageSize);
  newImageRegion.setStart(newImageIndex);
	
  image->resize(newImageSize);  
	
  Vector3D<double> imageSpacing(1,1,1);
  image->setSpacing(imageSpacing);
	
  // fill in values
  ImageIndexType imageIndex;
  for (unsigned int xIndex = 0; xIndex < newImageSize[0]; xIndex++)
    {
      imageIndex[0] = xIndex;
      for (unsigned int yIndex = 0; yIndex < newImageSize[1]; yIndex++)
	{
	  imageIndex[1] = yIndex;
	  for (unsigned int zIndex = 0; zIndex < newImageSize[2]; zIndex++)
	    {
	      imageIndex[2] = zIndex;
	      image->set(xIndex,yIndex,zIndex, array3DImage(xIndex,yIndex,zIndex));
	    }
	}
    }    
}




///////////////////////////////////////////
// _downsampleImageBy2WithGaussianFilter //
///////////////////////////////////////////

void
MultiScaleFluidWarp
::_downsampleImageBy2WithGaussianFilter(ImagePointer& image, ImagePointer& shrinkImage)
{
  Vector3D<double> scaleFactors(2,2,2);
  //Vector3D<double> sigma(1*scaleFactors.x,1*scaleFactors.y,1*scaleFactors.z);
  //Vector3D<double> kernelSize(2*scaleFactors.x,2*scaleFactors.y,2*scaleFactors.z);

  /** Downsample the image */
  DownsampleFilter3D filter;
  filter.SetInput(*image);
  filter.SetFactor((int)scaleFactors.x,(int)scaleFactors.y,(int)scaleFactors.z);
  filter.SetSigma(1*scaleFactors.x,1*scaleFactors.y,1*scaleFactors.z);
  filter.SetSize((int)(2*scaleFactors.x),(int)(2*scaleFactors.y),(int)(2*scaleFactors.z));
  filter.Update();
	
  /** Create new downsampled image */
  ImageSizeType spacing = image->getSpacing();
  shrinkImage = new ImageType(filter.GetNewSize());
  shrinkImage->setData(filter.GetOutput());
	

  /*ImageSizeType spacing = image->getSpacing();
    shrinkImage = new ImageType(image->getSize()/scaleFactors);
    Array3DUtils::gaussianDownsample(*image,
    *shrinkImage,
    scaleFactors,
    sigma,
    kernelSize);*/


  ImageSizeType imagesize(shrinkImage->getSizeX(),
			  shrinkImage->getSizeY(),
			  shrinkImage->getSizeZ());
  spacing.scale(scaleFactors.x,scaleFactors.y,scaleFactors.z);
	
  float origin_x= ((imagesize[0]/2)*spacing.x*(-1));
  float origin_y=((imagesize[1]/2)*spacing.y*(-1));
  float origin_z= ((imagesize[2]/2)*spacing.z*(-1));
	
  ImageIndexType origin(origin_x, origin_y,origin_z);
	
  shrinkImage->setOrigin(origin);
  shrinkImage->setSpacing(spacing);
}

//////////////////////////////
// _checkImageCompatibility //
//////////////////////////////

bool
MultiScaleFluidWarp
::_checkImageCompatibility(ImagePointer atlas,
			   ImagePointer subject)
{
  const double tolerence = 0.01;
	
  if (atlas == 0 || subject == 0)
    {
      return false;
    }
	
  ImageRegionType atlasRegion;
  atlasRegion.setSize(atlas->getSize());
  ImageSizeType atlasSize(atlasRegion.getSize());
	
  ImageRegionType subjectRegion;
  subjectRegion.setSize(subject->getSize());
  ImageSizeType subjectSize = subjectRegion.getSize();
	
  // see if dimensions match
  if (atlasSize != subjectSize)
    {
      return false;
    }
	
  // see if voxel spacing matches
  Vector3D<double> atlasSpacing = atlas->getSpacing();
  Vector3D<double> subjectSpacing = subject->getSpacing();
  if (fabs(atlasSpacing[0] - subjectSpacing[0]) >= tolerence ||
      fabs(atlasSpacing[1] - subjectSpacing[1]) >= tolerence ||
      fabs(atlasSpacing[2] - subjectSpacing[2]) >= tolerence)
    {
      return false;
    }
	
  // see if position (origin) matches
  Vector3D<double> atlasOrigin(atlas->getOrigin());
  Vector3D<double> subjectOrigin(subject->getOrigin());
  if (fabs(atlasOrigin[0] - subjectOrigin[0]) >= tolerence ||
      fabs(atlasOrigin[1] - subjectOrigin[1]) >= tolerence ||
      fabs(atlasOrigin[2] - subjectOrigin[2]) >= tolerence)
    {
      return false;
    }
	
  return true;
}

//////////////////
// resliceImage //
//////////////////
void
MultiScaleFluidWarp
::_resliceImage(ImagePointer& image3D, float newSpacing)
{

  //IMPORTANT : we need a newSpacing positive in this function
  // the orientation is not important, that's why we use abs()
	
  //determine the new number of slices
  //create the new image resliced
  unsigned int ImageSizeX = image3D->getSizeX();
  unsigned int ImageSizeY = image3D->getSizeY();

  float oldSpacing = image3D->getSpacing().z;
  float firstSlice = image3D->getOrigin().z;
  float lastSlice = image3D->getOrigin().z + (image3D->getSize().z-1)*(oldSpacing);
	
  int newNb_Slice = int( ( lastSlice - firstSlice ) / newSpacing);
  if (newNb_Slice != ( ( lastSlice - firstSlice ) / newSpacing))
    newNb_Slice+=1;//to count the first slice
	

  Vector3D<unsigned int> newSize(ImageSizeX,ImageSizeY,newNb_Slice);
  Image<float> image3DResliced(newSize);
  
  int newZ = 0 ; 
  
  for (newZ = 0; newZ < newNb_Slice; newZ++)
    {
      float oldZ = (newZ*newSpacing)/oldSpacing;
    
      int oldz1 = (int)floor(oldZ);
      int oldz2 = oldz1+1;
      for (unsigned int y=0 ; y < ImageSizeY ; y++) {
        for (unsigned int x=0 ; x < ImageSizeX ; x++) {
          image3DResliced.set(x,y,newZ,
                              (image3D->get(x,y,oldz1)*(oldz2-oldZ) +
                               image3D->get(x,y,oldz2)*(oldZ-oldz1)));
        }
      }
  }
			
		
	


  //reset image3D
  image3D->resize(ImageSizeX,ImageSizeY,newNb_Slice);
  image3D->setData(image3DResliced);
  image3D->setSpacing(image3D->getSpacing().x,image3D->getSpacing().y,newSpacing);


}

/////////////////
// resliceMask //
/////////////////
void
MultiScaleFluidWarp
::_resliceMask(MaskType& mask, unsigned int newNb_Slice)
{
  //create the new mask resliced
  unsigned int MaskSizeX = mask.getSizeX();
  unsigned int MaskSizeY = mask.getSizeY();
	
  Vector3D<unsigned int> newSize(MaskSizeX,MaskSizeY,newNb_Slice);
  MaskType maskResliced(newSize);
  
  float newSpacing = mask.getSizeZ()/newNb_Slice;
  unsigned int newZ = 0 ; 
  
  for (newZ = 0; newZ < newNb_Slice; newZ++)
    {
      float oldZ = (newZ*newSpacing);
    
      int oldz1 = (int)floor(oldZ);
      int oldz2 = oldz1+1;
      for (unsigned int y=0 ; y < MaskSizeY ; y++) {
        for (unsigned int x=0 ; x < MaskSizeX ; x++) {
          maskResliced.set(x,y,newZ,
                              (mask.get(x,y,oldz1)*(oldz2-oldZ) +
                               mask.get(x,y,oldz2)*(oldZ-oldz1)));
        }
      }
  }

  //reset mask
  mask.resize(MaskSizeX,MaskSizeY,newNb_Slice);
  mask.setData(maskResliced);
}

////////////////
// _saveImage // 
////////////////

void
MultiScaleFluidWarp
::_saveImage(ImagePointer image, const std::string& fileName)
{
  ImageIO *imageIO;
  imageIO = new ImageIO;
	
  std::cout << "Writing " << fileName << "...";
  try {
    imageIO->SaveImage(fileName, *image);
  }
  catch( ... ) {
    std::cerr << "Unknown exception" << std::endl;
  }
  std::cout<< "DONE"<<std::endl;
}

/////////////////
// _parseValue //
/////////////////

std::string
MultiScaleFluidWarp
::_parseValue(const std::string& str)
{
  //
  // return info between first and last quotes
  //
  size_t firstIndex = str.find_first_of('\"');
  size_t lastIndex  = str.find_last_of('\"');

  if ((firstIndex == std::string::npos) || 
      (lastIndex  == std::string::npos) ||
      (firstIndex == lastIndex))
    {
      throw std::invalid_argument("str does not contain a quote pair");
    }

  std::string result = str.substr(firstIndex + 1, lastIndex - firstIndex - 1);
  std::cerr << "parse result" << result << std::endl;
  return result;
}

/////////////////////////
// _createImageFromROI //
/////////////////////////

void
MultiScaleFluidWarp
::_createImageFromROI(const ImagePointer& inImage, ImagePointer& outImage)
{
  Array3D<VoxelType> roiArray;

  ROIUtils<VoxelType>::extractROIFromArray3D(*inImage, roiArray, _roi );
  *outImage = roiArray;

  outImage->setOrigin(inImage->getOrigin());
  outImage->setSpacing(inImage->getSpacing());
}
