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

//File Affine.cxx

//Affine registration
#include<Affine.h>

Affine::Affine(char* file1, char* file2,string type)
{
//	if(argc !=3 )
//	{
//		std::cout<<"Invalid Arguments"<<std::endl;
//		return 0;
//	}

        char* fileName=file1;
        char* fileName2=file2;
	ImagePointer fixedImage;
        fixedImage = new ImageType;
	//std::cout<<fileName;
        if (ApplicationUtils::ITKHasImageFileReader(fileName))
	{
	  ApplicationUtils::LoadImageITK(fileName, *fixedImage);
	  //int pos = name.find_last_of("/");
	  //name.erase(name.begin(), name.begin() + pos + 1 );
	  //pos = name.find_last_ImageRegionType getROI()
	  //name.erase(name.begin() + pos ,name.end());
	  // _imageNames.push_back(name.c_str());
	  //_imageFullFileNames.push_back(fileName);
//	  std::cout<<"FullFileName : "<<fileName<<std::endl;
	  //std::cout<<"Name : "<<name.c_str()<<std::endl;

	}
        ImagePointer movingImage;
        movingImage = new ImageType;

        if (ApplicationUtils::ITKHasImageFileReader(fileName2))
	{
        	  ApplicationUtils::LoadImageITK(fileName2, *movingImage);
//	          std::cout<<"FullFileName : "<<fileName2<<std::endl;
        	  //std::cout<<"Name : "<<name.c_str()<<std::endl;

        }
        ImageUtils::makeImageUnsigned(*fixedImage);
        ImageUtils::makeImageUnsigned(*movingImage);
	runAffine(fixedImage, movingImage,type);
	
}

Affine::Affine(ImagePointer fixedImage, ImagePointer movingImage, string type)
{
	runAffine(fixedImage, movingImage, type);
}

void Affine::runAffine(ImagePointer fixedImage, ImagePointer movingImage, string type)
{

       bool useIntensityWindowing = false;
        EstimateAffine::OutputMode verbosity = EstimateAffine::VERBOSE;
        EstimateAffine estimateAffine(fixedImage, movingImage, useIntensityWindowing, verbosity);
        EstimateAffine::ScheduleType pyramidSchedule = getPyramidSchedule();
        estimateAffine.SetShrinkSchedule(pyramidSchedule);

        if (useIntensityWindowing){
        estimateAffine.SetMinMax(0,1,0,1);
        }

        //estimateAffine.SetROI(getROI());
        std::string imageNameString;
        if(type=="Affine")
        {
                estimateAffine.RunEstimateAffine();
                imageNameString = "image->overlay affine";
        }
        else if(type=="Translation")
        {
                estimateAffine.RunEstimateTranslation();
                imageNameString = "image->overlay translation";

        }
        else if(type=="Rigid")
        {
                estimateAffine.RunEstimateRotation();
                imageNameString = "image->overlay rigid";
        }
        registrationTransform = estimateAffine.GetTransform();
//        ImagePointer finalImage;
        finalImage = _applyAffineTransform(registrationTransform,fixedImage,movingImage,imageNameString);
        std::cout<<registrationTransform<<std::endl;

}


Affine::ImagePointer Affine::_applyAffineTransform(const AffineTransform3D<double>& transform,
                        ImagePointer fixedImage,
                        ImagePointer movingImage,
                        std::string& imageName)
{

//  _updateStatusBuffer("Creating Transformed Image...");
  ImagePointer registeredImage = new ImageType(*fixedImage);

  EstimateAffine::ApplyTransformation(fixedImage, movingImage, registeredImage, transform);
//  _updateStatusBuffer("Creating Transformed Image...DONE");  

  // add translated surfaces and anastructs
 /* AffineTransform3D<double> invertedTransform = transform;
  if (!invertedTransform.invert()) {
    std::cerr << "Transform not invertible: Transformed "
              << "surfaces will not be created." << std::endl;
  } else {
    _updateStatusBuffer("Creating Transformed Surfaces...");  
    std::vector< ImAna > transformedImAnas;
    for (unsigned int surfaceIndex = 0; 
         surfaceIndex < _imAnaVectors[subjectIndex].size();
         ++surfaceIndex)
    {
      std::cerr << "###### translating surface ######" << std::endl;
      Surface transformedSurface(
        _imAnaVectors[subjectIndex][surfaceIndex].surface);
      transformedSurface.applyAffineTransform(invertedTransform);

      std::cerr << "######  creating anastruct ######" << std::endl;      
      Anastruct transformedAnastruct;
      _createAnastruct(transformedSurface,
                       _loadedImages[atlasIndex]->getOrigin(),
                       _loadedImages[atlasIndex]->getSpacing(),
                       _imAnaVectors[subjectIndex][surfaceIndex].anastruct.label,
                       transformedAnastruct);

      ImAna transformedImAna(transformedAnastruct,
                             transformedSurface,
                             _imAnaVectors[subjectIndex][surfaceIndex].visible,
                             _imAnaVectors[subjectIndex][surfaceIndex].color,
                             _imAnaVectors[subjectIndex][surfaceIndex].aspect,
                             _imAnaVectors[subjectIndex][surfaceIndex].opacity);
      transformedImAnas.push_back(transformedImAna);
    }
    _updateStatusBuffer("Creating Transformed Surfaces...DONE");  
    _imAnaVectors.push_back(transformedImAnas);
  }

  _imageNames.push_back(imageName);
  _imageFullFileNames.push_back(imageName);
  registeredImage->setDataType( Image<float>::Float );
  registeredImage->setOrientation(_loadedImages[atlasIndex]->getOrientation());
  _addImageToList(registeredImage);
*/
//  _updateStatusBuffer("Done applying Affine");
return registeredImage;
}



Affine::ImageRegionType Affine::getROI()
{
  int startX = static_cast<int>(10);
  int startY = static_cast<int>(10);
  int startZ = static_cast<int>(10);
  int stopX = static_cast<int>(20);
  int stopY = static_cast<int>(20);
  int stopZ = static_cast<int>(20);

  if (startX > stopX)
  {
    int tmp = startX;
    startX = stopX;
    stopX = tmp;
  }
  if (startY > stopY)
  {
    int tmp = startY;
    startY = stopY;
    stopY = tmp;
  }
  if (startZ > stopZ)
  {
    int tmp = startZ;
    startZ = stopZ;
    stopZ = tmp;
  }

  ImageIndexType roiStartIndex(startX,startY,startZ);

  ImageSizeType roiSize(stopX - startX + 1,
    stopY - startY + 1 ,
    stopZ - startZ + 1);
  ImageRegionType roi(roiStartIndex, roiSize);
  return roi;
}


Affine::ScheduleType Affine::getPyramidSchedule()
{

  // get values from gui
  std::vector< Vector3D<double> > pyramidScheduleAll(4);
  pyramidScheduleAll[0][0] = 8;
  pyramidScheduleAll[0][1] = 8;
  pyramidScheduleAll[0][2] = 2;  
  pyramidScheduleAll[1][0] = 4;
  pyramidScheduleAll[1][1] = 4;
  pyramidScheduleAll[1][2] = 2;  
  pyramidScheduleAll[2][0] = 2;
  pyramidScheduleAll[2][1] = 2;
  pyramidScheduleAll[2][2] = 1;  
  pyramidScheduleAll[3][0] = 1;
  pyramidScheduleAll[3][1] = 1;
  pyramidScheduleAll[3][2] = 1;  

  std::vector< Vector3D<double> > scheduleVec;

  // Make sure that each level (a) has no zeros, (b) is different from
  // the next coarser level, and (b) is consistently finer than the
  // next coarser level.
  for( int i = 3; i >= 0; --i ) {
    Vector3D<double>& scale = pyramidScheduleAll[i];
    if ( scale[0] > 0 && scale[1] > 0 && scale[2] > 0 &&
         (scheduleVec.empty() ||
          (scale != scheduleVec.back() &&
           scale[0] >= scheduleVec.back()[0] &&
           scale[1] >= scheduleVec.back()[1] &&
           scale[2] >= scheduleVec.back()[2]))) {
      scheduleVec.push_back(scale);
    }
  }
  if (scheduleVec.empty()) scheduleVec.push_back(Vector3D<double>(1,1,1));

  ScheduleType schedule( scheduleVec.size(), 3 );
  for (unsigned int i = 0; i < schedule.getSizeX(); ++i) {
    for (unsigned int j = 0; j < 3; ++j) {
      schedule(i, j) = scheduleVec[scheduleVec.size()-i-1][j];
    }
  }
  return schedule;
}

