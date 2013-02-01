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

// File: Affine.h
//

#ifndef _AFFINE_H_
#define _AFFINE_H_

#include <iostream>
#include <fstream>
#include<exception>
#include <ImageUtils.h>
#include <EstimateAffine.h>
#include <Array3DUtils.h>
#include<unistd.h>
#include<ImageIO.h>
#include <vector>
#include <string>


#include <Array2D.h>
#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

#include <AnastructUtils.h>
#include <Array3DUtils.h>
#include <ApplicationUtils.h>

#include <FL/Fl_Color_Chooser.H>
#include <FL/Fl_File_Chooser.H>
#include <Fl_Table.H>
#include <Fl_Table_Row.H>
#include <FL/fl_ask.H>

#include <time.h>
class Affine {

public:
  typedef float                     VoxelType;
  typedef Image<VoxelType>          ImageType;
  typedef ImageType*                ImagePointer;
  typedef ROI<int, unsigned int>    ImageRegionType;
  typedef Vector3D<double>          ImageSizeType;
  typedef Vector3D<double>          ImageIndexType;
  typedef Array2D<double>           ScheduleType;
  typedef Array3D<float>     MaskType;
  typedef Array3D<Vector3D<float> > HFieldType;

Affine(char* file1, char*file2, string type);
Affine(ImagePointer fixedImage,
	ImagePointer movingImage,string type);

AffineTransform3D<double> registrationTransform;
ImagePointer finalImage;

private:
void runAffine(ImagePointer,ImagePointer,string);
ScheduleType getPyramidSchedule();
ImageRegionType getROI();
ImagePointer _applyAffineTransform(const AffineTransform3D<double>& , ImagePointer , ImagePointer ,std::string& );

}; // Class Affine

#endif // #ifndef _AFFINE_H_
