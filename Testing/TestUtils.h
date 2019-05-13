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


#ifndef __TEST_UTILS_H__
#define __TEST_UTILS_H__

#include <cmath>

// include info needed for new routines
#include "AtlasWerksTypes.h"
#include "DataTypes/Image.h"
#include "ImageUtils.h"
#include "Array3D.h"
#include "Array3DUtils.h"
#include "HField3DIO.h"
#include "HField3DUtils.h"
#include "ApplicationUtils.h"
#include "Array3DToyImages.h"

#define RAND (((double)rand())/((double)RAND_MAX))

#define TEST_PASS 0
#define TEST_FAIL 1

class TestUtils{
public:
  // generate test data bullseye
  static void GenBullseye(RealImage &TestData, 
			  double r1, double r2, double r3);

  // generate test data
  static void GenXGradient(RealImage &TestData, double min, double max);

  // generate test data
  static void GenDilation(VectorField &TestData, double max);

  // generate test data
  static void GenDilation2(VectorField &TestData, double max);

  /**
   * 'Wavy' radial displacement (vField) pattern, amplitude amp and period p
   */
  static void GenWavy(VectorField &TestData, double amp=1.0, double p=4.0);

  // squared difference between vector fields
  static Real VecSquaredDiff(const VectorField &v1,
			     const VectorField &v2);
  
  // compare image to file
  static Real SquaredDiff(const RealImage &testData, 
			  const char *baselineFileName);
  
//   // compare image to file
//   static Real SquaredDiff(const VectorField &testData, 
// 			  const char *baselineFileNameBase);

  // compare image to file
  static Real SquaredDiff(const VectorField &testData, 
			  const char *baselineFileNameBase,
			  const char *baselineExtension=NULL);
  
  static void WriteHField(const char *filename, 
			  const VectorField &hField);

  static bool Test(const RealImage &testIm, 
		   const RealImage &baseline, 
		   Real eps,
		   const char *errName=NULL);
  
  static bool Test(const VectorField &testVF, 
		   const VectorField &baseline, 
		   Real eps,
		   const char *errName=NULL);
  
  static bool Test(const RealImage &testData, 
		   const char *fileName,
		   Real eps,
		   bool writeImagesOnFail=true);
  
  static bool Test(const VectorField &testData, 
		   const char *fileName,
		   Real eps,
		   bool writeImagesOnFail=true);
};

#endif
