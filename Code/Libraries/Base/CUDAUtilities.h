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


#ifndef __CUDA_UTILITIES_H__
#define __CUDA_UTILITIES_H__

#include "AtlasWerksTypes.h"
#include "cudaInterface.h"
#include "cudaVector3DArray.h"
#include "cudaReduce.h"

class CUDAUtilities{
public:
  /**
   * Copy a vector field from the device to the host.  If
   * useHostArraySize is true, the size of hostVF will be used to
   * determine the size of the data in deviceVF.  If tmp is specified,
   * it should be an array of size nVox (number of elements in
   * deviceVF)
   */
  static void CopyVectorFieldFromDevice(const cplVector3DArray &deviceVF, VectorField &hostVF, 
					bool useHostArraySize = false, float *tmp = NULL);
  /**
   * Copy a vector field from the host to the device.  If
   * useHostArraySize is true, the size of hostVF will be used to
   * determine the size of the data in deviceVF.  If tmp is specified,
   * it should be an array of size nVox (number of elements in
   * deviceVF)
   */
  static void CopyVectorFieldToDevice(const VectorField &hostVF, cplVector3DArray &deviceVF, 
				      bool useHostArraySize = false, float *tmp = NULL);

  /**
   * Set the CUDA device to be used, throw exception if it is not a
   * valid device.
   */
  static void SetCUDADevice(unsigned int deviceNum);

  /**
   * For debugging, copy an image back to host and save it
   */
  static void SaveDeviceImage(const char *fName, const float *dIm,
			      SizeType imSize,
			      OriginType imOrigin = OriginType(0.f,0.f,0.f), 
			      SpacingType imSpacing = SpacingType(1.f,1.f,1.f));

  /**
   * For debugging, copy an image back to host and save it
   */
  static void SaveDeviceVectorField(const char *fName, 
				    const cplVector3DArray &v,
				    SizeType size,
				    OriginType origin = OriginType(0.f,0.f,0.f), 
				    SpacingType spacing = SpacingType(1.f,1.f,1.f));

  /**
   * For debugging, just take the sum of image elements
   */
  static float DeviceImageSum(const float *dIm, unsigned int nElements);

  /**
   * For debugging, just take the sum of all vector elements
   */
  static float DeviceVectorSum(const cplVector3DArray &dV, unsigned int nElements);

  /**
   * Check for a CUDA error, and throw an exception
   * (AtlasWerksExceptioin) if one is found
   */
  static void CheckCUDAError(const char *file, int line);

  /**
   * Return CUDA memory usage as a string
   */
  static std::string GetCUDAMemUsage();

  /**
   * Test for minimum cuda capability version, throw an exception if
   * it is not met
   */
  static void AssertMinCUDACapabilityVersion(int devNum, int major, int minor);

  /**
   * Test current device for minimum cuda capability version, throw an
   * exception if it is not met
   */
  static void AssertMinCUDACapabilityVersion(int major, int minor);
};


#endif //  __CUDA_UTILITIES_H__
