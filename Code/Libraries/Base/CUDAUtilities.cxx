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


#include "CUDAUtilities.h"
#include "ApplicationUtils.h"
#include "StringUtils.h"
#include "log.h"

void 
CUDAUtilities::
CopyVectorFieldFromDevice(const cplVector3DArray &deviceVF, VectorField &hostVF, bool useHostArraySize, float *tmp)
{
  int nVox = 0;
  if(useHostArraySize){
    nVox = hostVF.getNumElements();
  }else{
    nVox = deviceVF.size();
  }
  bool deleteTmp = false;
  if(!tmp){
    deleteTmp = true;
    tmp = new float[nVox];
  }
  for(int dim=0;dim<3;dim++){
    copyArrayFromDevice<float>(tmp, deviceVF.elementArray(dim), nVox);
    Vector3D<float>*vecData=hostVF.getDataPointer(0);
    for(int i=0;i<nVox;i++){
      vecData[i][dim] = tmp[i];
    }
  }
  if(deleteTmp){
    delete [] tmp;
  }
}

void 
CUDAUtilities::
CopyVectorFieldToDevice(const VectorField &hostVF, cplVector3DArray &deviceVF, bool useHostArraySize, float *tmp)
{
  int nVox = 0;
  if(useHostArraySize){
    nVox = hostVF.getNumElements();
  }else{
    nVox = deviceVF.size();
  }
  bool deleteTmp = false;
  if(!tmp){
    deleteTmp = true;
    tmp = new float[nVox];
  }
  for(int dim=0;dim<3;dim++){
    const Vector3D<float>* vecData=hostVF.getDataPointer(0);
    for(int i=0;i<nVox;i++){
      tmp[i] = vecData[i][dim];
    }
    copyArrayToDevice<float>(deviceVF.elementArray(dim), tmp, nVox);
  }
  if(deleteTmp){
    delete [] tmp;
  }
}

void 
CUDAUtilities::
SetCUDADevice(unsigned int deviceNum)
{
  // determine the number of GPUs
  unsigned int systemGPUs = getNumberOfCapableCUDADevices();
  if(systemGPUs < 1){
    throw AtlasWerksException(__FILE__, __LINE__, "Error, no CUDA capable devices found.");
  }
  if(deviceNum >= systemGPUs){
    std::string err = StringUtils::strPrintf("Error, cannot select device %d, only %d devices available.",deviceNum, systemGPUs);
    throw AtlasWerksException(__FILE__, __LINE__, err);
  }
  
  cudaSetDevice(deviceNum);
}

void 
CUDAUtilities::
SaveDeviceImage(const char *fName, const float *dIm, 
		SizeType imSize, OriginType imOrigin, SpacingType imSpacing)
{
  RealImage tmp(imSize, imOrigin, imSpacing);
  unsigned int nVox = imSize.productOfElements();
  copyArrayFromDevice(tmp.getDataPointer(), dIm, nVox);
  ApplicationUtils::SaveImageITK(fName, tmp);
}

void 
CUDAUtilities::
SaveDeviceVectorField(const char *fName, 
		      const cplVector3DArray &dV,
		      SizeType size,
		      OriginType origin, 
		      SpacingType spacing)
{
  VectorField tmp(size);
  CopyVectorFieldFromDevice(dV,tmp,true);
  ApplicationUtils::SaveHFieldITK(fName, tmp, origin, spacing);
}

float
CUDAUtilities::
DeviceImageSum(const float *dIm, unsigned int nElements)
{
  cplReduce dReduce;
  float sum = dReduce.Sum(dIm, nElements);
  return sum;
}

float 
CUDAUtilities::
DeviceVectorSum(const cplVector3DArray &dV, unsigned int nElements)
{
  cplReduce dReduce;
  float sum = dReduce.Sum(dV.x, nElements);
  sum += dReduce.Sum(dV.y, nElements);
  sum += dReduce.Sum(dV.z, nElements);
  return sum;
}

void 
CUDAUtilities::
CheckCUDAError(const char *file, int line)
{
  std::string errMsg;
  if(hasCUDAError(errMsg)){
    ErrLog(file,line).Get(logERROR, true, true) 
      << errMsg;
  }
}

std::string
CUDAUtilities::
GetCUDAMemUsage()
{
  cuda_size free, total;
  cudaGetMemInfo(free, total);
  std::string rtn = StringUtils::strPrintf("CUDA Memory Usage: used %u of %u", (total - free) >> 20, total >> 20);
  return rtn;
}

void
CUDAUtilities::
AssertMinCUDACapabilityVersion(int devNum, int major, int minor)
{
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, devNum);
  if(deviceProp.major == 9999 && deviceProp.minor == 9999){
    std::string err = StringUtils::strPrintf("error, cuda-capable device %d not found", devNum);
    throw AtlasWerksException(__FILE__, __LINE__, err);
  }
  if(deviceProp.major < major || (deviceProp.major == major && deviceProp.minor < minor)){
    std::string err = StringUtils::strPrintf("error, cuda device %d supports cuda capability version %d.%d, "
					     "but %d.%d required", 
					     devNum, deviceProp.major, deviceProp.minor, major, minor);
    throw AtlasWerksException(__FILE__, __LINE__, err);
  }
}

void
CUDAUtilities::
AssertMinCUDACapabilityVersion(int major, int minor)
{
  int devNum;
  cudaGetDevice(&devNum);
  AssertMinCUDACapabilityVersion(devNum, major, minor);
}
