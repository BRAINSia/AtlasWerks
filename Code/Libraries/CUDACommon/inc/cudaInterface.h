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

#ifndef __CUDA_INTERFACE_H
#define __CUDA_INTERFACE_H

#include <cuda_runtime.h>
#include <cutil_math.h>
#include <string>

const int MAX_NUMBER_DEVICES = 256;
extern int cudaDeviceFlags[MAX_NUMBER_DEVICES];

// Compatibility for changes between CUDA 3.1 and 3.2
#if defined(CUDA_32) || defined(CUDA_40)
typedef size_t cuda_size;
#else
typedef uint cuda_size;
#endif
//

/**
 * Allocate / Free device memory with size in bytes. (Recommend to use the dmemAlloc/dmemFree instead)
 */
void allocateDeviceArray(void **devPtr, unsigned int size);
void freeDeviceArray(void *devPtr);

/**
 * Allocate / Free device memory with n is number of elements - Recommend to use.
 */
template<typename T>
void dmemAlloc(T*& d_ptr, uint n);
void dmemFree(void *devPtr);

/**
 * Allocate / Free pinned-memory (DMA enable) on host with n is number of elements.
 * This memory has much higher bandwith for data transfer between CPU and GPU
 * than regular new / malloc function
 */
template<typename T>
void allocatePinnedHostArray(T*& a, uint n);
void freePinnedHostArray(void* a);


/** @defgroup data_transfer Data transfer 
 *  Function to transfer the data from/to device.
 *  @{
 */
/** @brief Copy from device to host */
template<typename T>
void copyArrayFromDevice(T* host, const T* device,uint nElems);

/** @brief Copy from host to device */
template<typename T>
void copyArrayToDevice(T* device, const T* host, uint nElems);

/** @brief Copy from device to device */
template<typename T>
void copyArrayDeviceToDevice(T* d_dest, const T* d_src, uint nElems);

/** @brief Copy from device to device asynchronous*/
template<typename T>
void copyArrayDeviceToDeviceAsync(T* d_dest, const T* d_src, uint nElems, cudaStream_t stream=NULL);

/** @brief Copy from host to device asynchronous. REQUIRE: host to be pinned*/
template<typename T>
void copyArrayToDeviceAsync(T* device, const T* host, uint nElems, cudaStream_t stream=NULL);

/** @brief Copy from device to host asynchronous. REQUIRE: host to be pinned*/
template<typename T>
void copyArrayFromDeviceAsync(T* host, const T* device,uint nElems, cudaStream_t stream=NULL);

/** @brief Copy to constant memory from host */ 
template<typename T> 
void copyHostToConstant(const char* name, T* h_i, int cnt, cudaStream_t stream);

/** @brief Copy from constant memory to host */ 
template<typename T> 
void copyConstantToHost(T* h_i, const char* name, int cnt, cudaStream_t stream);

/** @brief Copy to constant memory from device */ 
template<typename T> 
void copyDeviceToConstant(const char* name, T* d_i, int cnt, cudaStream_t stream);

/** @brief Copy from constant memory to device */ 
template<typename T> 
void copyConstantToDevice(T* d_i, const char* name, int cnt, cudaStream_t stream);

/** @} */ // end of data transfer group

/** @defgroup cuda_information 
 *  CUDA device information querry
 *  @{
 */

/** @brief Get device infor (print > 0 if want to print out the status of the device*/ 
void getDeviceInfo(cudaDeviceProp& deviceProp, int dev, bool print=false);

/** @brief Get the number of CUDA capable device in the system*/ 
int  getNumberOfCapableCUDADevices();

/** @brief Get the identity number of the current device*/ 
int  getCurrentDeviceID();

void setDeviceFlag(int dev, int flag);
int  getDeviceFlag(int dev);
void setCurrentDeviceFlag(int flag);
int  getCurrentDeviceFlag();

/** @brief Check if the current device is capable of Zero Copy*/ 
bool isZeroCopyEnable();

/** @brief Check if the current device is capable of Zero Copy*/ 
bool isZeroCopyEnable(int dev);


/** @brief Get the information about memory of current device, free memory amount and total memory*/ 
void cudaGetMemInfo(cuda_size& free, cuda_size& total);

/** @brief Print meory usage at the point*/ 
void printCUDAMemoryUsage();

/** @} */ // end of information group

/** @defgroup Function error checking
 *  Check the execution of a CUDA function is successful or not
 *  @{
 */

/** @brief Check if the error happen. Recommend : don't use this one use cutilCheckMsg instead*/void checkCUDAError(const char *msg);

/** @brief Check if the error happen*/ 
bool hasCUDAError();

/** @brief Check if there is an error, if so return error in errMsg*/ 
bool hasCUDAError(std::string &errMsg);

/** @brief Get error message*/ 
std::string getCUDAError();

/**
 * Init cuda driver from the command line
 */
void cudaInit(int argc, char **argv);

/**
 * Thread synchronization for CUDA program
 */
void threadSync();


/** 
 * Simple supporting functions to check the correctness of CUDA implementation
 * should not been used as performance functions
 */

template<typename T>
T hostArraySum(T* a, int n) ;

template<typename T>
T deviceArraySum(T* a, int n) ;

float deviceArraySum(float* d_a, int n);
float getHostSumDouble(float* a, int n);
float getDeviceSumDouble(float* d_a, int n);

void getHostRange(float& minV, float& maxV, float* a, int n);
void getDeviceRange(float& minV, float& maxV, float* a, int n);

float getHostSum2Double(float* a, int n);
float getDeviceSum2Double(float* d_a, int n);

void writeHostArrayToFile(float* a, int n, char* name);
void writeDeviceArrayToFile(float* d_a, int n, char* name);
void readHostArrayFromFile(float** a, int* n, char* name);

#endif
