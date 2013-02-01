#CUDA Hiperformance image processing library 
INCLUDE_DIRECTORIES(${AtlasWerks_SOURCE_DIR}/Code/Libraries/CUDACommon/inc)

# Get rid of weird addAtomicity errors

IF(CUDA_ARCH_VERSION GREATER 11)
  SET(CUDA_NVCC_FLAGS -arch=sm_${CUDA_ARCH_VERSION})
ENDIF(CUDA_ARCH_VERSION GREATER 11)

SET(CUDA_SRCS
  VectorMath.cu
  VectorMathExt.cu
  comfunc.cpp
  cpuImage3D.cpp
  cpuRandomSupport.cpp
  cudaAccumulate.cu
  cudaAddScan.cu
  cudaAverage.cu
  cudaComplex.cu                            
  cudaComposition.cu
  cudaDataConvert.cu
  cudaDownSample.cpp
  cudaDownsizeFilter3D.cu
  cudaEventTimer.cu
  cudaFFTSolver.cu
  cudaFFTWrapper.cpp
  cudaGaussianFilter.cpp 
  cudaHField3DUtils.cu
  cudaHField3DUtilsSp.cu
  cudaHField3DUtilsExtra.cu
  cudaImage3D.cu
  cudaIndex3DArray.cu
  cudaInterface.cu
  cudaLOperator.cu
  cudaMap.cu
  cudaRandom.cu
  cudaRecursiveGaussianFilter.cu
  cudaReduce.cu
  cudaReduceStream.cu
  cudaSORSolver.cu
  cudaSeparableGaussFilter.cu
  cudaSort.cu
  cudaSortUtils.cu
  cudaSplatAtomic.cu
  cudaStream.cpp
  cudaStreamImages.cpp
  cudaTranspose.cu
  cudaTrilerp.cu
  cudaThreadSingleton.cpp
  cudaUpsample3D.cu
  cudaVector3DArray.cu
  cutil_comfunc.cpp
  imageDisplay.cu
  # gl/cudaBufferObject.cpp
  # gl/cudaMultiImageViewer.cpp
  # gl/GLRenderInterface.cpp
  # gl/GLSL_shader.cpp
  mesh/cudaMesh.cu
  mesh/cudaMeshOper.cu
  nddrUtils.cpp
  StringUtils.cxx
)  

IF(CUDA_ARCH_VERSION GREATER 11)
  SET(CUDA_SRCS
    ${CUDA_SRCS}
    cudaSplat.cu)
ENDIF(CUDA_ARCH_VERSION GREATER 11)


CUDA_ADD_LIBRARY(atlasWerksCUDACommon
  ${LIBTYPE} 
  ${CUDA_SRCS}
)

TARGET_LINK_LIBRARIES(atlasWerksCUDACommon
  ${CUDA_LIBRARIES}
  ${CUFFT_LIBRARIES}
  )


