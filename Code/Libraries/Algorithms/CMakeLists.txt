
INCLUDE_DIRECTORIES(../../../Testing)

#
# Algorithms (mostly legacy files)
#

SET(Algorithms_Misc_SRCS
  AtlasBuilder.cxx
  DiffeoImageGrowth.cxx  
  FluidWarp.cxx
  FluidWarpParameters.cxx
  ForwardSolve.cxx
  VectorAtlasBuilder.cxx
  )

SET(Algorithms_Misc_GPU_SRCS
  )

#
# Algorithms/Affine
#

SET(Algorithms_Affine_SRCS
  Affine/EstimateAffine.cxx
  )

SET(Algorithms_Affine_GPU_SRCS
  )

#
# Algorithms/Common
#

SET(Algorithms_Common_SRCS
  Common/DeformationIteratorInterface.cxx
  Common/AtlasBuilderInterface.cxx
  Common/DiffOper.cxx
  Common/Energy.cxx
  Common/EnergyHistory.cxx
  Common/ImagePCA.cxx
  Common/KernelFactory.cxx
  Common/MedianCalculator.cxx
  Common/MultiGaussianKernel.cxx
  Common/MultiscaleFluid.cxx
  Common/MultiscaleManager.cxx
  Common/WarpInterface.cxx
  )

SET(Algorithms_Common_GPU_SRCS
  Common/MultiscaleManagerGPU.cxx
  Common/DiffOperGPU.cxx
  Common/MultiGaussianKernelGPU.cxx
  )
  
#
# Algorithms/Greedy
#

SET(Algorithms_Greedy_SRCS
  Greedy/GreedyDeformationData.cxx
  Greedy/GreedyIteratorCPU.cxx
  Greedy/GreedyAtlasThreadCPU.cxx
  )

SET(Algorithms_Greedy_GPU_SRCS
  Greedy/GreedyDeformationDataGPU.cxx  
  Greedy/GreedyIteratorGPU.cxx
  Greedy/GreedyAtlasThreadGPU.cxx
  )

#
# Algorithms/LDMM
#

SET(Algorithms_LDMM_SRCS
  LDMM/LDMM.cxx                    
  LDMM/LDMMAtlasThreadCPU.cxx      
  LDMM/LDMMDeformationData.cxx     
  LDMM/LDMMEnergy.cxx              
  LDMM/LDMMGeodesicShooting.cxx
  LDMM/LDMMGeodesicShootingCPU.cxx
  LDMM/LDMMIteratorCPU.cxx         
  LDMM/LDMMParameters.cxx
  LDMM/LDMMShootingAtlasThreadCPU.cxx
  LDMM/LDMMShootingDefData.cxx
  LDMM/LDMMShootingIteratorCPU.cxx
  LDMM/LDMMVelShootingDefData.cxx
  LDMM/LDMMVelShootingIteratorCPU.cxx
  LDMM/LDMMAdShootingIteratorCPU.cxx
  LDMM/LDMMWarpCPU.cxx
  )

SET(Algorithms_LDMM_GPU_SRCS
  LDMM/LDMMAtlasThreadGPU.cxx      
  LDMM/LDMMDeformationDataGPU.cxx  
  LDMM/LDMMIteratorGPU.cxx
  LDMM/LDMMShootingAtlasThreadGPU.cxx
  LDMM/LDMMShootingDefDataGPU.cxx
  LDMM/LDMMShootingIteratorGPU.cxx
  LDMM/LDMMAdShootingIteratorGPU.cxx
  LDMM/LDMMWarpGPU.cxx
  LDMM/LDMMGeodesicShootingGPU.cxx
  )

#
# Set up overall source / lib list
#

SET(Algorithms_SRCS
  ${Algorithms_Affine_SRCS}
  ${Algorithms_Common_SRCS}
  ${Algorithms_Greedy_SRCS}
  ${Algorithms_LDMM_SRCS}
  ${Algorithms_Misc_SRCS}
  )

SET(Algorithms_LIBS
  atlasWerksBase
  atlasWerksUtilMatrix
  ${FFTWF_LIB}
  ${FFTWD_LIB}
  )

IF(CUDA_ENABLED)
  
# # We have to have at least sm_12 to compile the MMorph code
# # (required for splatting)
# IF(CUDA_ARCH_VERSION GREATER 11)
#   SET(LDMMGPU_SRCS
#     ${LDMMGPU_SRCS}
#     MMorphAtlasBuilderGPU.cxx
#     MMorphSingleGPUThread.cxx
#     MMorphIteratorGPU.cxx
#     MMorphWarpGPU.cxx)
# ENDIF(CUDA_ARCH_VERSION GREATER 11)
    
  SET(Algorithms_SRCS
    ${Algorithms_SRCS}
    ${Algorithms_Affine_GPU_SRCS}
    ${Algorithms_Common_GPU_SRCS}
    ${Algorithms_Greedy_GPU_SRCS}
    ${Algorithms_LDMM_GPU_SRCS}
    ${Algorithms_Misc_GPU_SRCS}
    )

  SET(Algorithms_LIBS
    ${Algorithms_LIBS}
    atlasWerksCUDACommon
    )

ENDIF(CUDA_ENABLED)

#
# Add target
#

ADD_LIBRARY(atlasWerksAlgorithms
  ${LIBTYPE} 
  ${Algorithms_SRCS}
  )

TARGET_LINK_LIBRARIES(atlasWerksAlgorithms
  ${Algorithms_LIBS}
  )

# Build MPI-enabled version of libraries if we have MPI
IF(MPI_FOUND)

  ADD_LIBRARY(atlasWerksAlgorithmsMPI
    ${LIBTYPE} 
    ${Algorithms_SRCS}
    )
  
  # Define MPI_ENABLED for MPI-based cluster code
  SET_TARGET_PROPERTIES(atlasWerksAlgorithmsMPI
    PROPERTIES 
    COMPILE_DEFINITIONS MPI_ENABLED
    )
  
  TARGET_LINK_LIBRARIES(atlasWerksAlgorithmsMPI
    ${Algorithms_LIBS}
    ${MPI_LIBRARY}
    )

ENDIF(MPI_FOUND)

IF(WRAP_PYTHON)
  # needed since this lib is linked with shared lib for python wrapping
  SET_TARGET_PROPERTIES(atlasWerksAlgorithms PROPERTIES COMPILE_FLAGS -fPIC)

  SET_SOURCE_FILES_PROPERTIES(Algorithms.i PROPERTIES CPLUSPLUS ON)
  SET_SOURCE_FILES_PROPERTIES(Algorithms.i PROPERTIES SWIG_FLAGS "-includeall")
  SWIG_ADD_MODULE(Algorithms python 
    Algorithms.i
    )
  SWIG_LINK_LIBRARIES(Algorithms 
    atlasWerksAlgorithms
    atlasWerksBase
    atlasWerksUtilDownsampleFilter
    ${ITK_LIBRARIES}
    ${FFTWF_LIB}
    ${FFTW_LIB}
    ${atlasWerks_LAPACK_LIBS}
    ${PYTHON_LIBRARIES})
ENDIF(WRAP_PYTHON)
