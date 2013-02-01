IF(CUDA_ENABLED)

  #
  # Compile the CUDA kernels separately
  #
  CUDA_ADD_LIBRARY(AtlasWerksCUDA
    GreedyExecutionFFT.cu
    )

  TARGET_LINK_LIBRARIES(AtlasWerksCUDA
    atlasWerksCUDACommon
    ${CUDA_LIBRARIES}
    )

  SET(AtlasWerks_SRC
    AtlasWerksMain.cxx
    GreedyAtlasBuilderCPU.cxx
    GreedyAtlasBuilderGPU.cxx
    GreedyAtlasBuilderSingleGPUThread.cxx
    )
  
  SET(AtlasWerks_LIBS
    atlasWerksUtilDownsampleFilter
    atlasWerksAlgorithms
    atlasWerksBase
    AtlasWerksCUDA
    atlasWerksCUDACommon
    ${ITK_LIBRARIES}
    ${FFTWF_LIB}
    ${FFTWD_LIB}
    ${atlasWerks_LAPACK_LIBS}
    )

ELSE(CUDA_ENABLED)
  
  #
  # Just build the CPU version
  #
  
  SET(AtlasWerks_SRC
    AtlasWerksMain.cxx
    GreedyAtlasBuilderCPU.cxx
    GreedyAtlasBuilderGPU_NullImpl.cxx
    GreedyAtlasBuilderSingleGPUThread_NullImpl.cxx
    )
  
  SET(AtlasWerks_LIBS
    atlasWerksUtilDownsampleFilter
    atlasWerksAlgorithms
    atlasWerksBase
    ${ITK_LIBRARIES}
    ${FFTWF_LIB}
    ${FFTWD_LIB}
    ${atlasWerks_LAPACK_LIBS}
    )
  
ENDIF(CUDA_ENABLED)

ADD_EXECUTABLE(AtlasWerks
  ${AtlasWerks_SRC}
  )

TARGET_LINK_LIBRARIES(AtlasWerks
  atlasWerksAlgorithms
  ${AtlasWerks_LIBS}
  )
  
IF(MPI_FOUND)
  # 
  # If MPI_FOUND is true, then USE_MPI is also true -- build cluster
  # code
  #
  
  #CUDA_ADD_EXECUTABLE(AtlasWerksCluster
  ADD_EXECUTABLE(AtlasWerksCluster
    ${AtlasWerks_SRC}
    )
  
  # Define MPI_ENABLED for MPI-based cluster code
  SET_TARGET_PROPERTIES(AtlasWerksCluster
    PROPERTIES 
    COMPILE_DEFINITIONS MPI_ENABLED
    )
  
  TARGET_LINK_LIBRARIES(AtlasWerksCluster
    atlasWerksAlgorithmsMPI
    ${AtlasWerks_LIBS}
    ${MPI_LIBRARY}
    )
ENDIF(MPI_FOUND)


ADD_TEST(AtlasWerks_Regression_Tests ${AtlasWerks_SOURCE_DIR}/Testing/Applications/Greedy/AtlasWerks/RegressionTest.sh ${EXECUTABLE_OUTPUT_PATH} ${AtlasWerks_SOURCE_DIR}/Testing)

IF(CUDA_ENABLED)
  ADD_TEST(AtlasWerks_Regression_Tests_GPU ${AtlasWerks_SOURCE_DIR}/Testing/Applications/Greedy/AtlasWerks/RegressionTest.sh ${EXECUTABLE_OUTPUT_PATH} ${AtlasWerks_SOURCE_DIR}/Testing --UseGPU true --nGPUs 1)
ENDIF(CUDA_ENABLED)
