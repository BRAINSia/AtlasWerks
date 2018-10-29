
#
# GreedyWarp
#
ADD_EXECUTABLE(GreedyWarp
  GreedyWarpMain.cxx
  )

SET(GreedyWarp_LIBS
  atlasWerksAlgorithms
  atlasWerksBase
  atlasWerksUtilDownsampleFilter
  ${ITK_LIBRARIES}
  ${FFTWF_LIB}
  ${atlasWerks_LAPACK_LIBS}
  )

TARGET_LINK_LIBRARIES(GreedyWarp
  ${GreedyWarp_LIBS}
  )

install(TARGETS GreedyWarp
  EXPORT AtlasWerksTargets
  RUNTIME DESTINATION bin
  )

#
# GreedyAtlas
#
ADD_EXECUTABLE(GreedyAtlas
  GreedyAtlasMain.cxx
  )

SET(GreedyAtlas_LIBS
  atlasWerksAlgorithms
  atlasWerksBase
  atlasWerksUtilDownsampleFilter
  ${ITK_LIBRARIES}
  ${FFTWF_LIB}
  ${atlasWerks_LAPACK_LIBS}
  )

TARGET_LINK_LIBRARIES(GreedyAtlas
  ${GreedyAtlas_LIBS}
  )

install(TARGETS GreedyAtlas
  EXPORT AtlasWerksTargets
  RUNTIME DESTINATION bin
  )

IF(MPI_FOUND)
  
  ADD_EXECUTABLE(GreedyAtlasCluster
    GreedyAtlasMain.cxx
    )
    
  # Define MPI_ENABLED for MPI-based cluster code
  SET_TARGET_PROPERTIES(GreedyAtlasCluster
    PROPERTIES 
    COMPILE_DEFINITIONS MPI_ENABLED
    )
  
  TARGET_LINK_LIBRARIES(GreedyAtlasCluster
    atlasWerksAlgorithmsMPI
    atlasWerksBase
    atlasWerksUtilDownsampleFilter
    ${ITK_LIBRARIES}
    ${FFTWF_LIB}
    ${atlasWerks_LAPACK_LIBS}
    )
    
ENDIF(MPI_FOUND)

#
# Testing
#

ADD_TEST(GreedyWarp_Regression_Tests ${AtlasWerks_SOURCE_DIR}/Testing/Applications/Greedy/GreedyWarp/RegressionTest.sh ${EXECUTABLE_OUTPUT_PATH} ${AtlasWerks_SOURCE_DIR}/Testing)

ADD_TEST(GreedyAtlas_Regression_Tests ${AtlasWerks_SOURCE_DIR}/Testing/Applications/Greedy/GreedyAtlas/RegressionTest.sh ${EXECUTABLE_OUTPUT_PATH} ${AtlasWerks_SOURCE_DIR}/Testing --nThreads 1)

IF(CUDA_ENABLED)
  
  ADD_TEST(GreedyWarp_GPU_Regression_Tests ${AtlasWerks_SOURCE_DIR}/Testing/Applications/Greedy/GreedyWarp/RegressionTest.sh ${EXECUTABLE_OUTPUT_PATH} ${AtlasWerks_SOURCE_DIR}/Testing --UseGPU true)
  
  ADD_TEST(GreedyAtlas_GPU_Regression_Tests ${AtlasWerks_SOURCE_DIR}/Testing/Applications/Greedy/GreedyAtlas/RegressionTest.sh ${EXECUTABLE_OUTPUT_PATH} ${AtlasWerks_SOURCE_DIR}/Testing --UseGPU true --nThreads 1)

ENDIF(CUDA_ENABLED)

