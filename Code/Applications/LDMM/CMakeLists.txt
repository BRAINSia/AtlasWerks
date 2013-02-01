
INCLUDE_DIRECTORIES(
  ${AtlasWerks_SOURCE_DIR}/Code/Applications/LDMM
  )

#
# LDMWarp
#

ADD_EXECUTABLE(LDMMWarp
  LDMMWarpMain.cxx
  )

SET(LDMMWarp_LIBS
  atlasWerksAlgorithms
  atlasWerksBase
  atlasWerksUtilDownsampleFilter
  ${ITK_LIBRARIES}
  ${FFTWF_LIB}
  ${atlasWerks_LAPACK_LIBS}
  )

#
# LDMMWarpOld
#

ADD_EXECUTABLE(LDMMWarpOld
  LDMMWarpOld.cxx
  )

SET(LDMMWarpOld_LIBS
  atlasWerksAlgorithms
  atlasWerksBase
  atlasWerksUtilDownsampleFilter
  ${ITK_LIBRARIES}
  ${FFTWF_LIB}
  ${atlasWerks_LAPACK_LIBS}
  )

#
# LDMMAtlas
#
ADD_EXECUTABLE(LDMMAtlas
  LDMMAtlasMain.cxx
  )

SET(LDMMAtlas_LIBS
  atlasWerksAlgorithms
  atlasWerksBase
  atlasWerksUtilDownsampleFilter
  ${ITK_LIBRARIES}
  ${FFTWF_LIB}
  ${atlasWerks_LAPACK_LIBS}
  )

IF(MPI_FOUND)
    
  ADD_EXECUTABLE(LDMMAtlasCluster
    LDMMAtlasMain.cxx
    )
    
  # Define MPI_ENABLED for MPI-based cluster code
  SET_TARGET_PROPERTIES(LDMMAtlasCluster
    PROPERTIES 
    COMPILE_DEFINITIONS MPI_ENABLED
    )
  
  TARGET_LINK_LIBRARIES(LDMMAtlasCluster
    atlasWerksAlgorithmsMPI
    atlasWerksBase
    atlasWerksUtilDownsampleFilter
    ${ITK_LIBRARIES}
    ${FFTWF_LIB}
    ${atlasWerks_LAPACK_LIBS}
    )
    
  ADD_EXECUTABLE(LDMMMultiWarpCluster
    LDMMMultiWarpMain.cxx
    )
  
  # Define MPI_ENABLED for MPI-based cluster code
  SET_TARGET_PROPERTIES(LDMMMultiWarpCluster
    PROPERTIES 
    COMPILE_DEFINITIONS MPI_ENABLED
    )
  
  TARGET_LINK_LIBRARIES(LDMMMultiWarpCluster
    atlasWerksAlgorithmsMPI
    atlasWerksBase
    atlasWerksUtilDownsampleFilter
    ${ITK_LIBRARIES}
    ${FFTWF_LIB}
    ${atlasWerks_LAPACK_LIBS}
    )

ENDIF(MPI_FOUND)

TARGET_LINK_LIBRARIES(LDMMAtlas
  ${LDMMAtlas_LIBS}
  )

TARGET_LINK_LIBRARIES(LDMMWarp
  ${LDMMWarp_LIBS}
  )

TARGET_LINK_LIBRARIES(LDMMWarpOld
  ${LDMMWarpOld_LIBS}
  )

#
# LDMMAtlasOld
#

ADD_EXECUTABLE(LDMMAtlasOld
  LDMMAtlasOld.cxx
  )

TARGET_LINK_LIBRARIES(LDMMAtlasOld
  atlasWerksAlgorithms
  atlasWerksBase
  atlasWerksUtilDownsampleFilter
  ${ITK_LIBRARIES}
  ${FFTWF_LIB}
  ${atlasWerks_LAPACK_LIBS}
  )

#
# LDMMAtlasOldCluster
#
IF(MPI_FOUND)

  ADD_EXECUTABLE(LDMMAtlasOldCluster
    LDMMAtlasOld.cxx
    )
  
  # Define MPI_ENABLED for MPI-based cluster code
  SET_TARGET_PROPERTIES(LDMMAtlasOldCluster
    PROPERTIES 
    COMPILE_DEFINITIONS MPI_ENABLED
    )
  
  TARGET_LINK_LIBRARIES(LDMMAtlasOldCluster
    atlasWerksAlgorithmsMPI
    atlasWerksBase
    atlasWerksUtilDownsampleFilter
    ${ITK_LIBRARIES}
    ${FFTWF_LIB}
    ${atlasWerks_LAPACK_LIBS}
    )

ENDIF(MPI_FOUND)

#
# AlphaFromVelocities
#

ADD_EXECUTABLE(AlphaFromVelocities
  AlphaFromVelocitiesMain.cxx
  )

TARGET_LINK_LIBRARIES(AlphaFromVelocities
  atlasWerksAlgorithms
  atlasWerksBase
  atlasWerksUtilDownsampleFilter
  ${ITK_LIBRARIES}
  ${FFTWF_LIB}
  ${atlasWerks_LAPACK_LIBS}
  )

ADD_EXECUTABLE(LDMMVelocityUtils
  LDMMVelocityUtilsMain.cxx
  )

TARGET_LINK_LIBRARIES(LDMMVelocityUtils
  atlasWerksAlgorithms
  atlasWerksBase
  atlasWerksUtilDownsampleFilter
  ${ITK_LIBRARIES}
  ${FFTWF_LIB}
  ${atlasWerks_LAPACK_LIBS}
  )

INSTALL(TARGETS 
  LDMMVelocityUtils
  DESTINATION bin
)

# LDMMWarp
ADD_TEST(LDMMWarp_Regression_Tests ${AtlasWerks_SOURCE_DIR}/Testing/Applications/LDMM/LDMMWarp/RegressionTest.sh ${EXECUTABLE_OUTPUT_PATH} ${AtlasWerks_SOURCE_DIR}/Testing)

# LDMMAtlas
ADD_TEST(LDMMAtlas_Regression_Tests ${AtlasWerks_SOURCE_DIR}/Testing/Applications/LDMM/LDMMAtlas/RegressionTest.sh ${EXECUTABLE_OUTPUT_PATH} ${AtlasWerks_SOURCE_DIR}/Testing --nThreads 1)

# LDMMShootingWarp
ADD_TEST(LDMMShootingWarp_Regression_Tests ${AtlasWerks_SOURCE_DIR}/Testing/Applications/LDMM/LDMMShootingWarp/RegressionTest.sh ${EXECUTABLE_OUTPUT_PATH} ${AtlasWerks_SOURCE_DIR}/Testing)

# LDMMShootingAtlas
# ADD_TEST(LDMMShootingAtlas_Regression_Tests ${AtlasWerks_SOURCE_DIR}/Testing/Applications/LDMM/LDMMShootingAtlas/RegressionTest.sh ${EXECUTABLE_OUTPUT_PATH} ${AtlasWerks_SOURCE_DIR}/Testing --nThreads 1)

# LDMMAdShootingWarp
ADD_TEST(LDMMAdShootingWarp_Regression_Tests ${AtlasWerks_SOURCE_DIR}/Testing/Applications/LDMM/LDMMAdShootingWarp/RegressionTest.sh ${EXECUTABLE_OUTPUT_PATH} ${AtlasWerks_SOURCE_DIR}/Testing)

# Run GPU tests?
IF(CUDA_ENABLED)

# LDMMWarp GPU
  ADD_TEST(LDMMWarp_GPU_Regression_Tests ${AtlasWerks_SOURCE_DIR}/Testing/Applications/LDMM/LDMMWarp/RegressionTest.sh ${EXECUTABLE_OUTPUT_PATH} ${AtlasWerks_SOURCE_DIR}/Testing --UseGPU true)

# LDMMAtlas GPU
  ADD_TEST(LDMMAtlas_GPU_Regression_Tests ${AtlasWerks_SOURCE_DIR}/Testing/Applications/LDMM/LDMMAtlas/RegressionTest.sh ${EXECUTABLE_OUTPUT_PATH} ${AtlasWerks_SOURCE_DIR}/Testing --UseGPU true --nThreads 1)

# LDMMShootingWarp GPU
  ADD_TEST(LDMMShootingWarp_GPU_Regression_Tests ${AtlasWerks_SOURCE_DIR}/Testing/Applications/LDMM/LDMMShootingWarp/RegressionTest.sh ${EXECUTABLE_OUTPUT_PATH} ${AtlasWerks_SOURCE_DIR}/Testing --UseGPU true)

# LDMMShootingAtlas GPU
# ADD_TEST(LDMMShootingAtlas_GPU_Regression_Tests ${AtlasWerks_SOURCE_DIR}/Testing/Applications/LDMM/LDMMShootingAtlas/RegressionTest.sh ${EXECUTABLE_OUTPUT_PATH} ${AtlasWerks_SOURCE_DIR}/Testing --UseGPU true --nThreads 1)

# LDMMAdShootingWarp GPU
  ADD_TEST(LDMMAdShootingWarp_GPU_Regression_Tests ${AtlasWerks_SOURCE_DIR}/Testing/Applications/LDMM/LDMMAdShootingWarp/RegressionTest.sh ${EXECUTABLE_OUTPUT_PATH} ${AtlasWerks_SOURCE_DIR}/Testing --UseGPU true)

ENDIF(CUDA_ENABLED)

