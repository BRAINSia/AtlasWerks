INCLUDE_DIRECTORIES(
  ${AtlasWerks_SOURCE_DIR}/Testing
)

# DiffOper Unit Tests
SET(DiffOper_Routine_Tests_SRCS
  DiffOperRoutineTests.cxx
  ${AtlasWerks_SOURCE_DIR}/Testing/TestUtils.cxx
  )

SET(DiffOper_Routine_Tests_LIBS
  atlasWerksAlgorithms
  atlasWerksUtilitiesDataTypes
  atlasWerksNumerics
  atlasWerksDataTypes
  atlasWerksBase
  atlasWerksUtilPowerCrust
  atlasWerksUtilDownsampleFilter
  ${ITK_LIBRARIES}
  ${FFTWF_LIB}
  ${atlasWerks_LAPACK_LIBS}
  )

IF(CUDA_ENABLED)
  SET(DiffOper_Routine_Tests_SRCS
    ${DiffOper_Routine_Tests_SRCS}
    )

  SET(DiffOper_Routine_Tests_LIBS
    ${DiffOper_Routine_Tests_LIBS}
    atlasWerksCUDACommon
    )
ENDIF(CUDA_ENABLED)

ADD_EXECUTABLE(DiffOper_Routine_Tests
  ${DiffOper_Routine_Tests_SRCS}
)

TARGET_LINK_LIBRARIES(DiffOper_Routine_Tests
  ${DiffOper_Routine_Tests_LIBS}
)

ADD_TEST(DiffOper_Routine_Tests ${EXECUTABLE_OUTPUT_PATH}/DiffOper_Routine_Tests)

# FluidWarp Unit Tests
ADD_EXECUTABLE(FluidWarp_Routine_Tests
  FluidWarpRoutineTests.cxx
  ${AtlasWerks_SOURCE_DIR}/Testing/TestUtils.cxx
)

TARGET_LINK_LIBRARIES(FluidWarp_Routine_Tests
  atlasWerksAlgorithms
  atlasWerksUtilitiesDataTypes
  atlasWerksNumerics
  atlasWerksDataTypes
  atlasWerksBase
  atlasWerksUtilPowerCrust
  atlasWerksUtilDownsampleFilter
  ${ITK_LIBRARIES}
  ${FFTWF_LIB}
  ${atlasWerks_LAPACK_LIBS}
)

ADD_TEST(FluidWarp_Routine_Tests ${EXECUTABLE_OUTPUT_PATH}/FluidWarp_Routine_Tests ${AtlasWerks_SOURCE_DIR}/Testing/Data)

# Mass Conservarion Test
ADD_EXECUTABLE(MassConservationTest
  MassConservationTest.cxx
)

TARGET_LINK_LIBRARIES(MassConservationTest
  atlasWerksAlgorithms
  atlasWerksUtilitiesDataTypes
  atlasWerksNumerics
  atlasWerksDataTypes
  atlasWerksBase
  atlasWerksUtilPowerCrust
  atlasWerksUtilDownsampleFilter
  ${ITK_LIBRARIES}
  ${FFTWF_LIB}
  ${atlasWerks_LAPACK_LIBS}
)

# Momentum Test
ADD_EXECUTABLE(ShootingRegistration
  ShootingRegistration.cxx
)

TARGET_LINK_LIBRARIES(ShootingRegistration
  atlasWerksAlgorithms
  atlasWerksUtilitiesDataTypes
  atlasWerksNumerics
  atlasWerksDataTypes
  atlasWerksBase
  atlasWerksUtilPowerCrust
  atlasWerksUtilDownsampleFilter
  ${ITK_LIBRARIES}
  ${FFTWF_LIB}
  ${atlasWerks_LAPACK_LIBS}
)

# ScaleManager
ADD_EXECUTABLE(ScaleManager_Routine_Tests
  MultiscaleManagerTests.cxx
  ${AtlasWerks_SOURCE_DIR}/Testing/TestUtils.cxx
)

TARGET_LINK_LIBRARIES(ScaleManager_Routine_Tests
  atlasWerksAlgorithms
  atlasWerksUtilitiesDataTypes
  atlasWerksNumerics
  atlasWerksDataTypes
  atlasWerksBase
  atlasWerksUtilPowerCrust
  atlasWerksUtilDownsampleFilter
  ${ITK_LIBRARIES}
  ${FFTWF_LIB}
  ${atlasWerks_LAPACK_LIBS}
)

ADD_TEST(ScaleManager_Routine_Tests ${EXECUTABLE_OUTPUT_PATH}/ScaleManager_Routine_Tests)

# LDMMIterator

IF(CUDA_ENABLED)
ADD_EXECUTABLE(LDMMIteratorTests
  LDMMIteratorTest.cxx
  ${AtlasWerks_SOURCE_DIR}/Testing/TestUtils.cxx
)

TARGET_LINK_LIBRARIES(LDMMIteratorTests
  atlasWerksAlgorithms
  atlasWerksCUDACommon
  atlasWerksUtilitiesDataTypes
  atlasWerksDataTypes
  atlasWerksBase
  atlasWerksUtilDownsampleFilter
  ${ITK_LIBRARIES}
  ${FFTWF_LIB}
  ${atlasWerks_LAPACK_LIBS}
)
ADD_TEST(LDMMIterator_Tests ${EXECUTABLE_OUTPUT_PATH}/LDMMIteratorTests)
ENDIF(CUDA_ENABLED)
