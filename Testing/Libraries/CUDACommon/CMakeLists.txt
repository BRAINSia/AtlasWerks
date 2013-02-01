
INCLUDE_DIRECTORIES(${AtlasWerks_SOURCE_DIR}/Testing)

ADD_EXECUTABLE(cudaHField3DUtilsRoutineTests
  cudaHField3DUtilsRoutineTests.cxx
  ${AtlasWerks_SOURCE_DIR}/Testing/TestUtils.cxx
  )

TARGET_LINK_LIBRARIES(cudaHField3DUtilsRoutineTests
  atlasWerksAlgorithms
  atlasWerksUtilitiesDataTypes
  atlasWerksCUDACommon
  atlasWerksNumerics
  atlasWerksDataTypes
  atlasWerksBase
  atlasWerksUtilPowerCrust
  atlasWerksUtilDownsampleFilter
  ${ITK_LIBRARIES}
  ${FFTWF_LIB}
  ${atlasWerks_LAPACK_LIBS}
  )

ADD_TEST(cudaHField3DUtils_Routine_Tests ${EXECUTABLE_OUTPUT_PATH}/cudaHField3DUtilsRoutineTests)

# cudaWarpTimingTest

ADD_EXECUTABLE(cudaWarpTimingTest
  cudaWarpTimingTest.cxx
  )

TARGET_LINK_LIBRARIES(cudaWarpTimingTest
  atlasWerksAlgorithms
  atlasWerksUtilitiesDataTypes
  atlasWerksCUDACommon
  atlasWerksNumerics
  atlasWerksDataTypes
  atlasWerksBase
  atlasWerksUtilPowerCrust
  atlasWerksUtilDownsampleFilter
  ${ITK_LIBRARIES}
  ${FFTWF_LIB}
  ${atlasWerks_LAPACK_LIBS}
  )

