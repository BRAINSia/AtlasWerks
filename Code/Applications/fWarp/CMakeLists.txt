SET(fWarp_SRCS
  main.cxx
)

ADD_EXECUTABLE(fWarp "${fWarp_SRCS}")

TARGET_LINK_LIBRARIES(fWarp
  atlasWerksAlgorithms
  atlasWerksUtilitiesDataTypes
  atlasWerksNumerics
  atlasWerksDataTypes
  atlasWerksBase
  ${ITK_LIBRARIES}
  atlasWerksUtilDownsampleFilter
  ${atlasWerks_LAPACK_LIBS}
  ${FFTWF_LIB}
)

ADD_TEST(fWarp_Regression_Tests ${AtlasWerks_SOURCE_DIR}/Testing/Applications/Greedy/fWarp/fWarpRegressionTest.sh ${EXECUTABLE_OUTPUT_PATH} ${AtlasWerks_SOURCE_DIR}/Testing)

