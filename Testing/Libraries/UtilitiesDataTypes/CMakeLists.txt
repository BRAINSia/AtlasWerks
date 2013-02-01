IF(COMPILE_EXPERIMENTAL)
  INCLUDE_DIRECTORIES(
    ${AtlasWerks_SOURCE_DIR}/Testing
    ${AtlasWerks_SOURCE_DIR}/Testing/Libraries/Experimental/Metamorphosis
    ./OrigCode
    )

  # ImageUtils Unit Tests
  ADD_EXECUTABLE(ImageUtils_Routine_Tests
    ImageUtilsRoutineTests.cxx
    ${AtlasWerks_SOURCE_DIR}/Testing/TestUtils.cxx
    ${AtlasWerks_SOURCE_DIR}/Testing/Libraries/Experimental/Metamorphosis/MetamorphosisTestUtils.cxx
    ./OrigCode/VectorField3D.cxx
    ./OrigCode/Image3D.cxx
    )

  TARGET_LINK_LIBRARIES(ImageUtils_Routine_Tests
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

  #ADD_TEST(ImageUtils_Routine_Tests ${EXECUTABLE_OUTPUT_PATH}/ImageUtils_Routine_Tests ${AtlasWerks_SOURCE_DIR}/Testing/Data)

  # HField3DUtils Unit Tests
  ADD_EXECUTABLE(HField3DUtils_Routine_Tests
    HField3DUtilsRoutineTests.cxx
    ${AtlasWerks_SOURCE_DIR}/Testing/TestUtils.cxx
    ${AtlasWerks_SOURCE_DIR}/Testing/Libraries/Experimental/Metamorphosis/MetamorphosisTestUtils.cxx
    ./OrigCode/VectorField3D.cxx
    ./OrigCode/Image3D.cxx
    )

  TARGET_LINK_LIBRARIES(HField3DUtils_Routine_Tests
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

  ADD_TEST(HField3DUtils_Routine_Tests ${EXECUTABLE_OUTPUT_PATH}/HField3DUtils_Routine_Tests)
ENDIF(COMPILE_EXPERIMENTAL)
