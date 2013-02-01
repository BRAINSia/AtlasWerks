INCLUDE_DIRECTORIES(
  ${AtlasWerks_SOURCE_DIR}/Testing
  ${AtlasWerks_SOURCE_DIR}/Code/Libraries/Base
)

# Unit Tests
ADD_EXECUTABLE(StringUtils_Routine_Test
  StringUtilsTest.cxx
)

TARGET_LINK_LIBRARIES(StringUtils_Routine_Test
  atlasWerksBase
)

ADD_TEST(StringUtils_Routine_Test ${EXECUTABLE_OUTPUT_PATH}/StringUtils_Routine_Test)

ADD_EXECUTABLE(Param_Routine_Tests
  ParamTests.cxx
)

TARGET_LINK_LIBRARIES(Param_Routine_Tests
  atlasWerksBase
  ${ITK_LIBRARIES}
)
