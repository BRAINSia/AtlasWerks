
INCLUDE_DIRECTORIES(${AtlasWerks_SOURCE_DIR}/Testing)

SET(TestFuncs_SRCS
  TestFuncs.cxx
  )

ADD_EXECUTABLE(TestFuncs 
  "${TestFuncs_SRCS}"
  ${AtlasWerks_SOURCE_DIR}/Testing/TestUtils.cxx
  )

TARGET_LINK_LIBRARIES(TestFuncs
  atlasWerksUtilitiesDataTypes
  atlasWerksNumerics
  atlasWerksDataTypes
  atlasWerksBase
  atlasWerksUtilPowerCrust
  ${ITK_LIBRARIES}
)

ADD_EXECUTABLE(GenDiff GenDiffMain.cxx)

TARGET_LINK_LIBRARIES(GenDiff
  atlasWerksUtilitiesDataTypes
  atlasWerksNumerics
  atlasWerksDataTypes
  atlasWerksBase
  ${ITK_LIBRARIES}
)

INSTALL(TARGETS 
  TestFuncs
  GenDiff
  DESTINATION bin
)

