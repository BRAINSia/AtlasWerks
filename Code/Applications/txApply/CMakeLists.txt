SET(txApply_SRCS
  txApplyMain.cxx
)

ADD_EXECUTABLE(txApply "${txApply_SRCS}")

TARGET_LINK_LIBRARIES(txApply
  atlasWerksUtilitiesDataTypes
  atlasWerksNumerics
  atlasWerksDataTypes
  atlasWerksBase
  atlasWerksUtilPowerCrust
  ${ITK_LIBRARIES}
  ${atlasWerks_LAPACK_LIBS}
#   gfortran
  vtkGraphics
  vtkCommon
  vtkRendering
)

INSTALL(TARGETS 
  txApply
  DESTINATION bin
)

