
SET(ImMap_SRCS
  HistogramDat.cxx
  HistogramWin.cxx
  WindowCS.cxx
  ImMap.cxx
  ImAna.cxx
  Window2D.cxx
  DicomPreview.cxx
  BYUTable.cxx
  SurfaceViewWindow.cxx
  main.cxx
)

SET(ImMap_GUI_SRCS
  ImMapGUI.fl
)

FLTK_WRAP_UI(ImMap ${ImMap_GUI_SRCS})

ADD_EXECUTABLE(ImMap ${ImMap_SRCS} ${ImMap_FLTK_UI_SRCS})

TARGET_LINK_LIBRARIES(ImMap
  atlasWerksAlgorithms
  atlasWerksBase
  atlasWerksUtilitiesDataTypes
  atlasWerksDataTypes
  atlasWerksNumerics  
  atlasWerksUtilImMapAttic
  atlasWerksUtilDownsampleFilter
  atlasWerksUtilMatrix
  atlasWerksUtilPowerCrust
  ${FLTK_LIBRARIES}
  vtkGraphics
  vtkRendering
  vtkFiltering
  vtkIO
  ${atlasWerks_LAPACK_LIBS}
  ${ITK_LIBRARIES}
  ${FFTWF_LIB}
)

IF(WIN32)
  TARGET_LINK_LIBRARIES(ImMap
    xdr
  )
ENDIF(WIN32)

INSTALL(TARGETS 
  ImMap
  DESTINATION bin
)

