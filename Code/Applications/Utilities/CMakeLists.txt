
#
# IntensityAverage
#

ADD_EXECUTABLE(ImageReadSliceWrite
  ImageReadSliceWriteMain.cxx
  )

TARGET_LINK_LIBRARIES(ImageReadSliceWrite
  atlasWerksBase
  ${ITK_LIBRARIES}
  )

#
# IntensityAverage
#

ADD_EXECUTABLE(IntensityAverage
  IntensityAverage.cxx
  )

TARGET_LINK_LIBRARIES(IntensityAverage
  atlasWerksBase
  ${ITK_LIBRARIES}
  )

#
# ForwardSolve
#

ADD_EXECUTABLE(ForwardSolve
  ForwardSolveMain.cxx
  )

TARGET_LINK_LIBRARIES(ForwardSolve
  atlasWerksBase
  atlasWerksAlgorithms
  atlasWerksUtilDownsampleFilter
  ${ITK_LIBRARIES}
  )

#
# VecUtils
#

ADD_EXECUTABLE(VecUtils
  VecUtilsMain.cxx
  )

TARGET_LINK_LIBRARIES(VecUtils
  atlasWerksBase
  atlasWerksAlgorithms
  ${ITK_LIBRARIES}
  )

#
# AlignCentroids
#

ADD_EXECUTABLE(AlignCentroids
  AlignCentroids.cxx
  )

TARGET_LINK_LIBRARIES(AlignCentroids
  atlasWerksBase
  atlasWerksAlgorithms
  atlasWerksUtilitiesDataTypes
  atlasWerksDataTypes
  atlasWerksUtilPowerCrust
  atlasWerksNumerics  
  vtkGraphics
  vtkFiltering
  vtkIO
  ${ITK_LIBRARIES}
  )

#
#Threshold
#
ADD_EXECUTABLE(Threshold
  Threshold.cxx
  )

TARGET_LINK_LIBRARIES(Threshold
  atlasWerksBase
  atlasWerksUtilDownsampleFilter
  ${ITK_LIBRARIES}
  )
#
# ExtractROI
#




ADD_EXECUTABLE(ExtractROI
  ExtractROIMain.cxx
  )

TARGET_LINK_LIBRARIES(ExtractROI
  atlasWerksBase
  atlasWerksUtilDownsampleFilter
  ${ITK_LIBRARIES}
  )
#
# DownsampleImage
#

ADD_EXECUTABLE(DownsampleImage
  DownsampleImageMain.cxx
  )

TARGET_LINK_LIBRARIES(DownsampleImage
  atlasWerksBase
  atlasWerksUtilDownsampleFilter
  ${ITK_LIBRARIES}
  )

#
# Test application for tricubic interpolation
#

ADD_EXECUTABLE(TestResample
  TestResampleMain.cxx
  )

TARGET_LINK_LIBRARIES(TestResample
  atlasWerksBase
  atlasWerksUtilDownsampleFilter
  ${ITK_LIBRARIES}
  )

#
# ImagePreprocessing
#

ADD_EXECUTABLE(ImagePreprocessing
  ImagePreprocessingMain.cxx
  )

TARGET_LINK_LIBRARIES(ImagePreprocessing
  atlasWerksBase
  atlasWerksUtilDownsampleFilter
  ${ITK_LIBRARIES}
  )

#
# PrintImageInfo
#

ADD_EXECUTABLE(PrintImageInfo
  PrintImageInfoMain.cxx
  )

TARGET_LINK_LIBRARIES(PrintImageInfo
  atlasWerksBase
  ${ITK_LIBRARIES}
  )

INSTALL(TARGETS 
  DownsampleImage
  ExtractROI
  ImagePreprocessing
  ImageReadSliceWrite
  IntensityAverage
  PrintImageInfo
  Threshold
  VecUtils
  DESTINATION bin
)

#
# DiffOper
#

ADD_EXECUTABLE(DiffOper
  DiffOperMain.cxx
  )

TARGET_LINK_LIBRARIES(DiffOper
  atlasWerksAlgorithms
  atlasWerksBase
  atlasWerksUtilDownsampleFilter
  ${ITK_LIBRARIES}
  ${FFTWF_LIB}
  ${atlasWerks_LAPACK_LIBS}
  )

