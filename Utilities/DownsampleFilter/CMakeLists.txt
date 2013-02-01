ADD_LIBRARY(atlasWerksUtilDownsampleFilter
  ${LIBTYPE} 
  DownsampleFilter3D.cxx
  DownsizeFilter3D.cxx
  GaussianFilter3D.cxx
  DownsampleFilter2D.cxx
  DownsizeFilter2D.cxx
  GaussianFilter2D.cxx
)

IF(WRAP_PYTHON)
  SET_TARGET_PROPERTIES(atlasWerksUtilDownsampleFilter PROPERTIES COMPILE_FLAGS -fPIC)
ENDIF(WRAP_PYTHON)
