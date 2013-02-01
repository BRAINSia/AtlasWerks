ADD_LIBRARY(atlasWerksUtilitiesDataTypes
  ${LIBTYPE} 
  AnastructUtils.cxx
  SurfaceUtils.cxx
  SurfaceUtilsVtk.cxx
  VtkUtils.cxx
)

TARGET_LINK_LIBRARIES(atlasWerksUtilitiesDataTypes
  planio
)

IF(WRAP_PYTHON)
  SET_SOURCE_FILES_PROPERTIES(UtilitiesDataTypes.i PROPERTIES CPLUSPLUS ON)
  SET_SOURCE_FILES_PROPERTIES(UtilitiesDataTypes.i PROPERTIES SWIG_FLAGS "-includeall")
  SWIG_ADD_MODULE(UtilitiesDataTypes python UtilitiesDataTypes.i)
  SWIG_LINK_LIBRARIES(UtilitiesDataTypes 
    ${PYTHON_LIBRARIES}     
    ${ITK_LIBRARIES}
    atlasWerksBase
    atlasWerksUtilDownsampleFilter
    ${FFTWF_LIB}
    ${atlasWerks_LAPACK_LIBS}
    )
ENDIF(WRAP_PYTHON)
