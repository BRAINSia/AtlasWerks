ADD_LIBRARY(atlasWerksDataTypes
#  VectorField.cxx
  Anastruct.cxx
  Color.cxx
  Contour.cxx
  Line3D.cxx
  Surface.cxx
  )

IF(WRAP_PYTHON)
  SET_SOURCE_FILES_PROPERTIES(DataTypes.i PROPERTIES CPLUSPLUS ON)
  SET_SOURCE_FILES_PROPERTIES(DataTypes.i PROPERTIES SWIG_FLAGS "-includeall")
  SWIG_ADD_MODULE(DataTypes python DataTypes.i)
  SWIG_LINK_LIBRARIES(DataTypes 
    atlasWerksBase
    ${PYTHON_LIBRARIES}
    ${ITK_LIBRARIES}
    ${atlasWerks_LAPACK_LIBS}
    )
ENDIF(WRAP_PYTHON)
