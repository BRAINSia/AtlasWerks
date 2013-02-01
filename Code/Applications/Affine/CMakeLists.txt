INCLUDE_DIRECTORIES(
  ${AtlasWerks_SOURCE_DIR}/Code/Applications/Affine
  )

SET(PROJ_NAME Affine)


SET(PROJ_SRCS
  ${PROJ_NAME}.cxx
  main.cxx
)

ADD_EXECUTABLE(${PROJ_NAME}
    ${PROJ_SRCS}
)

TARGET_LINK_LIBRARIES(Affine
  atlasWerksAlgorithms
  atlasWerksBase
  atlasWerksUtilitiesDataTypes
  atlasWerksDataTypes
  atlasWerksNumerics  
  atlasWerksUtilDownsampleFilter
  atlasWerksUtilMatrix
  ${atlasWerks_LAPACK_LIBS}
  ${ITK_LIBRARIES}
  ${FFTWF_LIB}
)

IF(WIN32)
  TARGET_LINK_LIBRARIES(Affine
    xdr
  )
ENDIF(WIN32)


