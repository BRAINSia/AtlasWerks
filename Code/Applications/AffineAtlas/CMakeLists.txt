INCLUDE_DIRECTORIES(
  ${AtlasWerks_SOURCE_DIR}/Code/Applications/AffineAtlas
  )

SET(PROJ_NAME AffineAtlas)


SET(PROJ_SRCS
  EstimateAffineAtlas.cxx
  AffineAtlas.cxx
  )

ADD_EXECUTABLE(AffineAtlas
    ${PROJ_SRCS}
    AffineAtlasMain.cxx
)

TARGET_LINK_LIBRARIES(AffineAtlas
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

 IF(MPI_FOUND)

ADD_EXECUTABLE(AffineAtlasCluster
    ${PROJ_SRCS}
    AffineAtlasBuilderCPU.cxx
    AffineAtlasClusterMain.cxx

)

  # Define MPI_ENABLED for MPI-based cluster code
  SET_TARGET_PROPERTIES(AffineAtlasCluster
    PROPERTIES
    COMPILE_DEFINITIONS MPI_ENABLED
    )

TARGET_LINK_LIBRARIES(AffineAtlasCluster
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
  atlasWerksAlgorithmsMPI
  ${MPI_LIBRARY}

)

 ENDIF(MPI_FOUND)

IF(WIN32)
  TARGET_LINK_LIBRARIES(AffineAtlas
    xdr
  )
ENDIF(WIN32)


