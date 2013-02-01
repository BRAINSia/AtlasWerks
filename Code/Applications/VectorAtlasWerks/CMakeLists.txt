SET(VectorAtlasWerks_SRCS
  main.cxx
  vectorDownsample.cxx
  vectorITKConversions.cxx
)

SET(maketestvecimages_SRCS
  maketestvecimages.cxx
)

ADD_EXECUTABLE(VectorAtlasWerks ${VectorAtlasWerks_SRCS})
ADD_EXECUTABLE(maketestvecimages ${maketestvecimages_SRCS})

TARGET_LINK_LIBRARIES(VectorAtlasWerks
  atlasWerksAlgorithms
  atlasWerksBase
  ${ITK_LIBRARIES}
  ${FFTWF_LIB}
)

TARGET_LINK_LIBRARIES(maketestvecimages
  ${ITK_LIBRARIES}
)
