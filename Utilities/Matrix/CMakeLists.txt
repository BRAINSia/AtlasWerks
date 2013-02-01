ADD_LIBRARY(atlasWerksUtilMatrix
  ${LIBTYPE} 
  Matrix.cxx
)
target_link_libraries(atlasWerksUtilMatrix
  ${atlasWerks_LAPACK_LIBS}
)

