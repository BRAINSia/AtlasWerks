SUBDIRS(
  Algorithms
  Base
  UtilitiesDataTypes
)

IF(CUDA_ENABLED)
  SUBDIRS(CUDACommon)
ENDIF(CUDA_ENABLED)

IF(COMPILE_EXPERIMENTAL)
  SUBDIRS(Experimental)
ENDIF(COMPILE_EXPERIMENTAL)
