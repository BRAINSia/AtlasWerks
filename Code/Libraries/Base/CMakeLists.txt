SET(atlasWerksBase_SRCS
  AtlasWerksDebug.cxx
  ApplicationUtils.cxx
  CmdLineParser.cxx
  ImagePreprocessor.cxx
  StringUtils.cxx
  Timer.cxx
  WeightedImageSet.cxx
  ../../../Utilities/tinyxml/tinyxml.cxx
  ../../../Utilities/tinyxml/tinyxmlparser.cxx
  ../../../Utilities/tinyxml/tinyxmlerror.cxx
)

SET(atlasWerksBase_LIBS
  ${ITK_LIBRARIES}
)

IF(CUDA_ENABLED)

  SET(atlasWerksBase_SRCS
    ${atlasWerksBase_SRCS}
    CUDAUtilities.cxx
    )

  SET(atlasWerksBase_LIBS
    ${atlasWerksBase_LIBS}
    atlasWerksCUDACommon
    )

ENDIF(CUDA_ENABLED)

ADD_LIBRARY(atlasWerksBase ${LIBTYPE} ${atlasWerksBase_SRCS})


TARGET_LINK_LIBRARIES(atlasWerksBase ${atlasWerksBase_LIBS})

IF(WRAP_PYTHON)
  # needed since this lib is linked with shared lib for python wrapping
  SET_TARGET_PROPERTIES(atlasWerksBase PROPERTIES COMPILE_FLAGS -fPIC)

  SET_SOURCE_FILES_PROPERTIES(Base.i PROPERTIES CPLUSPLUS ON)
  SET_SOURCE_FILES_PROPERTIES(Base.i PROPERTIES SWIG_FLAGS "-includeall")
  SWIG_ADD_MODULE(Base python Base.i ApplicationUtils.cxx)
  SWIG_LINK_LIBRARIES(Base 
    ${PYTHON_LIBRARIES}
    atlasWerksBase
    ${ITK_LIBRARIES}
    )
ENDIF(WRAP_PYTHON)
