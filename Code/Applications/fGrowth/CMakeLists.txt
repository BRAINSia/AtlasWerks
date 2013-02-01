SET(fGrowth_SRCS
  main.cxx
)

ADD_EXECUTABLE(fGrowth "${fGrowth_SRCS}")

TARGET_LINK_LIBRARIES(fGrowth
  atlasWerksUtilDownsampleFilter
  atlasWerksAlgorithms
  atlasWerksBase
  ${ITK_LIBRARIES}
  ${FFTWF_LIB}
  )

