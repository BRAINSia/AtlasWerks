INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR})

SET(ImageConvert_SRCS
  ImageConvert.cxx
)

ADD_EXECUTABLE(ImageConvert ${ImageConvert_SRCS})
TARGET_LINK_LIBRARIES(ImageConvert
  ${ITK_LIBRARIES}
)

INSTALL(TARGETS 
  ImageConvert
  DESTINATION bin
)
