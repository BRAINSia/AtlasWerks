SET(txWerks_SRCS
  txWerksMain.cxx
  )

ADD_EXECUTABLE(txWerks "${txWerks_SRCS}")

TARGET_LINK_LIBRARIES(txWerks
   atlasWerksBase
   ${ITK_LIBRARIES}
  ${atlasWerks_LAPACK_LIBS}
)

INSTALL(TARGETS 
  txWerks
  DESTINATION bin
)


