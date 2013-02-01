#CMakeLists.txt for plnuc planio library (loading of plan_im's,
#anastructs, etc)

# OS dependent stuff
IF (WIN32)
  ADD_DEFINITIONS(-DPLAN_WINNT)
ENDIF(WIN32)
IF (CMAKE_SYSTEM MATCHES "Linux*")
  ADD_DEFINITIONS(-DPLAN_LINUX)
ENDIF (CMAKE_SYSTEM MATCHES "Linux*")
IF (CMAKE_SYSTEM MATCHES "Sun*")
  ADD_DEFINITIONS(-DPLAN_SOLARIS2_4)
ENDIF (CMAKE_SYSTEM MATCHES "Sun*")
IF (CMAKE_SYSTEM MATCHES "Darwin*")
  ADD_DEFINITIONS(-DPLAN_DARWIN)
ENDIF (CMAKE_SYSTEM MATCHES "Darwin*")

ADD_LIBRARY(planio
  ${LIBTYPE} 
  get_filter.cxx
  get_lantis_data.cxx
  getopt.cxx
  get_phys_dat_dir.cxx
  get_plunc_root_dir.cxx
  get_slice.cxx
  get_time_calc.cxx
  get_tray.cxx
  get_unit.cxx
  plan_xdr_defs.cxx
  read_2d_grid.cxx
  read_anastruct.cxx
  read_beam.cxx
  read_byte.cxx
  read_contour.cxx
  read_filter.cxx
  read_grid.cxx
  read_image_header.cxx
  read_implant_desc.cxx
  read_many.cxx
  read_map.cxx
  read_norm.cxx
  read_object.cxx
  read_objects.cxx
  read_plan.cxx
  read_points.cxx
  read_sar.cxx
  read_scan.cxx
  read_time_calc.cxx
  read_tray.cxx
  read_unit.cxx
  read_weights.cxx
  realloc.cxx
  rpclib_name.cxx
  runlen.cxx
  sock_utils.cxx
  write_implant_desc.cxx
  write_object.cxx
  write_objects.cxx
  xdr_fd.cxx
  xdr_ll_planio.cxx
  )

