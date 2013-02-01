#-----------------------------------------------------------------------------
# Top level CMakeLists.txt file for AtlasWerks
CMAKE_MINIMUM_REQUIRED(VERSION 2.6)
PROJECT(AtlasWerks)

if(COMMAND cmake_policy)
  cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)

#-----------------------------------------------------------------------------
# enable testing
ENABLE_TESTING()

#-----------------------------------------------------------------------------
# Include the standard Dart testing module
# INCLUDE (${CMAKE_ROOT}/Modules/Dart.cmake)

#-----------------------------------------------------------------------------
# set cmake module path
SET(AtlasWerks_CMAKE_DIR "${AtlasWerks_SOURCE_DIR}/CMake")
SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${AtlasWerks_CMAKE_DIR}")

#-----------------------------------------------------------------------------
# set directory for compiled libraries and executables
SET (LIBRARY_OUTPUT_PATH
  ${PROJECT_BINARY_DIR}/bin
  CACHE PATH
  "Single Directory for all libraries.")
SET (EXECUTABLE_OUTPUT_PATH
  ${PROJECT_BINARY_DIR}/bin
  CACHE PATH
  "Single Directory for all executables.")
MARK_AS_ADVANCED (
  LIBRARY_OUTPUT_PATH
  EXECUTABLE_OUTPUT_PATH
)

#-----------------------------------------------------------------------------
# set unix specific variables
IF(UNIX)
ENDIF(UNIX)

CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

#-----------------------------------------------------------------------------
# should shared libraries be built?
MARK_AS_ADVANCED(BUILD_SHARED_LIBS)
OPTION(BUILD_SHARED_LIBS "NOT READY YET, LEAVE OFF -- Build shared libraries?" OFF)
IF(BUILD_SHARED_LIBS)
  SET(LIBTYPE SHARED)
ELSE(BUILD_SHARED_LIBS)
  SET(LIBTYPE STATIC)
ENDIF(BUILD_SHARED_LIBS)
# compute default library install dir

#-----------------------------------------------------------------------------
# Include debug messages?
IF(CMAKE_BUILD_TYPE STREQUAL "Debug")
  # logDEBUG4 = 7
  ADD_DEFINITIONS(-DERRLOG_MAX_LEVEL=7)
ELSE(CMAKE_BUILD_TYPE STREQUAL "Debug")
  # logINFO = 2
  ADD_DEFINITIONS(-DERRLOG_MAX_LEVEL=2)
ENDIF(CMAKE_BUILD_TYPE STREQUAL "Debug")


#-----------------------------------------------------------------------------
# Define LIBRARY_INSTALL_DIR
set (_DEFAULT_LIBRARY_INSTALL_DIR lib)
if (EXISTS "${CMAKE_INSTALL_PREFIX}/lib32/" AND CMAKE_SIZEOF_VOID_P EQUAL 4)
  set (_DEFAULT_LIBRARY_INSTALL_DIR lib32)
elif (EXISTS "${CMAKE_INSTALL_PREFIX}/lib64/" AND CMAKE_SIZEOF_VOID_P EQUAL 8)
  set (_DEFAULT_LIBRARY_INSTALL_DIR lib64)
endif (EXISTS "${CMAKE_INSTALL_PREFIX}/lib32/" AND CMAKE_SIZEOF_VOID_P EQUAL 4)
# the library install dir
set(LIBRARY_INSTALL_DIR "${_DEFAULT_LIBRARY_INSTALL_DIR}" CACHE PATH "Installation directory for libraries")
# make the library install dir an absolute path (can be important e.g. when using CONFIGURE_FILE to embed
# the library installation directory into a file)
if(NOT IS_ABSOLUTE "${LIBRARY_INSTALL_DIR}")
  set(LIBRARY_INSTALL_DIR "${CMAKE_INSTALL_PREFIX}/${LIBRARY_INSTALL_DIR}")
endif(NOT IS_ABSOLUTE "${LIBRARY_INSTALL_DIR}")


#-----------------------------------------------------------------------------
# Compile ImMap?
#MARK_AS_ADVANCED(atlasWerks_COMPILE_APP_IMMAP)
OPTION(atlasWerks_COMPILE_APP_IMMAP "Compile ImMap" ON)

#-----------------------------------------------------------------------------
# should we use SWIG to generate python wrappers?
OPTION(WRAP_PYTHON "Generate python wrapping using SWIG?" OFF)
IF(WRAP_PYTHON)
  FIND_PACKAGE(SWIG REQUIRED)
  INCLUDE(${SWIG_USE_FILE})
  
  FIND_PACKAGE(PythonLibs)
  INCLUDE_DIRECTORIES(${PYTHON_INCLUDE_PATH})
  
  INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})

  SET(CMAKE_SWIG_FLAGS "")

  SET(PYTHON_INSTALL_DIR
    "/usr/lib64/python"
    CACHE STRING
    "Location to install AtlasWerks python module"
    )
  # Set up package creation and installation
  INSTALL(CODE "EXECUTE_PROCESS(COMMAND chmod ug+x ${AtlasWerks_SOURCE_DIR}/InstallPythonModule.sh)")
  INSTALL(CODE "EXECUTE_PROCESS(COMMAND ${AtlasWerks_SOURCE_DIR}/InstallPythonModule.sh ${CMAKE_BINARY_DIR})")
  INSTALL(DIRECTORY ${CMAKE_BINARY_DIR}/python/AtlasWerks DESTINATION ${PYTHON_INSTALL_DIR})
ENDIF(WRAP_PYTHON)

#-----------------------------------------------------------------------------
# Should we try to use DCMTK?

MARK_AS_ADVANCED(USE_DCMTK)
OPTION(USE_DCMTK "Use DCMTK libraries?" OFF)
IF(USE_DCMTK)

  SET(DCMTK_FOUND TRUE CACHE INTERNAL "")

  MARK_AS_ADVANCED(DCMTK_LIBS_SEARCH_DIRS)
  SET(DCMTK_LIBS_SEARCH_DIRS
    "/usr/local/dicom/lib"
    "/usr/local/lib"
    CACHE STRING
    "Locations to search for DCMTK libs"
    )

  MARK_AS_ADVANCED(DCMTK_INCLUDE_SEARCH_DIRS)
  SET(DCMTK_INCLUDE_SEARCH_DIRS
    "/usr/local/dicom/include"
    CACHE STRING
    "Locations to search for DCMTK libs"
    )

  find_path(atlasWerks_DCMTK_LIBRARY libdcmdata.a PATHS ${DCMTK_LIBS_SEARCH_DIRS} ENV LD_LIBRARY_PATH)
  #message(STATUS "atlasWerks_DCMTK_LIBRARY is ${atlasWerks_DCMTK_LIBRARY}")

  if(NOT atlasWerks_DCMTK_LIBRARY)
    SET(DCMTK_FOUND FALSE CACHE INTERNAL "")
    message(SEND_ERROR "DCMTK libraries not found in search path: ${DCMTK_LIBS_SEARCH_DIRS}; Will not use DCMTK")
  endif(NOT atlasWerks_DCMTK_LIBRARY)

  if(DCMTK_FOUND)
    find_path(atlasWerks_DCMTK_INCLUDE
      NAMES "dcmtk/dcmdata/dctk.h"
      PATHS "${DCMTK_INCLUDE_SEARCH_DIRS}"
      )
    #message(STATUS "atlasWerks_DCMTK_INCLUDE is ${atlasWerks_DCMTK_INCLUDE}")

    if(NOT atlasWerks_DCMTK_INCLUDE)
      SET(DCMTK_FOUND FALSE CACHE INTERNAL "")
      message(SEND_ERROR "DCMTK include dir not found in: ${DCMTK_INCLUDE_SEARCH_DIRS}; Will not use DCMTK")
    endif(NOT atlasWerks_DCMTK_INCLUDE)

  endif(DCMTK_FOUND)

ENDIF(USE_DCMTK)

#-----------------------------------------------------------------------------
# Set up CUDA
OPTION(USE_CUDA "Attempt to compile CUDA code" OFF)
IF(USE_CUDA)

  SET(CMAKE_MODULE_PATH "${AtlasWerks_CMAKE_DIR}/cuda" ${CMAKE_MODULE_PATH})
  FIND_PACKAGE(CUDA)
  IF(NOT CUDA_FOUND)
    MESSAGE("Can't find CUDA, will not compile apps requiring CUDA")
  ENDIF(NOT CUDA_FOUND)
  IF(NOT CUDA_SDK_ROOT_DIR)
    MESSAGE("Can't find CUDA SDK, will not compile apps requiring CUDA")
  ENDIF(NOT CUDA_SDK_ROOT_DIR)

  # Add definition for preprocessor
  IF(CUDA_FOUND AND CUDA_SDK_ROOT_DIR)
    SET(CUDA_ENABLED TRUE)
    ADD_DEFINITIONS(-DCUDA_ENABLED)
    
    find_path(CUDA_CUT_INCLUDE_DIR
      cutil.h
      PATHS ${CUDA_SDK_SEARCH_PATH}
      PATH_SUFFIXES "common/inc"
      DOC "Location of cutil.h"
      NO_DEFAULT_PATH
      )
    # Now search system paths
    find_path(CUDA_CUT_INCLUDE_DIR cutil.h DOC "Location of cutil.h")
    mark_as_advanced(CUDA_CUT_INCLUDE_DIR)
    INCLUDE_DIRECTORIES(${CUDA_CUT_INCLUDE_DIR})

    if(CMAKE_SIZEOF_VOID_P EQUAL 8)
      set(cuda_cutil_name cutil_x86_64)
    else(CMAKE_SIZEOF_VOID_P EQUAL 8)
      set(cuda_cutil_name cutil32)
    endif(CMAKE_SIZEOF_VOID_P EQUAL 8)
    
    find_library(CUDA_CUT_LIBRARY
      NAMES cutil ${cuda_cutil_name}
      PATHS ${CUDA_SDK_SEARCH_PATH}
      # The new version of the sdk shows up in common/lib, but the old one is in lib
      PATH_SUFFIXES "common/lib" "lib"
      DOC "Location of cutil library"
      NO_DEFAULT_PATH
      )
    # Now search system paths
    find_library(CUDA_CUT_LIBRARY 
      NAMES cutil ${cuda_cutil_name} 
      DOC "Location of cutil library")
    mark_as_advanced(CUDA_CUT_LIBRARY)
    set(CUDA_CUT_LIBRARIES ${CUDA_CUT_LIBRARY})
    
    find_path(CUDA_CUDPP_INCLUDE_DIR
      "cudpp/cudpp.h"
      PATHS ${CUDA_SDK_SEARCH_PATH}
      PATH_SUFFIXES "common/inc"
      DOC "Location of cudpp.h"
      NO_DEFAULT_PATH
      )
    # Now search system paths
    find_path(CUDA_CUDPP_INCLUDE_DIR "cudpp/cudpp.h" PATH_SUFFIXES "common/inc" DOC "Location of cudpp.h")
    mark_as_advanced(CUDA_CUDPP_INCLUDE_DIR)
    INCLUDE_DIRECTORIES(${CUDA_CUDPP_INCLUDE_DIR})

    find_library(CUDPP_LIB
      NAMES cudpp_x86_64 cudpp
      PATHS ${CUDA_SDK_SEARCH_PATH}
      PATH_SUFFIXES "common/lib/linux"
      DOC "Location of cudpp library"
      )
    IF(NOT CUDPP_LIB)
      MESSAGE("Error, CUDPP library not found")
    ENDIF(NOT CUDPP_LIB)

    SET(CUDA_LIBRARIES 
      ${CUDA_LIBRARIES}
      ${CUDA_CUT_LIBRARIES} 
      ${CUDA_CUBLAS_LIBRARIES}
      ${CUDA_CUFFT_LIBRARIES}
      ${CUDPP_LIB}
      )

    # Definitions for the cuda version
    IF(${CUDA_VERSION} VERSION_EQUAL "4.0")
      ADD_DEFINITIONS(-DCUDA_40)
    ELSEIF(${CUDA_VERSION} VERSION_EQUAL "3.2")
      ADD_DEFINITIONS(-DCUDA_32)
    ELSEIF(${CUDA_VERSION} VERSION_EQUAL "3.1")
      ADD_DEFINITIONS(-DCUDA_31)
    ELSE(${CUDA_VERSION} VERSION_EQUAL "4.0")
      MESSAGE(SEND_ERROR
	"CUDA version 3.1 or greater required to build AtlasWerks GPU code")
    ENDIF(${CUDA_VERSION} VERSION_EQUAL "4.0")

    MARK_AS_ADVANCED(CUDA_ARCH_VERSION)
    SET(CUDA_ARCH_VERSION
      "12"
      CACHE STRING
      "Cuda architecture version, ie -arch sm_??")

  ELSE(CUDA_FOUND AND CUDA_SDK_ROOT_DIR)
    SET(CUDA_ENABLED FALSE)
  ENDIF(CUDA_FOUND AND CUDA_SDK_ROOT_DIR)

ELSE(USE_CUDA)
  # Not using CUDA, make sure this is false
  SET(CUDA_ENABLED FALSE)
ENDIF(USE_CUDA)

#-----------------------------------------------------------------------------
# Setup MPI
OPTION(USE_MPI "Attempt to compile MPI code" OFF)
IF(USE_MPI)

  find_package(MPI)
  IF(MPI_FOUND)
    INCLUDE_DIRECTORIES(${MPI_INCLUDE_PATH})
  ELSE(MPI_FOUND)
    MESSAGE("Could not find MPI directory.")
  ENDIF(MPI_FOUND)

ELSE(USE_MPI)

  # Not using MPI, make sure this is false
  SET(MPI_FOUND FALSE)

ENDIF(USE_MPI)

#-----------------------------------------------------------------------------
# Setup FFTW
# * include directories are added automatically
# * applications must add link libraries ${FFTWF_LIB} or ${FFTWF_THREADS_LIB}
MARK_AS_ADVANCED(USE_FFTWF)
OPTION(USE_FFTWF "Use single precision FFTW if found" ON)
MARK_AS_ADVANCED(USE_FFTWD)
OPTION(USE_FFTWD "Use single precision FFTW if found" ON)
FIND_PACKAGE(FFTW REQUIRED)
include_directories(${FFTW_INCLUDE})

#----------------------------------------------------------------------------
# Set default interpolation method
MARK_AS_ADVANCED(DEFAULT_INTERP_METHOD)
SET(DEFAULT_INTERP_METHOD
  "INTERP_LINEAR"
  CACHE STRING
  "Set the default interpolation method (INTERP_NN,INTERP_LINEAR,INTERP_CUBIC)"
  )
ADD_DEFINITIONS(-DDEFAULT_INTERP_METHOD=${DEFAULT_INTERP_METHOD})


#-----------------------------------------------------------------------------
# Setup LAPACK
# * include directories are added automatically
# * applications must add link libraries ${FFTWF_LIB} or ${FFTWF_THREADS_LIB}
MARK_AS_ADVANCED(LAPACK_LIBS)
SET(LAPACK_LIBS
  "lapack blas f2c"
  CACHE STRING
  "The LAPACK libraries (lapack, blas, possibly gfortran)"
  )

MARK_AS_ADVANCED(LAPACK_LIBS_SEARCH_DIRS)
SET(LAPACK_LIBS_SEARCH_DIRS
  "${LAPACK_DIR}/SRC;${LAPACK_DIR}/BLAS/SRC;${LAPACK_DIR}/F2CLIBS/libf2c"
  CACHE STRING
  "Locations to search for LAPACK libs"
  )

# clear the library var
set(atlasWerks_LAPACK_LIBS 
  "")
# turn space-separated list into CMAKE style
# semicolon separated list
separate_arguments(LAPACK_LIBS)
# find the path to each specified library
foreach(_library ${LAPACK_LIBS})
  #message(STATUS "finding library ${_library}")
  find_library(atlasWerks_${_library}_LIBRARY
    NAMES ${_library}
    PATHS ${LAPACK_LIBS_SEARCH_DIRS}
    NO_DEFAULT_PATH
    NO_CMAKE_PATH
    NO_CMAKE_ENVIRONMENT_PATH
    NO_SYSTEM_ENVIRONMENT_PATH
    NO_CMAKE_SYSTEM_PATH
    )
  if(atlasWerks_${_library}_LIBRARY)
    #message(STATUS "${_library} found!: ${atlasWerks_${_library}_LIBRARY}")
    set(atlasWerks_LAPACK_LIBS
      ${atlasWerks_LAPACK_LIBS}
      ${atlasWerks_${_library}_LIBRARY}
      CACHE INTERNAL ""
      )
  else(atlasWerks_${_library}_LIBRARY)
    message("${_library} not found!")
  endif(atlasWerks_${_library}_LIBRARY)
  mark_as_advanced(atlasWerks_${_library}_LIBRARY)
endforeach(_library ${LAPACK_LIBS})
message(STATUS "atlasWerks_LAPACK_LIBS: ${atlasWerks_LAPACK_LIBS}")

#FIND_PACKAGE(LAPACK)
#IF(LAPACK_FOUND)
#  # nothing, no include files for lapack
#ELSE(LAPACK_FOUND)
#  MESSAGE("Could not find LAPACK directory.")
#ENDIF(LAPACK_FOUND)

#-----------------------------------------------------------------------------
# Setup FLTK if building ImMap
IF(atlasWerks_COMPILE_APP_IMMAP)
  FIND_PACKAGE(FLTK)
  IF(FLTK_FOUND)
    INCLUDE_DIRECTORIES (${FLTK_INCLUDE_DIR})
  ELSE(FLTK_FOUND)
    MESSAGE("Could not find FLTK directory.")
  ENDIF(FLTK_FOUND)
ENDIF(atlasWerks_COMPILE_APP_IMMAP)

#-----------------------------------------------------------------------------
# Setup ITK
# * include directories are added automatically
# * applications must add appropriate libraries to link libraries
#MARK_AS_ADVANCED(USE_ITK)
#OPTION(USE_ITK "Use ITK if found" ON)
FIND_PACKAGE(ITK)
IF(ITK_FOUND)
  INCLUDE (${ITK_USE_FILE})
ELSE(ITK_FOUND)
  MESSAGE("Could not find ITK directory.")
ENDIF(ITK_FOUND)

#-----------------------------------------------------------------------------
# Setup VTK
# * include directories are added automatically
# * applications must add appropriate libraries to link libraries
#MARK_AS_ADVANCED(USE_VTK)
#OPTION(USE_VTK "Use VTK if found" ON)
FIND_PACKAGE(VTK)
IF(VTK_FOUND)
  INCLUDE (${VTK_USE_FILE})
ELSE(VTK_FOUND)
  MESSAGE("Could not find VTK directory.")
ENDIF(VTK_FOUND)

#-----------------------------------------------------------------------------
# Setup Doxygen
FIND_PACKAGE(Doxygen)
SET(DOXY_DIR "${CMAKE_CURRENT_SOURCE_DIR}/Docs")
SET(DOXY_CONFIG "${DOXY_DIR}/Doxyfile")
IF (DOXYGEN_FOUND)
  ADD_CUSTOM_TARGET(doc ${DOXYGEN_EXECUTABLE} ${DOXY_CONFIG}
    WORKING_DIRECTORY "${DOXY_DIR}")
ENDIF (DOXYGEN_FOUND)

#-----------------------------------------------------------------------------
# what to compile?
OPTION(AtlasWerks_COMPILE_LIBS    "Compile AtlasWerks libraries."     ON)
OPTION(AtlasWerks_COMPILE_APPS    "Compile AtlasWerks applications."  ON)
OPTION(AtlasWerks_COMPILE_TESTING "Compile AtlasWerks testing."       ON)

#-----------------------------------------------------------------------------
# add common include directories
INCLUDE_DIRECTORIES(
  ${AtlasWerks_SOURCE_DIR}/Utilities/DownsampleFilter
  ${AtlasWerks_SOURCE_DIR}/Utilities/ImMapAttic
  ${AtlasWerks_SOURCE_DIR}/Utilities/ImMapAttic/ImageIO
  ${AtlasWerks_SOURCE_DIR}/Utilities/ImMapAttic/ImageIO/Dicom
  ${AtlasWerks_SOURCE_DIR}/Utilities/ImMapAttic/ImageIO/PlanIM
  ${AtlasWerks_SOURCE_DIR}/Utilities/Matrix
  ${AtlasWerks_SOURCE_DIR}/Utilities/planio
  ${AtlasWerks_SOURCE_DIR}/Utilities/PowerCrust
  ${AtlasWerks_SOURCE_DIR}/Utilities/tclap
  ${AtlasWerks_SOURCE_DIR}/Utilities/tinyxml
  ${AtlasWerks_SOURCE_DIR}/Code/Applications/ImMap
  ${AtlasWerks_SOURCE_DIR}/Code/Applications/Affine
  ${AtlasWerks_SOURCE_DIR}/Code/Applications/AffineAtlas
  ${AtlasWerks_SOURCE_DIR}/Code/Libraries/Algorithms/
  ${AtlasWerks_SOURCE_DIR}/Code/Libraries/Algorithms/Affine
  ${AtlasWerks_SOURCE_DIR}/Code/Libraries/Algorithms/Common
  ${AtlasWerks_SOURCE_DIR}/Code/Libraries/Algorithms/Greedy
  ${AtlasWerks_SOURCE_DIR}/Code/Libraries/Algorithms/LDMM
  ${AtlasWerks_SOURCE_DIR}/Code/Libraries/Base
  ${AtlasWerks_SOURCE_DIR}/Code/Libraries/DataTypes
  ${AtlasWerks_SOURCE_DIR}/Code/Libraries/Numerics
  ${AtlasWerks_SOURCE_DIR}/Code/Libraries/UtilitiesDataTypes
)

OPTION(COMPILE_EXPERIMENTAL "Compile experimental code?" OFF)
IF(COMPILE_EXPERIMENTAL)
  INCLUDE_DIRECTORIES(
    ${AtlasWerks_SOURCE_DIR}/Code/Libraries/Experimental/Algorithms
    ${AtlasWerks_SOURCE_DIR}/Code/Libraries/Experimental/Algorithms/MMorph
    ${AtlasWerks_SOURCE_DIR}/Code/Libraries/Experimental/Metamorphosis
    ${AtlasWerks_SOURCE_DIR}/Code/Libraries/Experimental/ScanProjection
    ${AtlasWerks_SOURCE_DIR}/Code/Libraries/Experimental/ScanProjection/ConeBeam
    ${AtlasWerks_SOURCE_DIR}/Code/Libraries/Experimental/ScanProjection/ImageSlice
    ${AtlasWerks_SOURCE_DIR}/Code/Libraries/Experimental/ScanProjection/KSpaceLine3D
    )
ENDIF(COMPILE_EXPERIMENTAL)


IF(CUDA_ENABLED)
  CUDA_INCLUDE_DIRECTORIES(
    ${CUDA_CUT_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    ${AtlasWerks_SOURCE_DIR}/Code/Libraries/CUDACommon/inc
  )

  INCLUDE_DIRECTORIES(
    ${CUDA_CUT_INCLUDE_DIRS} 
    ${CUDA_INCLUDE_DIRS}
    ${AtlasWerks_SOURCE_DIR}/Code/Libraries/CUDACommon/inc
  )
ENDIF(CUDA_ENABLED)

#-----------------------------------------------------------------------------
# Build utilities
SUBDIRS(
  ${AtlasWerks_SOURCE_DIR}/Utilities/DownsampleFilter
  ${AtlasWerks_SOURCE_DIR}/Utilities/Matrix  
  ${AtlasWerks_SOURCE_DIR}/Utilities/planio
  ${AtlasWerks_SOURCE_DIR}/Utilities/PowerCrust
#  ${AtlasWerks_SOURCE_DIR}/Utilities/tinyxml
)

IF(atlasWerks_COMPILE_APP_IMMAP)
  SUBDIRS(${AtlasWerks_SOURCE_DIR}/Utilities/ImMapAttic)
ENDIF(atlasWerks_COMPILE_APP_IMMAP)

#-----------------------------------------------------------------------------
# go to subdirectories
IF(AtlasWerks_COMPILE_LIBS)
  SUBDIRS(${AtlasWerks_SOURCE_DIR}/Code/Libraries)
ENDIF(AtlasWerks_COMPILE_LIBS)

IF(AtlasWerks_COMPILE_APPS)
  SUBDIRS(${AtlasWerks_SOURCE_DIR}/Code/Applications)
ENDIF(AtlasWerks_COMPILE_APPS)

IF(AtlasWerks_COMPILE_TESTING)
  SUBDIRS(${AtlasWerks_SOURCE_DIR}/Testing)
ENDIF(AtlasWerks_COMPILE_TESTING)



