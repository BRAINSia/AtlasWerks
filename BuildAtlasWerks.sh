#!/bin/bash
#
# This script is still experimental, please report any problems.  It
# will download the AtlasWerk dependencies and build AtlasWerks
#

# Please make sure the following packages are installed:
# fftw3-devel
# fftw3-threads
# fftw3-threads-devel
# fltk
# fltk-devel
# gcc-fortran
# gcc-fortran-32bit
# lapack
# blas
# python-devel
# itcl

# the following requirements should already be installed:
# cmake, git, cvs, subversion

# Check if all needed packages are installed
# RPMSINSTALLED = 

#
# SETTINGS
#

# This is the location where the software will be installed
REPOSITORY="$HOME/tmp/software"

# This is the CMake build type -- Release, Debug, RelWithDebInfo, MinSizeRel
BUILDTYPE="RelWithDebInfo"

# Should CompOnc be built with CUDA support?
USECUDA="OFF"

# Maximum # of processors to use for building (will be fewer if # of
# cores < Max # of processors)
MAXPROCS=2

#
# END SETTINGS
#

ATLASWERKS_SRC_DIR=`pwd`
ATLASWERKS_BIN_DIR=${ATLASWERKS_SRC_DIR}/CompOnc-${BUILDTYPE}

# make repository if it doesn't exist
mkdir -p $REPOSITORY
if [ $? -ne 0 ]; then
    echo "Error creating repository $REPOSITORY"
    exit $?
fi

if [ `uname -p` == "x86_64" ]; then
    LIBDIR="lib64"
else
    LIBDIR="lib"
fi

USER_LOCAL="$HOME/usr/local"
mkdir -p $USER_LOCAL
PATH=$HOME/usr/local/bin:$PATH
LD_LIBRARY_PATH=$HOME/usr/local/$LIBDIR:$LD_LIBRARY_PATH

export PATH
export LD_LIBRARY_PATH

mkdir -p

CPUCOUNT=`cat /proc/cpuinfo | grep processor | wc -l`
NPROCS=$CPUCOUNT
if [ $NPROCS -gt $MAXPROCS ]; then
    NPROCS=$MAXPROCS
fi


CMAKE_SOURCE=cmake-2.6.4.tar.gz
CMAKE_SOURCE_DIR=cmake-2.6.4
CMAKE_SOURCE_URL=http://www.cmake.org/files/v2.6/${CMAKE_SOURCE}

ITK_BIN_DIR=${REPOSITORY}/itk-bin
#ITKAPP_BIN_DIR=${REPOSITORY}/itkapp-bin
VTK_BIN_DIR=${REPOSITORY}/vtk-bin
FLTK_BIN_DIR=/usr

#VTK_DATA_DIR=${REPOSITORY}/VTKData

ITK_VERSION="v3.18.0"
#ITKAPP_VERSION="v3.18.0-apps"
VTK_VERSION="v5.4.0"
#VTKDATA_VERSION="v5.4.0-data"

SYSTEM_CMAKE_26=$(cmake --version | grep -q 2\\.6\\.0; echo $?)

if [ $SYSTEM_CMAKE_26 -eq 0 ]; then
  echo "Using system cmake"
  CMAKE_BASE="cmake\
             -DCMAKE_BUILD_TYPE=$BUILDTYPE\
             -DBUILD_TESTING:BOOL=OFF\
             -DBUILD_EXAMPLES:BOOL=OFF\
             -DCMAKE_INSTALL_PREFIX=${HOME}/usr/local"
else
  echo "Using and/or building local cmake"
  CMAKE_BASE="${REPOSITORY}/${CMAKE_SOURCE_DIR}/bin/cmake\
              -DCMAKE_BUILD_TYPE=$BUILDTYPE\
              -DBUILD_TESTING:BOOL=OFF\
              -DBUILD_EXAMPLES:BOOL=OFF\
              -DCMAKE_INSTALL_PREFIX=${HOME}/usr/local"
fi

ITK_OPTS="\
           -DBUILD_SHARED_LIBS:BOOL=ON\
           -DUSE_FFTWD:BOOL=ON\
           -DUSE_FFTWF:BOOL=ON\
           -DITK_USE_SYSTEM_PNG:BOOL=ON\
           -DITK_USE_SYSTEM_TIFF:BOOL=ON\
           -DITK_USE_SYSTEM_ZLIB:BOOL=ON\
           -DITK_USE_REVIEW:BOOL=OFF\
           -DFFTWD_LIB:PATH=/usr/${LIBDIR}/libfftw3.so.3\
           -DFFTWD_THREADS_LIB:PATH=/usr/${LIBDIR}/libfftw3_threads.so.3\
           -DFFTWF_LIB:PATH=/usr/${LIBDIR}/libfftw3f.so.3\
           -DFFTWF_THREADS_LIB:PATH=/usr/${LIBDIR}/libfftw3f_threads.so.3\
           -DFFTW_INCLUDE_PATH:PATH=/usr/include"
           

VTK_OPTS="\
          -DBUILD_SHARED_LIBS:BOOL=ON\
          -DVTK_DATA_ROOT:PATH=${VTK_DATA_DIR}\
          -DVTK_WRAP_TCL:BOOL=ON\
          -DVTK_USE_RPATH:BOOL=ON\
          -DVTK_USE_GUISUPPORT:BOOL=ON\
          -DVTK_USE_SYSTEM_EXPAT:BOOL=ON\
          -DVTK_USE_SYSTEM_FREETYPE:BOOL=ON\
          -DVTK_USE_SYSTEM_JPEG:BOOL=ON\
          -DVTK_USE_SYSTEM_PNG:BOOL=ON\
          -DVTK_USE_LIBXML2:BOOL=ON\
          -DVTK_USE_SYSTEM_TIFF:BOOL=ON\
          -DVTK_USE_SYSTEM_ZLIB:BOOL=ON\
          -DDESIRED_QT_VERSION:INT=3\
          -DQT_MOC_EXECUTABLE:PATH=/usr/${LIBDIR}/qt3/bin/moc\
          -DQT_UIC_EXECUTABLE:PATH=/usr/${LIBDIR}/qt3/bin/uic\
          -DQT_QMAKE_EXECUTABLE:PATH=/usr/${LIBDIR}/qt3/bin/qmake\
          -DQT_QMAKE_EXECUTABLE:PATH=/usr/${LIBDIR}/qt3/bin/qmake\
          -DTK_INTERNAL_PATH:PATH=${REPOSITORY}/VTK/Utilities/TclTk/internals/tk8.4"
  
# ITKAPP_OPTS="\
#           -DBUILD_SHARED_LIBS:BOOL=ON\
#           -DITK_DIR:PATH=${ITK_BIN_DIR}\
#           -DVTK_DIR:PATH=${VTK_BIN_DIR}\
#           -DUSE_FLTK:BOOL=ON\
#           -DUSE_QT:BOOL=ON\
#           -DUSE_VTK:BOOL=ON\
#           -DFLTK_DIR:PATH=${FLTK_BIN_DIR}\
#           -DDESIRED_QT_VERSION:INT=3\
#           -DQT_MOC_EXECUTABLE:PATH=/usr/${LIBDIR}/qt3/bin/moc\
#           -DQT_UIC_EXECUTABLE:PATH=/usr/${LIBDIR}/qt3/bin/uic\
#           -DQT_QMAKE_EXECUTABLE:PATH=/usr/${LIBDIR}/qt3/bin/qmake\
#           -DQT_QMAKE_EXECUTABLE_FINDQT:PATH=/usr/${LIBDIR}/qt3/bin/qmake\
#           -DUSE_VolviewPlugIns:BOOL=OFF"

ATLASWERKS_OPTS="\
          -DUSE_CUDA:BOOL=$USECUDA\
          -DFLTK_DIR:PATH=${FLTK_BIN_DIR}\
          -DITK_DIR:PATH=${ITK_BIN_DIR}\         
          -DVTK_DIR:PATH=${VTK_BIN_DIR}"

echo 1 ${CMAKE_BASE}

function cvs_checkout_or_update()
{
    FREPOSITORY=$1
    FMODULE=$2
    FCVSUSER=$3
    FCVSPASS=$4
    FCVSSERV=$5
    FVERSION=$6
    
    FCVSROOT=:pserver:${FCVSUSER}:${FCVSPASS}@${FCVSSERV}
    echo $FCVSROOT

    if [ $FVERSION == "HEAD" ]; then
	CVSOPTIONS="-A"
    else
	CVSOPTIONS="-r $FVERSION"
    fi

    if [ -d ${FREPOSITORY}/${FMODULE} ]; then
	pushd ${FREPOSITORY}/${FMODULE}
	cvs -z3 update -dP ${CVSOPTIONS}
        if [ $? -ne 0 ]; then
            exit $?
        fi
	popd
    else
        pushd ${FREPOSITORY}
	cvs -d${FCVSROOT} login
        if [ $? -ne 0 ]; then
            exit $?
        fi
	cvs -z3 -d${FCVSROOT} co ${CVSOPTIONS} ${FMODULE}
        if [ $? -ne 0 ]; then
            exit $?
        fi
        popd
    fi
    
}

function svn_checkout_or_update()
{
    LOCALDIR=$1
    REPOSURL=$2
    SVNUSER=$3
    SVNPASS=$4
    REV=$5

    if [ ! $REV ]; then
	REV="HEAD"
    fi

    if [ -d $1  ]; then
	pushd $1
	svn update -r $REV
	popd
    else
	USERSWITCH=""
	if [ $SVNUSER ]; then
	    USERSWITCH="--username $SVNUSER"
	fi
	PASSSWITCH=""
	if [ $SVNPASS ]; then
	    PASSSWITCH="--password $SVNPASS"
	fi
	svn co $USERSWITCH $PASSSWITCH -r $REV $REPOSURL $LOCALDIR
    fi
}

function git_checkout_or_update()
{
    LOCALDIR=$1
    REPOSURL=$2
    TAG=$3

    if [ -d $1  ]; then
	pushd $1
	git pull .
	popd
    else
	git clone $REPOSURL $LOCALDIR
	pushd $1
	if [ -n $TAG ]; then
	    git checkout $TAG
	fi
	popd
    fi
}

# # Checkout or update appropriate directories
# 
# ITK
#                      Root        Module               CVSROOT                                                             Version
git_checkout_or_update "$REPOSITORY/ITK" git://itk.org/ITK.git ${ITK_VERSION}
git_checkout_or_update "$REPOSITORY/VTK" git://vtk.org/VTK.git ${VTK_VERSION}
#git_checkout_or_update "$REPOSITORY/VTKData" git://vtk.org/VTKData.git ${VTKDATA_VERSION}
#git_checkout_or_update "$REPOSITORY/ITKApps" git://itk.org/ITKApps.git ${ITKAPP_VERSION}
svn_checkout_or_update $REPOSITORY/AtlasWerks https://gforge.sci.utah.edu/svn/atlaswerks/trunk componc-guest componc



# ==================================================
# Configure and build cmake if cmake 2.6
# not the system cmake
# ==================================================

if [ $SYSTEM_CMAKE_26 -eq 1 ]; then
  if [ ! -e ${REPOSITORY}/${CMAKE_SOURCE_DIR}/bin/cmake ]; then
    echo "Cmake 2.6 not found installing"
    pushd $REPOSITORY
      wget -nc ${CMAKE_SOURCE_URL}
      tar zvxf  ${CMAKE_SOURCE}
      CMAKE_DIR=`basename ${CMAKE_SOURCE} .tar.gz`
      pushd ${CMAKE_DIR}
        sh ./bootstrap --parallel=${NPROCS} --prefix=${REPOSITORY} && make && make install
      popd
    popd
  fi
fi


# ===================================================
# Configure and build bin directories
# ===================================================

function configure_and_build()
{

    BIN_DIR=$1
    SRC_DIR=$2
    OPTS=$3
 
    if [ ! -f ${BIN_DIR}/CMakeCache.txt ]; then
	mkdir -p ${BIN_DIR}

	pushd ${BIN_DIR}

	${CMAKE_BASE} ${OPTS} ${SRC_DIR}
        if [ $? -ne 0 ]; then
            exit $?
        fi
	popd
    fi
    if [ `basename ${SRC_DIR}` == "InsightApplications" ]; then
        pushd ${BIN_DIR}/Auxiliary
    else
        pushd ${BIN_DIR}
    fi
    make -j${NPROCS}
    if [ $? -ne 0 ]; then
        exit $?
    fi
    make install
    popd 

}


#                     Binary Directory    Source Directory                   Options          
echo building ITK
configure_and_build   ${ITK_BIN_DIR}      ${REPOSITORY}/ITK              "${ITK_OPTS}"   
echo building VTK
configure_and_build   ${VTK_BIN_DIR}      ${REPOSITORY}/VTK                  "${VTK_OPTS}"
#echo building ITKapp
#configure_and_build   ${ITKAPP_BIN_DIR}   ${REPOSITORY}/ITKApps  "${ITKAPP_OPTS}"
echo building CompOnc
configure_and_build   ${ATLASWERKS_BIN_DIR}   ${ATLASWERKS_SRC_DIR}              "${ATLASWERKS_OPTS}" 
