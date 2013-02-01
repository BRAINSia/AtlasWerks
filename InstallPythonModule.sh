#!/bin/sh
BUILD_DIR=$1
INSTALLATION_LOCATION=$2

BIN_DIR=${BUILD_DIR}/bin
PYTHONDIR="${BUILD_DIR}/python"
if [ ! -d $PYTHONDIR ]; then
    mkdir $PYTHONDIR
fi

MODULEDIR="${PYTHONDIR}/AtlasWerks"
if [ ! -d $MODULEDIR ]; then
    mkdir $MODULEDIR
fi

#
# Copy in the python files
#
find ${BUILD_DIR} -not -path "*/python/*" -name "*.py" -exec cp {} $MODULEDIR \;

#
# Copy in the libraries
#
cp $BIN_DIR/_*.so $MODULEDIR

#
# create __init__.py
#
INITFILE="$MODULEDIR/__init__.py"
if [ ! -d $INITFILE ]; then
    rm $INITFILE
fi
MODULES=(`ls $MODULEDIR/*.py | xargs -L 1 basename | sed 's/\(.*\).py/\1/' | tr '\n' ' '`)
NMODULES=${#MODULES[@]}
ALLSTRING="__all__=["
for MODULEIDX in $(seq 0 $(($NMODULES-1)))
do
    ALLSTRING="${ALLSTRING} \"${MODULES[$MODULEIDX]}\""
    if [ $MODULEIDX -lt $(($NMODULES-1)) ]; then
	ALLSTRING="${ALLSTRING},"
    fi
done
ALLSTRING="${ALLSTRING}]"
echo $ALLSTRING > $INITFILE
