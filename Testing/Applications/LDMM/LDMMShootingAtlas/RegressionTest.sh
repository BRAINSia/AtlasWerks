#!/bin/bash

# Run the LDMM regression tests.  Takes the 'bin' directory
# containing the executables and the 'Testing' directory as inputs

BIN_DIR="$1"
TESTING_DIR="$2"

shift 2

TESTING_UTILS_DIR="$TESTING_DIR/TestUtilities"
TEST_DIR="$TESTING_DIR/Applications/LDMM/LDMMShootingAtlas"
TMP_DIR="$TEST_DIR/RegressionTemp"

# Source in some functions
source $TESTING_UTILS_DIR/TestFuncs.sh

# Set the PYTHONPATH to find AtlasWerksTestUtils
if [ -z "$PYTHONPATH" ]; then
    export PYTHONPATH="$TESTING_UTILS_DIR"
else
    export PYTHONPATH="$PYTHONPATH:$TESTING_UTILS_DIR"
fi

# Set path so unu can be found (needed for automatic builds)
export "PATH=$PATH:$BIN_DIR:/usr/local/bin"

echo "RUN_ATLAS=${RUN_ATLAS}"

if [ -z "$RUN_ATLAS" ]; then
    RUN_ATLAS="True"
fi

if [ $RUN_ATLAS = "True" ]; then
    
    _make_and_push_dir $TMP_DIR

    PARAMFILE="$TEST_DIR/LDMMAtlasParams.xml"
    echo "doing warp..."
    _execute_save_output "$BIN_DIR/LDMMAtlas -f $PARAMFILE --ShootingOptimization true $*" LDMMShootingAtlas.out
    echo "done."
    
    cat LDMMShootingAtlas.out | sed -n 's/.*Scale \(.\)\/. Iter \(.*\) energy: Energy = \(.*\) (total) = \(.*\) (image) + \(.*\) (vec.*/\1 \2 \3 \4 \5/p' > LDMMShootingAtlas.dat
    
    popd
    
fi

pushd $TEST_DIR

_gen_plot "$TMP_DIR/LDMMShootingAtlas.dat" "AtlasEnergy.png" "LDMM Atlas Energy"

if [ ! -d images ]; then
    mkdir images
fi

python GenHTML.py > index.html
_passtest $? "Tests failed"

popd
