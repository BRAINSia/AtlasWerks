#!/bin/bash

# Run the Metamorphosis regression tests.  Takes the 'bin' directory
# containing the executables and the 'Testing' directory as inputs

BIN_DIR="$1"
TESTING_DIR="$2"

TESTING_UTILS_DIR="$TESTING_DIR/TestUtilities"
TEST_DIR="$TESTING_DIR/Applications/Metamorphosis/MetamorphosisAverageMultiscale"
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

    PARAMFILE="$TEST_DIR/MetamorphosisAverageMultiscaleParams.xml"
    echo "doing warp..."
    _execute_save_output "$BIN_DIR/MetamorphosisAverageMultiscale -f $PARAMFILE --nThreads 4" MetamorphosisAverageMultiscale.out
    echo "done."
    
    cat MetamorphosisAverageMultiscale.out | sed -n 's/Scale \(.*\) Iter \(.*\) energy = \(.*\) = \(.*\) (image) + \(.*\) (vec.*/\1 \2 \3 \4 \5/p' > MetamorphosisAverageMultiscale.dat
    
    popd
    
fi

pushd $TEST_DIR

_gen_plot "$TMP_DIR/MetamorphosisAverageMultiscale.dat" "AtlasEnergy.png" "Metamorphosis Atlas Energy"

if [ ! -d images ]; then
    mkdir images
fi

python GenHTML.py > index.html
_passtest $? "Tests failed"

popd
