#!/bin/bash

# Run the Metamorphosis regression tests.  Takes the 'bin' directory
# containing the executables and the 'Testing' directory as inputs

BIN_DIR="$1"
TESTING_DIR="$2"

TESTING_UTILS_DIR="$TESTING_DIR/TestUtilities"
TEST_DIR="$TESTING_DIR/Applications/Metamorphosis/MetamorphosisMultiscale"
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

# Use externally-set RUN_WARP if available, set to true if not
echo "RUN_WARP=${RUN_WARP}"
if [ -z "$RUN_WARP" ]; then
    RUN_WARP="True"
fi

if [ $RUN_WARP = "True" ]; then
    
    _make_and_push_dir $TMP_DIR

    #
    # Run warp
    #
    PARAMFILE="$TEST_DIR/MetamorphosisMultiscaleParams.xml"
    echo "doing warp..."
    _execute_save_output "$BIN_DIR/MetamorphosisMultiscale -f $PARAMFILE" MetamorphosisMultiscale.out
    echo "done."
    
    #
    # Extract the energy from the output
    #
    cat MetamorphosisMultiscale.out | sed -n 's/Scale \(.*\) Iter \(.*\) energy = \(.*\) = \(.*\) (image) + \(.*\) (vec.*/\1 \2 \3 \4 \5/p' > MetamorphosisMultiscale.dat
    
    popd

fi

#
# Generate the energy plot
#
pushd $TEST_DIR

_gen_plot "$TMP_DIR/MetamorphosisMultiscale.dat" "WarpEnergy.png" "Metamorphosis Warp Energy"
    
if [ ! -d images ]; then
    mkdir images
fi

python GenHTML.py > index.html
_passtest $? "Tests failed"

popd