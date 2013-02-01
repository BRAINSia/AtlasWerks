#!/bin/bash

# Run the LDMM regression tests.  Takes the 'bin' directory
# containing the executables and the 'Testing' directory as inputs

BIN_DIR="$1"
TESTING_DIR="$2"

shift 2

TESTING_UTILS_DIR="$TESTING_DIR/TestUtilities"
TEST_DIR="$TESTING_DIR/Applications/LDMM/LDMMVelShootingWarp"
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
    PARAMFILE="$TEST_DIR/LDMMVelShootingWarpParams.xml"
    echo "doing warp..."
    _execute_save_output "$BIN_DIR/LDMMWarp -f $PARAMFILE --ShootingOptimization true --UseVelocityShootingOptimization true $*" LDMMVelShootingWarp.out
    echo "done."
    
    #
    # Extract the energy from the output
    #
    cat LDMMVelShootingWarp.out | sed -n 's/.*Scale \(.\)\/. Iter \(.*\) energy: Energy = \(.*\) (total) = \(.*\) (image) + \(.*\) (vec.*/\1 \2 \3 \4 \5/p' > LDMMVelShootingWarp.dat
    
    popd

fi

pushd $TEST_DIR

#
# Generate the energy plot
#
_gen_plot "$TMP_DIR/LDMMVelShootingWarp.dat" "WarpEnergy.png" "LDMM Vel Shooting Warp Energy"
    
if [ ! -d images ]; then
    mkdir images
fi

python GenHTML.py > index.html
_passtest $? "Tests failed"

popd