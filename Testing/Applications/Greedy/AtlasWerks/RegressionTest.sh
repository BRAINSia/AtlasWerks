#!/bin/bash

# Run the LDMM regression tests.  Takes the 'bin' directory
# containing the executables and the 'Testing' directory as inputs

BIN_DIR="$1"
TESTING_DIR="$2"

shift 2

TESTING_UTILS_DIR="$TESTING_DIR/TestUtilities"
TEST_DIR="$TESTING_DIR/Applications/Greedy/AtlasWerks"
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

    PARAMFILE="$TEST_DIR/AtlasWerksParams.xml"
    echo "doing warp..."
    _execute_save_output "$BIN_DIR/AtlasWerks -f $PARAMFILE $*" AtlasWerks.out
    echo "done."
    
    cat AtlasWerks.out | sed -n 's/\[\(.*\?\)\/.*RMSE=\(.*\) (\(.*\)\%)/\1 \2/p' > AtlasWerks.dat

    if [ `cat AtlasWerks.dat | wc -l` -eq 0 ]; then
	# GPU case
	cat AtlasWerks.out | sed -n 's/RMSE for iteration \(.*\) is \(.*\)/\1\/? ? \2/p' > AtlasWerks.dat
    fi
    
    popd
    
fi

pushd $TEST_DIR

PLOTFILE=plot.gnuplot
DATAFILE="$TMP_DIR/AtlasWerks.dat"
_print_gnuplot_header "AtlasEnergy.png" "Greedy Atlas Energy" > $PLOTFILE
echo "plot    \"$DATAFILE\" using 3 title 'RMSE' with linespoints" >> $PLOTFILE
echo "set out" >> $PLOTFILE
_execute gnuplot $PLOTFILE

if [ ! -d images ]; then
    mkdir images
fi

python GenHTML.py > index.html
_passtest $? "Tests failed"

popd
