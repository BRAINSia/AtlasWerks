#!/bin/sh

# Run the Metamorphosis regression tests.  Takes the 'bin' directory
# containing the executables and the 'Testing' directory as inputs

BIN_DIR="$1"
TESTING_DIR="$2"

TEST_DIR="$TESTING_DIR/Applications/Greedy/fWarp"
TMP_DIR="$TEST_DIR/RegressionTemp"
TEST_DATA_DIR="$TEST_DIR/Baseline"

EXIT_PASS=0
EXIT_FAIL=1

function passtest {
    if [ $1 -ne 0 ]; then
	echo "failing: $2"
	exit $EXIT_FAIL
    fi
}

function _execute {
    echo "running execute"
    echo "executing: $*"
    $*
}

#### Run MetamorphosisAverageMultiscale Tests

mkdir $TMP_DIR
pushd $TMP_DIR
_execute $BIN_DIR/fWarp `cat ../fWarpParams.txt`
popd

passtest $? "Problem running fWarp"
$BIN_DIR/TestFuncs -imgSqrDiff "$TMP_DIR/deformedImage_1.mhd" "$TEST_DATA_DIR/deformedImage_1.mhd"
passtest $? "fWarp mean image intensities different"
$BIN_DIR/TestFuncs -defSqrDiff "$TMP_DIR/deformationField_.mhd" "$TEST_DATA_DIR/deformationField_.mhd"
passtest $? "fWarp deformation field different"

#### End Tests

rm -rf $TMP_DIR
echo "Tests Passed."
exit $EXIT_PASS
