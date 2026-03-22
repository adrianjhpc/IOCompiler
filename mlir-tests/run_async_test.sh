#!/bin/bash
set -e
CLANG=$1
IO_OPT=$2
TEST_FILE=$3

$CLANG -fclangir -emit-cir $TEST_FILE -o async_test_cir.mlir
$IO_OPT async_test_cir.mlir --allow-unregistered-dialect --io-async-promotion -o async_test_opt.mlir

# Count occurrences of our async instructions in the entire file
SUBMIT_COUNT=$(grep -c "io.submit" async_test_opt.mlir || true)
WAIT_COUNT=$(grep -c "io.wait" async_test_opt.mlir || true)

if [ "$SUBMIT_COUNT" -eq 1 ] && [ "$WAIT_COUNT" -eq 1 ]; then
    echo -e "[PASS] async_happy successfully split into submit/wait, and async_hazard was bypassed!"
else
    echo -e "[FAIL] Expected exactly 1 io.submit and 1 io.wait, found $SUBMIT_COUNT submit and $WAIT_COUNT wait. Dumping MLIR:"
    cat async_test_opt.mlir
    exit 1
fi

rm -f async_test_cir.mlir async_test_opt.mlir
