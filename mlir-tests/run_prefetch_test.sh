#!/bin/bash
set -e
CLANG=$1
IO_OPT=$2
TEST_FILE=$3

$CLANG -fclangir -emit-cir $TEST_FILE -o prefetch_test_cir.mlir
$IO_OPT prefetch_test_cir.mlir --allow-unregistered-dialect --io-prefetch-injection -o prefetch_test_opt.mlir

# Count occurrences of io.prefetch in the entire file
PREFETCH_COUNT=$(grep -c "io.prefetch" prefetch_test_opt.mlir || true)

if [ "$PREFETCH_COUNT" -eq 1 ]; then
    echo -e "[PASS] prefetch_happy successfully injected read-ahead hints and fast loop was bypassed!"
else
    echo -e "[FAIL] Expected exactly 1 io.prefetch, found $PREFETCH_COUNT. Dumping MLIR:"
    cat prefetch_test_opt.mlir
    exit 1
fi
rm -f prefetch_test_cir.mlir prefetch_test_opt.mlir
