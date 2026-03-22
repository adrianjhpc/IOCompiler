#!/bin/bash
set -e
CLANG=$1
IO_OPT=$2
TEST_FILE=$3

$CLANG -fclangir -emit-cir $TEST_FILE -o mmap_test_cir.mlir
$IO_OPT mmap_test_cir.mlir --allow-unregistered-dialect --io-mmap-promotion -o mmap_test_opt.mlir

# Count occurrences of io.mmap in the entire file
MMAP_COUNT=$(grep -c "io.mmap" mmap_test_opt.mlir || true)

if [ "$MMAP_COUNT" -eq 1 ]; then
    echo -e "[PASS] mmap_happy successfully promoted, and mmap_mismatch was bypassed!"
else
    echo -e "[FAIL] Expected exactly 1 io.mmap, found $MMAP_COUNT. Dumping MLIR:"
    cat mmap_test_opt.mlir
    exit 1
fi
rm -f mmap_test_cir.mlir mmap_test_opt.mlir
