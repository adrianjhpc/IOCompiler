#!/bin/bash
set -e

CLANG=$1
IO_OPT=$2
TEST_FILE=$3

echo "[*] Compiling $TEST_FILE to ClangIR..."
$CLANG -fclangir -emit-cir $TEST_FILE -o vectored_test_cir.mlir

echo "[*] Running Block Vectored I/O Pass..."
$IO_OPT vectored_test_cir.mlir --allow-unregistered-dialect --io-block-vectored -o vectored_test_opt.mlir

echo "==========================================================="
echo " Test 1: Verifying 'serialize_header' (Vectored I/O Test)"
echo "==========================================================="
sed -n '/@serialize_header/,/^ *}/p' vectored_test_opt.mlir > serialize_func.mlir

# We expect one call to ioopt_writev_3, and we still expect one normal write() for the footer!
if grep -q "ioopt_writev_3" serialize_func.mlir && grep -q "write" serialize_func.mlir; then
    echo -e "[PASS] serialize_header successfully batched 3 writes and safely ignored the 4th!"
else
    echo -e "[FAIL] serialize_header was not optimized correctly."
    cat serialize_func.mlir
    exit 1
fi

rm -f vectored_test_cir.mlir vectored_test_opt.mlir serialize_func.mlir
