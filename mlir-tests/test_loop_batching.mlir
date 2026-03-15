// RUN: io-opt %s --io-loop-batching | %FileCheck %s

// ============================================================================
// TEST 1: The Contiguous Fast-Path
// ============================================================================
// CHECK-LABEL: func.func @test_contiguous_write
func.func @test_contiguous_write(%fd: i32, %base_ptr: !llvm.ptr) {
    %c0 = arith.constant 0 : index
    %c100 = arith.constant 100 : index
    %step = arith.constant 1 : index
    %write_size = arith.constant 1 : i64

    // Ensure the original scf.for loop is completely erased
    // CHECK-NOT: scf.for

    // Ensure we calculate the trip count and total size mathematically
    // CHECK: %[[TRIP_COUNT:.*]] = arith.divsi
    // CHECK: %[[TOTAL_SIZE:.*]] = arith.muli %[[TRIP_COUNT]], %{{.*}} : i64

    // Ensure we emit the massive batched write using the base pointer
    // CHECK: io.batch_write(%arg0, %arg1, %[[TOTAL_SIZE]]) : i32, !llvm.ptr, i64 -> i64
    
    scf.for %iv = %c0 to %c100 step %step {
        // Because step (1) == write_size (1), our verifySCEVOffset function 
        // will correctly flag this as perfectly contiguous memory!
        %ptr = llvm.getelementptr %base_ptr[%iv] : (!llvm.ptr, index) -> !llvm.ptr, i8
        %res = io.write(%fd, %ptr, %write_size) : i32, !llvm.ptr, i64 -> i64
    }

    return
}

// ============================================================================
// TEST 2: The Strided Vector Fallback
// ============================================================================
// CHECK-LABEL: func.func @test_strided_write
func.func @test_strided_write(%fd: i32, %base_ptr: !llvm.ptr) {
    %c0 = arith.constant 0 : index
    %c100 = arith.constant 100 : index
    %step = arith.constant 2 : index   // <-- WARNING: STEP IS 2
    %write_size = arith.constant 1 : i64

    // Ensure we allocate the tracking arrays
    // CHECK: memref.alloca
    // CHECK: memref.alloca

    // Ensure we generate a new loop just to calculate the addresses
    // CHECK: scf.for
    // CHECK: llvm.getelementptr
    // CHECK: memref.store
    // CHECK: memref.store

    // Ensure the io.write inside the loop is gone...
    // CHECK-NOT: io.write

    // ...and replaced with our Scatter/Gather I/O operation outside the loop
    // CHECK: io.batch_writev(%arg0, %{{.*}}, %{{.*}}, %{{.*}})
    
    scf.for %iv = %c0 to %c100 step %step {
        // Because step (2) != write_size (1), verifySCEVOffset will abort the 
        // contiguous path and trigger our Vector Fallback
        %ptr = llvm.getelementptr %base_ptr[%iv] : (!llvm.ptr, index) -> !llvm.ptr, i8
        %res = io.write(%fd, %ptr, %
