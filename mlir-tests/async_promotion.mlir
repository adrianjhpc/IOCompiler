// RUN: io-opt -io-async-promotion %s | %FileCheck %s

// Declare the external functions we expect to see
func.func private @read(i32, !llvm.ptr, i64) -> i32
func.func private @io_submit(i32, !llvm.ptr, i64) -> i32
func.func private @io_wait(i32) -> i32
func.func private @opaque_side_effect() -> ()

// ============================================================================
// TEST 1: The Happy Path (Successful Promotion)
// ============================================================================
// Expectation: The pass should split the read into submit/wait because there
// is an independent arithmetic operation between the read and the buffer usage.
// CHECK-LABEL: func @test_async_promotion
func.func @test_async_promotion(%fd: i32, %buf: !llvm.ptr, %size: i64, %other_val: i32) -> i32 {
    // CHECK-NEXT: %[[TOKEN:.*]] = call @io_submit(%arg0, %arg1, %arg2)
    // CHECK-NOT: call @read
    %bytes_read = func.call @read(%fd, %buf, %size) : (i32, !llvm.ptr, i64) -> i32

    // This is the independent compute! It should stay BETWEEN submit and wait.
    // CHECK-NEXT: %[[ADD:.*]] = arith.addi %arg3, %arg3
    %add = arith.addi %other_val, %other_val : i32

    // The hazard (we need the bytes_read result). Wait must be inserted right here.
    // CHECK-NEXT: %[[BYTES:.*]] = call @io_wait(%[[TOKEN]])
    // CHECK-NEXT: %[[RET:.*]] = arith.addi %[[BYTES]], %[[ADD]]
    %ret = arith.addi %bytes_read, %add : i32
    
    // CHECK-NEXT: return %[[RET]]
    func.return %ret : i32
}

// ============================================================================
// TEST 2: No Independent Compute (No Promotion)
// ============================================================================
// Expectation: The read is immediately followed by a dependent operation.
// The pass should realize `independentComputeCount == 0` and leave it alone.
// CHECK-LABEL: func @test_no_promotion
func.func @test_no_promotion(%fd: i32, %buf: !llvm.ptr, %size: i64) -> i32 {
    // CHECK-NEXT: %[[BYTES:.*]] = call @read
    // CHECK-NOT: call @io_submit
    %bytes_read = func.call @read(%fd, %buf, %size) : (i32, !llvm.ptr, i64) -> i32

    // Immediate dependency on the result.
    // CHECK-NEXT: %[[RET:.*]] = arith.addi %[[BYTES]], %[[BYTES]]
    %ret = arith.addi %bytes_read, %bytes_read : i32
    
    func.return %ret : i32
}

// ============================================================================
// TEST 3: The Opaque Hazard (Safe Fallback)
// ============================================================================
// Expectation: There is an independent compute instruction, but it is followed
// by an opaque function call that MIGHT do hidden I/O. The pass must insert
// the wait() BEFORE the opaque call to maintain strict safety.
// CHECK-LABEL: func @test_hazard_blocks_promotion
func.func @test_hazard_blocks_promotion(%fd: i32, %buf: !llvm.ptr, %size: i64, %val: i32) -> i32 {
    // CHECK-NEXT: %[[TOKEN:.*]] = call @io_submit
    %bytes_read = func.call @read(%fd, %buf, %size) : (i32, !llvm.ptr, i64) -> i32

    // Independent compute (safe to float)
    // CHECK-NEXT: arith.addi %arg3, %arg3
    %add = arith.addi %val, %val : i32

    // HAZARD! Opaque side effect. Wait must be forced here!
    // CHECK-NEXT: %[[BYTES:.*]] = call @io_wait(%[[TOKEN]])
    // CHECK-NEXT: call @opaque_side_effect()
    func.call @opaque_side_effect() : () -> ()

    // CHECK-NEXT: return %[[BYTES]]
    func.return %bytes_read : i32
}

// ============================================================================
// TEST 4: The Terminator Barrier
// ============================================================================
// Expectation: The pass must not let the pending async request "leak" out of 
// the basic block. It must insert the wait immediately before the return.
// CHECK-LABEL: func @test_terminator_barrier
func.func @test_terminator_barrier(%fd: i32, %buf: !llvm.ptr, %size: i64, %val: i32) -> i32 {
    // CHECK-NEXT: %[[TOKEN:.*]] = call @io_submit
    %bytes_read = func.call @read(%fd, %buf, %size) : (i32, !llvm.ptr, i64) -> i32

    // Independent compute
    // CHECK-NEXT: arith.addi %arg3, %arg3
    %add = arith.addi %val, %val : i32

    // HAZARD! Terminator. Wait must be forced here!
    // CHECK-NEXT: %[[BYTES:.*]] = call @io_wait(%[[TOKEN]])
    // CHECK-NEXT: return %[[BYTES]]
    func.return %bytes_read : i32
}
