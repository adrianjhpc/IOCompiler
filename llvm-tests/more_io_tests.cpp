// RUN: %ppclang -O2 -fno-inline -emit-llvm -S -c %s -o - | %opt -load-pass-plugin=%shlibdir/IOOpt%shlibext -passes=io-opt -S | %FileCheck %s

#include <unistd.h>
#include <sys/uio.h>

extern void opaque_side_effect();

// ============================================================================
// TEST 1: Contiguous Write Fast-Path
// ============================================================================
// CHECK-LABEL: define dso_local void @_Z21test_contiguous_writeiPc(
void test_contiguous_write(int fd, char* buffer) {
    // Prevent Clang from unrolling the loop before our pass gets to see it
    #pragma clang loop unroll(disable) vectorize(disable)
    for (int i = 0; i < 100; i++) {
        write(fd, &buffer[i], 1);
    }
    // CHECK-NOT: call i64 @write(i32{{.*}}, ptr{{.*}}, i64{{.*}}1)
    // CHECK: call i64 @write(i32{{.*}}, ptr{{.*}}, i64{{.*}}100)
}

// ============================================================================
// TEST 2: Strided Write (LLVM pass should safely ignore this)
// ============================================================================
// CHECK-LABEL: define dso_local void @_Z18test_strided_writeiPc(
void test_strided_write(int fd, char* buffer) {
    #pragma clang loop unroll(disable) vectorize(disable)
    for (int i = 0; i < 100; i += 2) {
        write(fd, &buffer[i], 1);
    }
    // The LLVM pass doesn't do writev, so it should leave the size-1 write alone!
    // CHECK: call i64 @write(i32{{.*}}, ptr{{.*}}, i64{{.*}}1)
    // CHECK-NOT: call i64 @writev
}

// ============================================================================
// TEST 3: Contiguous Read Fast-Path
// ============================================================================
// CHECK-LABEL: define dso_local void @_Z20test_contiguous_readiPc(
void test_contiguous_read(int fd, char* buffer) {
    #pragma clang loop unroll(disable) vectorize(disable)
    for (int i = 0; i < 100; i++) {
        read(fd, &buffer[i], 1);
    }
    // CHECK-NOT: call i64 @read(i32{{.*}}, ptr{{.*}}, i64{{.*}}1)
    // CHECK: call i64 @read(i32{{.*}}, ptr{{.*}}, i64{{.*}}100)
}

// ============================================================================
// TEST 4: Strided Read (LLVM pass should safely ignore this)
// ============================================================================
// CHECK-LABEL: define dso_local void @_Z17test_strided_readiPc(
void test_strided_read(int fd, char* buffer) {
    #pragma clang loop unroll(disable) vectorize(disable)
    for (int i = 0; i < 100; i += 2) {
        read(fd, &buffer[i], 1);
    }
    // The LLVM pass doesn't do readv, so it should leave the size-1 read alone!
    // CHECK: call i64 @read(i32{{.*}}, ptr{{.*}}, i64{{.*}}1)
    // CHECK-NOT: call i64 @readv
}

// ============================================================================
// TEST 5: The Hazard Bailout
// ============================================================================
// CHECK-LABEL: define dso_local void @_Z11test_hazardiPc(
void test_hazard(int fd, char* buffer) {
    #pragma clang loop unroll(disable) vectorize(disable)
    for (int i = 0; i < 100; i++) {
        // Because of the opaque function below, the compiler MUST leave this alone.
        // CHECK: call i64 @write(i32{{.*}}, ptr{{.*}}, i64{{.*}}1)
        write(fd, &buffer[i], 1);

        // CHECK: call void @_Z18opaque_side_effectv()
        opaque_side_effect();
    }
}
