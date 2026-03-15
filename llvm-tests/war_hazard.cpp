// RUN: %clang -O2 -emit-llvm -S %s -o - | %opt -load-pass-plugin=%shlibdir/IOOpt%shlibext -passes=io-opt -S | %FileCheck %s

#include <unistd.h>

// Use a wildcard to match the mangled C++ name: _Z15test_war_hazardiPc
// CHECK-LABEL: define {{.*}}@{{.*}}test_war_hazard
void test_war_hazard(int fd, char* buf) {
    // The pass MUST NOT merge these. We check they appear in the original order.

    // The first read
    // CHECK: {{.*}}call {{.*}} @read(i32 {{.*}}, ptr {{.*}}, i64 {{.*}}10)
    //
    // The memory hazard (the store)
    // CHECK: store i8 88
    
    // The second read must remain afterthe store
    // CHECK: {{.*}}call {{.*}} @read(i32 {{.*}}, ptr {{.*}}, i64 {{.*}}10)

    read(fd, buf, 10);
    buf[15] = 'X';
    read(fd, buf + 10, 10);
} 
