nclude <stdio.h>

// CHECK-LABEL: @test_hoist
int test_hoist(FILE *fp, int x, int y) {
    char buf[20];
    
    // The CPU bound math operations
    int z = x * y + 42;
    int w = z / 2;
    
    // In the C code, fread happens AFTER the math.
    // But our pass should hoist it to happen BEFORE the math in the IR.
    
    // CHECK: call i64 @fread
    // CHECK: mul nsw i32
    // CHECK: sdiv i32
    fread(buf, 1, 20, fp);
    
    return w + buf[0];
}
