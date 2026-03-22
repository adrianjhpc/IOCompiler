#include <unistd.h>

// TEST 1: The Happy Path
// A sequential loop with a read and heavy compute.
// EXPECTATION: io.prefetch should be injected before the loop's read.
void prefetch_happy(int fd, size_t size) {
    char buf[1024];
    for (int i = 0; i < 100; i++) {
        read(fd, buf, size);
        
        // Heavy Compute (>10 operations) to justify prefetching
        int a = i * 2;
        a += 1; a *= 2; a -= 3; a /= 2; a += 5;
        a += 1; a *= 2; a -= 3; a /= 2; a += 5;
        a += 1; a *= 2; a -= 3; a /= 2; a += 5;
        
        buf[0] = (char)a;
    }
}

// TEST 2: The Fast Loop Path
// A sequential loop, but almost no compute.
// EXPECTATION: Must abort. Prefetching would just add syscall overhead.
void prefetch_fast_loop(int fd, size_t size) {
    char buf[1024];
    for (int i = 0; i < 100; i++) {
        read(fd, buf, size);
        buf[0] = 'X';
    }
}
