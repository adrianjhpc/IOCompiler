#include <unistd.h>
#include <stdlib.h>

// TEST 1: The Happy Path
// Allocating a buffer and immediately reading the exact same size into it.
// EXPECTATION: malloc and read are erased, io.mmap is inserted.
void mmap_happy(int fd, size_t size) {
    char *buf = (char*)malloc(size);
    read(fd, buf, size);
    
    // Do something with it to prevent dead-code elimination
    buf[0] = 'X';
}

// TEST 2: Size Mismatch
// Allocating one size, but reading a different size.
// EXPECTATION: Must abort. mmap requires the sizes to perfectly match for safety.
void mmap_mismatch(int fd, size_t alloc_size, size_t read_size) {
    char *buf = (char*)malloc(alloc_size);
    read(fd, buf, read_size);
    
    buf[0] = 'X';
}
