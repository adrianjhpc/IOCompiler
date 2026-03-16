#include <unistd.h>
#include <fcntl.h>
#include <cstring>
#include <iostream>

// Defined in end_to_end_lib.cpp to test LTO boundary merging
extern void write_footer(int fd);

int main() {
    int fd = open("output_test.txt", O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) return 1;

    char buffer[] = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";

    // MLIR Test: Strided Loop Batching
    // The MLIR pass should convert this into a single writev() call
    #pragma clang loop unroll(disable) vectorize(disable)
    for (int i = 0; i < 10; i += 2) {
        write(fd, &buffer[i], 1);
    }

    // LTO Test: Cross-Module I/O
    // The LLVM LTO pass should inline and merge this cross-module write
    write_footer(fd);

    close(fd);
    return 0;
}
