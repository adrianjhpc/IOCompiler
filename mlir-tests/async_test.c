#include <unistd.h>

// TEST 1: The Happy Path
// The CPU performs independent math after the read before accessing the buffer.
// EXPECTATION: read is split into io.submit and io.wait.
void async_happy(int fd, char *buf, size_t size, int multiplier) {
    read(fd, buf, size);
    
    // INDEPENDENT COMPUTE: This should happen while the kernel fetches data!
    int compute = multiplier * 42;
    compute += 15;
    compute /= 2;
    
    // HAZARD: We finally need the data. io.wait should be inserted right above this.
    buf[0] = (char)compute;
}

// TEST 2: The Immediate Hazard Path
// The buffer is accessed immediately after the read.
// EXPECTATION: The compiler realizes splitting it is unprofitable/unsafe and leaves the read alone.
void async_hazard(int fd, char *buf, size_t size) {
    read(fd, buf, size);
    
    // IMMEDIATE HAZARD: No independent compute exists.
    buf[0] = 'X'; 
}
