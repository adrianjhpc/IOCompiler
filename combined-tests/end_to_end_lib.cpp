#include <unistd.h>
#include <cstring>

void write_footer(int fd) {
    const char* footer = "-FOOTER";
    // If io-lto-merge works, this might should merged with the final I/O
    // operations in main.cpp during the Link phase
    write(fd, footer, strlen(footer));
}
