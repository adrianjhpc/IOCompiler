#include <unistd.h>
#include <string.h>

// A helper function residing in a separate translation unit
void write_log_payload(int fd, const char* message) {
    // Write 2: The payload
    write(fd, message, strlen(message));
    // Write 3: The newline
    write(fd, "\n", 1);
}
