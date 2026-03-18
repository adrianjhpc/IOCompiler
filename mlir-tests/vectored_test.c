#include <unistd.h>

extern void dummy_hazard(void);

void serialize_header(int fd) {
    char header[10];
    char meta[10];
    char body[10];
    char footer[10];
    //char header[10] = "HEADER";
    //char meta[10]   = "META";
    //char body[10]   = "BODY";
    //char footer[10] = "FOOTER";

    // These three should become ioopt_writev_3
    write(fd, header, 10);
    write(fd, meta, 10);
    write(fd, body, 10);

    dummy_hazard(); // Breaks the batch!
    
    // Left alone
    write(fd, footer, 10); 
}
