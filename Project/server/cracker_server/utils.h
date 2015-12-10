#include <stdlib.h>
#include <unistd.h>

#ifndef DIS_SERVER_UTILS
#define DIS_SERVER_UTILS
#define REQUEST_LENGTH 512
ssize_t readFromFd(int fd, char ** data);
#endif
