#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/epoll.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdio.h>
#include <signal.h>
#include <unistd.h>
#include <netdb.h>
#include <errno.h>

#ifndef DIS_SERVER
#define DIS_SERVER
void initServer();
void reduceResult(const char * clientResult, size_t clientResultLen);
ssize_t getNextInput(char ** output);
void closeServer();
#endif
