#define _GNU_SOURCE
#include <string.h>
#include <unistd.h>

int connectToServer(const char * serverIP, const char * port); 
void sendToServer(const void * data, const size_t size);
size_t requestFromServer(const char * request, const size_t requestLen, const size_t hintLength, char ** outputBuffer);
int disconnectFromServer(); 
