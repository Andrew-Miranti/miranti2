#include "distributed.h"

const size_t MIN_LENGTH = 4096;
int serverFd = -1;

int connectToServer(const char * serverIP, const char * port) {
    serverFd = socket(AF_INET, SOCK_STREAM, 0);
    struct addrinfo hints, * result;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
	getaddrinfo(serverIP, port, &hints, &result);
    connect(serverFd, result->ai_addr, result->ai_addrlen);
    freeaddrinfo(result);
    return serverFd;
}

void sendToServer(const void * data, const size_t count) {
    write(serverFd, data, count);
}

size_t requestFromServer(const char * request, const size_t requestLen, const size_t hintLength, char ** outputBuffer) {
    sendToServer(request, requestLen);
    size_t requestLength = hintLength > MIN_LENGTH ? hintLength : MIN_LENGTH;
    size_t outputLength = 0;
    char * output = calloc(sizeof(char), requestLength);
    char * readBuffer = calloc(sizeof(char), requestLength);
    int done = 0;
    while (!done) {
        memset(readBuffer, 0, requestLength);
        size_t charactersRead = read(serverFd, readBuffer, requestLength);
        size_t oldOutputLength = outputLength;
        outputLength += charactersRead;
        if (charactersRead == requestLength) {
            requestLength *= 2;
            readBuffer = realloc(readBuffer, requestLength);
        } else {
            done = 1;
        }
        output = realloc(output, outputLength);
        memcpy(output+oldOutputLength, readBuffer, charactersRead);
    }
    *outputBuffer = output;
    free(readBuffer);
    return outputLength;
}

void disconnectFromServer() {
    close(serverFd);
    serverFd = -1;
}
