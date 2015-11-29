#include "distributed.h"

const size_t MIN_LENGTH = 4096;
int serverFd = -1;

int connectToServer(const char * serverIP, const char * port) {
    serverFd = socket(AF_INET, SOCK_STREAM, 0);
    struct addrinfo hints, * results;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    connect(serverFd, result->ai_addr, result->ai_addrlen);
    freeaddrinfo(results);
    return serverFd;
}

void sendToServer(const char * data, const size_t count) {
    write(serverFd, data, count);
}

size_t requestFromServer(const char * request, const size_t requestLen, const size_t hintLength, char ** outputBuffer) {
    sendToServer(request, requestLen);
    size_t requestLength = hintLength > MIN_LENGTH ? hintLength : MIN_LENGTH;
    size_t outputLength = 0;
    char * output = malloc(sizeof(char), requestLength);
    char * readBuffer = malloc(sizeof(char), requestLength);
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
        realloc(output, outputLength);
        memcpy(output+oldOutputLength, readBuffer, charactersRead);
    }
    *outputBuffer = output;
    free(readBuffer);
    return outputLength;
}

int disconnectFromServer() {
    close(serverFd);
    serverFd = -1;
}
