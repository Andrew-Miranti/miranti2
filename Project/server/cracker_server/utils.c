#include "utils.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

ssize_t readFromFd(int fd, char ** outputBuffer) {
	size_t outputLength = 0;
	size_t requestLength = REQUEST_LENGTH;
    char * output = calloc(sizeof(char), requestLength);
    char * readBuffer = calloc(sizeof(char), requestLength);
    int done = 0;
    while (!done) {
        memset(readBuffer, 0, requestLength);
        ssize_t charactersRead = read(fd, readBuffer, requestLength);
        size_t oldOutputLength = outputLength;
        outputLength += charactersRead;
        if (charactersRead < 0) {
			perror("");
            done = 1;
        } else {
        	output = realloc(output, outputLength + 1);
        	memcpy(output+oldOutputLength, readBuffer, charactersRead);
		}
    }
	if (outputLength > 0) {
		output[outputLength] = '\0';
    	*outputBuffer = output;
	} else {
		return -1;
	}
    free(readBuffer);
    return outputLength;
}
