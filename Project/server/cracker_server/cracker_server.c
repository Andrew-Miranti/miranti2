#define _GNU_SOURCE
#include "constants.h"
#include "server.h"
#include <stdio.h>

FILE * inputFile;

void initServer() {
	inputFile = fopen("in.txt", "r");
}

void reduceResult(const char * clientResult, size_t resultLength) {
	printf("Password cracked: %s", clientResult);
}

ssize_t getNextInput(char ** output) {
	char * lineptr = NULL;
	size_t len = 0;
	ssize_t result = getline(&lineptr, &len, inputFile);
	*output = lineptr;
	return result;
}

void closeServer() {
	fclose(inputFile);
	inputFile = NULL;
}
