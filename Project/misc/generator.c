#define _GNU_SOURCE
#define _XOPEN_SOURCE
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <crypt.h>
#include <math.h>

const char * names[] = {"Alice", "Bob", "Ted"};
const size_t NAMES_ARRAY_LENGTH = 3;
const size_t NAME_LENGTH = 8;
const size_t PASSWORD_LENGTH = 8;

int randomM(int max) {
	return rand() % max;
}

void randomName(char * name) {
	const char * nameBase = names[randomM(NAMES_ARRAY_LENGTH)];
	const int length = strlen(nameBase);
	const int numLength = NAME_LENGTH - length;
	const int max = (int)pow(10, numLength);
	sprintf(name, "%s%d", nameBase, randomM(max));
}

void randomPassword(char * password, size_t size) {
	for (size_t i = 0; i < size; ++i) {
		password[i] = randomM(26) + 'a';
	}
}

int main(int argc, char * argv[]) {
	if (argc <= 1 || argc > 4) {
		fprintf(stderr, "Usage: ./generate 40 [outputFile] [salt]\n");
		exit(EXIT_FAILURE);
	}
	FILE * outputFile = stdout;
	char * salt = "AM";
	if (argc >= 3) {
		outputFile = fopen(argv[2], "w");
	}
	if (argc == 4) {
		salt = argv[3];
	}

	long toGenerate = strtol(argv[1], NULL, 10);
	for (long i = 0; i < toGenerate; ++i) {
		char name[NAME_LENGTH + 1];
		char password[PASSWORD_LENGTH + 1];
		int randomOffset = randomM(PASSWORD_LENGTH - 1);
		memset(name, 0, sizeof(name));
		memset(password, 0, sizeof(password));
		randomName(name);
		randomPassword(password, PASSWORD_LENGTH);
		char * hash = crypt(password, salt);
		memset(password + randomOffset, '.', PASSWORD_LENGTH - randomOffset);
		fprintf(outputFile, "%s %s %s\n", name, hash, password);
	}
	fclose(outputFile);
}
