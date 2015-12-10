#define _GNU_SOURCE
#include <unistd.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <stdio.h>
#include "distributed.h"

extern char ** environ;
const char * OUTPUT_HEADER = "DIS_REDUCE ";

void runProgram(char * program, char * input, char * serverIP, char * serverPort, char * arguments[]) {
    int inputFile = open(input, O_RDONLY);
    int output[2];
    int error[2];
    pipe(output);
    pipe(error);
    pid_t childPid = fork();
    if (!childPid) {
        close(output[0]);
        close(error[0]);
        setenv("server_ip", serverIP, 0);
        dup2(inputFile,  0);
        dup2(output[1], 1);
        dup2(error[1], 2);
        execvpe(program, arguments, environ);
        perror("EXEC FAILED\n");
        exit(2);
    } else {
		connectToServer(serverIP, serverPort);	
        close(output[1]);
        close(error[1]);
        close(inputFile);
        FILE * clientOutput = fdopen(output[0], "r");
        FILE * clientError = fdopen(error[0], "r");

        char * line = NULL;
        size_t length = 0;
        ssize_t size;
		while ((size = getline(&line, &length, clientOutput)) > 0) {
            printf("Client: %s", line);
			char * result;
			asprintf(&result, "%s%s", OUTPUT_HEADER, line);
			sendToServer(line, size);
			free(result);
        }
        free(line);

        fclose(clientOutput);
        fclose(clientError);
        fflush(NULL);
    }
    waitpid(childPid, NULL, 0);
	disconnectFromServer();
}

void makeProgram(const char * target_path) {
    chdir(target_path);
    system("make");
}

int main(int argc, char * argv[]) {
    if (argc < 6) {
        fprintf(stderr, "Usage: ./client_wrapper target_path executable_path input_file_path serverip serverport arguments\n");
        exit(1);
    }

    printf("Make & Run %s in %s\n", argv[2], argv[1]);
    makeProgram(argv[1]);
    runProgram(argv[2], argv[3], argv[4], argv[5], &argv[6]);
    return 0;
}
