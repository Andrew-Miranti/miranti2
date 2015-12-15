#define _GNU_SOURCE
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <libssh2.h>
#include <libssh2_sftp.h>
#include <netdb.h>
#include <unistd.h>
#include <fcntl.h>
#include "distributed.h"
#include "DONOTSHOW.h"

const char * serverFilePath = "/home/miranti2/client.tar.gz";
const char * clientFilePath = "./client.tar.gz";
const char * client = "./clientFolder";
const char * executable = "./client";
const char * decompressCommand = "tar -xvzf %s -C ./clientFolder";
const char * dataRequest = "DIS_GET_INPUT\n";
const size_t requestLen = 14;
const char * wrapperCommand = "./client_wrapper ./clientFolder ./client ./inputFile.txt 127.0.0.1 2000";
const char * inputFile = "./inputFile.txt";

void initLibsshOrDie() {
    int result = libssh2_init(0);
    if (result) {
        fprintf(stderr, "Libssh2 initialization failed with code %d\n", result);
        exit(EXIT_FAILURE);
    } 
}

int makeSocketOrDie() {
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) {
        perror("Failed to open socket\n");
        exit(EXIT_FAILURE);
    }
    return sock;
}

void connectOrDie(int sockFd, const struct addrinfo * addr) {
    if (connect(sockFd, addr->ai_addr, addr->ai_addrlen)) {
        perror("Connection failed");
        exit(EXIT_FAILURE);
    }
}

struct addrinfo * getServerInfo(const char * serverAddress) {
    struct addrinfo hints, *result;
    result = NULL;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
	int addrerr = getaddrinfo(serverAddress, "22", &hints, &result);
    if (addrerr) {
        fprintf(stderr, "AddrInfo failed with %s\n", gai_strerror(addrerr));
        exit(EXIT_FAILURE);
    }
    return result;
}

LIBSSH2_SESSION * makeSession() {
    LIBSSH2_SESSION * session = libssh2_session_init();
    if (!session) {
        fprintf(stderr, "Failed to initiate libssh2 session\n");
        exit(EXIT_FAILURE);
    } 
    return session;
}

LIBSSH2_SFTP * makeSFTPSession(LIBSSH2_SESSION * session) {
    LIBSSH2_SFTP * result = libssh2_sftp_init(session);
    if (!result) {
        fprintf(stderr, "Failed to initiate sftp session\n");
		char * errorMessage;
		libssh2_session_last_error(session, &errorMessage, NULL, 0);
		fprintf(stderr, "Error %s\n", errorMessage);
        exit(EXIT_FAILURE);
    }
    return result;
}

void printSftpErrorAndDie(LIBSSH2_SESSION * session, LIBSSH2_SFTP * sftpSession) {
	char * errorMessage;
	libssh2_session_last_error(session, &errorMessage, NULL, 0);
	fprintf(stderr, "Error %s\n", errorMessage);
	long lastErr = libssh2_sftp_last_error(sftpSession);
	fprintf(stderr, "Error code: %ld\n", lastErr);
	fprintf(stderr, "No such file? %d\n", lastErr == LIBSSH2_FX_NO_SUCH_FILE);
    exit(EXIT_FAILURE);
}

void readFile(LIBSSH2_SESSION * session, LIBSSH2_SFTP * sftpSession, LIBSSH2_SFTP_HANDLE * fileHandle) {
	size_t totalRead = 0;
	if (!fileHandle) {
		printSftpErrorAndDie(session, sftpSession);	
	}
    LIBSSH2_SFTP_ATTRIBUTES fileAttributes;
    libssh2_sftp_fstat(fileHandle, &fileAttributes);
	size_t totalSize = (size_t)fileAttributes.filesize;
    char * filedata = malloc(totalSize);
	char * iter = filedata;
	while (totalRead < totalSize) {
		ssize_t readResult = libssh2_sftp_read(fileHandle, iter, fileAttributes.filesize);
		if (readResult < 0) {
			printSftpErrorAndDie(session, sftpSession);
		}
		iter += readResult;
		totalRead += readResult;
		printf("Read %ld bytes\n", readResult);
	}
	FILE * file = fopen(clientFilePath, "w");
	fwrite(filedata, totalRead, sizeof(char), file);
	printf("Wrote %lu bytes to %s\n", totalRead, clientFilePath);
    fclose(file);
    free(filedata);
    libssh2_sftp_close(fileHandle);

}

void getClient(char * serverAddress) {
    int sockFd = makeSocketOrDie();
    struct addrinfo * serverInfo = getServerInfo(serverAddress);
	printf("Connecting to server\n");
    connectOrDie(sockFd, serverInfo);
	printf("Connected to server.  Making LIBSSH2 session\n");
    LIBSSH2_SESSION * session = makeSession();
    libssh2_session_set_blocking(session, 1);
	libssh2_session_set_timeout(session, 5000);
	printf("Made session, handshaking\n");
    int result = libssh2_session_handshake(session, sockFd);
    //const char * fingerprint = libssh2_hostkey_hash(session, LIBSSH_HOSTKEY_HASH_SHA1);
    //TODO: Match the fingerprint against something.
    if (result) {
		char * errorMessage;
		libssh2_session_last_error(session, &errorMessage, NULL, 0);
		fprintf(stderr, "Error %s handshaking\n", errorMessage);
		exit(EXIT_FAILURE);
	}
    printf("Handshake completed, making SFTP Session\n");
	libssh2_userauth_password(session, NETID, PWD);
    LIBSSH2_SFTP * sftpSession = makeSFTPSession(session);
	printf("Started SFTP - Downloading file\n");
    LIBSSH2_SFTP_HANDLE * fileHandle = libssh2_sftp_open(sftpSession, serverFilePath, LIBSSH2_FXF_READ, 0);
	readFile(session, sftpSession, fileHandle);
    libssh2_sftp_shutdown(sftpSession);
    libssh2_session_disconnect(session, "Done.\n");
    libssh2_session_free(session);
    freeaddrinfo(serverInfo);
    close(sockFd);
}

#define PREDICTED_LENGTH 512

void executeClient(const char * serverAddress, char * port) {
	char * path = realpath(clientFilePath, NULL);
	char * command;
	asprintf(&command, decompressCommand, path);
	printf("Executing client command %s\n", command);
    system(command); 
	free(path);
	free(command);
	printf("Connecting to server\n");
	connectToServer(serverAddress, port);
	printf("Connected to server %s at port %s\n", serverAddress, port);
	char * result;
	size_t resultLen;
	result = NULL;
	printf("Requesting data from server\n");
	resultLen = requestFromServer(dataRequest, requestLen, PREDICTED_LENGTH, &result);
	printf("Got results of length %ld from server!, %s\n", resultLen, result);
	FILE * inputFilePtr = fopen(inputFile, "w");
	fprintf(inputFilePtr, "%s", result);
	fclose(inputFilePtr);
	free(result);
	pid_t pid = fork();
	if (pid == 0) {
		execl("./client_wrapper", "./client_wrapper", client, executable, inputFile, serverAddress, port, NULL);
		perror("ExecFailed\n");
		exit(1);  
	} else if (pid > 0) {
		int status;
		waitpid(pid, &status, 0);
	} else {
		perror("Fork failed\n");
		exit(EXIT_FAILURE);
	}
}

int main(int argc, char * argv[]) {
    initLibsshOrDie();
    if (argc == 4 && strcmp(argv[3], "-r") == 0) {
		printf("Getting client!\n");
        getClient(argv[1]);
    }
    libssh2_exit();
    executeClient(argv[1], argv[2]);
    return 0;
}
