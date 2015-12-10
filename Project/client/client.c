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

const char * serverAddress = "127.0.0.1";
const char * serverFilePath = "~/client.tar.gz";
const char * clientFilePath = "./client.tar.gz";
const char * client = "./client";
const char * port = "2000";
const char * dataPort = "2001";
const char * decompressCommand = "tar -xvfz client.tar.gz";
const char * dataRequest = "DIS_GET_INPUT";
const size_t requestLen = 14;
const char * wrapperCommand = "./client_wrapper . inputFile.txt 127.0.0.1";
const char * inputFile = "inputFile.txt";

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

struct addrinfo * getServerInfo() {
    struct addrinfo hints, *result;
    result = NULL;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    int addrerr = getaddrinfo(serverAddress, port, &hints, &result);
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
        exit(EXIT_FAILURE);
    }
    return result;
}

void getClient() {
    int sockFd = makeSocketOrDie();
    struct addrinfo * serverInfo = getServerInfo();
    connectOrDie(sockFd, serverInfo);
    LIBSSH2_SESSION * session = makeSession();
    libssh2_session_set_blocking(session, 1);
    libssh2_session_handshake(session, sockFd);
    //const char * fingerprint = libssh2_hostkey_hash(session, LIBSSH_HOSTKEY_HASH_SHA1);
    //TODO: Match the fingerprint against something.
    LIBSSH2_SFTP * sftpSession = makeSFTPSession(session);
    LIBSSH2_SFTP_HANDLE * fileHandle = libssh2_sftp_open(sftpSession, serverFilePath, LIBSSH2_FXF_READ, 0);
    LIBSSH2_SFTP_ATTRIBUTES fileAttributes;
    libssh2_sftp_fstat(fileHandle, &fileAttributes);
    char * filedata = malloc((size_t)fileAttributes.filesize);
    libssh2_sftp_read(fileHandle, filedata, fileAttributes.filesize);
    int fileFd = open(clientFilePath, O_CREAT | O_TRUNC, O_WRONLY);
    write(fileFd, filedata, fileAttributes.filesize);
    close(fileFd);
    free(filedata);
    libssh2_sftp_close(fileHandle);
    libssh2_sftp_shutdown(sftpSession);
    libssh2_session_disconnect(session, "Done.\n");
    libssh2_session_free(session);
    freeaddrinfo(serverInfo);
    close(sockFd);
}

#define PREDICTED_LENGTH 512

void executeClient() {
    system(decompressCommand); 
	connectToServer(serverAddress, dataPort);	
	char * result;
	size_t resultLen;
	do {
		result = NULL;
		resultLen = requestFromServer(dataRequest, requestLen, PREDICTED_LENGTH, &result);
		int inputFileFd = open(inputFile, O_CREAT | O_TRUNC);
		write(inputFileFd, result, resultLen);
		close(inputFileFd);
		free(result);
		pid_t pid = fork();
		if (pid == 0) {
			execl("./client_wrapper", "./client_wrapper", client, inputFile, serverAddress, NULL);
			perror("ExecFailed\n");
			exit(1);  
		} else if (pid > 0) {
			int status;
			waitpid(&pid, &status, 0);
		} else {
			perror("Fork failed\n");
			exit(EXIT_FAILURE);
		}
	} while (resultLen > 0);
}

int main(int argc, char * argv[]) {
    initLibsshOrDie();
    if (argc == 2 && strcmp(argv[1], "-r")) {
        getClient();
    }
    libssh2_exit();
    executeClient();
    return 0;
}
