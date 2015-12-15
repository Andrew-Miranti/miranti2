#define _GNU_SOURCE
#include "constants.h"
#include "server.h"
#include "utils.h"

#define min(a, b) (a < b ? a : b)`

volatile sig_atomic_t shouldStop = 0;
int serverFd = 0;

int makeServerSocket(const char * port) {
	
	int socketFd = socket(AF_INET, SOCK_STREAM, 0);
	if (socketFd < 0) {
		perror("Socket failed");
		exit(EXIT_FAILURE);
	}

	struct addrinfo info, * result;
	memset(&info, 0, sizeof(info));

	info.ai_family = AF_INET;
	info.ai_socktype = SOCK_STREAM;
	info.ai_flags = AI_PASSIVE;	

	fprintf(stderr, "Listening on port %s\n", port);
	int addrInfoError = getaddrinfo(NULL, port, &info, &result);
	if (addrInfoError) {
		fprintf(stderr, "GetAddrInfo failed: %s\n", gai_strerror(addrInfoError));
		exit(EXIT_FAILURE);
	}

	int optval = 1;
	setsockopt(socketFd, SOL_SOCKET, SO_REUSEPORT, &optval, sizeof(optval));

	int bindError = bind(socketFd, result->ai_addr, result->ai_addrlen);
	if (bindError) {
		perror("Bind Failed\n");
		exit(EXIT_FAILURE);
	}	

	int listenError = listen(socketFd, CONNECTION_QUEUE_SIZE);
	if (listenError) {
		perror("Listen Failed\n");
		exit(EXIT_FAILURE);
	}

	freeaddrinfo(result);
	return socketFd;
}

void stopFunction() {
	shouldStop = 1;
	close(serverFd);
	serverFd = 0;
}

void* acceptData(void * in) {
	int clientFd = (int)(long)in;
	if (clientFd < 0)
		return NULL;

	FILE * clientFile = fdopen(clientFd, "r+");	
	char * line = NULL;
	size_t size = 0;


	while (getline(&line, &size, clientFile) > 0) {
		if (strcmp(line, "DIS_GET_INPUT\n") == 0) {
			printf("GOT REQUEST\n");
			char * nextInput = NULL;
			size_t len = getNextInput(&nextInput);
			if (nextInput == NULL) {
				fprintf(stderr, "OUT OF INPUT\n");
				raise(SIGTERM);
				fprintf(clientFile, "DIS_DONE\n");
			} else {
				printf("SENDING INPUT: %s", nextInput);
				fprintf(clientFile, nextInput);
			}
		} else {
			printf("Got data from client %s", line);
		}
		if (shouldStop)
			break;
	}

	free(line);
	fclose(clientFile);
	return NULL;
}



int main(int argc, char *argv[]) {
	if (argc != 2) {
		fprintf(stderr, "Usage ./server port\n");
		exit(EXIT_FAILURE);
	} 
	char * port = argv[1];
	initServer(port);
	fprintf(stderr, "Server Started\n");
	struct sigaction stopper;
	memset(&stopper, 0, sizeof(stopper));
	stopper.sa_handler = stopFunction;
	sigaction(SIGINT, &stopper, NULL);
	sigaction(SIGTERM, &stopper, NULL);

	serverFd = makeServerSocket(port);
	fprintf(stderr, "Server socket open on port %s! with fd %d\n", port, serverFd);

	while (!shouldStop) {
		fprintf(stderr, "Now accepting new connections\n");
		int clientFd = accept(serverFd, NULL, NULL);
		fprintf(stderr, "New connection received with fd %d\n", clientFd);
		pthread_t newThread;
		pthread_create(&newThread, NULL, acceptData, (void*)(long)clientFd);
		pthread_detach(newThread);
		if (clientFd < 0 && !shouldStop) {
			perror("Accept failed");
			break;
		}
	}

	printf("SIGINT or SIGTERM received, stopping\n");
	closeServer();	
	pthread_exit(NULL);
	return 0;
}
