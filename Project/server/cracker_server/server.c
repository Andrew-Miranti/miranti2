#define _GNU_SOURCE
#include "constants.h"
#include "server.h"
#include "utils.h"

#define min(a, b) (a < b ? a : b)

typedef struct connectionData {
	struct connectionData * next;
	char * dataToSend, * dataIter;
	ssize_t dataSize;
	int fd;
} connectionData;

volatile sig_atomic_t shouldStop = 0;
int serverFd = 0;
connectionData * head = NULL;
int selfPipe[2];

connectionData * newConnectionData(int fd) {
	connectionData * result = calloc(sizeof(connectionData), 1);
	result->fd = fd;
	result->next = head;
	head = result;
	return result;
}

void cleanupConnectionData() {
	connectionData * iter = head;
	while (iter) {
		connectionData * nextiter = iter->next;
		free(iter->dataToSend);
		free(iter);
		iter = nextiter;
	}
	head = NULL;
}

int makeServerSocket(const char * port) {
	struct addrinfo info, * result;
	memset(&info, 0, sizeof(info));

	info.ai_family = AF_INET;
	info.ai_socktype = AI_PASSIVE;
	

	int addrInfoError = getaddrinfo(NULL, port, &info, &result);
	if (addrInfoError) {
		fprintf(stderr, "GetAddrInfo failed: %s\n", gai_strerror(addrInfoError));
		exit(EXIT_FAILURE);
	}

	//nt socketFd = socket(AF_INET, SOCK_STREAM | SOCK_NONBLOCK, result->ai_flags);
	int socketFd = socket(result->ai_family, result->ai_socktype, result->ai_flags);
	if (socketFd < 0) {
		perror("Socket failed");
		exit(EXIT_FAILURE);
	}

	int optval = 1;
	setsockopt(socketFd, SOL_SOCKET, SO_REUSEPORT, &optval, sizeof(optval));

	int bindError = bind(socketFd, (struct sockaddr*)result->ai_addr, result->ai_addrlen);
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
	write(selfPipe[1], "", 1);
}

void* acceptData(void * in) {
	int epollFd = (int)(long)in;
	while (!shouldStop) {
		struct epoll_event event;
		memset(&event, 0, sizeof(event));
		while(epoll_wait(epollFd, &event, 1, -1) < 0 && errno == EINTR && !shouldStop) {
			errno = 0;
		}
		if (shouldStop) break;
		fprintf(stderr, "Epoll event fired!\n");
	
		connectionData * data = event.data.ptr;

		if (event.events & EPOLLOUT) {
			fprintf(stderr, "Server Writing\n");
			if (data->dataToSend) {
				fprintf(stderr, "Writing %s to client!\n", data->dataIter);
				ssize_t written = write(data->fd, data->dataIter, data->dataSize);
				if (written >= 0) {
					data->dataIter += written;
					data->dataSize -= written;
				} else {
					perror("Write failed");
				}
				if (data->dataSize <= 0) {
					free(data->dataToSend);
					data->dataToSend = NULL;
					data->dataIter = NULL;
					event.events = EPOLLIN;
					epoll_ctl(epollFd, EPOLL_CTL_MOD, data->fd, &event);
				}
			}
		}
		if (event.events & EPOLLIN) {
			fprintf(stderr, "Server Reading\n");
			char * readFromClient = NULL;
			ssize_t size = readFromFd(data->fd, &readFromClient);
			fprintf(stderr, "Server done reading\n");
			if (size < 0) { // In EOF state.
				close(data->fd);
				epoll_ctl(epollFd, EPOLL_CTL_DEL, data->fd, &event);
				printf("Client disconnected\n");
			} else {
				fprintf(stderr, "From client: %s\n", readFromClient);
				char * body = readFromClient;
				while (*body && *body != ' ') body++;
				char * instruction = readFromClient;
				*body = '\0';
				body++;
				if (strcmp(instruction, REDUCE_INSTRUCTION) == 0) { 
					if (!body) {
						fprintf(stderr, "Malformed result of size %ld : %s ", size, body);
						exit(EXIT_FAILURE);
					}
					reduceResult(body, size);
				}
				else if (strcmp(instruction, GET_INSTRUCTION) == 0 && !data->dataToSend) {
					char * newInput;
					ssize_t inputSize = getNextInput(&newInput);
					fprintf(stderr, "About to send %s to client!\n", newInput);
					free(data->dataToSend);
					data->dataToSend = data->dataIter = newInput;
					data->dataSize = inputSize;
					event.events = EPOLLOUT | EPOLLIN;
					epoll_ctl(epollFd, EPOLL_CTL_MOD, data->fd, &event);
				}
			}
			free(readFromClient);
		}
	}
	close(epollFd);
	return NULL;
}



int main() {
	initServer();
	pipe(selfPipe); // Signal handlers acting weird with epoll, according to the internet apparently a pipe to yourself works.
	struct sigaction stopper;
	memset(&stopper, 0, sizeof(stopper));
	stopper.sa_handler = stopFunction;
	sigaction(SIGINT, &stopper, NULL);
	sigaction(SIGTERM, &stopper, NULL);

	serverFd = makeServerSocket(port);
	struct epoll_event event;
	memset(&event, 0, sizeof(event));
	event.events = EPOLLIN;
	int epollFd = epoll_create(1);
	epoll_ctl(epollFd, EPOLL_CTL_ADD, selfPipe[0], &event);
	
	pthread_t readerThread;
	pthread_create(&readerThread, NULL, acceptData, (void*)(long)epollFd);

	while (!shouldStop) {
		int newConnectionFd = accept4(serverFd, NULL, NULL, SOCK_NONBLOCK);
		printf("New connection received with fd %d\n", newConnectionFd);
		connectionData * data = newConnectionData(newConnectionFd);
		event.data.ptr = data;
		if (newConnectionFd >= 0)
			epoll_ctl(epollFd, EPOLL_CTL_ADD, newConnectionFd, &event);
		else if (!shouldStop) {
			perror("Accept failed");
			break;
		}
	}

	printf("SIGINT or SIGTERM received, stopping\n");
	pthread_join(readerThread, NULL);
	printf("Finished waiting for reader, exiting\n");
	cleanupConnectionData();
	close(selfPipe[0]);
	close(selfPipe[1]);
	closeServer();	
	return 0;
}
