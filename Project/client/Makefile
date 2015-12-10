EXECS=client_wrapper client
CFLAGS=-g -Wall -Wextra -Wno-unused-parameter -pthread -std=c99
CC=gcc $(CFLAGS)
LIBS=-lcrypt -lrt -lssh2

all: $(EXECS)

client: client.o distributed.o
	$(CC) $^ $(LIBS) -o $@

client.o: client.c

client_wrapper: client_wrapper.o distributed.o
	$(CC) $^ $(LIBS) -o $@

client_wrapper.o: client_wrapper.c

distributed.o: distributed.c

%.o: %.c
	$(CC) -c $<

clean:
	rm -f *.o $(EXECS) *~ client.tar.gz
	rm -rf clientFolder/*
