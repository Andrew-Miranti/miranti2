EXECS=server
CFLAGS=-g -Wall -Wextra -Wno-unused-parameter -pthread -std=c99
CC=gcc $(CFLAGS)
LIBS=-lcrypt -lrt -lssh2

all: $(EXECS)

server: server.o cracker_server.o utils.o constants.o
	$(CC) $^ $(LIBS) -o $@

server.o: server.c

cracker_server.o: cracker_server.c

utils.o: utils.c

constants.o: constants.c

%.o: %.c
	$(CC) -c $<

clean:
	rm -f *.o $(EXECS) *~
