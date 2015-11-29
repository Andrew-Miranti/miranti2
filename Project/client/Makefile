EXECS=client_wrapper client
CFLAGS=-g -Wall -Wextra -Wno-unused-parameter -pthread -std=c99
# CC=clang $(CFLAGS)
CC=gcc $(CFLAGS)
LIBS=-lcrypt -lrt

all: $(EXECS)

client: client_wrapper client.o
	$(CC) $^ $(LIBS) -o $@

client.o: client.c

client_wrapper: client_wrapper.o
	$(CC) $^ $(LIBS) -o $@

client_wrapper.o: distributed.h client_wrapper.c

distributed.o: distributed.c

%.o: %.c
	$(CC) -c $<

clean:
	rm -f *.o $(EXECS) *~
