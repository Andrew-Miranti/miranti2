EXECS=client
CFLAGS=-g -Wall -Wextra -Wno-unused-parameter -pthread -std=c99
# CC=clang $(CFLAGS)
CC=gcc $(CFLAGS)
LIBS=-lcrypt -lrt

all: $(EXECS)

client: cracker.o common.o 
	$(CC) $^ $(LIBS) -o $@

%.o: %.c
	$(CC) -c $<

clean:
	rm -f *.o $(EXECS) *~
