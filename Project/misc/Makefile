EXECS=generator
CFLAGS=-g -Wall -Wextra -Wno-unused-parameter -pthread -std=c99
# CC=clang $(CFLAGS)
CC=gcc $(CFLAGS)
LIBS=-lcrypt -lrt -lm

all: $(EXECS)

generator: generator.o
	$(CC) $^ $(LIBS) -o $@

%.o: %.c
	$(CC) -c $<

clean:
	rm -f *.o $(EXECS) *~
