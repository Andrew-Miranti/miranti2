#include <string.h>

#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <ctype.h>
#include <sys/types.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <omp.h>

#include "tbb/tick_count.h"
using tbb::tick_count;

char* map_file(char *filename, int *length_out) 
{
	struct stat file_stat;
	int fd = open(filename, O_RDWR);
	if (fd == -1) 
	{
		printf("failed to open file: %s\n", filename); 
		exit(1);
	}
	if (fstat(fd, &file_stat) != 0) 
	{
		printf("failed to stat file: %s\n", filename); 
		exit(1);
	}
	off_t length = file_stat.st_size;
	void *file = mmap(0, length, PROT_WRITE, MAP_SHARED, fd, 0);
	if (file == (void *)-1) 
	{
		printf("failed to stat file: %s\n", filename); 
		exit(1);
	}

	*length_out = length;
	return (char *)file;
}

int main(int argc, char *argv[]) 
{
	int length = 0;
	char *file = map_file(argv[1], &length);

	tick_count start = tick_count::now();

	// This is what you will be implementing!
	for (int i = 0 ; i < length ; i ++) 
	{
		file[i] = toupper(file[i]);
	}

	tick_count end = tick_count::now();
	printf("time = %f seconds\n", (end - start).seconds());  
}
