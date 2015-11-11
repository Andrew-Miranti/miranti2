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
	int fd = open(filename, O_RDONLY);
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
	void *file = mmap(0, length, PROT_WRITE, MAP_PRIVATE, fd, 0);
	if (file == (void *)-1) 
	{
		printf("failed to stat file: %s\n", filename); 
		exit(1);
	}

	*length_out = length;
	return (char *)file;
}

__global__ void makeUpper(char * file, int length, int total) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int startIndex = index * length / total;
    int endIndex = (index+1) * length / total;
    for (int i = startIndex; i < endIndex; ++i) {
        file[i] = (file[i] >= 'a' && file[i] <= 'z') ? (file[i] - 'a' + 'A') : file[i];
    }
}

int main(int argc, char *argv[]) 
{
	int length = 0;
	char *file = map_file(argv[1], &length);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    char * cudaFile = NULL;
    cudaMalloc((void**)&cudaFile, length);
    cudaMemcpy(cudaFile, file, length, cudaMemcpyHostToDevice);
    int numblocks = 4096;
    int numthreads = 512;
    cudaEventRecord(start);
    makeUpper<<<numblocks, numthreads>>>(cudaFile, length, numblocks * numthreads);
    cudaEventRecord(stop);
    cudaMemcpy(file, cudaFile, length, cudaMemcpyDeviceToHost);
    cudaFree(cudaFile);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

	printf("time = %f milliseconds\n", milliseconds);  
}
