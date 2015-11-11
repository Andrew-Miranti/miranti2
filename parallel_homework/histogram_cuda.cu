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

#define HISTOGRAM_SIZE 256
#define HISTOGRAM_COUNT 1

__global__ void countLetters(char * file, unsigned * allHistograms, int length, int total) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned * histogram = allHistograms + ((HISTOGRAM_COUNT - 1) & index) * HISTOGRAM_SIZE; 
    for (int i = index; i < length; i += total) {
        atomicAdd(&histogram[file[i]], 1);
    }
}


int main(int argc, char *argv[]) 
{
	int length = 0;
	bool print = false;
	if (strncmp(argv[1], "-print", 7) == 0) 
	{
		print = true;
		argv = &argv[1];
	}
	char *file = map_file(argv[1], &length);
	unsigned histogram[HISTOGRAM_SIZE] = {0};

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);


    char * deviceFile = NULL;
    cudaMalloc((void**)&deviceFile, length);
    cudaMemcpy(deviceFile, file, length, cudaMemcpyHostToDevice);
    unsigned * histograms = NULL;
    int numBlocks = 65535;
    int numThreads = 512;
    int totalThreads = numBlocks * numThreads;
    size_t allHistogramSize = sizeof(unsigned) * HISTOGRAM_COUNT * HISTOGRAM_SIZE;
    cudaMalloc((void**)&histograms, allHistogramSize);
    cudaMemset(histograms, 0, allHistogramSize);
    cudaEventRecord(start);
    countLetters<<<numBlocks, numThreads>>>(deviceFile, histograms, length, totalThreads);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    unsigned * hostHistograms = (unsigned*)malloc(allHistogramSize);
    memset(hostHistograms, 0, allHistogramSize);

    cudaMemcpy(hostHistograms, histograms, allHistogramSize, cudaMemcpyDeviceToHost);
    cudaFree(deviceFile);
    cudaFree(histograms);

    for (int j = 0; j < HISTOGRAM_COUNT; ++j) {
        for (int i = 0; i < HISTOGRAM_SIZE; ++i) {
            histogram[i] += hostHistograms[HISTOGRAM_SIZE * j + i];
        }  
    }
    free(hostHistograms);

	float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("%f seconds\n", milliseconds / 1000);

	if (print) 
	{
		for (int i = 0 ; i < 128 ; i ++) 
		{
			if (histogram[i] != 0) 
			{
				printf("%c (%d): %d\n", i, i, histogram[i]);
			}
		}
	}
}
