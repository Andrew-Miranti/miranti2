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

__global__ void countLetters(char * file, unsigned * allHistogram, int length, int total) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned * histogram = allHistogram + index * HISTOGRAM_SIZE;
    int startIndex = index * length / total;
    int endIndex = (index+1) * length / total;
    for (int i = startIndex; i < endIndex; ++i) {
        histogram[file[i]]++;
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

	tick_count start = tick_count::now();

    char * deviceFile = NULL;
    cudaMalloc((void**)&deviceFile, length);
    cudaMemcpy(deviceFile, file, length, cudaMemcpyHostToDevice);
    unsigned * histograms = NULL;
    int numBlocks = 4;
    int numThreads = 4;
    int totalThreads = numBlocks * numThreads;
    size_t allHistogramSize = sizeof(unsigned) * totalThreads * HISTOGRAM_SIZE;
    cudaMalloc((void**)&histograms, allHistogramSize);
    cudaMemset(histograms, 0, allHistogramSize);
    countLetters<<<numBlocks, numThreads>>>(deviceFile, histograms, length, totalThreads);
    unsigned * hostHistograms = (unsigned*)malloc(allHistogramSize);
    memset(hostHistograms, 0, allHistogramSize);

    cudaMemcpy(hostHistograms, histograms, allHistogramSize, cudaMemcpyDeviceToHost);
    cudaFree(deviceFile);
    cudaFree(histograms);

    for (int j = 0; j < totalThreads; ++j) {
        for (int i = 0; i < HISTOGRAM_SIZE; ++i) {
            histogram[i] += hostHistograms[HISTOGRAM_SIZE * j + i];
        }  
    }
    free(hostHistograms);

	tick_count end = tick_count::now();

	printf("time = %f seconds\n", (end - start).seconds());  

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
