#define _GNU_SOURCE

#include<string.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <sys/sysinfo.h>
#include "time_util.h"

/* part A */
/* Checks the time that takes to write to the memory for different sizes to obtain memory bandwidth. */
void calculateMemoryBandWidth(){

	FILE *fp = fopen("memoryBandWidth.csv", "w+");
	fprintf(fp, "write_size,bandwidth,time\n");

	unsigned int arr_size = 1024 * 1024 * 1024;
	char *arr = malloc(arr_size * sizeof(char));
	memset(arr, (char)0, arr_size * sizeof(char));

	unsigned int step = 512;
	struct timespec start_time;
	struct timespec end_time;
	int i = 0;
	while (step <= arr_size){

		clock_gettime(CLOCK_MONOTONIC, &start_time);
		memset(arr, (char)i, step * sizeof(char));
		clock_gettime(CLOCK_MONOTONIC, &end_time);
		double time = timespec_to_nsec(difftimespec(end_time, start_time));
		fprintf(fp, "%d, %f, %f\n", step, step / time, time);
		step *= 2;
		i++;
	}
	fclose(fp);
	free(arr);
}

/* Part B */
/* Checks total time of a constant number of accesses to arrays of different sizes. This information
is used to calculate the cache sizes and thier latency. */
void calculateCacheSizes(){

	FILE *fp = fopen("cache_sizes.csv", "w+");
	fprintf(fp, "array_size,time,avg_time_per_item\n");

	int max_steps = 16 * 1024 * 1024;
	// check for array sizes from 512 Bytes to 1 Gigabytes
	for (unsigned int arr_size = 512; arr_size <= 1024 * 1024 * 1024; arr_size <<= 1)
	{
		max_steps = max_steps / 2;
		char *arr = malloc(arr_size * sizeof(char));
		// access the elements of the array once to eliminate the overhead of address translation
		for (unsigned int i = 0; i < arr_size; i++)
		{
			arr[i] = 0;
		}

		// Let the cache line size be 64 bytes
		struct timespec start_time;
		struct timespec end_time;
		clock_gettime(CLOCK_MONOTONIC, &start_time);

		// max_steps * arr_size will be constant, and it is number of times to check each array
		for (int t = 0; t < max_steps; t++)
		{
			for (unsigned int i = 0; i < arr_size; i += 64){
				arr[i] += 1;
			}
		}

		clock_gettime(CLOCK_MONOTONIC, &end_time);
		double time = timespec_to_nsec(difftimespec(end_time, start_time));
		free(arr);

		// write the max time it took to process this array in the csv file
		fprintf(fp, "%d,%f,%f\n", arr_size, time, (double)(time / (double)((arr_size / 64) * max_steps)));
	}
	fclose(fp);
}

/* Calculates the cache latencies based on cache sizes. */
void calculateCacheLatencies()
{

	FILE *fp = fopen("cacheLatency.csv", "w+");
	fprintf(fp, "array_size,Latency\n");

	// We calcuated the size of different cache levels in calculateMemoryBandWidth()
	// now using that info we can obtain the time latency of each of them
	// We found the size of caches to be: L1: 32 KiB, L2: 256 KiB, L3: 20 MiB
	int cacheSizes[4] = {32 * 1024, 256 * 1024, 20 * 1024 * 1024, 128 * 1024 * 1024};

	// array sizes to test accesses on
	unsigned int arraySizes[4] = {16 * 1024, 128 * 1024, 19 * 1024 * 1024, 1024 * 1024 * 1024};

	for (int i = 0; i < 4; i++)
	{
		int prevCacheSizes = 0;
		for (int prev = 0; prev < i; prev++)
		{
			prevCacheSizes += cacheSizes[prev];
		}
		char *testArr = malloc(arraySizes[i] * sizeof(char));
		char *invalidatingArr = malloc(prevCacheSizes * sizeof(char));
		// fill the caches with the lines of this testArr
		for (int t = 0; t < arraySizes[i]; t += 64)
		{
			testArr[t] = 'a';
		}

		// invalidate the caches lines from the testArr lines so that testArr lines go one cache down
		// the hierarchy
		int accesses = 16 * 1024; // an arbitary number
		for (int n = 0; n < accesses; n++)
		{
			for (int t = 0; t < prevCacheSizes; t += 64)
			{
				invalidatingArr[t] += 1;
			}
		}

		// now time the accesses to testArr a fixed num of times
		// note the lines of testArr is now located in the cache level i which we are measuring its latency
		struct timespec start_time;
		struct timespec end_time;
		clock_gettime(CLOCK_MONOTONIC, &start_time);
		for (int t = 0; t < accesses; t += 64)
		{
			testArr[t] += 1;
		}
		clock_gettime(CLOCK_MONOTONIC, &end_time);
		double time = timespec_to_nsec(difftimespec(end_time, start_time));

		fprintf(fp, "%d, %f\n", arraySizes[i], time / (accesses / 64));
		free(testArr);
		free(invalidatingArr);
	}
	fclose(fp);
}

/* Calculates cache line size */
void calculateCacheLineSize(){

	FILE *fp = fopen("cache_line.csv", "w+");
	fprintf(fp, "array_size,time\n");

	int numAccess = 2 * 1024;
	int arrSize = 4;
	// check arrays of size up to 2 * 1024
	while(arrSize < 2 * 1024){
		char *arr = malloc(arrSize * sizeof(char));
		struct timespec start_time;
		struct timespec end_time;
		clock_gettime(CLOCK_MONOTONIC, &start_time);

		// access each array constant num of times equal to numAccess * arrSize which is 8 * 1024
		for (int i = 0; i < numAccess; i++){
			for(int t = 0; t < arrSize; t++){
				arr[t] = 'a';
			}
		}
		clock_gettime(CLOCK_MONOTONIC, &end_time);
		double time = timespec_to_nsec(difftimespec(end_time, start_time));
		fprintf(fp, "%d, %f\n", arrSize, time / (numAccess * arrSize));
		arrSize *= 2;
		numAccess = numAccess / 2;
		free(arr);
	}
	fclose(fp);
}

/******* main ******/
int main(int argc, char *argv[])
{
	// part A
	calculateMemoryBandWidth();

	//part B
	calculateCacheSizes();
	calculateCacheLatencies();

	// bonus - cache line size
	calculateCacheLineSize();

	return 0;
}
