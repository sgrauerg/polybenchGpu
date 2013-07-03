/**
 * twodconv.c: This file is part of the PolyBench/GPU 1.0 test suite.
 *
 *
 * Contact: Scott Grauer-Gray <sgrauerg@gmail.com>
 * Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 */

#include <unistd.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>

#include "../common/polybenchUtilFuncts.h"

//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

/* Problem dimensions */
#define NI 4096
#define NJ 4096

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;




void conv2D(DATA_TYPE A[NI][NJ], DATA_TYPE B[NI][NJ])
{
	int i, j, k;
	DATA_TYPE c11, c12, c13, c21, c22, c23, c31, c32, c33;

	c11 = +2;  c21 = +5;  c31 = -8;
	c12 = -3;  c22 = +6;  c32 = -9;
	c13 = +4;  c23 = +7;  c33 = +10;

	#pragma acc kernels
	{
		for (i = 1; i < NI - 1; ++i) // 0
		{
			for (j = 1; j < NJ - 1; ++j) // 1
			{
				B[i][j] = c11 * A[i - 1][j - 1]  +  c12 * A[i + 0][j - 1]  +  c13 * A[i + 1][j - 1]
					+ c21 * A[i - 1][j + 0]  +  c22 * A[i + 0][j + 0]  +  c23 * A[i + 1][j + 0] 
					+ c31 * A[i - 1][j + 1]  +  c32 * A[i + 0][j + 1]  +  c33 * A[i + 1][j + 1];
			}
		}
	}
}


void conv2DCpu(DATA_TYPE A[NI][NJ], DATA_TYPE B[NI][NJ])
{
	int i, j, k;
	DATA_TYPE c11, c12, c13, c21, c22, c23, c31, c32, c33;

	c11 = +2;  c21 = +5;  c31 = -8;
	c12 = -3;  c22 = +6;  c32 = -9;
	c13 = +4;  c23 = +7;  c33 = +10;

	{
		for (i = 1; i < NI - 1; ++i) // 0
		{
			for (j = 1; j < NJ - 1; ++j) // 1
			{
				B[i][j] = c11 * A[i - 1][j - 1]  +  c12 * A[i + 0][j - 1]  +  c13 * A[i + 1][j - 1]
					+ c21 * A[i - 1][j + 0]  +  c22 * A[i + 0][j + 0]  +  c23 * A[i + 1][j + 0] 
					+ c31 * A[i - 1][j + 1]  +  c32 * A[i + 0][j + 1]  +  c33 * A[i + 1][j + 1];
			}
		}
	}
}

void init(DATA_TYPE A[NI][NJ])
{
	int i, j;

	for (i = 0; i < NI; ++i)
	{
		for (j = 0; j < NJ; ++j)
		{
			A[i][j] = i % 12 + 2 * (j % 7);
		}
	}
}


void compareResults(DATA_TYPE B[NI][NJ], DATA_TYPE B_outputFromGpu[NI][NJ])
{
	int i, j, fail;
	fail = 0;
	
	// Compare a and b
	for (i=1; i < (NI-1); i++) 
	{
		for (j=1; j < (NJ-1); j++) 
		{
			if (percentDiff(B[i][j], B_outputFromGpu[i][j]) > PERCENT_DIFF_ERROR_THRESHOLD) 
			{
				fail++;
			}
		}
	}
	
	// Print results
	printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);
	
}


int main(int argc, char *argv[])
{
	double t_start, t_end;

	DATA_TYPE A[NI][NJ];
	DATA_TYPE B[NI][NJ];  // CPU target results
	DATA_TYPE B_outputFromGpu[NI][NJ];  // GPU exec results

	//initialize the arrays
	init(A);
	

	
	t_start = rtclock();
	

	conv2D(A, B_outputFromGpu);


    t_end = rtclock();
    fprintf(stderr, "GPU Runtime: %0.6lf\n", t_end - t_start);
	

	t_start = rtclock();
	conv2DCpu(A, B);
    t_end = rtclock();
    fprintf(stderr, "CPU Runtime: %0.6lf\n", t_end - t_start);
	
	compareResults(B, B_outputFromGpu);

	return 0;
}
