/**
 * atax.c: This file is part of the PolyBench/GPU 1.0 test suite.
 *
 *
 * Contact: Scott Grauer-Gray <sgrauerg@gmail.com>
 * Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <stdlib.h>
#include <stdarg.h>

#include "../common/polybenchUtilFuncts.h"

//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

/* Problem size. */
#define NX 4096
#define NY 4096

/* Constant for pi */
#define M_PI 3.14159

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;



void ataxLoopa(DATA_TYPE a[NX][NY], DATA_TYPE x[NY], DATA_TYPE y[NY], DATA_TYPE tmp[NX])
{
	int i, j;

	#pragma acc kernels
	{
		for (i= 0; i < NX; i++)
		{
			y[i] = 0;
		}


		for (i = 0; i < NX; i++)
		{
			tmp[i] = 0;


			for (j = 0; j < NY; j++)
			{
				tmp[i] = tmp[i] + a[i][j] * x[j];
			}
		}
	}
}


void ataxLoopb(DATA_TYPE a[NX][NY], DATA_TYPE y[NY], DATA_TYPE tmp[NX])
{
	int i, j;

	#pragma acc kernels
	{
		for(i = 0; i < NY; i++)
		{
			for (j = 0; j < NX; j++)
			{
				y[j] = y[j] + a[i][j] * tmp[i];
			}
		}
	}
}


void ataxLoopaCpu(DATA_TYPE a[NX][NY], DATA_TYPE x[NY], DATA_TYPE y[NY], DATA_TYPE tmp[NX])
{
	int i, j;

	{
		for (i= 0; i < NX; i++)
		{
			y[i] = 0;
		}


		for (i = 0; i < NX; i++)
		{
			tmp[i] = 0;


			for (j = 0; j < NY; j++)
			{
				tmp[i] = tmp[i] + a[i][j] * x[j];
			}
		}
	}
}


void ataxLoopbCpu(DATA_TYPE a[NX][NY], DATA_TYPE y[NY], DATA_TYPE tmp[NX])
{
	int i, j;

	{
		for(i = 0; i < NY; i++)
		{
			for (j = 0; j < NX; j++)
			{
				y[j] = y[j] + a[i][j] * tmp[i];
			}
		}
	}
}

void init_array(DATA_TYPE A[NX][NY], DATA_TYPE x[NX])
{
	int i, j;

	for (i = 0; i < NX; i++)
	{
		x[i] = i * M_PI;
		for (j = 0; j < NY; j++)
		{
			A[i][j] = ((DATA_TYPE) i*j) / NX;
		}
	}
}


void compareResults(DATA_TYPE y[NY], DATA_TYPE y_outputFromGpu[NY])
{
	int i, fail;
	fail = 0;

	for (i=0; i<NY; i++)
	{
		if (percentDiff(y[i], y_outputFromGpu[i]) > PERCENT_DIFF_ERROR_THRESHOLD)
		{
			fail++;
		}		
	}
	
	// print results
	printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);

}



int main(int argc, char** argv)
{
	double t_start, t_end;
  
	/* Array declaration */
	DATA_TYPE A[NX][NY];
	DATA_TYPE tmp[NX];
	DATA_TYPE tmp_Gpu[NX];
	DATA_TYPE x[NY];
	DATA_TYPE y[NY];
	DATA_TYPE y_outputFromGpu[NY];


	/* Initialize array. */
	init_array(A, x);


	t_start = rtclock();

	ataxLoopa(A, x, y_outputFromGpu, tmp_Gpu);

	ataxLoopb(A, y_outputFromGpu, tmp_Gpu);

	t_end = rtclock();
	fprintf(stderr, "GPU Runtime: %0.6lfs\n", t_end - t_start);
    

    
	t_start = rtclock();
 
	ataxLoopaCpu(A, x, y, tmp);
	ataxLoopbCpu(A, y, tmp);

	t_end = rtclock();
	fprintf(stderr, "CPU Runtime: %0.6lfs\n", t_end - t_start);

	compareResults(y, y_outputFromGpu);

	return 0;
}
