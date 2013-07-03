#include <stdio.h>
#include <unistd.h>
#include <time.h>
#include <sys/time.h>
#include <string.h>
#include <stdlib.h>
#include <stdarg.h>
#include <math.h>

#include "../common/polybenchUtilFuncts.h"

//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

/* Problem size. */
#define TSTEPS 10000
#define N 1024

/* Thread block dimensions */
#define DIM_THREAD_BLOCK_X 32
#define DIM_THREAD_BLOCK_Y 8

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;



void jacobi2D(DATA_TYPE A[N][N], DATA_TYPE B[N][N])
{
	int t, i, j;
	int tsteps = 100;
	int n = N;

	#pragma acc kernels
	{
		for (t = 0; t < tsteps; t++)
		{
			for (i = 2; i < n - 1; i++)
			{
				for (j = 2; j < n - 1; j++)
				{
					B[i][j] = 0.2 * (A[i][j] + A[i][j-1] + A[i][1+j] + A[1+i][j] + A[i-1][j]);
				}
			}
		
			for (i = 2; i < n-1; i++)
			{
				for (j = 2; j < n-1; j++)
				{
					A[i][j] = B[i][j];
				}
			}
		}
	}
}


void jacobi2DCpu(DATA_TYPE A[N][N], DATA_TYPE B[N][N])
{
	int t, i, j;
	int tsteps = 100;
	int n = N;

	#pragma acc kernels
	{
		for (t = 0; t < tsteps; t++)
		{
			for (i = 2; i < n - 1; i++)
			{
				for (j = 2; j < n - 1; j++)
				{
					B[i][j] = 0.2 * (A[i][j] + A[i][j-1] + A[i][1+j] + A[1+i][j] + A[i-1][j]);
				}
			}
		
			for (i = 2; i < n-1; i++)
			{
				for (j = 2; j < n-1; j++)
				{
					A[i][j] = B[i][j];
				}
			}
		}
	}
}

void init_array(DATA_TYPE A[N][N], DATA_TYPE B[N][N], DATA_TYPE C[N][N], DATA_TYPE D[N][N])
{ 
	int i, j;

	for (i = 0; i < N; i++)
	{
		for (j = 0; j < N; j++)
		{ 
			A[i][j] = ((DATA_TYPE) i*j + 10) / N;
			B[i][j] = ((DATA_TYPE) i*j + 11) / N;
			C[i][j] = ((DATA_TYPE) i*j + 10) / N;
			D[i][j] = ((DATA_TYPE) i*j + 11) / N;
		}
	}
}


void compareResults(DATA_TYPE a[N][N], DATA_TYPE b[N][N], DATA_TYPE c[N][N], DATA_TYPE d[N][N])
{
	int i, j, fail;
	fail = 0;   

       // Compare a and c
       for (i=0; i<N; i++) 
	{
		for (j=0; j<N; j++)
		{
			if (percentDiff(a[i][j], c[i][j]) > PERCENT_DIFF_ERROR_THRESHOLD)
			{
                    		fail++;
			}
              }
       }

	// Compare b and d
	for (i=0; i<N; i++) 
	{
		for (j=0; j<N; j++)
		{
			if (percentDiff(d[i][j], b[i][j]) > PERCENT_DIFF_ERROR_THRESHOLD)
                     {
				fail++;
			}
		}
	}

	// Print results
	printf("Number of misses: %d\n", fail);
}


int main(int argc, char** argv)
{
	double t_start, t_end;

	/* Array declaration */
	DATA_TYPE A[N][N];
	DATA_TYPE B[N][N];
	DATA_TYPE C[N][N];
	DATA_TYPE D[N][N];

	/* Initialize array. */
	init_array(A, B, C, D);



        // Run GPU code

        t_start = rtclock();


        jacobi2D(A,B);



	t_end = rtclock();
	fprintf(stderr, "GPU Runtime: %0.6lfs\n", t_end - t_start);



        t_start = rtclock();

        jacobi2DCpu(C,D);

	t_end = rtclock();
	fprintf(stderr, "CPU Runtime: %0.6lfs\n", t_end - t_start);

        compareResults(A, B, C, D);

	return 0;
}

