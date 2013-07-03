/**
 * gramschmidt.c: This file is part of the PolyBench/GPU 1.0 test suite.
 *
 *
 * Contact: Scott Grauer-Gray <sgrauerg@gmail.com>
 * Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <stdlib.h>
#include <stdarg.h>

#include "../common/polybenchUtilFuncts.h"

//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

/* Problem size. */
#define M 2048
#define N 2048

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;



void runGramSchmidt(DATA_TYPE pA[M][N], DATA_TYPE pR[M][N], DATA_TYPE pQ[M][N])
{
	int i, j, k;
	int m = M;
	int n = N;

	DATA_TYPE pnrm;

	#pragma acc kernels
	{
		for (k = 0; k < n; k++)
		{
			pnrm = 0;

			for (i = 0; i < m; i++)
			{
				pnrm += pA[i][k] * pA[i][k];
			}
		
			pR[k][k] = sqrtf(pnrm);
		
			for (i = 0; i < m; i++)
			{
				pQ[i][k] = pA[i][k] / pR[k][k];
			}
	 
			for (j = k + 1; j < n; j++)
			{
				pR[k][j] = 0;
		    
				for (i = 0; i < m; i++)
				{
					pR[k][j] += pQ[i][k] * pA[i][j];
				}
		    
				for (i = 0; i < m; i++)
				{
					pA[i][j] = pA[i][j] - pQ[i][k] * pR[k][j];
				}
			}
		}
	}
}


void runGramSchmidtCpu(DATA_TYPE pA[M][N], DATA_TYPE pR[M][N], DATA_TYPE pQ[M][N])
{
	int i, j, k;
	int m = M;
	int n = N;

	DATA_TYPE pnrm;

	{
		for (k = 0; k < n; k++)
		{
			pnrm = 0;

			for (i = 0; i < m; i++)
			{
				pnrm += pA[i][k] * pA[i][k];
			}
		
			pR[k][k] = sqrtf(pnrm);
		
			for (i = 0; i < m; i++)
			{
				pQ[i][k] = pA[i][k] / pR[k][k];
			}
	 
			for (j = k + 1; j < n; j++)
			{
				pR[k][j] = 0;
		    
				for (i = 0; i < m; i++)
				{
					pR[k][j] += pQ[i][k] * pA[i][j];
				}
		    
				for (i = 0; i < m; i++)
				{
					pA[i][j] = pA[i][j] - pQ[i][k] * pR[k][j];
				}
			}
		}
	}
}


void init_array(DATA_TYPE A[M][N], DATA_TYPE A_Gpu[M][N])
{
	int i, j;

	for (i = 0; i < M; i++)
	{
		for (j = 0; j < N; j++)
		{
			A[i][j] = ((DATA_TYPE) (i+1)*(j+1)) / (M+1);
			A_Gpu[i][j] = ((DATA_TYPE) (i+1)*(j+1)) / (M+1);
		}
	}
}


void compareResults(DATA_TYPE A[M][N], DATA_TYPE A_outputFromGpu[M][N])
{
	int i, j, fail;
	fail = 0;

	for (i=0; i < M; i++) 
	{
		for (j=0; j < N; j++) 
		{
			if (percentDiff(A[i][j], A_outputFromGpu[i][j]) > PERCENT_DIFF_ERROR_THRESHOLD) 
			{				
				fail++;
			}
		}
	}
	
	// Print results
	printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);

}


int main(int argc, char** argv)
{
	double t_start, t_end;
	
	/* Array declaration */
	DATA_TYPE A[M][N];
	DATA_TYPE A_outputFromGpu[M][N];
	DATA_TYPE R[M][N];
	DATA_TYPE R_Gpu[M][N];
	DATA_TYPE Q[M][N];
	DATA_TYPE Q_Gpu[M][N];

	/* Initialize array. */
	init_array(A, A_outputFromGpu);
    

	t_start = rtclock();
	

	runGramSchmidt(A_outputFromGpu, R_Gpu, Q_Gpu);
    


	t_end = rtclock();
	fprintf(stderr, "GPU Runtime: %0.6lfs\n", t_end - t_start);
    
	
	t_start = rtclock();
	
	runGramSchmidtCpu(A, R, Q);
	
	t_end = rtclock();
	fprintf(stderr, "CPU Runtime: %0.6lfs\n", t_end - t_start);
	
	compareResults(A, A_outputFromGpu);

	return 0;
}
