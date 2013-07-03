/**
 * threemm.c: This file is part of the PolyBench/GPU 1.0 test suite.
 *
 *
 * Contact: Scott Grauer-Gray <sgrauerg@gmail.com>
 * Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 */

#include <unistd.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <stdlib.h>
#include <stdarg.h>
#include <math.h>

#include "../common/polybenchUtilFuncts.h"

//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

/* Problem size. */
#define NI 512
#define NJ 512
#define NK 512
#define NL 512
#define NM 512

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;




void threeMMloopa(DATA_TYPE a[NI][NK], DATA_TYPE b[NK][NJ], DATA_TYPE e[NI][NJ])
{
	int i, j, k;

	#pragma acc kernels
	{
		/* E := A*B */
		for (i = 0; i < NI; i++)
		{
			for (j = 0; j < NJ; j++)
			{
				e[i][j] = 0;
		 
				for (k = 0; k < NK; ++k)
				{
					e[i][j] += a[i][k] * b[k][j];
				}
			}
		}
	}
}


void threeMMloopb(DATA_TYPE c[NJ][NM], DATA_TYPE d[NM][NL], DATA_TYPE f[NJ][NL])
{
	int i, j, k; 


	#pragma acc kernels
	{
		/* F := C*D */
		for (i = 0; i < NJ; i++)
		{
			for (j = 0; j < NL; j++)
			{
				f[i][j] = 0;

				for (k = 0; k < NM; ++k)
				{
					f[i][j] += c[i][k] * d[k][j];
				}
			}
		}
	}
}


void threeMMloopc(DATA_TYPE e[NI][NJ], DATA_TYPE f[NJ][NL], DATA_TYPE g[NI][NL])
{
	int i, j, k;


	#pragma acc kernels
	{

		/* G := E*F */
		for (i = 0; i < NI; i++)
		{      


			for (j = 0; j < NL; j++)
			{
				g[i][j] = 0;
		  
				for (k = 0; k < NJ; ++k)
				{
					g[i][j] += e[i][k] * f[k][j];
				}
			}
		}
	}
}



void threeMMloopaCpu(DATA_TYPE a[NI][NK], DATA_TYPE b[NK][NJ], DATA_TYPE e[NI][NJ])
{
	int i, j, k;

	{
		/* E := A*B */
		for (i = 0; i < NI; i++)
		{
			for (j = 0; j < NJ; j++)
			{
				e[i][j] = 0;
		 
				for (k = 0; k < NK; ++k)
				{
					e[i][j] += a[i][k] * b[k][j];
				}
			}
		}
	}
}


void threeMMloopbCpu(DATA_TYPE c[NJ][NM], DATA_TYPE d[NM][NL], DATA_TYPE f[NJ][NL])
{
	int i, j, k; 

	{
		/* F := C*D */
		for (i = 0; i < NJ; i++)
		{
			for (j = 0; j < NL; j++)
			{
				f[i][j] = 0;

				for (k = 0; k < NM; ++k)
				{
					f[i][j] += c[i][k] * d[k][j];
				}
			}
		}
	}
}


void threeMMloopcCpu(DATA_TYPE e[NI][NJ], DATA_TYPE f[NJ][NL], DATA_TYPE g[NI][NL])
{
	int i, j, k;

	{
		/* G := E*F */
		for (i = 0; i < NI; i++)
		{      


			for (j = 0; j < NL; j++)
			{
				g[i][j] = 0;
		  
				for (k = 0; k < NJ; ++k)
				{
					g[i][j] += e[i][k] * f[k][j];
				}
			}
		}
	}
}

void compareResults(DATA_TYPE G[NI][NL], DATA_TYPE G_outputFromGpu[NI][NL])
{
	int i,j,fail;
	fail = 0;

	for (i=0; i < NI; i++)
	{
		for (j=0; j < NL; j++)
		{
			if (percentDiff(G[i][j], G_outputFromGpu[i][j]) > PERCENT_DIFF_ERROR_THRESHOLD)
			{
				fail++;				
			}
		}
	}
	
	// print results
	printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);

}


void init_array(DATA_TYPE A[NI][NK], DATA_TYPE B[NK][NJ], DATA_TYPE C[NJ][NM], DATA_TYPE D[NM][NL])
{
	int i, j;

	for (i = 0; i < NI; i++)
	{
		for (j = 0; j < NK; j++)
		{
			A[i][j] = ((DATA_TYPE) i*j) / NI;
		}
	}
  
	for (i = 0; i < NK; i++)
	{
		for (j = 0; j < NJ; j++)
		{
			B[i][j] = ((DATA_TYPE) i*(j+1)) / NJ;
		}
	}
  
	for (i = 0; i < NJ; i++)
	{
		for (j = 0; j < NM; j++)
		{
			C[i][j] = ((DATA_TYPE) i*(j+3)) / NL;
		}
	}
  
	for (i = 0; i < NM; i++)
	{
		for (j = 0; j < NL; j++)
		{
			D[i][j] = ((DATA_TYPE) i*(j+2)) / NK;
		}
	}
}

int main(int argc, char** argv)
{
	double t_start, t_end;

	/* Array declaration */
	DATA_TYPE A[NI][NK];
	DATA_TYPE B[NK][NJ];
	DATA_TYPE C[NJ][NM];
	DATA_TYPE D[NM][NL];
	DATA_TYPE E[NI][NJ];
	DATA_TYPE E_gpu[NI][NJ];	
	DATA_TYPE F[NJ][NL];
	DATA_TYPE F_gpu[NJ][NL];
	DATA_TYPE G[NI][NL];
	DATA_TYPE G_outputFromGpu[NI][NL];

	/* Initialize array. */
	init_array(A, B, C, D);
    

	t_start = rtclock();

	threeMMloopa(A, B, E_gpu);

	threeMMloopb(C, D, F_gpu);

	threeMMloopc(E_gpu, F_gpu, G_outputFromGpu);


	t_end = rtclock();
	fprintf(stderr, "GPU Runtime: %0.6lfs\n", t_end - t_start);
    

	
	t_start = rtclock();

	threeMMloopaCpu(A, B, E);
	threeMMloopbCpu(C, D, F);
	threeMMloopcCpu(E, F, G);

	t_end = rtclock();
	fprintf(stderr, "CPU Runtime: %0.6lfs\n", t_end - t_start);

	compareResults(G, G_outputFromGpu);

	return 0;
}
