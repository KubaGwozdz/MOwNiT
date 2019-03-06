#include <stdio.h>
#include <gsl/gsl_ieee_utils.h>
#include <stdlib.h>
#include <time.h>
#include <gsl/gsl_blas.h>

/* 
- naiwną metodę mnożenia macierzy (wersja 1) 
- ulepszoną za pomocą zamiany pętli metodę mnożenia macierzy (wersja 2), pamiętając, że w C macierz przechowywana jest wierszami (row major order tzn A11,A12, ..., A1m, A21, A22,...,A2m, ..Anm), inaczej niż w Julii ! 
- skorzystać z  możliwości BLAS dostępnego w GSL(wersja 3). 
*/


void printMatrix(double** A, int size){
	for(int i=0; i<size; i++){
		printf("\n");
		for(int j=0; j<size; j++){
			printf("%f ", A[i][j]);
		}
	}
}

void better_multiplication(double** A, double** B, double ** result, int size){
	for (int i=0; i< size; i++){
		for(int k=0; k<size; k++){
			for(int j=0; j<size; j++){
				result[i][j] = result[i][j]+(A[i][k]*B[k][j]);
			}
		}
	}
}


void naive_multiplication(double** A, double** B, double ** result, int size){
	for (int i=0; i< size; i++){
		for(int j=0; j<size; j++){
			for(int k=0; k<size; k++){
				result[i][j] = result[i][j]+(A[i][k]*B[k][j]);
			}
		}
	}
}



int main(void){
	srand(time(NULL));
	FILE *f = fopen("results.csv", "w");
  	if(f==NULL) return -1;

  	clock_t naiveStart, naiveEnd, betterStart, betterEnd, blasStart, blasEnd;
  	double nDiv, btrDiv, blasDiv;

  	fprintf(f, "%s, %s, %s, %s\n","size","naive", "better", "blas");

	for(int size=100;size<=1000;size+=100){
		printf("size: %d\n", size);

		double** matA = malloc(size*sizeof(double*));
		double** matB = malloc(size*sizeof(double*)); 
		double** result = malloc(size*sizeof(double*));
		// blas
		double *m1 = malloc(size*size*sizeof(double));
		double *m2 = malloc(size*size*sizeof(double));
		double *res = malloc(size*size*sizeof(double));

		for(int at=0; at<10; at++){


			for(int i=0; i<size; i++){
				matA[i] = malloc(size*sizeof(double));
				matB[i] = malloc(size*sizeof(double));
				result[i] = malloc(size*sizeof(double));

				for(int j=0; j<size; j++){
					 matA[i][j] = (double)(rand()*5.0/RAND_MAX);
					 matB[i][j] = (double)(rand()*5.0/RAND_MAX);
					 result[i][j] = 0;

				}
			}


		    for(int i=0; i<size*size; i++){
		    	m1[i] = (double)(rand()*5.0/RAND_MAX);
		    	m2[i] = (double)(rand()*5.0/RAND_MAX);
		    	res[i] = 0;
		    }

		    gsl_matrix_view M1 = gsl_matrix_view_array(m1, size, size);
			gsl_matrix_view M2 = gsl_matrix_view_array(m2, size, size);
			gsl_matrix_view RES = gsl_matrix_view_array(res, size, size);

			blasStart = clock();
			gsl_blas_dgemm (CblasNoTrans, CblasNoTrans,
	                  1.0, &M1.matrix, &M2.matrix,
	                  0.0, &RES.matrix);
			blasEnd = clock();

			//printMatrix(matA, size);
			//printf("\n");
			//printMatrix(matB, size);
			//printf("\n");
			naiveStart = clock();
			naive_multiplication(matA, matB, result, size);
			naiveEnd = clock();

			for(int i=0; i<size; i++){
				for(int j=0; j<size; j++){
					 matA[i][j] = (double)(rand()*5.0/RAND_MAX);
					 matB[i][j] = (double)(rand()*5.0/RAND_MAX);
					 result[i][j] = 0;
				}
			}

			betterStart = clock();
			better_multiplication(matA, matB, result, size);
			betterEnd = clock();

			nDiv = ((double) (naiveEnd - naiveStart)) / CLOCKS_PER_SEC;
			btrDiv = ((double) (betterEnd - betterStart)) / CLOCKS_PER_SEC;
			blasDiv = ((double) (blasEnd - blasStart)) / CLOCKS_PER_SEC;

			fprintf(f, "%d, %f, %f, %f\n",size ,nDiv, btrDiv, blasDiv);
		}
		//printMatrix(result, size);
		free(matA);
		free(matB);
		free(result);
		free(m1);
		free(m2);
		free(res);
		
		}
	}







