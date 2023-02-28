#include <math.h>
#include <string.h>

#define gridSize 128

int main(int argc, char** argv)
{
    const int iter_max = 1000;
    
    const double tol = 1.0e-6;
    double error = 1.0;
    
     // Allocate two 2D arrays
    double** matrixA = (double**)calloc(1, sizeof(double*));
    double** matrixB = (double**)calloc(1, sizeof(double*));
    for (size_t i = 0; i < gridSize; i++)
    {
        matrixA[i] = (double*)calloc(gridSize, sizeof(double));
        matrixB[i] = (double*)calloc(gridSize, sizeof(double));
    }
        
    for (int j = 0; j < gridSize; j++)
    {
        matrixA[j][0] = 1.0;
        matrixB[j][0] = 1.0;
    }

    int iter = 0;
    

    while ( error > tol && iter < iter_max )
    {
        error = 0.0;

#pragma acc kernels
        for( int j = 1; j < gridSize-1; j++)
        {
            for( int i = 1; i < gridSize-1; i++ )
            {
                matrixB[j][i] = 0.25 * ( matrixA[j][i+1] + matrixA[j][i-1]
                                    + matrixA[j-1][i] + matrixA[j+1][i]);
                error = fmax( error, fabs(matrixB[j][i] - matrixA[j][i]));
            }
        }
        
#pragma acc kernels
        for( int j = 1; j < gridSize-1; j++)
        {
            for( int i = 1; i < gridSize-1; i++ )
            {
                matrixA[j][i] = matrixB[j][i];    
            }
        }

        if(iter % 100 == 0) printf("%5d, %0.6f\n", iter, error);
        
        iter++;
    }

}