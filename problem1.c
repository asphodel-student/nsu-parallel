#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

#define MAX_SIZE 10000000

void fillArr(double* arr, size_t len)
{
    double step = 3.1415 * 2 / MAX_SIZE;
    double phi = 0;

    #pragma acc parallel loop
    for(size_t i = 0; i < MAX_SIZE; i++)
    {
        arr[i] = sin(phi);
        phi += step;
    }
}

int main()
{
    double time_spent = 0.0;
    clock_t begin = clock();

    double* arr = (double*)calloc(MAX_SIZE, sizeof(double));
    fillArr(arr, MAX_SIZE);

    clock_t end = clock();
    time_spent += (double)(end - begin) / CLOCKS_PER_SEC;
    printf("The elapsed time is %f seconds\n", time_spent);

    return 0;
}