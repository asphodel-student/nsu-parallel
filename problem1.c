#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

#define MAX_SIZE 10000000

void fillArr(double* arr, size_t len)
{
    double step = 3.141592653589783 * 2 / MAX_SIZE;
    
    #pragma acc data copyin(step)
    #pragma acc parallel loop vector vector_length(128) gang, present_or_copyin(arr)
    for(size_t i = 0; i < len; i++)
    {
        arr[i] = sin(step * i);
    }
}

//#pragma acc routine
double sumArr(double* arr, size_t len)
{
    double sum = 0.0;
    
    #pragma acc data copy(sum)
    #pragma acc parallel loop reduction(+:sum) present(arr)
    for(size_t i = 0; i < len; i++)
    {
        sum += arr[i];
    }

    return sum;
}


int main()
{
    double time_spent = 0.0;
    
    double* arr = (double*)malloc(MAX_SIZE * sizeof(double));
    #pragma acc data create(arr[0:MAX_SIZE])
    {
    clock_t begin = clock();

    fillArr(arr, MAX_SIZE);
    printf("Sum = %0.23lf\n", sumArr(arr, MAX_SIZE));

    clock_t end = clock();
    time_spent += (double)(end - begin) / CLOCKS_PER_SEC;
    printf("The elapsed time is %lf seconds\n", time_spent);

    free(arr);
    }
    return 0;
}
