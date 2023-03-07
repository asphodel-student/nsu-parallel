#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

#define MAX_SIZE 10000000

#ifdef FLOAT_TYPE
#define TYPE float
#define FORMAT "%.23f"
#define SIN sinf
#else
#define TYPE double
#define FORMAT "%.23lf"
#define SIN sin
#endif

void fillArr(TYPE* arr, size_t len)
{
    TYPE step = 3.141592653589 * 2.0 / MAX_SIZE;
    
    #pragma acc parallel loop vector vector_length(128) gang num_gangs(1024) present(arr)
    for(size_t i = 0; i < len; i++)
    {
        arr[i] = SIN(step * i);
    }
}

TYPE sumArr(TYPE* arr, size_t len)
{
    TYPE sum = 0.0;
    
    #pragma acc data copy(sum)
    #pragma acc parallel loop gang num_gangs(1024) reduction(+:sum) present(arr)
    for(size_t i = 0; i < len; i++)
    {
        sum += arr[i];
    }

    return sum;
}


int main()
{
    double time_spent = 0.0;
    clock_t begin = clock();

    TYPE* arr = (TYPE*)malloc(MAX_SIZE * sizeof(TYPE));
    #pragma acc data create(arr[0:MAX_SIZE])
    {
    fillArr(arr, MAX_SIZE);
    printf("Sum = ");
    printf(FORMAT, sumArr(arr, MAX_SIZE));

    clock_t end = clock();
    time_spent += (double)(end - begin) / CLOCKS_PER_SEC;
    printf("\nThe elapsed time is %lf seconds\n", time_spent);

    free(arr);
    }
    return 0;
}
