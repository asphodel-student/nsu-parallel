#include "../Inc/Functions.cuh"

__global__
void _sigmoid(float* data, size_t size)
{
    size_t x = blockDim.x * blockIdx.x + threadIdx.x;

    if (x < size)
    {
        data[x] = 1.0 / (1.0 + exp(data[x]));
    }
}

void sigmoid(float* data, size_t size)
{
    size_t threads = 16;
    size_t blocks = size / threads;

    dim3 blockSize(threads);
    dim3 gridSize(blocks);

    _sigmoid<<<gridDim, blockDim>>>(data, size);
}


