#include "../Inc/Functions.cuh"

// CUDA kernel to apply the sigmoid function to each element in the array
__global__
void _sigmoid(float* data, int size)
{
    size_t x = blockDim.x * blockIdx.x + threadIdx.x;

    if (x < size)
    {
        // Apply the sigmoid function to the current element
        data[x] = 1.0 / (1.0 + expf(-data[x]));
    }
}

// Host function to call the sigmoid kernel
__host__
void sigmoid(float* data, int size)
{
    // Set the number of threads per block
    size_t threads = 16;

    // Calculate the number of blocks needed based on the array size
    size_t blocks = std::ceil(1.0 * size / threads);

    // Set the block size and grid size for the kernel
    dim3 blockSize(threads);
    dim3 gridSize(blocks);

    // Launch the sigmoid kernel
    _sigmoid<<<gridSize, blockSize>>>(data, size);
}
