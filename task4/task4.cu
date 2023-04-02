#include <iostream>
#include <cstring>
#include <cmath>
#include <ctime>
#include <iomanip>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cub/cub.cuh>
#include <cub/block/block_load.cuh>
#include <cub/block/block_store.cuh>
#include <cub/block/block_reduce.cuh>

#define CORNER1 10
#define CORNER2 20
#define CORNER3 30
#define CORNER4 20

__global__
void calculateMatrix(double* matrixA, double* matrixB, size_t size)
{
	size_t i = blockIdx.x;
	size_t j = threadIdx.x;

	if(!(blockIdx.x == 0 || threadIdx.x == 0))
	{
		matrixB[i * size + j] = 0.25 * (matrixA[i * size + j - 1] + matrixA[(i - 1) * size + j] +
							matrixA[(i + 1) * size + j] + matrixA[i * size + j + 1]);		
	}
}

/*template <int BLOCK_THREADS, 
int ITEMS_PER_THREAD,
cub::BlockReduceAlgorithm ALGORITHM>
__global__ void BlockReduceKernel(double *matrixA, double* matrixB, double *d_out)    
{
    
    typedef BlockReduce<double, BLOCK_THREADS, ALGORITHM> BlockReduceT;
    
    __shared__ typename BlockReduceT::TempStorage temp_storage;
   
    double data_1[ITEMS_PER_THREAD];
	double data_2[ITEMS_PER_THREAD];

    LoadDirectStriped<BLOCK_THREADS>(threadIdx.x, matrixA, data_1);
	LoadDirectStriped<BLOCK_THREADS>(threadIdx.x, matrixB, data_2);
   
	for (size_t i = 0; i < ITEMS_PER_THREAD; i++)
	{
		data_1[i] =  std::abs(matrixB[i] - matrixA[i]);
		data_1
	}

    // Compute sum
    int aggregate = BlockReduceT(temp_storage).Sum(data);
   
    // Store aggregate and elapsed clocks
    if (threadIdx.x == 0)
    {
        *d_out = aggregate;
    }
}

struct CustomMax
{
	__device__ __forceinline__
	double operator()(double &a, double &b)
	{
		return (a < b) ? a : b;
	}
};
*/

int main(int argc, char** argv)
{
	const double minError = std::pow(10, -std::stoi(argv[1]));
	const int size = std::stoi(argv[2]);
	const int maxIter = std::stoi(argv[3]);
	const size_t totalSize = size * size;

	std::cout << "Parameters: " << std::endl <<
		"Min error: " << minError << std::endl <<
		"Maximal number of iteration: " << maxIter << std::endl <<
		"Grid size: " << size << std::endl;

	double* matrixA = new double[totalSize];
	double* matrixB = new double[totalSize];
	
	std::memset(matrixA, 0, totalSize * sizeof(double));

	// Adding the border conditions
	matrixA[0] = CORNER1;
	matrixA[size - 1] = CORNER2;
	matrixA[size * size - 1] = CORNER3;
	matrixA[size * (size - 1)] = CORNER4;

	const double step = 1.0 * (CORNER2 - CORNER1) / (size - 1);
	for (int i = 1; i < size - 1; i++)
	{
		matrixA[i] = CORNER1 + i * step;
		matrixA[i * size] = CORNER1 + i * step;
		matrixA[size - 1 + i * size] = CORNER2 + i * step;
		matrixA[size * (size - 1) + i] = CORNER4 + i * step;
	}

	std::memcpy(matrixB, matrixA, totalSize * sizeof(double));

	cudaSetDevice(3);

	double* deviceMatrixAPtr, *deviceMatrixBPtr, *deviceErrorsMatrix; 
	cudaError_t cudaStatus_1 = cudaMalloc((void**)(&deviceMatrixAPtr), sizeof(double) * totalSize);
	cudaError_t cudaStatus_2 = cudaMalloc((void**)(&deviceMatrixBPtr), sizeof(double) * totalSize);
	
	if (cudaStatus_1 != 0 || cudaStatus_2 != 0)
	{
		std::cout << "Memory allocation error" << std::endl;
		return -1;
	}

	cudaStatus_1 = cudaMemcpy(deviceMatrixAPtr, matrixA, sizeof(double) * totalSize, cudaMemcpyHostToDevice);
	cudaStatus_2 = cudaMemcpy(deviceMatrixBPtr, matrixB, sizeof(double) * totalSize, cudaMemcpyHostToDevice);

	if (cudaStatus_1 != 0 || cudaStatus_2 != 0)
	{
		std::cout << "Memory transfering error" << std::endl;
		return -1;
	}

	cublasHandle_t handler;
	cublasStatus_t status;
	status = cublasCreate(&handler);

	int iter = 0, idx = 0, storageSize = 0; 
	double error = 1.0, alpha = -1.0;

	clock_t begin = clock();
	while(iter < maxIter && error > minError)
	{
		iter++;

		calculateMatrix<<<size - 1, size - 1>>>(deviceMatrixAPtr, deviceMatrixBPtr, size);

		if(iter % 100 == 0)
		{
			status = cublasDaxpy(handler, size * size, &alpha, deviceMatrixBPtr, 1, deviceMatrixAPtr, 1);
			status = cublasIdamax(handler, size * size, deviceMatrixAPtr, 1, &idx);
			cudaMemcpy(&error, &deviceMatrixAPtr[idx - 1], sizeof(double), cudaMemcpyDeviceToHost);
			error = std::abs(error);
			status = cublasDcopy(handler, size * size, deviceMatrixBPtr, 1, deviceMatrixAPtr, 1);
		}

		std::swap(deviceMatrixAPtr, deviceMatrixBPtr);
	}

	clock_t end = clock();
	std::cout << "Time: " << 1.0 * (end - begin) / CLOCKS_PER_SEC << std::endl;
	std::cout << "Iter: " << iter << " Error: " << error << std::endl;

	cudaStatus_1 = cudaMemcpy((void*)matrixA, (void*)deviceMatrixAPtr, sizeof(double) * totalSize, cudaMemcpyDeviceToHost);

	std::cout << cudaStatus_1 << std::endl;

	/*for (size_t i = 0; i < size; i++)
	{
		for (size_t j = 0; j < size; j++)
		{
			std::cout << matrixA[i * size + j] << " ";
		}

		std::cout << std::endl;
	}*/

	cudaFree(deviceMatrixAPtr);
	cudaFree(deviceMatrixBPtr);

	delete[] matrixA;
	delete[] matrixB;

	return 0;
}
