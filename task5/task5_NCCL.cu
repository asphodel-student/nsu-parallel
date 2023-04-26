#include <iostream>
#include <cstring>
#include <cmath>
#include <ctime>
#include <iomanip>

#include <cuda_runtime.h>
#include <cub/cub.cuh>

#include "nccl.h"
#include "mpi.h"

#define CORNER1 10
#define CORNER2 20
#define CORNER3 30
#define CORNER4 20


// Главная функция - расчёт поля 
__global__
void calculateMatrix(double* matrixA, double* matrixB, size_t size, size_t sizePerGpu)
{
	unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int i = blockIdx.y * blockDim.y + threadIdx.y;
	
	if(!(j == 0 || i == 0 || j == size - 1 || i == sizePerGpu - 1))
	{
		matrixB[i * size + j] = 0.25 * (matrixA[i * size + j - 1] + matrixA[(i - 1) * size + j] +
							matrixA[(i + 1) * size + j] + matrixA[i * size + j + 1]);	
	}
}

// Функция, подсчитывающая разницу матриц
__global__
void getErrorMatrix(double* matrixA, double* matrixB, double* outputMatrix, size_t size)
{
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

	outputMatrix[idx] = std::abs(matrixB[idx] - matrixA[idx]);
}

int main(int argc, char** argv)
{
	int rank, sizeOfTheGroup;
    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &sizeOfTheGroup);

	cudaSetDevice(rank);

       	// Получаем значения из командной строки
	const double minError = std::pow(10, -std::stoi(argv[1]));
	const int size = std::stoi(argv[2]);
	const int maxIter = std::stoi(argv[3]);
	const size_t totalSize = size * size;

	if (rank == 0)
	{
		std::cout << "Parameters: " << std::endl <<
		"Min error: " << minError << std::endl <<
		"Maximal number of iteration: " << maxIter << std::endl <<
		"Grid size: " << size << std::endl;
	}

    if (rank!=0)
        cudaDeviceEnablePeerAccess(rank - 1, 0);
    if (rank!=sizeOfTheGroup-1)
        cudaDeviceEnablePeerAccess(rank + 1, 0);

    ncclUniqueId id;
    ncclComm_t comm;
    if (rank == 0) ncclGetUniqueId(&id);
    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
    ncclCommInitRank(&comm, sizeOfTheGroup, id, rank);

	// Размечаем границы между устройствами
	size_t sizeOfAreaForOneProcess = size / sizeOfTheGroup;
	size_t startYIdx = sizeOfAreaForOneProcess * rank;

	// Выделение памяти на хосте
	double* matrixA = new double[totalSize];
	double* matrixB = new double[totalSize];

	std::memset(matrixA, 0, size * size * sizeof(double));

	// Заполнение граничных условий
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

	double* deviceMatrixAPtr, *deviceMatrixBPtr, *deviceError, *errorMatrix, *tempStorage = NULL;

	// Расчитываем, сколько памяти требуется процессу
	if (rank != 0 && rank != sizeOfTheGroup - 1)
	{
		sizeOfAreaForOneProcess += 2;
	}
	else 
	{
		sizeOfAreaForOneProcess += 1;
	}

	size_t sizeOfAllocatedMemory = size * sizeOfAreaForOneProcess;

	// Выделяем память на девайсе
	cudaMalloc((void**)&deviceMatrixAPtr, sizeOfAllocatedMemory * sizeof(double));
	cudaMalloc((void**)&deviceMatrixBPtr, sizeOfAllocatedMemory * sizeof(double));
	cudaMalloc((void**)&errorMatrix, sizeOfAllocatedMemory * sizeof(double));
	cudaMalloc((void**)&deviceError, sizeof(double));

	// Копируем часть заполненной матрицы в выделенную память, начиная с 1 строки
	size_t offset = (rank != 0) ? size : 0;
 	cudaMemcpy(deviceMatrixAPtr, matrixA + (startYIdx * size) - offset, sizeof(double) * sizeOfAllocatedMemory, cudaMemcpyHostToDevice);
	cudaMemcpy(deviceMatrixBPtr, matrixB + (startYIdx * size) - offset, sizeof(double) * sizeOfAllocatedMemory, cudaMemcpyHostToDevice);

	// Здесь мы получаем размер временного буфера для редукции и выделяем память для этого буфера
	size_t tempStorageSize = 0;
	cub::DeviceReduce::Max(tempStorage, tempStorageSize, errorMatrix, deviceError, size * sizeOfAreaForOneProcess);
	cudaMalloc((void**)&tempStorage, tempStorageSize);

	int iter = 0; 
	double error = 1.0;

    unsigned int threads_x = (size < 1024) ? size : 1024;
    unsigned int blocks_y = sizeOfAreaForOneProcess;
    unsigned int blocks_x = size / threads_x;

    dim3 blockDim(threads_x, 1);
    dim3 gridDim(blocks_x, blocks_y);

    cudaStream_t stream, memoryStream;
    cudaStreamCreate(&stream);
    cudaStreamCreate(&memoryStream);

	// Главный алгоритм 
	clock_t begin = clock();
	while(iter < maxIter && error > minError)
	{
		iter++;

		// Расчет матрицы
		calculateMatrix<<<gridDim, blockDim, 0, stream>>>(deviceMatrixAPtr, deviceMatrixBPtr, size, sizeOfAreaForOneProcess);
		
		// Расчитываем ошибку каждую сотую итерацию
		if (iter % 100 == 0)
		{
			getErrorMatrix<<<blocks_x * blocks_y, threads_x,  0, stream>>>(deviceMatrixAPtr, deviceMatrixBPtr, errorMatrix, size);
            //cudaStreamSynchronize(stream);
			cub::DeviceReduce::Max(tempStorage, tempStorageSize, errorMatrix, deviceError, sizeOfAllocatedMemory, stream);

            ncclGroupStart();
            ncclAllReduce((void*)deviceError, (void*)deviceError, 1, ncclDouble, ncclMax, comm, stream);
            ncclGroupEnd();

            cudaMemcpyAsync(&error, deviceError, sizeof(double), cudaMemcpyDeviceToHost, stream);
        }
		
		// Обмен "граничными" условиями каждой области
		// Обмен верхней границей
        ncclGroupStart();
        if (rank != 0)
		{
            ncclSend(deviceMatrixBPtr + size + 1, size - 2, ncclDouble, rank - 1, comm, stream); 
            ncclRecv(deviceMatrixBPtr + 1, size - 2, ncclDouble, rank - 1, comm, stream);

		}
        // Обмен нижней границей
		if (rank != sizeOfTheGroup - 1)
		{
            ncclSend(deviceMatrixBPtr + (sizeOfAreaForOneProcess - 2) * size + 1, 
				size - 2, ncclDouble, rank + 1, comm, stream);
            ncclRecv(deviceMatrixBPtr + (sizeOfAreaForOneProcess - 1) * size + 1, 
				size - 2, ncclDouble, rank + 1, comm, stream);
		}
        ncclGroupEnd();

		// Обмен указателей
		std::swap(deviceMatrixAPtr, deviceMatrixBPtr);
	}

	clock_t end = clock();
	if (rank == 0)
	{
		std::cout << "Time: " << 1.0 * (end - begin) / CLOCKS_PER_SEC << std::endl;
		std::cout << "Iter: " << iter << " Error: " << error << std::endl;
	}

	// Высвобождение памяти
	cudaFree(deviceMatrixAPtr);
	cudaFree(deviceMatrixBPtr);
	cudaFree(errorMatrix);
	cudaFree(tempStorage);

	delete[] matrixA;
	delete[] matrixB;

	MPI_Finalize();

	return 0;
}
