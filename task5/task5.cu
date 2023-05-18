#include <iostream>
#include <cstring>
#include <cmath>
#include <ctime>
#include <iomanip>

#include <cuda_runtime.h>
#include <cub/cub.cuh>

#include <mpi.h>

#define CORNER1 10
#define CORNER2 20
#define CORNER3 30
#define CORNER4 20

#define GET_CUDA_STATUS(status) { gpuAssert((status), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t status, const char *file, int line)
{
   if (status != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(status), file, line);
      std::exit(status);
   }
}

#define GET_MPI_STATUS(status) { mpiAssert((status), __FILE__, __LINE__); }
inline void mpiAssert(int status, const char *file, int line)
{
	if (status != MPI_SUCCESS)
	{
		fprintf(stderr, "MPIassert: %s %s %d\n", status, file, line);
		std::exit(status);
	}
}

// Заводим глобальные указатели для матриц
double 	*matrixA 			= nullptr, 
		*matrixB			= nullptr,
	 	*deviceMatrixAPtr 	= nullptr, 
		*deviceMatrixBPtr	= nullptr, 
		*deviceError 		= nullptr, 
		*errorMatrix 		= nullptr, 
		*tempStorage 		= nullptr;

void freeMemoryHandler()
{
	if (deviceMatrixAPtr) 	cudaFree(deviceMatrixAPtr);
	if (deviceMatrixBPtr) 	cudaFree(deviceMatrixBPtr);
	if (errorMatrix)	  	cudaFree(errorMatrix);
	if (tempStorage) 		cudaFree(tempStorage);
	if (matrixA) 			cudaFree(matrixA);
	if (matrixB) 			cudaFree(matrixB);
}

#define CALCULATE(matrixA, matrixB, size, i, j) \
	matrixB[i * size + j] = 0.25 * (matrixA[i * size + j - 1] + matrixA[(i - 1) * size + j] + \
			matrixA[(i + 1) * size + j] + matrixA[i * size + j + 1]);	

__global__
void calculateBoundaries(double* matrixA, double* matrixB, size_t size, size_t sizePerGpu)
{
	unsigned int idxUp = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int idxDown = blockIdx.x * blockDim.x + threadIdx.x;

	if (idxUp == 0 || idxUp > size - 2) return;
	
	if(idxUp < size)
	{
		CALCULATE(matrixA, matrixB, size, 1, idxUp);
		CALCULATE(matrixA, matrixB, size, (sizePerGpu - 2), idxDown);
	}
}

// Главная функция - расчёт поля 
__global__
void calculateMatrix(double* matrixA, double* matrixB, size_t size, size_t sizePerGpu)
{
	unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int i = blockIdx.y * blockDim.y + threadIdx.y;
	
	if(!(j < 1 || i < 2 || j > size - 2 || i > sizePerGpu - 2))
	{
		CALCULATE(matrixA, matrixB, size, i, j);
	}
}

// Функция, подсчитывающая разницу матриц
__global__
void getErrorMatrix(double* matrixA, double* matrixB, double* outputMatrix, size_t size, size_t sizePerGpu)
{
	unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int i = blockIdx.y * blockDim.y + threadIdx.y;

	size_t idx = i * size + j;
	if(!(j == 0 || i == 0 || j == size - 1 || i == sizePerGpu - 1))
	{
		outputMatrix[idx] = std::abs(matrixB[idx] - matrixA[idx]);
	}
}

int findNearestPowerOfTwo(size_t num) {
    int power = 1;
    while (power < num) {
        power <<= 1;
    }
    return power;
}

int main(int argc, char** argv)
{
	auto atExifStatus = std::atexit(freeMemoryHandler);
	if (atExifStatus != 0)
	{
		std::cout << "Register error" << std::endl;
		exit(-1);
	}

	if (argc != 4)
	{
		std::cout << "Invalid parameters" << std::endl;
		std::exit(-1);
	}

	// Получаем значения из командной строки
	const double minError = std::pow(10, -std::stoi(argv[1]));
	const int size = std::stoi(argv[2]);
	const int maxIter = std::stoi(argv[3]);
	const size_t totalSize = size * size;

	int rank, sizeOfTheGroup;
    GET_MPI_STATUS(MPI_Init(&argc, &argv));
    GET_MPI_STATUS(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    GET_MPI_STATUS(MPI_Comm_size(MPI_COMM_WORLD, &sizeOfTheGroup));

	int numOfDevices = 0;
	cudaGetDeviceCount(&numOfDevices);
	if (sizeOfTheGroup > numOfDevices || sizeOfTheGroup < 1)
	{
		std::cout << "Invalid number of devices!";
		std::exit(-1);
	}

	GET_CUDA_STATUS(cudaSetDevice(rank));

	if (rank == 0)
	{
		std::cout << "Parameters: " << std::endl <<
		"Min error: " << minError << std::endl <<
		"Maximal number of iteration: " << maxIter << std::endl <<
		"Grid size: " << size << std::endl;
	}

	// Размечаем границы между устройствами
	size_t sizeOfAreaForOneProcess = size / sizeOfTheGroup;
	size_t startYIdx = sizeOfAreaForOneProcess * rank;

	// Выделение памяти на хосте
    GET_CUDA_STATUS(cudaMallocHost(&matrixA, sizeof(double) * totalSize));
    GET_CUDA_STATUS(cudaMallocHost(&matrixB, sizeof(double) * totalSize));

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
	GET_CUDA_STATUS(cudaMalloc((void**)&deviceMatrixAPtr, sizeOfAllocatedMemory * sizeof(double)));
	GET_CUDA_STATUS(cudaMalloc((void**)&deviceMatrixBPtr, sizeOfAllocatedMemory * sizeof(double)));
	GET_CUDA_STATUS(cudaMalloc((void**)&errorMatrix, sizeOfAllocatedMemory * sizeof(double)));
	GET_CUDA_STATUS(cudaMalloc((void**)&deviceError, sizeof(double)));

	// Копируем часть заполненной матрицы в выделенную память, начиная с 1 строки
	size_t offset = (rank != 0) ? size : 0;
 	GET_CUDA_STATUS(cudaMemcpy(deviceMatrixAPtr, matrixA + (startYIdx * size) - offset, 
					sizeof(double) * sizeOfAllocatedMemory, cudaMemcpyHostToDevice));
	GET_CUDA_STATUS(cudaMemcpy(deviceMatrixBPtr, matrixB + (startYIdx * size) - offset, 
					sizeof(double) * sizeOfAllocatedMemory, cudaMemcpyHostToDevice));

	// Здесь мы получаем размер временного буфера для редукции и выделяем память для этого буфера
	size_t tempStorageSize = 0;
	cub::DeviceReduce::Max(tempStorage, tempStorageSize, errorMatrix, deviceError, size * sizeOfAreaForOneProcess);
	GET_CUDA_STATUS(cudaMalloc((void**)&tempStorage, tempStorageSize));

	double* error;
	cudaMallocHost(&error, sizeof(double));
	*error = 1.0;

	cudaStream_t stream, matrixCalculationStream;
	GET_CUDA_STATUS(cudaStreamCreate(&stream));
	GET_CUDA_STATUS(cudaStreamCreate(&matrixCalculationStream));

	unsigned int threads_x = std::min(findNearestPowerOfTwo(size), 1024);
    unsigned int blocks_y = sizeOfAreaForOneProcess;
    unsigned int blocks_x = size / threads_x;

    dim3 blockDim(threads_x, 1);
    dim3 gridDim(blocks_x, blocks_y);


	int iter = 0; 
	
	// Главный алгоритм 
	clock_t begin = clock();
	while((iter < maxIter) && (*error) > minError)
	{
		iter++;

		// Расчитываем границы, которые потом будем отправлять другим процессам
		calculateBoundaries<<<size, 1, 0, stream>>>(deviceMatrixAPtr, deviceMatrixBPtr, 
										size, sizeOfAreaForOneProcess);

		cudaStreamSynchronize(stream);
		// Расчет матрицы
		calculateMatrix<<<gridDim, blockDim, 0, matrixCalculationStream>>>
							(deviceMatrixAPtr, deviceMatrixBPtr, size, sizeOfAreaForOneProcess);


		// Расчитываем ошибку каждую сотую итерацию
		if (iter % 100 == 0)
		{
			getErrorMatrix<<<gridDim, blockDim, 0, matrixCalculationStream>>>(deviceMatrixAPtr, deviceMatrixBPtr, errorMatrix,
															size, sizeOfAreaForOneProcess);

			cub::DeviceReduce::Max(tempStorage, tempStorageSize, errorMatrix, deviceError, sizeOfAllocatedMemory, matrixCalculationStream);
			
			GET_CUDA_STATUS(cudaStreamSynchronize(matrixCalculationStream));
			
			// Находим максимальную ошибку среди всех и передаём её всем процессам
			GET_MPI_STATUS(MPI_Allreduce((void*)deviceError, (void*)deviceError, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD));

			GET_CUDA_STATUS(cudaMemcpyAsync(error, deviceError, sizeof(double), cudaMemcpyDeviceToHost, matrixCalculationStream));
		}
		
		// Обмен "граничными" условиями каждой области
		// Обмен верхней границей
		if (rank != 0)
		{
		    GET_MPI_STATUS(MPI_Sendrecv(deviceMatrixBPtr + size + 1, size - 2, MPI_DOUBLE, rank - 1, 0, 
			deviceMatrixBPtr + 1, size - 2, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
		}
		// Обмен нижней границей
		if (rank != sizeOfTheGroup - 1)
		{
		    GET_MPI_STATUS(MPI_Sendrecv(deviceMatrixBPtr + (sizeOfAreaForOneProcess - 2) * size + 1, size - 2, MPI_DOUBLE, rank + 1, 0,
							deviceMatrixBPtr + (sizeOfAreaForOneProcess - 1) * size + 1, 
							size - 2, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
		}
	
		GET_CUDA_STATUS(cudaStreamSynchronize(matrixCalculationStream));
		// Обмен указателей
		std::swap(deviceMatrixAPtr, deviceMatrixBPtr);
	}

	clock_t end = clock();
	if (rank == 0)
	{
		std::cout << "Time: " << 1.0 * (end - begin) / CLOCKS_PER_SEC << std::endl;
		std::cout << "Iter: " << iter << " Error: " << *error << std::endl;
	}

	GET_MPI_STATUS(MPI_Finalize());

	return 0;
}
