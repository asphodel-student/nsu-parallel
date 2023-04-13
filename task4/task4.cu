#include <iostream>
#include <cstring>
#include <cmath>
#include <ctime>

#include <cuda_runtime.h>
#include <cub/cub.cuh>

#define CORNER1 10
#define CORNER2 20
#define CORNER3 30
#define CORNER4 20


// Главная функция - расчёт поля 
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

// Функция, подсчитывающая разницу матриц
__global__
void getErrorMatrix(double* matrixA, double* matrixB, double* outputMatrix)
{
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

	if(!(blockIdx.x == 0 || threadIdx.x == 0))
	{
		outputMatrix[idx] = std::abs(matrixB[idx] - matrixA[idx]);
	}
}


int main(int argc, char** argv)
{
	// Получаем значения из коммандной строки
	const double minError = std::pow(10, -std::stoi(argv[1]));
	const int size = std::stoi(argv[2]);
	const int maxIter = std::stoi(argv[3]);
	const size_t totalSize = size * size;

	std::cout << "Parameters: " << std::endl <<
		"Min error: " << minError << std::endl <<
		"Maximal number of iteration: " << maxIter << std::endl <<
		"Grid size: " << size << std::endl;

	// Выделение памяти на хосте
	double* matrixA = new double[totalSize];
	double* matrixB = new double[totalSize];
	
	std::memset(matrixA, 0, totalSize * sizeof(double));

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

	// Выбор устройства
	cudaSetDevice(3);

	// Выделяем папять на девайсе и копируем память с хоста
	double* deviceMatrixAPtr, *deviceMatrixBPtr, *deviceError, *errorMatrix, *tempStorage = NULL;
	size_t tempStorageSize = 0;

	cudaError_t cudaStatus_1 = cudaMalloc((void**)(&deviceMatrixAPtr), sizeof(double) * totalSize);
	cudaError_t cudaStatus_2 = cudaMalloc((void**)(&deviceMatrixBPtr), sizeof(double) * totalSize);
	cudaMalloc((void**)&deviceError, sizeof(double));
	cudaStatus_1 = cudaMalloc((void**)&errorMatrix, sizeof(double) * totalSize);
	
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

	// Здесь мы получаем размер временного буфера для редукции
	cub::DeviceReduce::Max(tempStorage, tempStorageSize, errorMatrix, deviceError, totalSize);
	
	// Выделяем память для буфера
	cudaMalloc((void**)&tempStorage, tempStorageSize);

	int iter = 0; 
	double error = 1.0;

	// Главный алгоритм 
	clock_t begin = clock();
	while(iter < maxIter && error > minError)
	{
		iter++;
		// Расчет матрицы
		calculateMatrix<<<size - 1, size - 1>>>(deviceMatrixAPtr, deviceMatrixBPtr, size);

		// Расчитываем ошибку каждую сотую итерацию
		if(iter % 100 == 0)
		{
			getErrorMatrix<<<size - 1, size - 1>>>(deviceMatrixAPtr, deviceMatrixBPtr, errorMatrix);
			cub::DeviceReduce::Max(tempStorage, tempStorageSize, errorMatrix, deviceError, totalSize);
			cudaMemcpy(&error, deviceError, sizeof(double), cudaMemcpyDeviceToHost);
		}
		
		// Обмен указателей
		std::swap(deviceMatrixAPtr, deviceMatrixBPtr);
	}

	clock_t end = clock();
	std::cout << "Time: " << 1.0 * (end - begin) / CLOCKS_PER_SEC << std::endl;
	std::cout << "Iter: " << iter << " Error: " << error << std::endl;

	// Высвобождение памяти
	cudaFree(deviceMatrixAPtr);
	cudaFree(deviceMatrixBPtr);
	cudaFree(errorMatrix);
	cudaFree(tempStorage);

	delete[] matrixA;
	delete[] matrixB;

	return 0;
}
