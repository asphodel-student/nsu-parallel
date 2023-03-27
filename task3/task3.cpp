#include <iostream>
#include <cstring>
#include <cmath>
#include <ctime>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CORNER1 10
#define CORNER2 20
#define CORNER3 30
#define CORNER4 20

int main(int argc, char** argv)
{
	const double minError = std::pow(10, -std::stoi(argv[1]));
	const int size = std::stoi(argv[2]);
	const int maxIter = std::stoi(argv[3]);

	std::cout << "Parameters: " << std::endl <<
		"Min error: " << minError << std::endl <<
		"Maximal number of iteration: " << maxIter << std::endl <<
		"Grid size: " << size << std::endl;

	double* matrixA = new double[size * size];
	double* matrixB = new double[size * size];
	
	std::memset(matrixA, 0, size * size * sizeof(double));

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
		matrixA[(size - 1) * i] = CORNER2 + i * step;
		matrixA[size * (size - 1) + i] = CORNER4 + i * step;
	}

	std::memcpy(matrixB, matrixA, size * size * sizeof(double));
	size_t totalSize = size * size;
	cublasHandle_t handler;
	cublasStatus_t status;
	cudaError err;
	double error = 1.0;
	int iter = 0, idx = 0;

	status = cublasCreate(&handler);

	std::cout << "Start: " << std::endl;

// Main algorithm
#pragma acc enter data copyin(matrixA[0:totalSize], matrixB[0:totalSize]) 
	{
		clock_t begin = clock();
		int idx = 0;
		double alpha = -1.0;

		while (error > minError && iter < maxIter)
		{
			iter++;

#pragma acc data present(matrixA, matrixB)
#pragma acc parallel loop independent collapse(2) vector vector_length(256) gang num_gangs(256) async
			for (int i = 1; i < size - 1; i++)
			{
				for (int j = 1; j < size - 1; j++)
				{
					matrixB[i * size + j] = 0.25 *
						(matrixA[i * size + j - 1] +
							matrixA[(i - 1) * size + j] +
							matrixA[(i + 1) * size + j] +
							matrixA[i * size + j + 1]);
				}
			}

			if (iter % 100 == 0)
			{
// Ищем максимум из разницы
#pragma acc data present (matrixA, matrixB) wait
#pragma acc host_data use_device(matrixA, matrixB)
				{
			status = cublasDaxpy(handler, size * size, &alpha, matrixB, 1, matrixA, 1);
			status = cublasIdamax(handler, size * size, matrixA, 1, &idx);
				}

// Возвращаем ошибку на host
#pragma acc update host(matrixA[idx - 1]) 
			error = std::abs(matrixA[idx - 1]);

// 'Восстанавливаем' матрицу A		
#pragma acc host_data use_device(matrixA, matrixB)
			status = cublasDcopy(handler, size * size, matrixB, 1, matrixA, 1);
			}

			double* temp = matrixA;
			matrixA = matrixB;
			matrixB = temp;
		}

		clock_t end = clock();
		std::cout << "Time: " << 1.0 * (end - begin) / CLOCKS_PER_SEC << std::endl;
	}


	std::cout << "Iter: " << iter << " Error: " << error << std::endl;

	delete[] matrixA;
	delete[] matrixB;

	return 0;
}
