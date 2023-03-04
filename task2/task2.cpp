#include <iostream>
#include <cstring>
#include <cmath>
#include <ctime>

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

	matrixA[0] = CORNER1;
	matrixA[size - 1] = CORNER2;
	matrixA[size * size - 1] = CORNER3;
	matrixA[size * (size - 1) - 1] = CORNER4;


	size_t totalSize = size * size;
	const double step = (CORNER2 - CORNER1) / size;
//	#pragma acc enter data copyin(matrixA[0:totalSize]) create(matrixB[0:totalSize])
//	#pragma acc parallel loop 
	for (int i = 1; i < size - 1; i++)
	{	matrixA[i] = CORNER1 + i * step;
		matrixA[i * size] = CORNER1 + i * step;
		matrixA[(size - 1) * i] = CORNER2 + i * step;
		matrixA[size * (size - 1) + i] = CORNER4 + i * step;
	}

	std::memcpy(matrixB, matrixA, size * size * sizeof(double));


	double error = 1.0;
	int iter = 0;

	std::cout << "Start: " << std::endl;

	#pragma acc enter data copyin(matrixA[0:totalSize], matrixB[0:totalSize], error)
	{
	clock_t begin = clock();
	while (error > minError && iter < maxIter)
	{
	        iter++;

		if(iter % 100 == 0)
		{
		#pragma acc kernels async(1)
		error = 0.0;
		#pragma acc update device(error) async(1)
		}

		#pragma acc data present(matrixA, matrixB, error)
		#pragma acc parallel loop independent collapse(2) vector vector_length(256) gang num_gangs(256) reduction(max:error) async(1)
		for (int i = 1; i < size - 1; i++)
		{
			for (int j = 1; j < size - 1; j++)
			{
				matrixB[i * size + j] = 0.25 * 
					(matrixA[i * size + j - 1] +
					matrixA[(i - 1) * size + j] +
					matrixA[(i + 1) * size + j] +
					matrixA[i * size + j + 1]);

					error = fmax(error, matrixB[i * size + j] - matrixA[i * size + j]);
			}
		}
	if(iter % 100 == 0)
	{
		#pragma acc update host(error) async(1)	
		#pragma acc wait(1)
	}
		double* temp = matrixA;
		matrixA = matrixB;
		matrixB = temp;
	}
	clock_t end = clock();
	std::cout << "Time: " << 1.0 * (end - begin) / CLOCKS_PER_SEC << std::endl; 
	}


	std::cout << "Iter: " << iter << " Error: " << error << std::endl;

	return 0;
}
