#include <iostream>
#include <string>

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

	const double step = (CORNER2 - CORNER1) / size;
	for (int i = 1; i < size - 1; i++)
	{
		matrixA[i] = CORNER1 + i * step;
		matrixA[i * size] = CORNER1 + i * step;
		matrixA[(size - 1) * i] = CORNER2 + i * step;
		matrixA[size * (size - 1) + i] = CORNER4 + i * step;
	}

	std::memcpy(matrixB, matrixA, size * size * sizeof(double));

	double error = 1.0;
	int iter = 0;

	while (error > minError && iter < maxIter)
	{
		error = 0.0;
		iter++;
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

		double* temp = matrixA;
		matrixA = matrixB;
		matrixB = temp;
	}

	std::cout << "Iter: " << iter << " Error: " << error;

	return 0;
}
