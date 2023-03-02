#include <iostream>
#include <cstring>
#include <cmath>
#include <ctime>

#define NUM_OF_PARAM 4

#define CORNER1 10
#define CORNER2 20
#define CORNER3 30
#define CORNER4 20

double matrixA[4096][4096] = { 0 };
double matrixB[4096][4096] = { 0 };

// Function for computing the Poisson equation
double computeArray(size_t gridSize, double error)
{
   error = 0.0;

    #pragma acc parallel loop seq vector vector_length(256) gang num_gangs(256) reduction(max:error) \
	present(matrixA[0:gridSize][0:gridSize], matrixB[0:gridSize][0:gridSize]) 
    for (size_t i = 1; i < gridSize - 1; i++)
    {
        for (size_t j = 1; j < gridSize - 1; j++)
        {
            matrixB[i][j] = 0.25 * (matrixA[i + 1][j] + matrixA[i - 1][j] + matrixA[i][j - 1] + matrixA[i][j + 1]);
            error = fmax(error, fabs(matrixB[i][j] - matrixA[i][j]));
        }
    }

    return error;
}

// Function for array updating
void updateArrays(size_t gridSize)
{
    #pragma acc parallel loop seq vector vector_length(256) gang num_gangs(256) \
	present(matrixA[0:gridSize][0:gridSize], matrixB[0:gridSize][0:gridSize])
    for (size_t i = 1; i < gridSize; i++)
    {
        for (size_t j = 0; j < gridSize; j++)
        {
            matrixA[i][j] = matrixB[i][j];
        }
    }
}

int main(int argc, char** argv)
{
    if (argc < NUM_OF_PARAM)
    {
        std::cout << "Number of argument should be 3." << std::endl;
        std::cout << "Arguments: accuracy, grid size, number of iterations" << std::endl;
        return -1;
    }

    double minError = std::pow(10, -std::stoi(argv[1]));
    int gridSize = std::stoi(argv[2]);
    int numOfIter = std::stoi(argv[3]);

    if (gridSize < 128 || gridSize > 4096 || numOfIter < 100 || numOfIter > 10000000) return -1;

    std::cout << "Min error " << minError << std::endl;
    std::cout << "Grid size: " << gridSize << std::endl;
    std::cout << "Max number of iteration: " << numOfIter << std::endl;

    // Adding a boundary conditions
    matrixA[0][0] = matrixB[0][0] = CORNER1;
    matrixA[0][gridSize - 1] = matrixB[0][gridSize - 1] = CORNER2;
    matrixA[gridSize - 1][0] = matrixB[gridSize - 1][0] = CORNER4;
    matrixA[gridSize - 1][gridSize - 1] = matrixB[gridSize - 1][gridSize - 1] = CORNER3;

    double step = 1.0 * (CORNER2 - CORNER1) / gridSize;

    clock_t initBegin = clock();
      
    #pragma acc data copy (matrixB[0:gridSize][0:gridSize]), copy (matrixA[0:gridSize][0:gridSize]) 
    {
    #pragma acc parallel loop seq gang num_gangs(256) vector vector_length(256)
    for (size_t i = 1; i < gridSize - 1; i++)
    {
        matrixA[0][i] = matrixB[0][i] = CORNER1 + step * i;
        matrixA[i][0] = matrixB[i][0] = CORNER1 + step * i;
        matrixA[gridSize - 1][i] = matrixB[gridSize - 1][i] = CORNER3 + step * i;
        matrixA[i][gridSize - 1] = matrixB[i][gridSize - 1] = CORNER2 + step * i;
    }

    clock_t initEnd = clock();
    std::cout << "Initialization time: " << 1.0 * (initEnd - initBegin) / CLOCKS_PER_SEC << std::endl;

    clock_t algBegin = clock();


    // Main algorithm
    std::cout << "-----------Start-----------" << std::endl;
    double error = 1.0; int iter = 0;
    while (minError < error && iter < numOfIter)
    {
        error = computeArray(gridSize, error);
        updateArrays(gridSize);
	iter++;
    }

    clock_t algEnd = clock();

    std::cout << "Number of iteration: " << iter << ", error:  " << error << std::endl;
    std::cout << "Time of computation: " << 1.0 * (algEnd - algBegin) / CLOCKS_PER_SEC << std::endl;
    }
    
    return 0;
}
