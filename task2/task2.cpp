#include <iostream>
#include <string>

#define NUM_OF_PARAM 4

#define CORNER1 10
#define CORNER2 20
#define CORNER3 30
#define CORNER4 20

int main(int argc, char** argv)
{
    if(argc < NUM_OF_PARAM)
    {
        std::cout << "Number of argument should be 3." << std::endl;
        std::cout << "Arguments: accuracy, grid size, number of iterations" << std::endl;
        return -1;
    }

    // Проверки  ...

    int maxError = std::stoi(argv[1]);
    size_t gridSize = std::stoi(argv[2]);
    size_t numOfIter = std::stoi(argv[3]);

    double** matrixA = new double*[gridSize];
    for(size_t i = 0; i < gridSize; i++)
        matrixA[i] = new double[gridSize];

    double** matrixB = new double*[gridSize];
    for(size_t i = 0; i < gridSize; i++)
        matrixB[i] = new double[gridSize];

    matrixA[0][0] = matrixB[0][0] = CORNER1;
    matrixA[0][gridSize - 1] = matrixB[0][gridSize - 1] = CORNER2;
    matrixA[gridSize - 1][0] = matrixB[gridSize - 1][0] = CORNER3;
    matrixA[gridSize - 1][gridSize - 1] = matrixB[gridSize - 1][gridSize - 1] = CORNER4;

    double step = 10.0 / gridSize;
    for(size_t i = 1; i < gridSize - 1; i++)
    {
        matrixA[i][0] = matrixA[0][i] = CORNER1 * step * i;
        matrixA[k][gridSize - 1] = matrixA[gridSize - 1][i] = CORNER2 * step * i;
    }

    int error = 0, iter = 0;
    while(maxError > error and iter < numOfIter)
    {
        for(size_t j = 1; j < gridSize - 1; j++)
        {
            for(size_t i = 1; i < gridSize - 1; i++)
            {
                matrixB[i][j] = matrixA[i + 1][j] + matrixA[i - 1][j] + matrixA[i][j - 1] + matrixA[i][j + 1];
                error = std::max(error, matrixB[i][j] - matrixA[i][j]);
            }
        }
        iter++;
    }

    std::cout << iter << " " << error << std::endl;

    for(size_t i = 0; i < gridSize; i++)
        delete[] matrixA[i];

    for(size_t i = 0; i < gridSize; i++)
        delete[] matrixB[i];

    delete[] matrixA; delete[] matrixB;

    return 0;
}