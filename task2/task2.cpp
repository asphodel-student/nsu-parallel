#include <iostream>
#include <string>

#define NUM_OF_PARAM 4

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

    int** matrixA = new int*[gridSize];
    for(size_t i = 0; i < gridSize; i++)
        matrixA[i] = new int[gridSize];

    int** matrixB = new int*[gridSize];
    for(size_t i = 0; i < gridSize; i++)
        matrixB[i] = new int[gridSize];

    matrixA[0][0] = matrixB[0][0] = 10;
    matrixA[0][gridSize - 1] = matrixB[0][gridSize - 1] = 20;
    matrixA[gridSize - 1][0] = matrixB[gridSize - 1][0] = 30;
    matrixA[gridSize - 1][gridSize - 1] = matrixB[gridSize - 1][gridSize - 1] = 20;

    int error = 0, iter = 0;
    while(maxError > error || iter < numOfIter)
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