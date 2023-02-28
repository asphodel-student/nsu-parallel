#include <iostream>
#include <string>
#include <cmath>

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

    double minError = std::pow(10, -std::stoi(argv[1]));

    std::cout << "Min error " << minError << std::endl; 
   
    size_t gridSize = std::pow(std::stoi(argv[2]), 2);
    size_t numOfIter = std::stoi(argv[3]);

    std::cout << "Grid size: " << gridSize << std::endl;
    std::cout << "Max number of iteration: " << numOfIter << std::endl;

    // Allocate two 2D arrays
    double** matrixA = new double*[gridSize];
    for(size_t i = 0; i < gridSize; i++)
        matrixA[i] = new double[gridSize];

    double** matrixB = new double*[gridSize];
    for(size_t i = 0; i < gridSize; i++)
        matrixB[i] = new double[gridSize];

    matrixA[0][0] = matrixB[0][0] = CORNER1;
    matrixA[0][gridSize - 1] = matrixB[0][gridSize - 1] = CORNER2;
    matrixA[gridSize - 1][0] = matrixB[gridSize - 1][0] = CORNER4;
    matrixA[gridSize - 1][gridSize - 1] = matrixB[gridSize - 1][gridSize - 1] = CORNER3;

    double step1 = 1.0 * (CORNER2 - CORNER1) / gridSize, //From (0, 0) to (gridSize, 0)
	   step2 = 1.0 * (CORNER1 - CORNER4) / gridSize, //From (0, 0) to (0, gridSize) 
	   step3 = 1.0 * (CORNER2 - CORNER3) / gridSize, //From (gridSize, 0) to (gridSize, gridSize)
	   step4 = 1.0 * (CORNER3 - CORNER4) / gridSize; //From (0, gridSize) to (gridSize, gridSize)

    // Writing a boundary condition
    for(size_t i = 1; i < gridSize - 1; i++)
    {
        matrixA[0][i] = step1 * i;
	matrixA[i][0] = step2 * i;
	matrixA[gridSize - 1][i] = step4 * i;
        matrixA[i][gridSize - 1] =  step3 * i;
    }

    //for (size_t i = 0; i < gridSize; i++)
//	    std::cout << matrixA[0][i] << std::endl;

    std::cout << "Start" << std::endl;
    // Main algorithm
    
    double error = 1; int iter = 0;

    while(minError < error and iter < numOfIter)
    {
        for(size_t j = 1; j < gridSize - 1; j++)
        {
            for(size_t i = 1; i < gridSize - 1; i++)
            {
                matrixB[i][j] = 0.25 * (matrixA[j + 1][i] + matrixA[j - 1][i] + matrixA[j][i - 1] + matrixA[j][i + 1]);
                error = std::max(error, matrixB[j][i] - matrixA[j][i]);
            }
        }
        iter++;
	//std::cout << "Iteration: " << iter << std::endl;
    }

    std::cout << iter << " " << error << std::endl;

    // Free memory
    for(size_t i = 0; i < gridSize; i++)
        delete[] matrixA[i];

    for(size_t i = 0; i < gridSize; i++)
        delete[] matrixB[i];

    delete[] matrixA; delete[] matrixB;

    return 0;
}
