#include <iostream>
#include <cstring>
#include <cmath>
#include <ctime>

#define NUM_OF_PARAM 4

#define CORNER1 10
#define CORNER2 20
#define CORNER3 30
#define CORNER4 20

double matrixA[4096][4096] = {0};
double matrixB[4096][4096] = {0};

int main(int argc, char** argv)
{
    if (argc < NUM_OF_PARAM)
    {
        std::cout << "Number of argument should be 3." << std::endl;
        std::cout << "Arguments: accuracy, grid size, number of iterations" << std::endl;
        return -1;
    }

    double minError = std::pow(10, -std::stoi(argv[1]));
    size_t gridSize = std::stoi(argv[2]);
    size_t numOfIter = std::stoi(argv[3]);

    if(gridSize < 128 || gridSize > 4096 || numOfIter < 0 || numOfIter > 10000000) return -1;

    std::cout << "Min error " << minError << std::endl;
    std::cout << "Grid size: " << gridSize << std::endl;
    std::cout << "Max number of iteration: " << numOfIter << std::endl;

/*    // Allocate two 2D arrays
    double** matrixA = new double* [gridSize];
    double** matrixB = new double* [gridSize];

    //#pragma acc kernels
    for (size_t i = 0; i < gridSize; i++)
    {
        matrixA[i] = new double[gridSize];
        matrixB[i] = new double[gridSize];
        memset(matrixA[i], 0, sizeof(double) * gridSize);
        memset(matrixB[i], 0, sizeof(double) * gridSize);
    }
*/
    matrixA[0][0] = matrixB[0][0] = CORNER1;
    matrixA[0][gridSize - 1] = matrixB[0][gridSize - 1] = CORNER2;
    matrixA[gridSize - 1][0] = matrixB[gridSize - 1][0] = CORNER4;
    matrixA[gridSize - 1][gridSize - 1] = matrixB[gridSize - 1][gridSize - 1] = CORNER3;

    double step1 = 1.0 * (CORNER2 - CORNER1) / gridSize, //From (0, 0) to (gridSize, 0)
           step2 = 1.0 * (CORNER1 - CORNER4) / gridSize, //From (0, 0) to (0, gridSize) 
           step3 = 1.0 * (CORNER2 - CORNER3) / gridSize, //From (gridSize, 0) to (gridSize, gridSize)
           step4 = 1.0 * (CORNER3 - CORNER4) / gridSize; //From (0, gridSize) to (gridSize, gridSize)

    clock_t initBegin = clock();

    // Writing a boundary condition  
    #pragma acc parallel loop vector vector_length(128) gang num_gangs(128)
    for (size_t i = 1; i < gridSize - 1; i++)
    {
       matrixA[0][i] = matrixB[0][i] = CORNER1 + step1 * i;
       matrixA[i][0] = matrixB[i][0] = CORNER1 + step2 * i;
       matrixA[gridSize - 1][i] = matrixB[gridSize - 1][i] = CORNER3 + step4 * i;
       matrixA[i][gridSize - 1] = matrixB[i][gridSize - 1] = CORNER2 + step3 * i;
    }

    clock_t initEnd = clock();
    std::cout << "Initialization time: " << 1.0 * (initEnd - initBegin) / CLOCKS_PER_SEC << std::endl;

    clock_t algBegin = clock();
  
    #pragma acc data create (matrixA[0:gridSize][0:gridSize]) 
    #pragma acc data create (matrixB[0:gridSize][0:gridSize])
    {
    // Main algorithm
    std::cout << "Start" << std::endl;
    double error = 1.0; int iter = 0;
    while (minError < error && iter < numOfIter)
    {
        error = 0.0;
        #pragma acc parallel loop vector vector_length(128) gang num_gangs(128)
        for (size_t i = 1; i < gridSize - 1; i++)
        {
            for (size_t j = 1; j < gridSize - 1; j++)
            {
                matrixB[i][j] = 0.25 * (matrixA[i + 1][j] + matrixA[i - 1][j] + matrixA[i][j - 1] + matrixA[i][j + 1]);
                error = fmax(error, matrixB[i][j] - matrixA[i][j]);
            }
        }
        iter++;

        //double* temp = matrixA;
        //matrixA = matrixB;
        //matrixB = temp;
        #pragma acc parallel loop vector vector_length(1024) gang num_gangs(128)
	for (size_t i = 1; i < gridSize; i++)
	{
	    for(size_t j = 0; j < gridSize; j++)
	    {
	        matrixA[i][j] = matrixB[i][j];
	    }
	}
     }
    
    clock_t algEnd = clock();

    std::cout << "Number of iteration: " << iter << ", error:  " << error << std::endl;

    std::cout << "Time of computation: " << 1.0 * (algEnd - algBegin) / CLOCKS_PER_SEC << std::endl;
    }
    // Free memory
   // for (size_t i = 0; i < gridSize; i++)
    //{
      //  delete[] matrixA[i];
       // delete[] matrixB[i];
   // }
   // delete[] matrixA; delete[] matrixB;

    return 0;
}
