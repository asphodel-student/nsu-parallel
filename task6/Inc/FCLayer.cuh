#pragma once

#include <string>

#include <cublas_v2.h>

class Linear
{
public:
    Linear(cublasHandle_t handle, const char* pathToWeights, int in, int out);
    ~Linear();

    void forward(float* input, float** output);

    int getInputSize();
    int getOutputSize();

private:
    float* input;
    float* output;
    float* weights;
    int sizeX, sizeY;
    cublasHandle_t cublasHandle;
};