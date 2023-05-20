#pragma once

#include <string>

#include <cublas_v2.h>

class Linear
{
public:
    Linear(cublasHandle_t handle, std::string pathToWeights, size_t in, size_t out);
    ~Linear();

    void forward(float* input, float* output);

    size_t getInputSize();
    size_t getOutputSize();

private:
    float* input;
    float* output;
    float* weights;
    size_t sizeX, sizeY;
    cublasHandle_t cublasHandle;
};