#pragma once

#include <string>

#include <cublas_v2.h>

struct LinearArguments
{
public:
    LinearArguments(const char* pathToWeights, int inSize, int outSize);

    const char* getPath();
    int getInputSize();
    int getOutputSize();
    
private:
    const char* _pathToWeights;
    int _inSize, _outSize;
};

class Linear
{
public:
    Linear(cublasHandle_t handle, LinearArguments args);
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