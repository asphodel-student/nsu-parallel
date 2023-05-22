#include <iostream>
#include <cstdio>

#include <cuda_runtime.h>

#include "../Inc/FCLayer.cuh"
#include "../Inc/Functions.cuh"
#include "../Inc/Errors.cuh"


LinearArguments::LinearArguments(const char* pathToWeights, int inSize, int outSize) : 
        _pathToWeights(pathToWeights), _inSize(inSize), _outSize(outSize)  {}

const char* LinearArguments::getPath() { return this->_pathToWeights; }

int LinearArguments::getInputSize() { return this->_inSize; }

int LinearArguments::getOutputSize() { return this->_outSize; }


Linear::Linear(cublasHandle_t handle, LinearArguments args) : cublasHandle(handle)
{
    this->sizeX = args.getOutputSize();
    this->sizeY = args.getInputSize();

    // Allocate memory
    float* tempBufferForWeights;
    GET_CUDA_STATUS(cudaMallocHost(&tempBufferForWeights, sizeof(float) * this->sizeY * this->sizeX));
    GET_CUDA_STATUS(cudaMalloc(&this->weights, sizeof(float) * this->sizeY * this->sizeX));
    GET_CUDA_STATUS(cudaMalloc(&this->output, sizeof(float) * this->sizeX));

    // Here we will write weights from 'pathToWeights' file
    FILE* fin = std::fopen(args.getPath(), "rb");
    if (!fin)
    {
        std::cout << "There's no such file: " << args.getPath() << std::endl;
        std::exit(-1);
    }

    std::fread(tempBufferForWeights, sizeof(float), this->sizeY * this->sizeX, fin);

    GET_CUDA_STATUS(cudaMemcpy(
        (void*)this->weights, 
        (void*)tempBufferForWeights, 
        sizeof(float) * this->sizeY * this->sizeX,
        cudaMemcpyHostToDevice));

    // Delete temp buffer 
    GET_CUDA_STATUS(cudaFreeHost(tempBufferForWeights));
    std::fclose(fin);
}

Linear::~Linear()
{
    if (this->output)   GET_CUDA_STATUS(cudaFree(this->output));
    if (this->weights)  GET_CUDA_STATUS(cudaFree(this->weights));
}

void Linear::forward(float* input, float** output)
{
    const float alpha = 1.0, beta = 0.0;
    GET_CUBLAS_STATUS(cublasSgemv_v2(
        this->cublasHandle,
        CUBLAS_OP_T,
        this->sizeY, 
        this->sizeX,
        &alpha,
        this->weights,
        this->sizeY,
        input, 
        1,
        &beta,
        this->output,
        1));

    *output = this->output;
}

 int Linear::getInputSize()
 {
    return this->sizeY;
 }

 int Linear::getOutputSize()
 {
    return this->sizeX;
 }

 

