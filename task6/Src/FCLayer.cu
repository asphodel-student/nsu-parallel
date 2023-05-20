#include <iostream>
#include <cstdio>

#include <cuda_runtime.h>

#include "../Inc/FCLayer.cuh"
#include "../Inc/Functions.cuh"
#include "../Inc/Errors.cuh"

Linear(cublasHandle_t handle, std::string pathToWeights, size_t in, size_t out) : 
        cublasHandle(handle), sizeY(out), sizeX(in);
{
    // Allocate memory
    float* tempBufferForWeights;
    GET_CUDA_STATUS(cudaMallocHost(&temp, sizeof(float) * in));
    GET_CUDA_STATUS(cudaMalloc(&this->weights, sizeof(float) * in * out));
    GET_CUDA_STATUS(cudaMalloc(&this->output, sizeof(float) * out));

    // Here we will write weights from 'pathToWeights' file
    FILE* fin = std::fopen(pathToWeights.c_str(), "rb");
    std::fread(tempBufferForWeights, sizeof(float), in * out, fin);

    GET_CUDA_STATUS(cudaMemcpy(
        this->weights, 
        tempBufferForWeights, 
        sizeof(float) * in * out,
        cudaMemcpyHostToDevice));

    // Delete temp buffer 
    GET_CUDA_STATUS(cudaFree(tempBufferForWeights));
    std::fclose(fin);
}

Linear::~Linear()
{
    if (this->output)   GET_CUDA_STATUS(cudaFree(this->output));
    if (this->weigths)  GET_CUDA_STATUS(this->weigths);
}

void Linear::forward(float* input, float* output);
{
    const float alpha = 1.0, beta = 0.0;
    GET_CUBLAS_STATUS(cublasSgemm(
        this->handle, 
        CUBLAS_OP_N, 
        CUBLAS_OP_N, 
        this->sizeY, 
        1, 
        this->sizeX, 
        &alpha,
        this->weights,
        1,
        input,
        1,
        &beta,
        this->output,
        1));

        output = this->output;
}

 size_t Linear::getInputSize()
 {
    return this->sizeX;
 }

 size_t Linear::getOutputSize()
 {
    return this->sizeY;
 }

 

