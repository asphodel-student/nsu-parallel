#pragma once

#include <cublas_v2.h>

static const char *_cudaGetErrorEnum(cublasStatus_t error)
{
    switch (error)
    {
        case CUBLAS_STATUS_SUCCESS:
            return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE:
            return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR:
            return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR";
    }
    return "<unknown>";
}

#define GET_CUDA_STATUS(status) { gpuAssert((status), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t status, const char *file, int line)
{
   if (status != cudaSuccess) 
   {
      fprintf(stderr,"CUDA error: %s %s %d\n", cudaGetErrorString(status), file, line);
      std::exit(status);
   }
}

#define GET_CUBLAS_STATUS(status) { cublasAssert((status), __FILE__, __LINE__); }
inline void cublasAssert(cublasStatus_t status, const char *file, int line)
{
   if (status != CUBLAS_STATUS_SUCCESS) 
   {
      fprintf(stderr,"Cublas error: %s %s %d\n", _cudaGetErrorEnum(status), file, line);
      std::exit(status);
   }
}
