cmake_minimum_required(VERSION 3.0)
project(task6 CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CUBLAS_LIB cublas)

add_executable(task6 main.cu ../task6/Src/FCLayer.cu ../task6/Src/Functions.cu)
target_link_libraries(task6 ${CUBLAS_LIB})