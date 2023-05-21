#include <iostream>
#include <memory>

#include "../Inc/FCLayer.cuh"
#include "../Inc/Functions.cuh"

// Class representing a neural network
class Network
{
public:
    // Constructor that initializes the network 
    Network(cublasHandle_t handle)
    {
        // Create three fully connected layers with specified input and output sizes
        this->fc1 = std::make_unique<Linear>(handle, "../task6/Weights/weights1", 32 * 32, 16 * 16);
        this->fc2 = std::make_unique<Linear>(handle, "../task6/Weights/weights2", 16 * 16, 4 * 4);
        this->fc3 = std::make_unique<Linear>(handle, "../task6/Weights/weights3", 4 * 4, 1);
    }

    ~Network() = default;

    void forward(float* input, float* output)
    {
        float* layerOutputPtr = nullptr;
        
        // Perform forward pass through the network layers
        this->fc1->forward(input, &layerOutputPtr);
        sigmoid(layerOutputPtr, this->fc1->getOutputSize());

        this->fc2->forward(layerOutputPtr, &layerOutputPtr);
        sigmoid(layerOutputPtr, this->fc2->getOutputSize());

        this->fc3->forward(layerOutputPtr, &layerOutputPtr);
        sigmoid(layerOutputPtr, this->fc3->getOutputSize());

        // Copy the final output from device to host memory
        cudaMemcpy(output, layerOutputPtr, sizeof(float), cudaMemcpyDeviceToHost);
    }

private:
    std::unique_ptr<Linear> fc1, fc2, fc3;
};

int main()
{
    // Creating cublas handler
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Here we will read weights from a given file
    FILE* fin = std::fopen("../task6/Weights/input", "rb");
    if (!fin)
    {
        std::exit(-1);
    }

    // Allocating memory for input data on host and device
    float* input, *devInput;
    cudaMallocHost(&input, sizeof(float) * 32 * 32);
    std::fread(input, sizeof(float), 32 * 32, fin);
    
    cudaMalloc(&devInput, sizeof(float) * 32 * 32);
    cudaMemcpy(devInput, input, sizeof(float) * 32 * 32, cudaMemcpyHostToDevice);

    // Creating an instance of our "network"
    Network* net =  new Network(handle);

    // Forward pass
    float out = 0.0;
    net->forward(devInput, &out);

    // See the result
    std::cout << "Output: " << out << std::endl;

    delete net;
    cudaFreeHost(input);
    cudaFree(devInput);

    return 0;
}