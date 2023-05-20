#include  <memory>

#include "../Inc/FCLayer.cuh"
#include "../Inc/Functions.cuh"

class Network
{
public:
    Network()
    {
        this->fc1 = std::make_unique<Linear>("../task6/weights/weights1", 32 * 32, 16 * 16);
        this->fc2 = std::make_unique<Linear>("../task6/weights/weights2", 16 * 16, 4 * 4);
        this->fc3 = std::make_unique<Linear>("../task6/weights/weights3", 4 * 4, 1);
    }

    ~Network() = default;

    void forward(float* input, float* output)
    {
        float* layerOutputPtr;

        this->fc1->forward(input, layerOutputPtr);
        sigmoid(layerOutputPtr, this->fc1->getOutputSize());

        this->fc2->forward(layerOutputPtr, layerOutputPtr);
        sigmoid(layerOutputPtr, this->fc2->getOutputSize());

        this->fc3->forward(layerOutputPtr, layerOutputPtr);
        sigmoid(layerOutputPtr, this->fc3->getOutputSize());

        output = layerOutputPtr;
    }

private:
    std::unique_ptr<Linear> fc1, fc2, fc3;
};

int main()
{
    Network net();

}